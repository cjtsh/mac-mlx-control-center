"""Unit checks for mlx_stream_guard/stream_guard.py (no model, no fastapi needed).

Run: python3 tests/test_stream_guard.py
"""

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="mlx-guard-test-")
os.environ["MLX_GUARD_LOG"] = os.path.join(TMP, "stream_guard.log")
os.environ["MLX_GUARD_STATE"] = os.path.join(TMP, "stream_guard.json")
os.environ["MLX_GUARD_KEEPALIVE_SECS"] = "1"
os.environ["MLX_GUARD_HEARTBEAT"] = "auto"

sys.path.insert(0, os.path.join(ROOT, "mlx_stream_guard"))
import stream_guard as sg  # noqa: E402

FAILS = []


def check(name, condition, detail=""):
    if condition:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILS.append(name)


def state():
    sg._cfg()["state"] = os.environ["MLX_GUARD_STATE"]
    with open(os.environ["MLX_GUARD_STATE"]) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- 1
def test_heartbeat_and_terminator():
    """Silence between items must be filled, and a missing [DONE] appended once."""

    async def slow():
        yield 'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        await asyncio.sleep(2.4)
        yield 'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'

    async def collect():
        return [item async for item in sg.guard_stream(slow(), "/v1/chat/completions", 1)]

    items = asyncio.run(collect())
    blob = "".join(items)
    check("passes chunks through", "content" in blob and '"a"' in blob and '"b"' in blob)
    check("injects keepalives during silence", blob.count("chat.completion.chunk") >= 2, blob)
    check("ends with exactly one [DONE]", blob.count("[DONE]") == 1, repr(blob[-80:]))
    check("terminator is last", blob.rstrip().endswith("[DONE]"), repr(blob[-40:]))


def test_no_duplicate_done():
    """An upstream that already terminates must not get a second [DONE]."""

    async def done():
        yield 'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
        yield "data: [DONE]\n\n"

    async def collect():
        return [item async for item in sg.guard_stream(done(), "/v1/chat/completions", 2)]

    blob = "".join(asyncio.run(collect()))
    check("keeps a single [DONE] when upstream sent one", blob.count("[DONE]") == 1, blob)


def test_anthropic_uses_ping():
    """The Anthropic endpoint must not receive OpenAI chat chunks."""

    async def slow():
        await asyncio.sleep(1.6)
        yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    async def collect():
        return [item async for item in sg.guard_stream(slow(), "/v1/messages", 3)]

    blob = "".join(asyncio.run(collect()))
    check("anthropic heartbeat is a ping event", "event: ping" in blob, blob)
    check("anthropic stream gets no [DONE]", "[DONE]" not in blob, blob)


def test_unknown_path_is_comment():
    async def slow():
        await asyncio.sleep(1.6)
        yield "data: {}\n\n"

    async def collect():
        return [item async for item in sg.guard_stream(slow(), "/v1/embeddings-stream", 4)]

    blob = "".join(asyncio.run(collect()))
    check("unknown stream style uses an SSE comment", "mlx keepalive" in blob, blob)


def test_upstream_error_propagates():
    async def boom():
        yield "data: {}\n\n"
        raise RuntimeError("upstream exploded")

    async def collect():
        out = []
        try:
            async for item in sg.guard_stream(boom(), "/v1/chat/completions", 5):
                out.append(item)
        except RuntimeError as exc:
            return "".join(out), str(exc)
        return "".join(out), ""

    blob, error = asyncio.run(collect())
    check("upstream exception is not swallowed", "exploded" in error, error)
    check("no fake [DONE] after a failed stream", "[DONE]" not in blob, blob)


# --------------------------------------------------------------------------- 2
class FakeWfile:
    """Stands in for a socket file object."""

    def __init__(self):
        self.bytes_out = bytearray()

    def write(self, data):
        self.bytes_out.extend(data)
        return len(data)

    def flush(self):
        pass


class FakeHandler:
    """Anything the patched handle_one_request needs: a path and a wfile."""

    def __init__(self, path="/v1/chat/completions"):
        self.path = path
        self.wfile = FakeWfile()


def with_patched_handler(behaviour):
    """Install the real hook around a fake request, run it, restore the class."""
    import http.server as http_server

    base = http_server.BaseHTTPRequestHandler
    saved = base.handle_one_request
    had_guard = getattr(base, "_mlx_guarded", False)
    base.handle_one_request = behaviour
    try:
        sg._install_http_server()
        handler = FakeHandler()
        sink = handler.wfile
        base.handle_one_request(handler)
        return handler, bytes(sink.bytes_out)
    finally:
        base.handle_one_request = saved
        if not had_guard:
            try:
                del base._mlx_guarded
            except AttributeError:
                pass


def test_http_server_proxy():
    """mlx-lm writes to raw wfile: header sniffing, keepalive, [DONE], no dupes."""
    header = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n"

    def behaviour(handler):
        handler.wfile.write(header)
        handler.wfile.flush()
        handler.wfile.write(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
        handler.wfile.flush()
        time.sleep(1.4)
        handler.wfile.write(b'data: {"choices":[{"delta":{"content":"there"}}]}\n\n')
        handler.wfile.flush()

    handler, body = with_patched_handler(behaviour)
    check("plain writes pass through", body.startswith(b"HTTP/1.1 200 OK"), body[:40])
    check("writes a keepalive while the model is silent", b"chat.completion.chunk" in body, body)
    check("synthesises [DONE] for mlx-lm streams", body.count(b"[DONE]") == 1, body[-80:])
    check("terminator is last", body.rstrip().endswith(b"[DONE]"), body[-40:])
    check("records the stream in the state file", state()["streams"] >= 1, state())
    check("counts keepalives in the state file", state()["keepalives"] >= 1, state())
    check("nothing is written through the proxy afterwards", len(body) == len(handler.wfile._real.bytes_out))


def test_upstream_done_not_duplicated():
    def behaviour(handler):
        handler.wfile.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
        handler.wfile.write(b'data: {"choices":[{"delta":{}}]}\n\n')
        handler.wfile.write(b"data: [DONE]\n\n")

    _, body = with_patched_handler(behaviour)
    check("upstream [DONE] is not duplicated", body.count(b"[DONE]") == 1, body[-60:])


def test_non_sse_untouched():
    payload = b'{"choices":[{"message":{"content":"hi"}}]}'

    def behaviour(handler):
        handler.wfile.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n")
        handler.wfile.write(payload)

    _, body = with_patched_handler(behaviour)
    check("json responses are not modified", body.endswith(payload), body[-60:])
    check("no keepalive for a json response", b"chat.completion.chunk" not in body, body)


def test_zero_chunk_stream_gets_terminator():
    """A stream that died before producing anything still terminates cleanly."""

    def behaviour(handler):
        handler.wfile.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")

    _, body = with_patched_handler(behaviour)
    check(
        "empty SSE stream is closed (no bytes after headers)",
        body.endswith(b"\r\n\r\n"),
        body,
    )
    check("empty stream is counted", state()["empty_streams"] >= 1, state())


# --------------------------------------------------------------------------- 9
def test_max_completion_tokens_bridge():
    """A local OpenAI client sends max_completion_tokens; mlx-vlm only reads max_tokens."""
    raw = json.dumps(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 8192}
    ).encode()
    fixed = sg._rename_max_tokens(raw)
    payload = json.loads(fixed)
    check("bridges max_completion_tokens onto max_tokens", payload.get("max_tokens") == 8192, payload)
    check("keeps the original field", payload.get("max_completion_tokens") == 8192, payload)

    explicit = json.dumps({"max_tokens": 512, "max_completion_tokens": 8192}).encode()
    check(
        "an explicit max_tokens wins",
        json.loads(sg._rename_max_tokens(explicit)).get("max_tokens") == 512,
    )
    check("non-json bodies pass through", sg._rename_max_tokens(b"not json") == b"not json")
    check("missing field passes through", sg._rename_max_tokens(b'{"a":1}') == b'{"a":1}')
    for bad in (True, "8192", 0, -5, 8.5, None):
        blob = json.dumps({"max_completion_tokens": bad}).encode()
        check("ignores max_completion_tokens=%r" % (bad,), sg._rename_max_tokens(blob) == blob)

    # The ASGI wrapper must present the rewritten body (and length) to the app.
    saved = sg._cfg().get("fix_max_tokens")
    sg._cfg()["fix_max_tokens"] = True
    try:
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"content-length", str(len(raw)).encode())],
        }
        queue = [
            {"type": "http.request", "body": raw[:10], "more_body": True},
            {"type": "http.request", "body": raw[10:], "more_body": False},
        ]

        async def receive():
            return queue.pop(0) if queue else {"type": "http.disconnect"}

        before = state().get("max_tokens_fixed", 0)
        message = asyncio.run(sg._rewriting_receive(scope, receive)())
        check(
            "multi-chunk body arrives bridged and re-framed",
            message.get("body") == sg._rename_max_tokens(raw) and message.get("more_body") is False,
            message,
        )
        headers = dict(scope["headers"])
        check(
            "content-length matches the new body",
            headers.get(b"content-length") == str(len(message["body"])).encode(),
            scope["headers"],
        )
        check("the bridge is counted", state().get("max_tokens_fixed", 0) == before + 1, state())
    finally:
        sg._cfg()["fix_max_tokens"] = saved


if __name__ == "__main__":
    print("stream_guard unit checks")
    test_heartbeat_and_terminator()
    test_no_duplicate_done()
    test_anthropic_uses_ping()
    test_unknown_path_is_comment()
    test_upstream_error_propagates()
    test_http_server_proxy()
    test_non_sse_untouched()
    test_zero_chunk_stream_gets_terminator()
    test_max_completion_tokens_bridge()
    print("\n%d failure(s)" % len(FAILS))
    sys.exit(1 if FAILS else 0)
