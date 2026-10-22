"""In-process stream guard for the MLX model servers.

Why this exists
---------------
Both Apple servers are silent while they work:

* ``mlx_vlm.server`` suppresses tool-call markup from the content delta and only
  flushes a tool call once the whole call is written, so a 3.7k-token tool call
  produces *no bytes at all* on the wire for ~100 s (observed in
  ``~/.mlx_logs/sh0wie_..._9999.log``: 3736 tokens / 96.8 s, finish_reason=tool_calls).
* ``mlx_lm.server`` writes nothing while it prefills, and nothing at all while the
  text state machine is inside a tool call (``server.py``:1644-1656).
* Neither server ever writes an SSE keepalive, and ``mlx_lm.server`` can close a
  stream without ``data: [DONE]`` if generation raises after the headers are sent.

A harness that watches for bytes on the wire therefore sees a dead connection for
tens of seconds, and a stream that ends without a terminal event looks like a
*finished* turn — which is exactly the "the model just stopped, say hello again"
symptom.

What it does
------------
This module is imported by ``sitecustomize.py`` inside the model-server process
(``PYTHONPATH`` + ``MLX_STREAM_GUARD=1``, set by the launcher) and wraps the SSE
response at the framework boundary:

* writes a keepalive every ``MLX_GUARD_KEEPALIVE_SECS`` (default 5 s) of silence,
* guarantees exactly one ``data: [DONE]`` for OpenAI chat streams,
* records first byte, longest gap, chunk count and terminator status for every
  stream in ``~/.mlx_logs/stream_guard.log`` plus a rolling summary in
  ``~/.mlx_pids/stream_guard.json``.

Everything is defensive: if a patch cannot be applied the servers behave exactly
as before, and ``MLX_STREAM_GUARD=0`` disables the guard completely.
"""

import asyncio
import contextvars
import json
import os
import sys
import threading
import time
import traceback

_DONE = "[DONE]"
_SSE = "text/event-stream"

_PATH = contextvars.ContextVar("mlx_guard_path", default="")

_CFG = {}
_INSTALLED = False
_INSTALL_ERRORS = []
_STATE_LOCK = threading.Lock()
_STATE = {
    "installed": False,
    "engine_hooks": [],
    "pid": os.getpid(),
    "started": time.time(),
    "streams": 0,
    "keepalives": 0,
    "synthetic_done": 0,
    "empty_streams": 0,
    "errors": 0,
    "max_tokens_fixed": 0,
    "last": None,
}


def _env_float(name, default):
    try:
        return max(1.0, float(os.environ.get(name, "").strip()))
    except Exception:
        return default


def _cfg():
    if not _CFG:
        _CFG["interval"] = _env_float("MLX_GUARD_KEEPALIVE_SECS", 5.0)
        _CFG["mode"] = (os.environ.get("MLX_GUARD_HEARTBEAT") or "auto").strip().lower()
        _CFG["fix_max_tokens"] = _env_flag("MLX_GUARD_FIX_MAX_TOKENS", True)
        _CFG["log"] = os.path.expanduser(
            os.environ.get("MLX_GUARD_LOG") or "~/.mlx_logs/stream_guard.log"
        )
        _CFG["state"] = os.path.expanduser(
            os.environ.get("MLX_GUARD_STATE") or "~/.mlx_pids/stream_guard.json"
        )
        _CFG["max_log"] = 1024 * 1024
    return _CFG


# --------------------------------------------------------------------------- io


def _write_log(line):
    """Append one line to the guard log, rotating it once it passes 1 MB."""
    cfg = _cfg()
    try:
        path = cfg["log"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            if os.path.getsize(path) > cfg["max_log"]:
                os.replace(path, path + ".1")
        except OSError:
            pass
        with open(path, "a") as fh:
            fh.write(time.strftime("%Y-%m-%d %H:%M:%S") + " " + line + "\n")
    except Exception:
        pass


def _write_state():
    """Publish the rolling counters so the dashboard can show them."""
    cfg = _cfg()
    try:
        path = cfg["state"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(_STATE, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        pass


def _record(stats):
    with _STATE_LOCK:
        _STATE["streams"] += 1
        _STATE["keepalives"] += stats.get("keepalives", 0)
        if stats.get("synthetic_done"):
            _STATE["synthetic_done"] += 1
        if stats.get("chunks", 0) == 0:
            _STATE["empty_streams"] += 1
        if stats.get("error"):
            _STATE["errors"] += 1
        _STATE["last"] = stats
        _STATE["updated"] = time.time()
        _write_state()
    gap = stats.get("max_gap") or 0.0
    _write_log(
        "stream id=%s path=%s chunks=%d bytes=%d first=%.2fs max_gap=%.1fs "
        "keepalives=%d done=%s synthetic_done=%s total=%.2fs error=%s"
        % (
            stats.get("id"),
            stats.get("path"),
            stats.get("chunks", 0),
            stats.get("bytes", 0),
            stats.get("first_byte") or 0.0,
            gap,
            stats.get("keepalives", 0),
            "yes" if stats.get("done") else "no",
            "yes" if stats.get("synthetic_done") else "no",
            stats.get("total") or 0.0,
            (stats.get("error") or "")[:200] or "-",
        )
    )


def _note_error(where, exc):
    _INSTALL_ERRORS.append("%s: %r" % (where, exc))
    _write_log("install-error %s: %r" % (where, exc))
    try:
        _STATE["install_errors"] = _INSTALL_ERRORS[-5:]
    except Exception:
        pass


# ------------------------------------------------------------------ heartbeats


def _heartbeat(path, mode):
    """The bytes to write during silence, or b"" to write none.

    An empty ``delta`` chunk is used for chat completions because a client-side
    idle watchdog only resets when an event it understands arrives; a bare SSE
    comment (valid, but ignored by most parsers) would keep the socket alive and
    still let the watchdog fire. Anywhere else a comment is used, because it can
    never confuse a protocol-specific parser.
    """
    if mode == "off":
        return b""
    streaming_style = "comment"
    if mode in ("auto", "chunk"):
        if path.endswith("/messages") or "/messages" in path:
            streaming_style = "anthropic"
        elif not path or "/chat/completions" in path:
            streaming_style = "openai" if path else "comment"
        if mode == "chunk" and streaming_style == "comment":
            streaming_style = "openai"
    if streaming_style == "openai":
        chunk = {
            "id": "chatcmpl-mlx-guard",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "mlx-guard",
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        }
        return ("data: " + json.dumps(chunk) + "\n\n").encode()
    if streaming_style == "anthropic":
        return b'event: ping\ndata: {"type": "ping"}\n\n'
    return b": mlx keepalive\n\n"


def _terminator(path):
    """The terminal event to synthesise when the server forgot to send one."""
    if not path:
        # Protocol unknown (no request path was captured): never invent a
        # terminator that a strict parser might reject.
        return b""
    if path.endswith("/messages") or "/messages" in path:
        return b""
    return ("data: " + _DONE + "\n\n").encode()


# ------------------------------------------------------- starlette / fastapi


async def guard_stream(inner, path="", stream_id=0):
    """Async generator that passes ``inner`` through with keepalives and a finish.

    The upstream generator is pumped by its own task and read through a queue on
    purpose: cancelling ``__anext__`` (what ``asyncio.wait_for`` would do on a
    timeout) would inject ``CancelledError`` into the server's token loop and can
    abort a running generation, so the timer path must never touch ``inner``.
    """
    cfg = _cfg()
    interval = cfg["interval"]
    mode = cfg["mode"]
    hb = _heartbeat(path, mode)
    end = _terminator(path)
    started = time.monotonic()
    stats = {
        "id": stream_id,
        "path": path,
        "chunks": 0,
        "bytes": 0,
        "first_byte": None,
        "max_gap": 0.0,
        "keepalives": 0,
        "done": False,
        "synthetic_done": False,
        "error": "",
    }
    queue = asyncio.Queue(maxsize=256)

    async def pump():
        try:
            async for item in inner:
                await queue.put(("item", item))
        except BaseException as exc:  # noqa: BLE001 - reported, then re-raised
            await queue.put(("error", exc))
        finally:
            await queue.put(("end", None))

    task = asyncio.ensure_future(pump())
    last = time.monotonic()
    # ASGI SSE generators yield str; keep the injected events the same type as the
    # upstream so the client never sees a mixed str/bytes stream.
    text_mode = True

    def typed(blob):
        return blob.decode("utf-8") if text_mode else blob

    try:
        while True:
            try:
                kind, value = await asyncio.wait_for(queue.get(), interval)
            except asyncio.TimeoutError:
                if hb:
                    now = time.monotonic()
                    stats["max_gap"] = max(stats["max_gap"], now - last)
                    stats["keepalives"] += 1
                    last = now
                    yield typed(hb)
                continue
            if kind == "item":
                now = time.monotonic()
                stats["max_gap"] = max(stats["max_gap"], now - last)
                last = now
                if stats["first_byte"] is None:
                    stats["first_byte"] = now - started
                    text_mode = not isinstance(value, (bytes, bytearray))
                stats["chunks"] += 1
                if isinstance(value, str):
                    stats["bytes"] += len(value)
                    if _DONE in value:
                        stats["done"] = True
                elif isinstance(value, (bytes, bytearray)):
                    stats["bytes"] += len(value)
                    if _DONE.encode() in bytes(value):
                        stats["done"] = True
                yield value
            elif kind == "error":
                stats["error"] = repr(value)
                raise value
            else:
                break
        if end and not stats["done"]:
            stats["synthetic_done"] = True
            yield typed(end)
    finally:
        if not task.done():
            task.cancel()
        stats["total"] = time.monotonic() - started
        _record(stats)


def _install_starlette():
    """Wrap StreamingResponse so FastAPI/Starlette SSE gets the guard."""
    from starlette.responses import StreamingResponse

    if getattr(StreamingResponse, "_mlx_guarded", False):
        return True

    orig_init = StreamingResponse.__init__
    counter = {"n": 0}

    def __init__(self, content, *args, **kwargs):
        try:
            media = kwargs.get("media_type")
            if media is None and len(args) >= 3:
                media = args[2]
            if (
                media
                and _SSE in str(media)
                and hasattr(content, "__aiter__")
                and not isinstance(content, (str, bytes, bytearray, dict, list))
            ):
                counter["n"] += 1
                content = guard_stream(content, _PATH.get(), counter["n"])
        except Exception as exc:  # noqa: BLE001 - never break the response
            _note_error("starlette.init", exc)
        orig_init(self, content, *args, **kwargs)

    StreamingResponse.__init__ = __init__
    StreamingResponse._mlx_guarded = True

    # Remember the request path so a heartbeat can match the wire protocol.
    for mod_name, cls_name in (
        ("fastapi", "FastAPI"),
        ("starlette.applications", "Starlette"),
    ):
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
        except Exception:
            continue
        orig_call = cls.__call__
        if getattr(orig_call, "_mlx_guarded", False):
            continue

        async def __call__(self, scope, receive, send, _orig=orig_call):
            try:
                if scope.get("type") == "http":
                    _PATH.set(scope.get("path") or "")
                    if (
                        _cfg().get("fix_max_tokens")
                        and (scope.get("method") or "").upper() == "POST"
                    ):
                        receive = _rewriting_receive(scope, receive)
            except Exception as exc:  # noqa: BLE001 - never break the request
                _note_error("starlette.call", exc)
            return await _orig(self, scope, receive, send)

        __call__._mlx_guarded = True
        cls.__call__ = __call__
    return True


# ------------------------------------------------- request body (max_tokens)


def _env_flag(name, default):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _rename_max_tokens(raw):
    """Copy max_completion_tokens onto max_tokens, which is all mlx-vlm reads.

    The OpenAI clients used by Pi and DSH only send ``max_tokens`` to an
    allowlist of hosted providers; against a local base URL they send
    ``max_completion_tokens`` instead.  mlx-vlm's request normaliser looks for
    ``max_tokens`` / ``max_output_tokens`` and otherwise falls back to the
    --max-tokens launch flag, so the harness's own limit is silently replaced
    by the control center's default.  Returning the original bytes on any doubt
    keeps this a no-op when the request is already well formed.
    """
    if not raw or b"max_completion_tokens" not in raw:
        return raw
    try:
        payload = json.loads(raw)
    except Exception:
        return raw
    if not isinstance(payload, dict) or payload.get("max_tokens") is not None:
        return raw
    value = payload.get("max_completion_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return raw
    payload["max_tokens"] = value
    try:
        return json.dumps(payload).encode("utf-8")
    except Exception:
        return raw


def _fix_content_length(scope, body):
    headers = scope.get("headers")
    if not headers:
        return
    fixed = []
    for key, value in headers:
        if key.lower() == b"content-length":
            fixed.append((key, str(len(body)).encode("ascii")))
        else:
            fixed.append((key, value))
    scope["headers"] = fixed


def _count_max_tokens_fix(value):
    try:
        with _STATE_LOCK:
            _STATE["max_tokens_fixed"] = _STATE.get("max_tokens_fixed", 0) + 1
            _STATE["last_max_tokens"] = value
            _write_state()
        _write_log("max_tokens bridged from max_completion_tokens=%s" % (value,))
    except Exception:
        pass


def _rewriting_receive(scope, receive):
    """Wrap ASGI receive so the JSON body can be adjusted before the app parses it."""
    state = {"done": False}

    async def wrapped():
        if state["done"]:
            return await receive()
        state["done"] = True
        message = await receive()
        if not isinstance(message, dict) or message.get("type") != "http.request":
            return message
        chunks = [message.get("body") or b""]
        while message.get("more_body"):
            message = await receive()
            if not isinstance(message, dict) or message.get("type") != "http.request":
                break
            chunks.append(message.get("body") or b"")
        raw = b"".join(chunks)
        try:
            fixed = _rename_max_tokens(raw)
        except Exception as exc:  # noqa: BLE001
            _note_error("request.body", exc)
            fixed = raw
        if fixed is not raw and fixed != raw:
            try:
                _fix_content_length(scope, fixed)
                payload = json.loads(fixed)
                _count_max_tokens_fix(payload.get("max_tokens"))
            except Exception:
                pass
        return {"type": "http.request", "body": fixed, "more_body": False}

    return wrapped


# --------------------------------------------------------- http.server (mlx-lm)


class _WriterProxy:
    """A wfile stand-in that notices an SSE response and keeps it alive.

    ``mlx_lm.server`` writes straight to ``self.wfile``; there is no framework
    object to wrap, so the guard watches the bytes and writes its keepalives
    through the same lock as the server to avoid interleaving.
    """

    def __init__(self, real, state):
        self._real = real
        self._state = state

    def write(self, data):
        state = self._state
        with state.lock:
            written = self._real.write(data)
            try:
                state.observe(data)
            except Exception:
                pass
        return written

    def flush(self):
        with self._state.lock:
            return self._real.flush()

    def __getattr__(self, name):
        return getattr(self._real, name)


class _HandlerState:
    def __init__(self, handler):
        self.handler = handler
        self.lock = threading.RLock()
        self.header = b""
        self.sse = False
        self.done = False
        self.chunks = 0
        self.bytes = 0
        self.keepalives = 0
        self.started = time.monotonic()
        self.first_byte = None
        self.max_gap = 0.0
        self.last_write = time.monotonic()
        self.path = ""
        self.closed = False
        self.stop = threading.Event()
        self.thread = None

    def observe(self, data):
        if not isinstance(data, (bytes, bytearray)):
            return
        data = bytes(data)
        if not self.sse:
            self.header += data
            head, sep, rest = self.header.partition(b"\r\n\r\n")
            if not sep:
                return
            lowered = head.lower()
            if _SSE.encode() not in lowered:
                self.header = b""
                return
            self.sse = True
            self.header = b""
            data = rest
            try:
                request = getattr(self.handler, "path", "") or ""
                self.path = request.split("?")[0]
            except Exception:
                self.path = ""
            self._start_heartbeat()
            if not data:
                return
        self.last_write = time.monotonic()
        if self.first_byte is None:
            self.first_byte = self.last_write - self.started
        if data:
            self.chunks += 1
            self.bytes += len(data)
            if _DONE.encode() in data:
                self.done = True

    def _start_heartbeat(self):
        cfg = _cfg()
        hb = _heartbeat(self.path, cfg["mode"])
        if not hb:
            return
        interval = cfg["interval"]

        def run():
            while not self.stop.wait(min(1.0, interval)):
                if self.closed:
                    return
                if time.monotonic() - self.last_write < interval:
                    continue
                with self.lock:
                    if self.closed:
                        return
                    gap = time.monotonic() - self.last_write
                    try:
                        self.handler.wfile_guard_write(hb)
                    except Exception:
                        return
                    self.keepalives += 1
                    self.max_gap = max(self.max_gap, gap)
                    self.last_write = time.monotonic()

        self.thread = threading.Thread(target=run, name="mlx-stream-guard", daemon=True)
        self.thread.start()

    def finish(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.stop.set()
            if not self.sse:
                return
            end = _terminator(self.path)
            synthetic = False
            if end and not self.done and self.chunks:
                try:
                    self.handler.wfile_guard_write(end)
                    synthetic = True
                except Exception:
                    pass
            stats = {
                "id": id(self),
                "path": self.path,
                "chunks": self.chunks,
                "bytes": self.bytes,
                "first_byte": self.first_byte,
                "max_gap": self.max_gap,
                "keepalives": self.keepalives,
                "done": self.done,
                "synthetic_done": synthetic,
                "total": time.monotonic() - self.started,
                "error": "",
            }
        _record(stats)


def _install_http_server():
    """Wrap every BaseHTTPRequestHandler request so mlx-lm streams get the guard."""
    import http.server as http_server

    base = http_server.BaseHTTPRequestHandler
    if getattr(base, "_mlx_guarded", False):
        return True
    orig = base.handle_one_request

    def handle_one_request(self, _orig=orig):
        state = _HandlerState(self)
        real_wfile = None
        try:
            real_wfile = self.wfile
            self.wfile = _WriterProxy(real_wfile, state)

            def guard_write(data, _real=real_wfile, _state=state):
                with _state.lock:
                    written = _real.write(data)
                    try:
                        _state.observe(data)
                    except Exception:
                        pass
                return written

            self.wfile_guard_write = guard_write
        except Exception as exc:  # noqa: BLE001
            _note_error("http.server.setup", exc)
        try:
            return _orig(self)
        finally:
            try:
                state.finish()
            except Exception as exc:  # noqa: BLE001
                _note_error("http.server.finish", exc)

    base.handle_one_request = handle_one_request
    base._mlx_guarded = True
    return True


# ---------------------------------------------------------------------- entry


def install():
    """Install whichever hooks this interpreter can use. Never raises."""
    global _INSTALLED
    if _INSTALLED:
        return True
    if os.environ.get("MLX_STREAM_GUARD", "1").strip() == "0":
        _write_log("disabled by MLX_STREAM_GUARD=0")
        return False
    hooks = []
    try:
        if _install_starlette():
            hooks.append("starlette")
    except Exception as exc:  # noqa: BLE001
        _note_error("install.starlette", exc)
    try:
        if _install_http_server():
            hooks.append("http.server")
    except Exception as exc:  # noqa: BLE001
        _note_error("install.http_server", exc)
    _INSTALLED = bool(hooks)
    _STATE["installed"] = _INSTALLED
    _STATE["engine_hooks"] = hooks
    _STATE["python"] = sys.executable
    _STATE["interval"] = _cfg()["interval"]
    _STATE["mode"] = _cfg()["mode"]
    with _STATE_LOCK:
        _write_state()
    _write_log(
        "installed hooks=%s pid=%d interval=%.1fs mode=%s"
        % (",".join(hooks) or "-", os.getpid(), _cfg()["interval"], _cfg()["mode"])
    )
    return _INSTALLED
