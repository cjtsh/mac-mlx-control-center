#!/usr/bin/env python3
"""The API & Chat Test panel is engine-agnostic: any OpenAI-compatible target."""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import mlx_helper as h

failures = 0


def check(name, ok, detail=""):
    global failures
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"  <- {detail}"))
    if not ok:
        failures += 1


def main():
    # MLX: no base URL, default port, the server ignores the model name
    url, model = h.openai_target("", port=9999, model="")
    check("MLX default target", url == "http://127.0.0.1:9999/v1/chat/completions", url)
    check("MLX default model id", model == "default_model", model)

    # Ollama: a base URL plus the exact tag (tags have no '/')
    url, model = h.openai_target("http://127.0.0.1:11434/v1/",
                                 model="qwen2.5:72b-instruct-q4_K_M")
    check("Ollama target URL", url == "http://127.0.0.1:11434/v1/chat/completions", url)
    check("Ollama keeps the tag", model == "qwen2.5:72b-instruct-q4_K_M", model)

    # A custom endpoint (vLLM, LM Studio, anything)
    url, model = h.openai_target("http://127.0.0.1:8000/v1", model="mistral-7b")
    check("custom endpoint target", url == "http://127.0.0.1:8000/v1/chat/completions", url)

    # Benchmarking must use the same resolution and pass the model through
    seen = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}}],
                               "usage": {"prompt_tokens": 3, "completion_tokens": 5}}).encode()

    real = h.urllib.request.urlopen

    def fake(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        return _Resp()

    try:
        h.urllib.request.urlopen = fake
        res = h.run_model_benchmark("hi", "qwen2.5:72b-instruct-q4_K_M",
                                    port=11434, base_url="http://127.0.0.1:11434/v1")
        check("benchmark hits the base URL",
              seen.get("url") == "http://127.0.0.1:11434/v1/chat/completions", seen.get("url"))
        check("benchmark sends the Ollama tag",
              seen.get("body", {}).get("model") == "qwen2.5:72b-instruct-q4_K_M",
              seen.get("body", {}).get("model"))
        check("benchmark returns stats", isinstance(res, dict) and res.get("status") != "error", res)
    finally:
        h.urllib.request.urlopen = real

    print(f"\n{'All test-panel checks passed.' if not failures else f'{failures} failure(s)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
