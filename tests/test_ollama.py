#!/usr/bin/env python3
"""Ollama engine wiring: GGUF discovery, harness sync, import validation.

No daemon, no network and no model needed — every call is pointed at a fake HTTP
layer or a temporary config file.
Run: python3 tests/test_ollama.py
"""
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import mlx_helper as h  # noqa: E402

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")


PATCH = """\
- id: llm-pi-ai
  name: "@deepseek-ai/dsh-llm-pi-ai"
  config:
    providers:
      mlx-local:
        displayName: MLX (this Mac)
        api: openai-completions
        baseURL: http://127.0.0.1:9999/v1
        models:
          - id: mlx-community/Qwen2.5-1.5B-Instruct-4bit
            name: Qwen2.5 1.5B (MLX)
            contextWindow: 32768
        apiKeyEnv: MLX_LOCAL_API_KEY
- id: agent-default-model
  name: "@deepseek-ai/dsh-agent-default-model"
  config:
    provider: deepseek-official
    model: deepseek-flash
"""

FAKE_MODELS = [
    {"id": "qwen3-coder:30b", "contextWindow": 262144, "maxTokens": 8192,
     "size_gb": 17.28, "family": "qwen3moe", "is_vision": False},
    {"id": "qwen2.5vl:7b", "contextWindow": 128000, "maxTokens": 8192,
     "size_gb": 5.56, "family": "qwen25vl", "is_vision": True},
]


def run():
    tmp = tempfile.mkdtemp(prefix="ollama_test_")
    patch = os.path.join(tmp, "cordis.patch.yml")
    with open(patch, "w") as f:
        f.write(PATCH)
    real_models = h.ollama_harness_models
    real_expanduser = os.path.expanduser
    real_dsh_models = h.ollama_harness_models
    try:
        h.ollama_harness_models = lambda port=None: list(FAKE_MODELS)

        # ---- GGUF quant labels come out of plain GGUF names ----------------
        check("Q4_K_M -> 4-bit", h._detect_quant_label("Qwen2.5-7B-Instruct-Q4_K_M.gguf") == "4-bit")
        check("IQ3_XS -> 3-bit", h._detect_quant_label("model-iq3_xs.gguf") == "3-bit")
        check("Q8_0 -> 8-bit", h._detect_quant_label("x/Q8_0.gguf") == "8-bit")
        check("repo name still works", h._detect_quant_label("mlx-community/foo-4bit") == "4-bit")

        # ---- a single file is fetched with --include, not the whole repo ---
        cmd = h.get_download_cmd("bartowski/Qwen2.5-7B-GGUF",
                                 include=["Qwen2.5-7B-Q4_K_M.gguf"])
        check("include pattern reaches hf download",
              cmd[-2:] == ["--include", "Qwen2.5-7B-Q4_K_M.gguf"], cmd)
        cmd2 = h.get_download_cmd("org/repo", subfolder="sub", include=["a.gguf", "b.gguf"])
        check("subfolder and file patterns combine", cmd2.count("--include") == 3, cmd2)

        # ---- GGUF listing parses the HF tree endpoint ----------------------
        payload = [
            {"type": "file", "path": "model-Q4_K_M.gguf", "size": 4 * 1024 ** 3},
            {"type": "file", "path": "model-Q8_0.gguf", "size": 8 * 1024 ** 3},
            {"type": "file", "path": "README.md", "size": 100},
            {"type": "directory", "path": "old"},
        ]

        class _Resp:
            def __init__(self, body):
                self._body = body

            def read(self):
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        real_urlopen = h.urllib.request.urlopen
        h.urllib.request.urlopen = lambda *a, **k: _Resp(json.dumps(payload).encode())
        try:
            files = h.hf_repo_gguf_files("org/repo")
        finally:
            h.urllib.request.urlopen = real_urlopen
        check("only .gguf files are listed", [f["path"] for f in files] ==
              ["model-Q4_K_M.gguf", "model-Q8_0.gguf"], files)
        check("sizes convert to GB", files[0]["size_gb"] == 4.0, files[0])
        check("quant detected in listing", [f["quant"] for f in files] == ["4-bit", "8-bit"], files)

        # ---- the free-space guard counts the selected quant, not the repo --
        h._REPO_SIZE_CACHE.clear()
        real_urlopen = h.urllib.request.urlopen
        h.urllib.request.urlopen = lambda *a, **k: _Resp(json.dumps(payload).encode())
        try:
            whole = h.get_repo_files_size("org/repo")
            one = h.get_repo_files_size("org/repo", include=["model-Q4_K_M.gguf"])
            other = h.get_repo_files_size("org/repo", include=["nomatch/*"])
        finally:
            h.urllib.request.urlopen = real_urlopen
        check("whole-repo size sums every file", whole == 12 * 1024 ** 3 + 100, whole)
        check("include narrows the free-space guard", one == 4 * 1024 ** 3, one)
        check("include with no match is zero, not the repo", other == 0, other)

        # ---- DSH gets a second provider, and MLX is left alone -------------
        res = h.sync_dsh_ollama(patch_path=patch)
        check("provider inserted", res.get("ok") and res.get("action") == "inserted", res)
        text = open(patch).read()
        check("ollama provider present", "      ollama-local:" in text)
        check("ollama baseURL", "baseURL: http://127.0.0.1:11434/v1" in text)
        check("mlx baseURL untouched", "baseURL: http://127.0.0.1:9999/v1" in text)
        check("models written", "- id: qwen3-coder:30b" in text and "- id: qwen2.5vl:7b" in text)
        check("vision model keeps image input", "input: [ text, image ]" in text)
        check("apiKeyEnv present", "apiKeyEnv: OLLAMA_LOCAL_API_KEY" in text)
        # the agent-default-model block must still follow the providers mapping
        check("next top-level block intact", "\n- id: agent-default-model" in text)

        before = open(patch).read()
        res2 = h.sync_dsh_ollama(patch_path=patch)
        check("second run is a no-op", res2.get("ok") and res2.get("changed") is False, res2)
        check("no duplicate provider", open(patch).read().count("      ollama-local:") == 1)
        check("file unchanged on repeat", open(patch).read() == before)

        # models changing rewrites the block in place
        h.ollama_harness_models = lambda port=None: [FAKE_MODELS[0]]
        res3 = h.sync_dsh_ollama(patch_path=patch)
        text3 = open(patch).read()
        check("removed model leaves the list", "qwen2.5vl:7b" not in text3, text3)
        check("remaining model kept", "qwen3-coder:30b" in text3)
        check("still exactly one provider", text3.count("      ollama-local:") == 1)
        check("rewrite reported as changed", res3.get("changed") is True, res3)

        # no models at all leaves a valid empty list rather than vanishing
        h.ollama_harness_models = lambda port=None: []
        h.sync_dsh_ollama(patch_path=patch)
        check("empty store writes models: []", "models: []" in open(patch).read())

        # ---- Pi Coder provider, with MyMac untouched ----------------------
        pi_dir = os.path.join(tmp, ".pi", "agent")
        os.makedirs(pi_dir, exist_ok=True)
        pi_file = os.path.join(pi_dir, "models.json")
        with open(pi_file, "w") as f:
            json.dump({"providers": {"MyMac": {"baseUrl": "http://127.0.0.1:9999/v1"}}}, f)

        def fake_expand(path):
            if path.startswith("~/.pi/"):
                return path.replace("~", tmp, 1)
            if path.startswith("~/.config/opencode") or path == "~/opencode.jsonc":
                return os.path.join(tmp, "missing-opencode.jsonc")
            return real_expanduser(path)

        h.os.path.expanduser = fake_expand
        try:
            h.ollama_harness_models = lambda port=None: list(FAKE_MODELS)
            done = h.sync_pi_ollama()
        finally:
            h.os.path.expanduser = real_expanduser
        check("pi provider written", "pi" in done, done)
        md = json.load(open(pi_file))
        check("MyMac provider kept", md["providers"]["MyMac"]["baseUrl"] == "http://127.0.0.1:9999/v1")
        check("Ollama provider added", "Ollama" in md["providers"], list(md["providers"]))
        check("Ollama baseUrl", md["providers"]["Ollama"]["baseUrl"] == "http://127.0.0.1:11434/v1")
        ids = [m["id"] for m in md["providers"]["Ollama"]["models"]]
        check("pi models listed", ids == ["qwen3-coder:30b", "qwen2.5vl:7b"], ids)
        vision = [m for m in md["providers"]["Ollama"]["models"] if m["id"] == "qwen2.5vl:7b"][0]
        check("pi vision input", vision["input"] == ["text", "image"], vision)

        # ---- import validation --------------------------------------------
        res = h.ollama_import("", None)
        check("import rejects an empty path", res.get("ok") is False, res)
        res = h.ollama_import(os.path.join(tmp, "nope.gguf"))
        check("import rejects a missing file", res.get("ok") is False, res)
        not_gguf = os.path.join(tmp, "model.safetensors")
        open(not_gguf, "w").write("x")
        res = h.ollama_import(not_gguf)
        check("import rejects a non-GGUF file", res.get("ok") is False, res)

        # ---- status is honest when the daemon is absent --------------------
        real_req, real_pid = h._ollama_http, h.ollama_pid
        h._ollama_http = lambda *a, **k: None
        h.ollama_pid = lambda port=None: None
        try:
            st = h.ollama_status(with_models=True)
        finally:
            h._ollama_http, h.ollama_pid = real_req, real_pid
        check("status reports not running", st.get("running") is False, st)
        check("status still names the store", bool(st.get("store")), st)
    finally:
        h.ollama_harness_models = real_models
        os.path.expanduser = real_expanduser

    passed = sum(1 for _, ok, _ in CHECKS if ok)
    print(f"\n{passed}/{len(CHECKS)} checks passed")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(run())
