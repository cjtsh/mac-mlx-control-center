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

        # ---- GGUFs that ship no chat template get the family's real one ----
        check("qwen2 template bundled", "if .Messages" in h.ollama_chat_template_for("qwen2"))
        check("family lookup normalises dots", h.ollama_chat_template_for("qwen2.5") ==
              h.ollama_chat_template_for("qwen2"))
        check("unknown family gets no template", h.ollama_chat_template_for("llama") == "")

        # ---- an already-imported GGUF must never get a second copy ----
        check("tag normalisation ignores ':' vs '-'",
              h._normalise_tag("qwen2.5:72b-instruct-q4_K_M") ==
              h._normalise_tag("qwen2.5-72b-instruct-q4_k_m:latest"))
        real_models = h.ollama_models
        real_show = h._ollama_http
        real_digest = h.gguf_digest
        try:
            h.ollama_models = lambda port=None: [{"id": "qwen2.5:72b-instruct-q4_K_M"}]
            h._ollama_http = lambda *a, **k: {"details": {"parent_model": "/x/blobs/sha256-abc"}}
            h.gguf_digest = lambda path: "abc"
            check("existing tag found by name",
                  h.ollama_tag_for_gguf("/models/Qwen2.5-72B-Instruct-Q4_K_M.gguf") ==
                  "qwen2.5:72b-instruct-q4_K_M")
            h.ollama_models = lambda port=None: [{"id": "something-else:latest"}]
            h._ollama_http = lambda *a, **k: {"details": {"parent_model": "/x/blobs/sha256-abc"}}
            check("existing tag found by digest",
                  h.ollama_tag_for_gguf("/models/other-name.gguf") == "something-else:latest")
            h.ollama_models = lambda port=None: []
            check("no tag when store is empty", h.ollama_tag_for_gguf("/models/x.gguf") == "")
            h.ollama_models = lambda port=None: [{"id": "qwen2.5:72b-instruct-q4_K_M"}]
            h.gguf_digest = lambda path: "abc"
            h._ollama_http = lambda *a, **k: {"details": {"parent_model": "/x/blobs/sha256-abc"}}
            import tempfile as _tf
            fh = _tf.NamedTemporaryFile(suffix=".gguf", delete=False); fh.write(b"x"); fh.close()
            res = h.ollama_import(fh.name)
            check("import reuses an existing tag", res.get("reused") is True and
                  res.get("tag") == "qwen2.5:72b-instruct-q4_K_M", res)
            os.unlink(fh.name)
        finally:
            h.ollama_models, h._ollama_http, h.gguf_digest = real_models, real_show, real_digest
        real_run = h.subprocess.run

        class _Run:
            def __init__(self, out, rc=0):
                self.stdout, self.stderr, self.returncode = out, "", rc

        try:
            h.subprocess.run = lambda *a, **k: _Run("FROM foo\nTEMPLATE {{ .Prompt }}\n")
            check("raw passthrough detected", h.ollama_has_chat_template("x") is False)
            h.subprocess.run = lambda *a, **k: _Run('TEMPLATE """{{ .Messages }}"""\n')
            check("real template detected", h.ollama_has_chat_template("x") is True)
        finally:
            h.subprocess.run = real_run

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
        # ---- Start Model = load the tag, with a keep_alive ----
        seen = {}
        real_http = h._ollama_http
        try:
            def fake_run(path, port=None, timeout=None, payload=None):
                if path == "/api/generate":
                    seen["path"] = path
                    seen["payload"] = payload
                return {} if path == "/api/generate" else {"models": []}
            h._ollama_http = fake_run
            res = h.ollama_run("qwen2.5:72b-instruct-q4_K_M", keep_alive="30m")
            check("run posts to /api/generate", seen.get("path") == "/api/generate", seen)
            check("run sends the tag and keep_alive",
                  seen.get("payload", {}).get("model") == "qwen2.5:72b-instruct-q4_K_M"
                  and seen.get("payload", {}).get("keep_alive") == "30m", seen.get("payload"))
            check("run asks for no generation", seen.get("payload", {}).get("stream") is False)
            check("empty tag is refused", h.ollama_run("")["ok"] is False)
        finally:
            h._ollama_http = real_http

        # ---- vision support is what the engine reports, not the name ----
        real_http = h._ollama_http
        try:
            h._ollama_http = lambda path, port=None, timeout=None, payload=None: (
                {"capabilities": ["completion", "vision"]}
                if path == "/api/show" else {"models": []})
            h._OLLAMA_CAPS_CACHE.clear()
            check("a vision tag is recognised",
                  "vision" in h.ollama_model_capabilities("qwen2.5vl:7b"), True)
            h._OLLAMA_CAPS_CACHE.clear()
            h._ollama_http = lambda path, port=None, timeout=None, payload=None: (
                {"capabilities": ["completion", "tools"]}
                if path == "/api/show" else {"models": []})
            check("a text-only tag is not called vision",
                  "vision" not in h.ollama_model_capabilities("qwen2.5:72b"), False)
        finally:
            h._OLLAMA_CAPS_CACHE.clear()
            h._ollama_http = real_http

        # ---- Stop Model and the kill switch both reach Ollama ----
        calls = []
        real_http = h._ollama_http
        try:
            def fake_unload(path, port=None, timeout=None, payload=None):
                calls.append((path, payload))
                return {"models": []}
            h._ollama_http = fake_unload
            res = h.ollama_unload("qwen2.5:72b-instruct-q4_K_M")
            check("unload sends keep_alive 0",
                  any(p and p.get("keep_alive") == 0 for _, p in calls), calls)
            check("unload reports nothing left", res.get("still_loaded") == [], res)
            calls.clear()
            h.ollama_unload()
            check("unload with no tag only asks the engine what is loaded",
                  all(p == "/api/ps" for p, _ in calls), calls)
        finally:
            h._ollama_http = real_http

        # ---- a projector beside the weights is picked up automatically ----
        import tempfile as _tf
        d = _tf.mkdtemp()
        open(os.path.join(d, "model-Q4_K_M.gguf"), "w").write("x")
        open(os.path.join(d, "mmproj-model-f16.gguf"), "w").write("x")
        check("the mmproj is found next to the weights",
              h.gguf_projector_for(os.path.join(d, "model-Q4_K_M.gguf")).endswith("mmproj-model-f16.gguf"))
        plain = _tf.mkdtemp()
        open(os.path.join(plain, "other.gguf"), "w").write("x")
        open(os.path.join(plain, "other-Q8_0.gguf"), "w").write("x")
        check("another quant is not mistaken for a projector",
              h.gguf_projector_for(os.path.join(plain, "other.gguf")) == "")

        # ---- "is it actually running?" comes from /api/ps, not from the daemon ----
        real_http = h._ollama_http
        try:
            h._ollama_http = lambda *a, **k: {"models": [
                {"name": "qwen2.5:72b-instruct-q4_K_M", "size": 61800000000,
                 "size_vram": 61800000000, "context_length": 32768}]}
            ps = h.ollama_ps()
            check("ps lists the loaded model", [m["id"] for m in ps] == ["qwen2.5:72b-instruct-q4_K_M"], ps)
            check("ps reports memory in GB", ps[0]["vram_gb"] > 50, ps[0])
            h._ollama_http = lambda *a, **k: None
            check("ps is empty when the daemon is down", h.ollama_ps() == [])
        finally:
            h._ollama_http = real_http

        check("status still names the store", bool(st.get("store")), st)
    finally:
        h.ollama_harness_models = real_models
        os.path.expanduser = real_expanduser

    # ---- HF search: a lower bound on model size, not just an upper one ----
    real_urlopen = h.urllib.request.urlopen

    class _SearchResp:
        def __init__(self, payload):
            self.payload = payload
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return json.dumps(self.payload).encode()

    seen_urls = []

    def fake_search(req, timeout=None):
        seen_urls.append(req.full_url if hasattr(req, "full_url") else str(req))
        return _SearchResp([
            {"id": "org/Qwen2.5-VL-72B-Instruct-GGUF", "tags": ["gguf"], "pipeline_tag": "image-text-to-text"},
            {"id": "org/Qwen2.5-7B-Instruct-GGUF", "tags": ["gguf"], "pipeline_tag": "text-generation"},
            {"id": "org/mystery-GGUF", "tags": ["gguf", "multimodal"], "pipeline_tag": "image-text-to-text"},
        ])

    try:
        h.urllib.request.urlopen = fake_search
        res = h.search_hf_api("qwen", limit=10, filter_tags=["gguf", "vision"],
                              min_params=60, max_params=80, enrich=False)
        check("search asks HF for the parameter range",
              any("num_parameters=min:60B,max:80B" in u for u in seen_urls), seen_urls)
        ids = [r["id"] for r in res]
        check("a 72B model survives a 60-80B filter", any("72B" in i for i in ids), ids)
        check("a 7B model is dropped by a 60-80B filter", not any("7B-" in i for i in ids), ids)
        check("a model with no stated size is kept", any("mystery" in i for i in ids), ids)

        seen_urls.clear()
        h.search_hf_api("qwen", limit=10, filter_tags=["gguf"], min_params=13, enrich=False)
        check("a lower bound alone is sent",
              any("num_parameters=min:13B" in u for u in seen_urls), seen_urls)
    finally:
        h.urllib.request.urlopen = real_urlopen

    passed = sum(1 for _, ok, _ in CHECKS if ok)
    print(f"\n{passed}/{len(CHECKS)} checks passed")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(run())
