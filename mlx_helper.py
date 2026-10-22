#!/usr/bin/env python3
import os, sys, json, hashlib, fnmatch, tempfile, hashlib, signal, shlex, subprocess, re, shutil, urllib.request, urllib.parse, urllib.error, socket, time, threading

_REPO_SIZE_CACHE = {}
_SPEED_CACHE = {}
_SPEED_HIST = {}
_DL_WATCH = {}
_STALL_SECONDS = 90
_STALL_MIN_BYTES = 1 << 20
_STALL_MAX_RETRIES = 3
_WATCH_THREAD = None
_DOWNLOAD_LOG = os.path.expanduser("~/.mlx_logs/downloads.log")
PAUSED_FILE = os.path.expanduser("~/.mlx_pids/paused_downloads.json")
SETTINGS_FILE = os.path.expanduser("~/.mlx_pids/settings.json")
SIZE_INFO_CACHE_FILE = os.path.expanduser("~/.mlx_pids/repo_sizes.json")
SIZE_INFO_TTL = 14 * 86400  # re-check repo sizes every 2 weeks
_SIZE_INFO_CACHE = {}
_SIZE_INFO_LOCK = threading.Lock()
WEIGHT_FILE_EXTS = (".safetensors", ".gguf", ".bin", ".pt", ".pth", ".npz", ".h5")

def load_global_settings():
    default_settings = {
        "default_context_window": 0,
        "show_thinking": False,
        "thinking_budget": 0,
        "default_port": 9999,
        "default_host": "127.0.0.1",
        "default_temp": 0.0,
        "default_max_tokens": 0,  # 0 = derive from the loaded model's context window
        "auto_sync_agents": True,
        "turbo_downloads": True,
        "xet_downloads": False,
        "prefix_cache": True,          # reuse the KV of a prompt prefix shared by two turns
        "prefix_cache_blocks": 4096,   # 16 tokens per block -> 65,536 reusable tokens
        "prefix_cache_disk_gb": 20,    # warm-disk tier that survives a restart
        "prefix_cache_entries": 4,     # whole-prefix snapshots in RAM (hybrid/VLM models)
        "prefix_cache_memory_gb": 16,  # ceiling for the in-RAM cache (mlx-vlm defaults to 8)
        "ollama_port": 11434,          # the daemon listens here; every Ollama model shares it
        "ollama_store": "",            # "" = ~/.ollama/models (read by the daemon at start)
        "ollama_auto_start": True,
        "ollama_context": 32768,       # OLLAMA_CONTEXT_LENGTH for daemons we launch
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                st = json.load(f)
                default_settings.update(st)
        except Exception:
            pass
    return default_settings

def save_global_settings(settings):
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Hugging Face credentials (never shipped in the repo: per-user, stored locally)
# ---------------------------------------------------------------------------
KEYCHAIN_SERVICE = "mlx-control-center"
SECRETS_FILE = os.path.expanduser("~/.mlx_pids/secrets.json")
HF_HOST = "https://huggingface.co"
HF_OAUTH_DEVICE = HF_HOST + "/oauth/device"
HF_OAUTH_TOKEN = HF_HOST + "/oauth/token"
HF_OAUTH_REGISTER = HF_HOST + "/oauth/register"
HF_OAUTH_SCOPES = "openid profile read-repos"
_SECRET_LOCK = threading.Lock()

def _keychain_available():
    return sys.platform == "darwin" and bool(shutil.which("security"))

def _secret_read_keychain():
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", "hf", "-w"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return json.loads(out.stdout.strip())
    except Exception:
        pass
    return None

def _secret_write_keychain(data):
    try:
        out = subprocess.run(
            ["security", "add-generic-password", "-U", "-s", KEYCHAIN_SERVICE, "-a", "hf",
             "-l", "macOS MLX Control Center - Hugging Face", "-w", json.dumps(data)],
            capture_output=True, text=True, timeout=10)
        return out.returncode == 0
    except Exception:
        return False

def _secret_delete_keychain():
    try:
        subprocess.run(["security", "delete-generic-password", "-s", KEYCHAIN_SERVICE, "-a", "hf"],
                       capture_output=True, text=True, timeout=10)
    except Exception:
        pass

def _secret_read_file():
    try:
        if os.path.exists(SECRETS_FILE):
            with open(SECRETS_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return None

def _secret_write_file(data):
    try:
        os.makedirs(os.path.dirname(SECRETS_FILE), exist_ok=True)
        fd = os.open(SECRETS_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False

def load_secrets():
    """Token + OAuth client id, from Keychain when possible, else a 0600 JSON file."""
    with _SECRET_LOCK:
        data = _secret_read_keychain() or _secret_read_file() or {}
    if not isinstance(data, dict):
        data = {}
    return data

def save_secrets(data):
    with _SECRET_LOCK:
        stored = _secret_write_keychain(data) if _keychain_available() else False
        if not stored:
            _secret_write_file(data)
        elif os.path.exists(SECRETS_FILE):
            try:
                os.remove(SECRETS_FILE)  # do not leave a plaintext copy behind
            except Exception:
                pass
    return True

def hf_storage_backend():
    return "keychain" if _keychain_available() else "file"

def get_hf_token():
    token = (load_secrets().get("token") or os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    # A revoked token breaks every anonymous call that carries it (401 instead of
    # a real answer), so an unusable token is the same as no token here.
    if token and not hf_token_usable(token):
        return ""
    return token

_HF_TOKEN_STATE = {"token": "", "usable": None, "status": 0, "ts": 0.0}

def hf_token_usable(token=None, ttl=600):
    """Is this stored token still accepted by Hugging Face?

    A revoked token makes `hf download` answer "Model not found" for public repos,
    which looks like a broken model instead of a broken credential. Probe once per
    token and keep the answer for a few minutes; a network failure counts as usable
    so an offline Mac can still download from a warm cache.
    """
    token = token or get_hf_token()
    if not token:
        return False
    now = time.time()
    st = _HF_TOKEN_STATE
    if st["token"] == token and st["usable"] is not None and now - st["ts"] < ttl:
        return st["usable"]
    state, usable = 0, True
    try:
        req = urllib.request.Request(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": "Bearer " + token, "User-Agent": "mlx-control-center"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            state = getattr(resp, "status", 200)
            usable = state == 200
    except urllib.error.HTTPError as exc:
        state = exc.code
        usable = exc.code not in (401, 403)
    except Exception:
        state, usable = 0, True
    _HF_TOKEN_STATE.update({"token": token, "usable": usable, "status": state, "ts": now})
    return usable

def hf_token_status():
    """What the dashboard needs to warn about a dead credential."""
    token = get_hf_token()
    if not token:
        return {"present": False, "usable": False, "status": 0}
    return {"present": True, "usable": hf_token_usable(token),
            "status": _HF_TOKEN_STATE.get("status", 0)}

def hf_token_env():
    token = get_hf_token()
    if not token:
        return {}
    if not hf_token_usable(token):
        # Sending a revoked token turns every public download into "not found".
        return {}
    return {"HF_TOKEN": token, "HUGGING_FACE_HUB_TOKEN": token}

def set_hf_token(token, kind="token", user=""):
    data = load_secrets()
    data.update({"token": token.strip(), "kind": kind, "user": user, "saved_at": int(time.time())})
    save_secrets(data)
    return data

def clear_hf_token():
    data = load_secrets()
    keep = {k: v for k, v in data.items() if k == "client_id"}
    if keep:
        save_secrets(keep)
    else:
        with _SECRET_LOCK:
            _secret_delete_keychain()
            try:
                if os.path.exists(SECRETS_FILE):
                    os.remove(SECRETS_FILE)
            except Exception:
                pass
    return True

def _http_json(url, payload=None, method=None, timeout=30, token=None):
    """Minimal JSON-over-HTTP helper (form or JSON payloads) with no extra deps."""
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        if isinstance(payload, dict) and payload.get("__json__"):
            body = {k: v for k, v in payload.items() if k != "__json__"}
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        else:
            data = urllib.parse.urlencode(payload).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, data=data, headers=headers, method=method or ("POST" if data else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
            return r.status, (json.loads(body) if body.strip().startswith(("{", "[")) else {"raw": body})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = {"error": body[:400]}
        return e.code, parsed
    except Exception as e:
        return 0, {"error": str(e)}

def hf_whoami(token=None):
    """Validate a token against the Hub and return its identity."""
    token = token or get_hf_token()
    if not token:
        return {"ok": False, "error": "no token", "anonymous": True}
    status, data = _http_json(HF_HOST + "/api/whoami-v2", token=token, timeout=20)
    if status == 200 and data.get("name"):
        return {"ok": True, "user": data.get("name"), "type": data.get("type", "user"),
                "email": data.get("email", ""), "orgs": [o.get("name") for o in (data.get("orgs") or [])]}
    if status in (401, 403):
        return {"ok": False, "error": "invalid or expired token", "status": status}
    return {"ok": False, "error": data.get("error", f"HTTP {status}"), "status": status}

def hf_auth_state():
    """What the UI needs: signed in or not, as which user, where the secret lives."""
    token = get_hf_token()
    secrets = load_secrets()
    state = {
        "signed_in": bool(token),
        "user": secrets.get("user", ""),
        "kind": secrets.get("kind", ""),
        "storage": hf_storage_backend(),
        "device_flow": True,
    }
    if token and not state["user"]:
        who = hf_whoami(token)
        if who.get("ok"):
            state["user"] = who["user"]
            set_hf_token(token, secrets.get("kind", "token"), who["user"])
        else:
            state["invalid"] = who.get("error", "invalid token")
    return state

def hf_ensure_client_id():
    """Register this install as its own OAuth client (public, no secret, not in the repo)."""
    data = load_secrets()
    if data.get("client_id"):
        return data["client_id"]
    status, resp = _http_json(HF_OAUTH_REGISTER, {
        "__json__": True,
        "client_name": "macOS MLX Control Center",
        "redirect_uris": [],
        "grant_types": ["urn:ietf:params:oauth:grant-type:device_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
        "scope": HF_OAUTH_SCOPES,
    }, timeout=25)
    if status in (200, 201) and resp.get("client_id"):
        data["client_id"] = resp["client_id"]
        save_secrets(data)
        return resp["client_id"]
    raise RuntimeError(resp.get("error") or f"client registration failed (HTTP {status})")

def hf_device_start():
    """Begin a device-code login: returns the code the user types in the browser."""
    client_id = hf_ensure_client_id()
    status, resp = _http_json(HF_OAUTH_DEVICE, {
        "client_id": client_id,
        "scope": HF_OAUTH_SCOPES,
    }, timeout=25)
    if status != 200 or not resp.get("device_code"):
        return {"ok": False, "error": resp.get("error") or f"HTTP {status}"}
    return {
        "ok": True,
        "device_code": resp.get("device_code"),
        "user_code": resp.get("user_code"),
        "verification_uri": resp.get("verification_uri") or (HF_HOST + "/oauth/device"),
        "expires_in": resp.get("expires_in", 300),
        "interval": resp.get("interval", 5),
    }

def hf_device_poll(device_code):
    """One poll of the device flow. pending/slow_down/expired/accepted."""
    client_id = load_secrets().get("client_id") or hf_ensure_client_id()
    status, resp = _http_json(HF_OAUTH_TOKEN, {
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        "device_code": device_code,
        "client_id": client_id,
    }, timeout=25)
    if status == 200 and resp.get("access_token"):
        token = resp["access_token"]
        who = hf_whoami(token)
        user = who.get("user", "") if who.get("ok") else ""
        set_hf_token(token, "oauth", user)
        return {"status": "accepted", "user": user, "ok": bool(user), "who": who}
    err = (resp.get("error") or "").lower()
    if err in ("authorization_pending", "pending"):
        return {"status": "pending"}
    if err == "slow_down":
        return {"status": "slow_down"}
    if err in ("expired_token", "expired"):
        return {"status": "expired"}
    if err in ("access_denied", "denied"):
        return {"status": "denied"}
    return {"status": "error", "error": resp.get("error_description") or resp.get("error") or f"HTTP {status}"}

def get_mlx_env():
    """Builds environment with full PATH resolution for Apple Silicon and standard macOS binaries."""
    env = os.environ.copy()
    paths = [
        "/opt/homebrew/bin",
        "/opt/homebrew/sbin",
        "/usr/local/bin",
        os.path.expanduser("~/.local/bin"),
        os.path.expanduser("~/.cargo/bin"),
        os.path.expanduser("~/bin"),
        env.get("PATH", "")
    ]
    env["PATH"] = ":".join([p for p in paths if p])
    env["PYTHONUNBUFFERED"] = "1"
    # huggingface_hub >= 1.0 downloads through Xet; hf_transfer is gone.
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    # Xet can stall at 0 bytes on a large single-file blob (seen on a 44 GB GGUF:
    # four minutes with an empty .incomplete file), while plain HTTP resume runs at
    # 45 MB/s. Off by default; set xet_downloads in settings.json to re-enable.
    try:
        xet_on = bool(load_global_settings().get("xet_downloads", False))
    except Exception:
        xet_on = False
    if xet_on:
        env["HF_XET_HIGH_PERFORMANCE"] = "1"
        env.pop("HF_HUB_DISABLE_XET", None)
    else:
        env.pop("HF_XET_HIGH_PERFORMANCE", None)
        env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"
    env["HF_HUB_ETAG_TIMEOUT"] = "20"
    # User's own credentials, never baked into the repo.
    for k, v in hf_token_env().items():
        env[k] = v
    return env

STREAM_GUARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlx_stream_guard")
STREAM_GUARD_STATE = os.path.expanduser("~/.mlx_pids/stream_guard.json")
STREAM_GUARD_LOG = os.path.expanduser("~/.mlx_logs/stream_guard.log")

def stream_guard_env(env=None):
    """Make the launched model server import `mlx_stream_guard/sitecustomize.py`.

    The guard has to live *inside* the server process: proxying would mean a second
    port plus a baseURL change in every harness config, and would need its own
    supervision. Putting this directory first on PYTHONPATH leaves the command line,
    the port, the pid file and every existing health check untouched — `python -m
    mlx_vlm.server` simply imports the hook at interpreter start.

    Set MLX_STREAM_GUARD=0 to launch the old, unguarded way.
    """
    env = dict(env if env is not None else os.environ)
    if (os.environ.get("MLX_STREAM_GUARD") or "").strip() == "0":
        env["MLX_STREAM_GUARD"] = "0"
        return env
    if not os.path.exists(os.path.join(STREAM_GUARD_DIR, "sitecustomize.py")):
        return env
    previous = env.get("PYTHONPATH", "")
    parts = [STREAM_GUARD_DIR]
    parts += [p for p in previous.split(os.pathsep) if p and p != STREAM_GUARD_DIR]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["MLX_STREAM_GUARD"] = "1"
    env.setdefault("MLX_GUARD_KEEPALIVE_SECS", "5")
    env["MLX_GUARD_LOG"] = env.get("MLX_GUARD_LOG") or STREAM_GUARD_LOG
    env["MLX_GUARD_STATE"] = env.get("MLX_GUARD_STATE") or STREAM_GUARD_STATE
    return env

def stream_guard_status():
    """What the guard saw, for the dashboard and for `mlx.sh status`."""
    enabled = (os.environ.get("MLX_STREAM_GUARD") or "").strip() != "0"
    available = os.path.exists(os.path.join(STREAM_GUARD_DIR, "sitecustomize.py"))
    info = {
        "available": available,
        "enabled": enabled and available,
        "state_file": STREAM_GUARD_STATE,
        "log": STREAM_GUARD_LOG,
        "active": False,
    }
    try:
        with open(STREAM_GUARD_STATE) as fh:
            state = json.load(fh)
    except Exception:
        return info
    age = None
    try:
        age = max(0.0, time.time() - os.path.getmtime(STREAM_GUARD_STATE))
    except Exception:
        pass
    pid = state.get("pid")
    alive = False
    if pid:
        try:
            os.kill(int(pid), 0)
            alive = True
        except Exception:
            alive = False
    info["pid"] = pid
    info["active"] = bool(state.get("installed")) and alive
    info["stale"] = bool(state.get("installed")) and not alive
    info["streams"] = state.get("streams", 0)
    info["keepalives"] = state.get("keepalives", 0)
    info["synthetic_done"] = state.get("synthetic_done", 0)
    info["max_tokens_fixed"] = state.get("max_tokens_fixed", 0)
    info["empty_streams"] = state.get("empty_streams", 0)
    info["errors"] = state.get("errors", 0)
    info["hooks"] = state.get("engine_hooks") or []
    info["interval"] = state.get("interval")
    info["mode"] = state.get("mode")
    info["last"] = state.get("last")
    info["age_seconds"] = None if age is None else round(age, 1)
    return info

# ---------------------------------------------------------------------------
# Prefix cache (mlx-vlm's "Automatic Prefix Caching", APC)
# ---------------------------------------------------------------------------
# mlx-vlm ships APC, but the OpenAI-compatible server only builds it when
# APC_ENABLED is set; the default is off. With it off, every turn re-prefills the
# whole conversation (cached_tokens=0 in every log line) — 49 s of silence for a
# 63k-token prompt, even when the only new part is "hello". Turning it on lets a
# turn reuse the KV of the prefix it shares with the previous turn and prefill just
# the suffix. mlx-lm has its own prompt cache and ignores these variables.
APC_CACHE_HOME = os.path.expanduser("~/.cache/mlx-vlm")
APC_DEFAULT_BLOCKS = 4096      # 16 tokens/block -> 65,536 reusable tokens in RAM
APC_BLOCK_SIZE = 16
APC_DEFAULT_DISK_GB = 20
APC_DEFAULT_ENTRIES = 4        # whole-prefix snapshots kept in RAM (VLM layouts)
APC_DEFAULT_MEMORY_GB = 16     # ceiling for the in-RAM cache; mlx-vlm auto-caps at 8

def prefix_cache_settings():
    """Prefix-cache knobs from settings.json, with safe fallbacks."""
    st = load_global_settings()
    enabled = st.get("prefix_cache", True)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() not in ("0", "false", "no", "off", "")
    def _num(key, default, cast):
        try:
            return cast(st.get(key, default))
        except Exception:
            return cast(default)
    blocks = _num("prefix_cache_blocks", APC_DEFAULT_BLOCKS, int)
    blocks = blocks if blocks > 0 else APC_DEFAULT_BLOCKS
    disk_gb = max(0.0, _num("prefix_cache_disk_gb", APC_DEFAULT_DISK_GB, float))
    entries = _num("prefix_cache_entries", APC_DEFAULT_ENTRIES, int)
    entries = entries if entries > 0 else APC_DEFAULT_ENTRIES
    memory_gb = max(0.0, _num("prefix_cache_memory_gb", APC_DEFAULT_MEMORY_GB, float))
    return {
        "enabled": bool(enabled),
        "blocks": blocks,
        "disk_gb": disk_gb,
        "entries": entries,
        "memory_gb": memory_gb,
    }

def apc_env(env=None, engine="mlx-vlm"):
    """Environment that switches on mlx-vlm's prefix cache.

    Two shapes of cache exist. Dense models reuse 16-token "pages" sized by
    APC_NUM_BLOCKS; hybrid/rotating-attention models (most VLMs) cannot stitch pages
    back together and keep whole-prefix snapshots instead — for those the dials are
    APC_EXACT_CACHE_ENTRIES and APC_MEMORY_MAX_GB, which mlx-vlm otherwise caps at
    2 snapshots and 8 GB. Both are set here, so either layout gets a sane budget.

    Set MLX_VLM_APC=0 to launch without it. Non-VLM engines already have a prompt
    cache of their own, so they are left untouched.
    """
    env = dict(env if env is not None else os.environ)
    if engine != "mlx-vlm":
        return env
    if (env.get("MLX_VLM_APC") or os.environ.get("MLX_VLM_APC") or "").strip() == "0":
        return env
    cfg = prefix_cache_settings()
    if not cfg["enabled"]:
        return env
    env.setdefault("APC_ENABLED", "1")
    env.setdefault("APC_BLOCK_SIZE", str(APC_BLOCK_SIZE))
    env.setdefault("APC_NUM_BLOCKS", str(cfg["blocks"]))
    env.setdefault("APC_EXACT_CACHE_ENTRIES", str(cfg["entries"]))
    if cfg["memory_gb"] > 0:
        env.setdefault("APC_MEMORY_MAX_GB", str(cfg["memory_gb"]))
    env.setdefault("MLX_VLM_CACHE_HOME", APC_CACHE_HOME)
    if cfg["disk_gb"] > 0:
        env.setdefault("APC_DISK_MAX_GB", str(cfg["disk_gb"]))
    else:
        env.setdefault("APC_DISK_ENABLED", "0")
    return env

def apc_status(port=None, timeout=1.5):
    """Prefix-cache state for the dashboard: configured, and what the server reports."""
    cfg = prefix_cache_settings()
    info = {
        "enabled": cfg["enabled"],
        "blocks": cfg["blocks"],
        "disk_gb": cfg["disk_gb"],
        "entries": cfg["entries"],
        "memory_gb": cfg["memory_gb"],
        "cache_home": APC_CACHE_HOME,
        "server_running": False,
        "server_enabled": False,
        "hit_rate": None,
        "matched_tokens": 0,
    }
    if not port:
        return info
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{int(port)}/v1/cache/stats")
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            data = json.load(fh)
        info["server_running"] = True
        info["server_enabled"] = bool(data.get("enabled"))
        if info["server_enabled"]:
            info["hit_rate"] = data.get("token_hit_rate")
            info["matched_tokens"] = data.get("matched_tokens") or 0
            info["lookups_hit"] = data.get("lookups_hit") or 0
            info["lookups_miss"] = data.get("lookups_miss") or 0
            info["resident_bytes"] = data.get("resident_bytes")
            info["disk_bytes"] = data.get("disk_bytes")
            info["memory_max_bytes"] = data.get("memory_max_bytes")
            info["exact_hits"] = data.get("exact_hits")
    except Exception as exc:
        info["error"] = str(exc)
    return info

# ---------------------------------------------------------------------------
# Ollama engine
# ---------------------------------------------------------------------------
# Ollama is a daemon: one process serves every model on one port (11434 by
# default), unlike the MLX engines which are one process per model. Its store is
# a content-addressed blob directory, not a folder of GGUF files, so the only way
# to choose where weights live is OLLAMA_MODELS at daemon start. Everything here
# therefore talks to the daemon (adopt it when it is already up) and treats the
# store as a location, not a model list.
OLLAMA_DEFAULT_PORT = 11434
OLLAMA_DEFAULT_HOST = "127.0.0.1"
OLLAMA_STORE_DEFAULT = os.path.expanduser("~/.ollama/models")
OLLAMA_LAUNCH_AGENT = os.path.expanduser("~/Library/LaunchAgents/com.ollama.ollama.plist")
OLLAMA_PID_DIR = os.path.expanduser("~/.mlx_pids")
OLLAMA_LOG_DIR = os.path.expanduser("~/.mlx_logs")

def ollama_bin():
    """Path to the ollama CLI, or None."""
    found = shutil.which("ollama")
    if found:
        return found
    for cand in ("/opt/homebrew/bin/ollama", "/usr/local/bin/ollama", "/usr/bin/ollama"):
        if os.path.exists(cand):
            return cand
    return None

def ollama_settings():
    """Engine settings from settings.json."""
    st = load_global_settings()
    def _num(key, default, cast):
        try:
            return cast(st.get(key, default))
        except Exception:
            return cast(default)
    port = _num("ollama_port", OLLAMA_DEFAULT_PORT, int)
    if port <= 0 or port > 65535:
        port = OLLAMA_DEFAULT_PORT
    store = str(st.get("ollama_store") or OLLAMA_STORE_DEFAULT)
    return {
        "port": port,
        "store": store,
        "auto_start": bool(st.get("ollama_auto_start", True)),
        "context": max(0, _num("ollama_context", 32768, int)),
    }

def ollama_version(timeout=4):
    exe = ollama_bin()
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=timeout)
        blob = (out.stdout or "") + (out.stderr or "")
        m = re.search(r"(\d+\.\d+\.\d+)", blob)
        return m.group(1) if m else blob.strip()[:40] or None
    except Exception:
        return None

def _ollama_http(path, port=None, payload=None, timeout=3.0):
    cfg = ollama_settings()
    port = int(port or cfg["port"])
    url = f"http://127.0.0.1:{port}{path}"
    data = None
    headers = {"User-Agent": "mlx-control-center"}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode() or "{}")

def ollama_pid(port=None):
    """PID listening on the Ollama port, or None."""
    cfg = ollama_settings()
    port = int(port or cfg["port"])
    try:
        out = subprocess.run(["lsof", "-ti", f":{port}", "-sTCP:LISTEN"],
                             capture_output=True, text=True, timeout=5)
        for line in out.stdout.split():
            if line.strip().isdigit():
                return int(line.strip())
    except Exception:
        pass
    pid_file = os.path.join(OLLAMA_PID_DIR, f"ollama_{port}.pid")
    try:
        with open(pid_file) as fh:
            return int(fh.read().strip())
    except Exception:
        return None

def ollama_runner_rss_gb():
    """Resident memory of the engine's runner processes, in GB.

    Ollama reports `size`/`size_vram` as allocated capacity (weights + the whole
    KV reservation), which can exceed what the machine is actually holding. The
    dashboard's other numbers are resident sizes, so this keeps them comparable.
    """
    try:
        out = subprocess.check_output(["ps", "-Ao", "rss,args"], text=True)
    except Exception:
        return 0.0
    total_kb = 0
    for line in out.splitlines()[1:]:
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        args = parts[1]
        if "ollama" not in args:
            continue
        if "runner" not in args and "llama-server" not in args:
            continue
        try:
            total_kb += int(parts[0])
        except ValueError:
            continue
    return round(total_kb / (1024**2), 2)

def ollama_unload(tag=None, port=None, timeout=60):
    """Unload one tag (or every loaded model) from the engine's memory.

    keep_alive: 0 tells Ollama to drop the model now; an empty prompt means no
    generation is requested, only the unload.
    """
    tags = [tag] if tag else [m["id"] for m in ollama_ps(port=port)]
    done = []
    for t in tags:
        try:
            _ollama_http("/api/generate", port=port, timeout=timeout,
                         payload={"model": t, "prompt": "", "stream": False, "keep_alive": 0})
            done.append(t)
        except Exception:
            pass
    # The engine drops the model asynchronously: give it a moment before
    # reporting, or every Stop looks like it failed.
    remaining = [m["id"] for m in ollama_ps(port=port)]
    deadline = time.time() + 6
    while remaining and time.time() < deadline:
        time.sleep(0.4)
        remaining = [m["id"] for m in ollama_ps(port=port)]
    return {"ok": True, "unloaded": done, "still_loaded": remaining}

def ollama_run(tag, port=None, keep_alive="30m", timeout=900):
    """Load a model into the engine's memory (Ollama's equivalent of Start Model).

    Uses /api/generate with an empty prompt: the engine loads the weights and
    returns, no tokens are produced. keep_alive then decides how long it stays.
    """
    if not tag:
        return {"ok": False, "error": "no model tag given"}
    try:
        data = _ollama_http("/api/generate", port=port, timeout=timeout,
                            payload={"model": tag, "prompt": "", "stream": False,
                                     "keep_alive": keep_alive or "30m"}) or {}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    if not isinstance(data, dict) or data.get("error"):
        return {"ok": False, "error": (data or {}).get("error") or "engine did not load the model"}
    loaded = ollama_ps(port=port)
    mine = next((m for m in loaded if m["id"] == tag or m["id"].split(":")[0] == tag), None)
    return {"ok": True, "tag": tag, "keep_alive": keep_alive or "30m",
            "loaded": bool(mine), "size_gb": (mine or {}).get("size_gb", 0)}

def ollama_ps(port=None, timeout=2.0):
    """Models the Ollama engine currently holds in memory (`/api/ps`).

    This is the only way to tell the engine is serving something: the daemon
    runs whether or not a model is loaded.
    """
    try:
        data = _ollama_http("/api/ps", port=port, timeout=timeout) or {}
    except Exception:
        return []
    out = []
    for m in (data.get("models") or []):
        if not isinstance(m, dict):
            continue
        size = m.get("size") or 0
        vram = m.get("size_vram") or 0
        out.append({
            "id": m.get("name") or m.get("model") or "",
            "size_gb": round(size / (1024**3), 2) if size else 0,
            "vram_gb": round(vram / (1024**3), 2) if vram else 0,
            "expires_at": m.get("expires_at") or "",
            "context": m.get("context_length") or 0,
        })
    return out

def ollama_models(port=None, timeout=3.0):
    """Models the running daemon can serve: id, size, family, params, context."""
    try:
        data = _ollama_http("/api/tags", port=port, timeout=timeout)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    out = []
    for m in data.get("models", []) or []:
        name = m.get("name") or m.get("model") or ""
        if not name:
            continue
        details = m.get("details") or {}
        caps = ollama_model_capabilities(name, port=port)
        out.append({
            "id": name,
            "name": name,
            "size_gb": round((m.get("size") or 0) / (1024 ** 3), 2),
            "modified": str(m.get("modified_at") or "")[:19].replace("T", " "),
            "family": details.get("family") or "",
            "parameter_size": details.get("parameter_size") or "",
            "quantization": details.get("quantization_level") or "",
            "capabilities": caps,
            "is_vision": "vision" in caps,
            "supports_tools": "tools" in caps,
        })
    out.sort(key=lambda x: x["id"])
    return out

_OLLAMA_CAPS_CACHE = {}

def ollama_model_capabilities(name, port=None, timeout=3.0):
    """What the engine says this model can do: e.g. ['completion','tools','vision'].

    This is the only reliable way to know whether a tag is multimodal: the name
    and the family are guesses, the capability list is the engine's own answer.
    """
    if not name:
        return []
    key = (name, port)
    now = time.time()
    hit = _OLLAMA_CAPS_CACHE.get(key)
    if hit and now - hit[0] < 300:
        return hit[1]
    try:
        data = _ollama_http("/api/show", port=port, payload={"model": name}, timeout=timeout) or {}
    except Exception:
        return []
    caps = data.get("capabilities") if isinstance(data, dict) else None
    caps = [str(c) for c in (caps or []) if c]
    _OLLAMA_CAPS_CACHE[key] = (now, caps)
    return caps

def ollama_context_length(name, port=None, timeout=3.0):
    """Trained context length for one model, from /api/show; None if unknown."""
    try:
        data = _ollama_http("/api/show", port=port, payload={"model": name}, timeout=timeout)
    except Exception:
        return None
    info = data.get("model_info") or {}
    for key, val in info.items():
        if key.endswith("context_length") and isinstance(val, int) and val > 0:
            return val
    for key, val in info.items():
        if key.endswith(".context_length") and isinstance(val, int) and val > 0:
            return val
    return None

_OLLAMA_STORE_CACHE = {"ts": 0.0, "path": "", "gb": 0.0}

def ollama_store_gb(path=None, ttl=60):
    """Size of the Ollama blob store. Cached: `du` on a 70 GB store is not free."""
    cfg = ollama_settings()
    path = path or cfg["store"]
    now = time.time()
    if (_OLLAMA_STORE_CACHE["path"] == path
            and now - _OLLAMA_STORE_CACHE["ts"] < ttl):
        return _OLLAMA_STORE_CACHE["gb"]
    try:
        out = subprocess.run(["du", "-sk", path], capture_output=True, text=True, timeout=15)
        gb = round(int(out.stdout.split()[0]) / (1024 ** 2), 1)
    except Exception:
        gb = 0.0
    _OLLAMA_STORE_CACHE.update({"ts": now, "path": path, "gb": gb})
    return gb

def ollama_status(port=None, with_models=True):
    """Everything the dashboard needs to render the Ollama panel."""
    cfg = ollama_settings()
    port = int(port or cfg["port"])
    exe = ollama_bin()
    info = {
        "installed": bool(exe),
        "binary": exe or "",
        "version": ollama_version() if exe else None,
        "port": port,
        "host": OLLAMA_DEFAULT_HOST,
        "base_url": f"http://{OLLAMA_DEFAULT_HOST}:{port}/v1",
        "store": cfg["store"],
        "store_default": OLLAMA_STORE_DEFAULT,
        "store_gb": 0.0,
        "running": False,
        "pid": None,
        "models": [],
        "model_count": 0,
        "context_default": cfg["context"],
        "launch_agent": os.path.exists(OLLAMA_LAUNCH_AGENT),
    }
    if not exe:
        return info
    pid = ollama_pid(port)
    if pid:
        try:
            os.kill(pid, 0)
            info["running"] = True
            info["pid"] = pid
        except Exception:
            info["running"] = False
    info["store_gb"] = ollama_store_gb(cfg["store"])
    if info["running"] and with_models:
        info["models"] = ollama_models(port)
        info["model_count"] = len(info["models"])
        # A running daemon says nothing about whether a model is in memory.
        info["loaded"] = ollama_ps(port)
        info["loaded_count"] = len(info["loaded"])
        info["loaded_gb"] = round(sum(m["vram_gb"] or m["size_gb"] for m in info["loaded"]), 2)
    return info

def ollama_start(port=None, store=None, context=None, wait_seconds=25):
    """Start the daemon if it is not already up. Returns a status dict."""
    cfg = ollama_settings()
    port = int(port or cfg["port"])
    store = store or cfg["store"]
    exe = ollama_bin()
    if not exe:
        return {"ok": False, "error": "ollama is not installed"}
    status = ollama_status(port, with_models=False)
    if status.get("running"):
        return {"ok": True, "adopted": True, "pid": status.get("pid"), "port": port,
                "message": f"Using the Ollama daemon already listening on port {port}"
                            + (" (managed by launchd)" if status.get("launch_agent") else "")}
    os.makedirs(OLLAMA_PID_DIR, exist_ok=True)
    os.makedirs(OLLAMA_LOG_DIR, exist_ok=True)
    if store:
        os.makedirs(store, exist_ok=True)
    log_path = os.path.join(OLLAMA_LOG_DIR, f"ollama_{port}.log")
    env = dict(os.environ)
    env["OLLAMA_HOST"] = f"{OLLAMA_DEFAULT_HOST}:{port}"
    if store:
        env["OLLAMA_MODELS"] = store
    ctx = int(context if context is not None else cfg["context"] or 0)
    if ctx > 0:
        env["OLLAMA_CONTEXT_LENGTH"] = str(ctx)
    try:
        with open(log_path, "ab", buffering=0) as log:
            proc = subprocess.Popen([exe, "serve"], stdout=log, stderr=log, env=env,
                                    start_new_session=True)
    except Exception as exc:
        return {"ok": False, "error": f"could not start ollama: {exc}"}
    try:
        with open(os.path.join(OLLAMA_PID_DIR, f"ollama_{port}.pid"), "w") as fh:
            fh.write(str(proc.pid))
    except Exception:
        pass
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        try:
            _ollama_http("/api/tags", port=port, timeout=1.0)
            return {"ok": True, "adopted": False, "pid": proc.pid, "port": port,
                    "message": f"Ollama listening on port {port}, store {store}"}
        except Exception:
            if proc.poll() is not None:
                return {"ok": False, "error": f"ollama exited (see {log_path})"}
            time.sleep(0.5)
    return {"ok": False, "error": f"ollama did not answer on port {port} within {wait_seconds}s"}

def ollama_stop(port=None):
    """Stop a daemon this app started. A launchd-managed daemon is left alone."""
    cfg = ollama_settings()
    port = int(port or cfg["port"])
    pid_file = os.path.join(OLLAMA_PID_DIR, f"ollama_{port}.pid")
    try:
        with open(pid_file) as fh:
            pid = int(fh.read().strip())
    except Exception:
        pid = None
    if not pid:
        return {"ok": False, "error": "no daemon started by this app on that port"
                                      + (" (launchd owns it — unload the agent to stop it)"
                                         if os.path.exists(OLLAMA_LAUNCH_AGENT) else "")}
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(40):
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except Exception:
                break
        else:
            os.kill(pid, signal.SIGKILL)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    try:
        os.unlink(pid_file)
    except Exception:
        pass
    return {"ok": True, "message": f"Stopped Ollama on port {port}"}

def ollama_model_tag(path_or_name):
    """Turn a GGUF path or repo name into a usable Ollama tag."""
    base = os.path.basename(str(path_or_name)).replace(".gguf", "")
    base = base.lower()
    base = re.sub(r"[^a-z0-9._-]+", "-", base).strip("-.")
    return (base or "local-model") + ":latest"

# A GGUF converted without its tokenizer config has no chat template, and Ollama
# then falls back to a raw "{{ .Prompt }}" passthrough: no <|im_start|> tokens, no
# tool calling, and answers that look like a text-completion demo. Re-create the
# model with the family's real template when that happens.
CHAT_TEMPLATES = {
    "qwen2": r'''{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}''',
}


def ollama_has_chat_template(tag, port=None):
    """True when the model's TEMPLATE is a real chat template, not a passthrough."""
    exe = ollama_bin()
    if not exe:
        return None
    env = dict(os.environ)
    env["OLLAMA_HOST"] = f"{OLLAMA_DEFAULT_HOST}:{int(port or ollama_settings()['port'])}"
    try:
        out = subprocess.run([exe, "show", "--modelfile", tag], capture_output=True,
                             text=True, env=env, timeout=20)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    for line in (out.stdout or "").splitlines():
        if line.startswith("TEMPLATE"):
            body = line[len("TEMPLATE"):].strip()
            return bool(body) and body not in ("{{ .Prompt }}", "{{.Prompt}}")
    return False

def ollama_chat_template_for(family):
    """Go template for a GGUF family, used only when the file carries none.

    Ollama's TEMPLATE is a Go template, not Jinja: the Hugging Face chat_template
    from tokenizer_config.json fails with "malformed character constant".
    """
    fam = (family or "").lower().replace("-", "").replace("_", "").replace(".", "")
    for key, tmpl in CHAT_TEMPLATES.items():
        if key in fam:
            return tmpl
    return ""

def gguf_projector_for(gguf_path):
    """The mmproj/clip projector shipped beside a vision GGUF, if there is one.

    Ollama needs it as a second FROM line in the Modelfile; without it the file
    loads as a text-only model and image prompts are rejected.
    """
    try:
        folder = os.path.dirname(os.path.abspath(gguf_path))
        for fn in sorted(os.listdir(folder)):
            low = fn.lower()
            if not low.endswith(".gguf") or fn == os.path.basename(gguf_path):
                continue
            if "mmproj" in low or "projector" in low or low.startswith("clip"):
                return os.path.join(folder, fn)
    except Exception:
        pass
    return ""

def ollama_import(gguf_path, name=None, port=None, wait_seconds=180):
    """Create an Ollama model from a local GGUF file.

    Ollama copies the file into its blob store (deduplicated by digest), so this
    can double the disk cost of the file. The Modelfile pins num_ctx to the
    configured context so a harness prompt is not silently truncated.
    """
    exe = ollama_bin()
    if not exe:
        return {"ok": False, "error": "ollama is not installed"}
    if not os.path.isfile(gguf_path):
        return {"ok": False, "error": f"no such file: {gguf_path}"}
    cfg = ollama_settings()
    tag = name or ollama_model_tag(gguf_path)
    if not name:
        existing = ollama_tag_for_gguf(gguf_path, port=port)
        if existing:
            return {"ok": True, "tag": existing, "reused": True,
                    "message": f"Already in Ollama as {existing}",
                    "output": ""}
    # Vision GGUF repos keep the multimodal projector in a second file; Ollama
    # needs both FROM lines or the model comes up text-only.
    projector = gguf_projector_for(gguf_path)
    modelfile = f"FROM {gguf_path}\n"
    if projector:
        modelfile += f"FROM {projector}\n"
    if cfg["context"] > 0:
        modelfile += f"PARAMETER num_ctx {cfg['context']}\n"
    env = dict(os.environ)
    env["OLLAMA_HOST"] = f"{OLLAMA_DEFAULT_HOST}:{int(port or cfg['port'])}"
    if cfg["store"]:
        env["OLLAMA_MODELS"] = cfg["store"]
    # `ollama create -f -` ignores stdin, so the Modelfile has to be a real file.
    tmpdir = tempfile.mkdtemp(prefix="ollama_import_")
    modelfile_path = os.path.join(tmpdir, "Modelfile")
    try:
        with open(modelfile_path, "w") as fh:
            fh.write(modelfile)
        out = subprocess.run([exe, "create", tag, "-f", modelfile_path], text=True,
                             capture_output=True, env=env, timeout=wait_seconds)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    if out.returncode != 0:
        return {"ok": False, "error": (out.stderr or out.stdout or "ollama create failed").strip()[:400]}

    # No template in the GGUF? Put the family's real chat template back, or the
    # model answers as a raw completion and no harness can call tools on it.
    template_fixed = False
    if ollama_has_chat_template(tag, port) is False:
        try:
            info = _ollama_http("/api/show", port=port, payload={"model": tag}, timeout=20) or {}
            family = (info.get("details") or {}).get("family", "")
        except Exception:
            family = ""
        tmpl = ollama_chat_template_for(family)
        if tmpl:
            tmpdir = tempfile.mkdtemp(prefix="ollama_template_")
            mf = os.path.join(tmpdir, "Modelfile")
            try:
                with open(mf, "w") as fh:
                    fh.write(modelfile + 'TEMPLATE """' + tmpl + '"""\n')
                fixed = subprocess.run([exe, "create", tag, "-f", mf], text=True,
                                       capture_output=True, env=env, timeout=wait_seconds)
                template_fixed = fixed.returncode == 0
                if not template_fixed:
                    _write_log_line("ollama template fix failed: " + (fixed.stderr or "")[:300])
            except Exception:
                template_fixed = False
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

    return {"ok": True, "tag": tag, "message": f"Imported as {tag}",
            "template_fixed": template_fixed,
            "output": (out.stdout or "").strip()[-400:]}

GGUF_DIGEST_CACHE = os.path.join(os.path.expanduser("~/.mlx_pids"), "gguf_digests.json")

def gguf_digest(path):
    """sha256 of a GGUF file, cached by (path, size, mtime).

    Ollama's blob names are exactly this digest, so it is how we tell whether a
    downloaded file is already inside the model store.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    cache = {}
    try:
        with open(GGUF_DIGEST_CACHE) as fh:
            cache = json.load(fh) or {}
    except Exception:
        cache = {}
    if cache.get("key") == key and cache.get("sha256"):
        return cache["sha256"]
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(8 << 20), b""):
                digest.update(block)
        value = digest.hexdigest()
    except Exception:
        return None
    try:
        with open(GGUF_DIGEST_CACHE, "w") as fh:
            json.dump({"key": key, "sha256": value}, fh)
    except Exception:
        pass
    return value

def _normalise_tag(tag):
    """Compare tags loosely: ':' 'latest' '-' '_' '.' are all just separators."""
    text = str(tag or "").lower().replace(":latest", "").replace(":", "-")
    return re.sub(r"[-_.]+", "-", text).strip("-")

def ollama_tag_for_gguf(gguf_path, port=None):
    """The Ollama tag that already holds this exact file, or "".

    A name match is checked first (cheap); the content hash confirms it. A tag
    created by hand may spell the same model with ':' where the control center
    uses '-', which is how a duplicate tag becomes easy to create by accident.
    """
    try:
        models = ollama_models(port=port)
    except Exception:
        return ""
    if not models:
        return ""
    wanted = _normalise_tag(ollama_model_tag(gguf_path))
    for m in models:
        tag = m.get("id", "")
        if tag and _normalise_tag(tag) == wanted:
            return tag
    # Different name, same bytes? Ollama names its blobs by the file's sha256.
    digest = gguf_digest(gguf_path)
    if not digest:
        return ""
    for m in models:
        tag = m.get("id", "")
        if not tag:
            continue
        try:
            info = _ollama_http("/api/show", port=port, payload={"model": tag}, timeout=10) or {}
        except Exception:
            continue
        parent = str((info.get("details") or {}).get("parent_model") or "")
        if digest in parent:
            return tag
    return ""

def local_gguf_files(model_dir):
    """The .gguf files inside one downloaded model snapshot (usually exactly one)."""
    out = []
    try:
        for name in sorted(os.listdir(model_dir)):
            if name.lower().endswith(".gguf"):
                full = os.path.join(model_dir, name)
                try:
                    size = os.path.getsize(full)
                except Exception:
                    size = 0
                out.append({"path": full, "name": name, "size_gb": round(size / (1024 ** 3), 2)})
    except Exception:
        return []
    return out

def ollama_engine_for(model_name_or_path):
    """'ollama' for GGUF files and Ollama tags, otherwise 'mlx'."""
    name = str(model_name_or_path or "")
    if name.lower().endswith(".gguf"):
        return "ollama"
    if re.match(r"^[\w.-]+:[\w.-]+$", name) and "/" not in name:
        return "ollama"
    return "mlx"

def ollama_harness_models(port=None):
    """Models to advertise to the coding tools: id + real context window."""
    cfg = ollama_settings()
    out = []
    for m in ollama_models(port=port):
        ctx = ollama_context_length(m["id"], port=port) or cfg["context"] or 8192
        out.append({
            "id": m["id"],
            "contextWindow": int(ctx),
            "maxTokens": output_token_budget(m["id"]),
            "size_gb": m.get("size_gb"),
            "family": m.get("family"),
            # Trust the engine's capabilities; fall back to the name only if the
            # engine did not report any (older versions).
            "is_vision": m.get("is_vision", False) if m.get("capabilities")
                          else bool(re.search(r"vl|vision|llava|qwen2\.5vl", m["id"], re.I)),
        })
    return out

STORAGE_FILE = os.path.expanduser("~/.mlx_pids/storage.json")
LIBRARY_DIRNAME = "MLX-Models"
MIN_DRIVE_GB = 8
_DRIVE_CACHE = {"ts": 0.0, "drives": []}
DRIVE_CACHE_TTL = 20
_TM_CACHE = {"ts": 0.0, "mounts": set()}
_TM_EXCLUDED = {}

def load_storage_config():
    """Persisted storage preferences: default drive, known drives (by UUID), disabled drives."""
    cfg = {"default_drive": "", "known": {}, "disabled": []}
    if os.path.exists(STORAGE_FILE):
        try:
            with open(STORAGE_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                cfg.update(data)
        except Exception:
            pass
    if not isinstance(cfg.get("known"), dict):
        cfg["known"] = {}
    if not isinstance(cfg.get("disabled"), list):
        cfg["disabled"] = []
    return cfg

def save_storage_config(cfg):
    try:
        os.makedirs(os.path.dirname(STORAGE_FILE), exist_ok=True)
        with open(STORAGE_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass

def _diskutil_field(mount, field):
    """Reads a single field from `diskutil info` for a mounted volume."""
    try:
        out = subprocess.check_output(["diskutil", "info", mount], text=True,
                                      timeout=6, stderr=subprocess.DEVNULL)
    except Exception:
        return ""
    m = re.search(r"^\s*" + re.escape(field) + r"\s*:\s*(.+)$", out, re.M)
    return m.group(1).strip() if m else ""

def tm_exclusion_state(path):
    """True when Time Machine skips `path`, False when it would back it up.

    Model weights are multi-GB blobs that are trivially re-downloadable, so they never
    belong in a backup set. Unknown/unqueryable paths report True so we stay quiet.
    """
    if not path:
        return True
    try:
        if path in _TM_EXCLUDED:
            return _TM_EXCLUDED[path]
    except Exception:
        pass
    # tmutil can only answer for existing paths; inherit the state of the closest
    # existing ancestor (an excluded volume protects everything inside it).
    probe = path
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe.rstrip("/"))
        if parent == probe:
            break
        probe = parent
    try:
        out = subprocess.run(["tmutil", "isexcluded", probe or path], capture_output=True,
                             text=True, timeout=8)
        state = "[Excluded]" in (out.stdout or "")
    except Exception:
        state = True
    try:
        _TM_EXCLUDED[path] = state
    except Exception:
        pass
    return state

def exclude_from_time_machine(path):
    """Best effort: adds `path` to the Time Machine exclusion list. Returns True if excluded."""
    if not path or not os.path.exists(path):
        return False
    try:
        if tm_exclusion_state(path):
            return True
        subprocess.run(["tmutil", "addexclusion", path], capture_output=True,
                       text=True, timeout=15)
        try:
            _TM_EXCLUDED.pop(path, None)
        except Exception:
            pass
        return tm_exclusion_state(path)
    except Exception:
        return False

def protect_model_storage():
    """Keeps every place we may store weights out of Time Machine backups."""
    internal_lib = os.path.expanduser(os.path.join("~", LIBRARY_DIRNAME))
    try:
        # Create the internal library up front so its exclusion is in place before
        # any download could land there.
        os.makedirs(internal_lib, exist_ok=True)
    except Exception:
        pass
    targets = [internal_lib]
    targets.append(os.path.expanduser("~/.cache/huggingface"))
    drives = list_storage_drives()
    default = get_default_drive(drives)
    for d in drives:
        lib = d.get("library") or ""
        if not lib:
            continue
        if default and d.get("id") == default.get("id") and d.get("connected"):
            try:
                os.makedirs(lib, exist_ok=True)
            except Exception:
                pass
        targets.append(os.path.dirname(lib))
    done = []
    for t in targets:
        if t and os.path.isdir(t) and not tm_exclusion_state(t):
            if exclude_from_time_machine(t):
                done.append(t)
    return done

def _time_machine_mounts():
    """Mount points that are Time Machine destinations — offered but flagged, never the default."""
    now = time.time()
    if now - _TM_CACHE["ts"] < 120:
        return _TM_CACHE["mounts"]
    mounts = set()
    try:
        out = subprocess.check_output(["tmutil", "destinationinfo"], text=True,
                                      timeout=10, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            m = re.search(r"Mount Point\s*:\s*(.+)$", line)
            if m:
                mounts.add(os.path.realpath(m.group(1).strip()))
    except Exception:
        pass
    _TM_CACHE["ts"] = now
    _TM_CACHE["mounts"] = mounts
    return mounts

def _volume_record(mount, name, internal=False, probe_path=None, library=None):
    """Builds a drive record for a mounted volume, or None if it cannot host a model library.

    The internal drive is probed through $HOME: "/" is the sealed system volume, which is
    read-only and reports no free space, while user data lives on a separate volume.
    """
    probe = probe_path or mount
    if not os.path.isdir(probe):
        return None
    try:
        st = os.statvfs(probe)
    except Exception:
        return None
    total_bytes = st.f_frsize * st.f_blocks
    free_bytes = st.f_frsize * st.f_bavail
    total_gb = total_bytes / (1024 ** 3)
    if total_gb < MIN_DRIVE_GB:
        return None
    if st.f_flag & getattr(os, "ST_RDONLY", 1):
        return None
    if not os.access(probe, os.W_OK):
        return None

    fs = _diskutil_field(mount, "File System Personality")
    fs_l = fs.lower()
    # Hugging Face caches rely on symlinks: only APFS/HFS+ can host a library.
    if not ("apfs" in fs_l or "hfs" in fs_l):
        return None
    if "ntfs" in fs_l or "exfat" in fs_l or "ms-dos" in fs_l or "fat32" in fs_l:
        return None

    uuid = _diskutil_field(mount, "Volume UUID")
    drive_id = uuid if uuid and uuid not in ("Not applicable", "-") else f"mount:{mount}"

    library = library or os.path.join(mount, LIBRARY_DIRNAME, "hub")
    model_count = 0
    if os.path.isdir(library):
        try:
            model_count = len([d for d in os.listdir(library) if d.startswith("models--")])
        except Exception:
            model_count = 0

    is_tm = os.path.realpath(mount) in _time_machine_mounts()

    return {
        "id": drive_id,
        "name": name,
        "mount": mount,
        "internal": bool(internal),
        "fs": fs or "unknown",
        "total_gb": round(total_gb, 1),
        "free_gb": round(free_bytes / (1024 ** 3), 1),
        "library": library,
        "library_exists": os.path.isdir(library),
        "models": model_count,
        "is_time_machine": is_tm,
        "tm_excluded": tm_exclusion_state(library),
        "connected": True,
    }

def list_storage_drives(force=False, include_known=True):
    """Live list of mounted volumes usable as model libraries (discovered, never configured)."""
    now = time.time()
    if not force and _DRIVE_CACHE["drives"] and (now - _DRIVE_CACHE["ts"] < DRIVE_CACHE_TTL):
        drives = [dict(d) for d in _DRIVE_CACHE["drives"]]
    else:
        drives = []
        root = _volume_record("/", "Macintosh HD", internal=True,
                              probe_path=os.path.expanduser("~"),
                              library=os.path.expanduser(os.path.join("~", LIBRARY_DIRNAME, "hub")))
        if root:
            drives.append(root)

        try:
            entries = sorted(os.listdir("/Volumes"))
        except Exception:
            entries = []
        for entry in entries:
            if entry.startswith("."):
                continue
            mount = os.path.join("/Volumes", entry)
            if not os.path.ismount(mount):
                continue
            rec = _volume_record(mount, entry)
            if rec:
                drives.append(rec)

        _DRIVE_CACHE["ts"] = now
        _DRIVE_CACHE["drives"] = [dict(d) for d in drives]

    cfg = load_storage_config()
    disabled = set(cfg.get("disabled") or [])
    seen = {d["id"] for d in drives}
    known = cfg.get("known") or {}

    # Remember connected drives so an unplugged one can still be shown as "not connected".
    changed = False
    for d in drives:
        prev = known.get(d["id"]) or {}
        entry = {
            "name": d["name"],
            "mount": d["mount"],
            "internal": d["internal"],
            "models": d["models"],
            "total_gb": d["total_gb"],
            "is_time_machine": d["is_time_machine"],
        }
        if prev != entry:
            known[d["id"]] = entry
            changed = True
    if changed:
        cfg["known"] = known
        save_storage_config(cfg)

    for d in drives:
        d["disabled"] = d["id"] in disabled
        d["is_default"] = (d["id"] == cfg.get("default_drive"))

    if include_known:
        for did, info in known.items():
            if did in seen:
                continue
            mount = info.get("mount", "") or ""
            library = os.path.join(mount, LIBRARY_DIRNAME, "hub") if mount else ""
            # Only keep offline rows that still have something to show: a model
            # count persisted in the config is only trusted while the library
            # directory actually exists (otherwise the drive was wiped).
            offline_models = info.get("models", 0) if library and os.path.isdir(library) else 0
            if not offline_models and not info.get("pinned"):
                continue
            drives.append({
                "id": did,
                "name": info.get("name", "Unknown drive"),
                "mount": mount,
                "internal": bool(info.get("internal")),
                "fs": "unknown",
                "total_gb": info.get("total_gb", 0),
                "free_gb": 0,
                "library": library,
                "library_exists": bool(library and os.path.isdir(library)),
                "models": offline_models,
                "is_time_machine": bool(info.get("is_time_machine")),
                "connected": False,
                "disabled": did in disabled,
                "is_default": (did == cfg.get("default_drive")),
            })

    drives.sort(key=lambda d: (not d.get("is_default"), d.get("internal", False),
                              not d.get("connected", True), d.get("name", "")))
    return drives

def get_drive(drive_id):
    """Looks up a drive by UUID (or mount path / name as a fallback)."""
    for d in list_storage_drives():
        if d["id"] == drive_id or d["mount"] == drive_id or d["name"] == drive_id:
            return d
    return None

def pick_default_drive(drives=None):
    """Prefers a connected external (non-Time-Machine) drive with the most free space."""
    drives = drives if drives is not None else list_storage_drives()
    connected = [d for d in drives if d.get("connected") and not d.get("disabled")]
    if not connected:
        return None
    ranked = sorted(
        connected,
        key=lambda d: (
            bool(d.get("is_time_machine")),
            bool(d.get("internal")),
            -float(d.get("free_gb") or 0),
        ),
    )
    return ranked[0]

def get_default_drive(drives=None):
    """Configured default drive, falling back to the best available volume."""
    cfg = load_storage_config()
    configured = cfg.get("default_drive") or ""
    drives = drives if drives is not None else list_storage_drives()
    if configured:
        for d in drives:
            if d["id"] == configured and d.get("connected") and not d.get("disabled"):
                return d
    chosen = pick_default_drive(drives)
    if chosen and chosen["id"] != configured:
        cfg["default_drive"] = chosen["id"]
        save_storage_config(cfg)
    return chosen

def set_default_drive(drive_id):
    cfg = load_storage_config()
    cfg["default_drive"] = drive_id or ""
    save_storage_config(cfg)
    _DRIVE_CACHE["ts"] = 0.0
    return get_default_drive(list_storage_drives(force=True))

def resolve_library_hub(drive_id=None):
    """Returns (hub_path, drive_record) without creating anything."""
    drive = get_drive(drive_id) if drive_id else get_default_drive()
    if not drive:
        return os.path.expanduser("~/.cache/huggingface/hub"), None
    if not drive.get("connected"):
        return drive.get("library") or "", drive
    return drive["library"], drive

def ensure_library_hub(drive_id=None):
    """Creates <drive>/MLX-Models/hub on demand and returns (hub_path, drive)."""
    hub, drive = resolve_library_hub(drive_id)
    if hub and drive and drive.get("connected"):
        try:
            os.makedirs(hub, exist_ok=True)
        except Exception:
            pass
        # Multi-GB weights are re-downloadable: keep them out of Time Machine.
        exclude_from_time_machine(os.path.dirname(hub))
    return hub, drive

def library_hub_paths():
    """Hub directories for every connected drive, plus the legacy internal cache."""
    paths = []
    legacy = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.isdir(legacy):
        paths.append(legacy)
    for d in list_storage_drives():
        if d.get("connected") and d.get("library") and d["library"] not in paths:
            paths.append(d["library"])
    return paths

def drive_for_path(path):
    """Maps an absolute model path to the drive record that hosts it."""
    if not path:
        return None
    real = os.path.realpath(path)
    best = None
    for d in list_storage_drives():
        lib = d.get("library") or ""
        if not lib or not d.get("connected"):
            continue
        base = os.path.dirname(lib)
        if real == os.path.realpath(base) or real.startswith(os.path.realpath(base) + os.sep):
            if best is None or len(base) > len(os.path.dirname(best.get("library") or "")):
                best = d
    return best

DSH_PATCH_CANDIDATES = [
    os.path.expanduser("~/.dsh/profiles/desktop/cordis.patch.yml"),
    os.path.expanduser("~/.dsh/profiles/web/cordis.patch.yml"),
]
DSH_PROVIDER_ID = "mlx-local"

def dsh_patch_path():
    """Where the DeepSeek Harness user patch file lives (override with DSH_PATCH_FILE)."""
    override = os.environ.get("DSH_PATCH_FILE")
    if override:
        return os.path.expanduser(override)
    for cand in DSH_PATCH_CANDIDATES:
        if os.path.exists(cand):
            return cand
    return DSH_PATCH_CANDIDATES[0]

def dsh_model_label(repo_id):
    """'mlx-community/gemma-4-E4B-it-MLX-4bit' -> 'gemma 4 E4B it (MLX)'."""
    stem = repo_id.split("/")[-1]
    quant = ""
    m = re.search(r"-(\d+)bit$", stem, flags=re.I)
    if m:
        quant = " " + m.group(1) + "-bit"
        stem = stem[:m.start()]
    stem = re.sub(r"-MLX$", "", stem, flags=re.I)
    pretty = stem.replace("-", " ").replace("_", " ").strip()
    if pretty:
        pretty = pretty[0].upper() + pretty[1:]
    return "%s%s (MLX)" % (pretty, quant)

def _yaml_quote(value):
    """Quote only when YAML needs it, so hand diffs stay readable."""
    text = str(value)
    if re.match(r"^[A-Za-z0-9_./][A-Za-z0-9_./ ()+-]*$", text) and not text.endswith(" "):
        return text
    return '"' + text.replace('"', '\\"') + '"'

def _format_dsh_models(entries):
    lines = []
    for e in entries:
        lines.append("          - id: " + _yaml_quote(e["id"]))
        lines.append("            name: " + _yaml_quote(e["name"]))
        for extra in e.get("extra") or []:
            if extra.split(":")[0].strip() in ("id", "name"):
                continue
            lines.append("            " + extra)
    return lines

def read_dsh_models(provider_id=DSH_PROVIDER_ID, patch_path=None):
    """Parse the provider's existing model entries out of the DSH patch file."""
    path = patch_path or dsh_patch_path()
    entries, prov_idx, models_idx = [], None, None
    try:
        raw = open(path).read().splitlines()
    except Exception:
        return {"ok": False, "patch": path, "entries": [], "error": "patch file unreadable"}
    prov_re = re.compile(r"^ {6}" + re.escape(provider_id) + r":\s*$")
    for i, line in enumerate(raw):
        if prov_re.match(line):
            prov_idx = i
            break
    if prov_idx is None:
        return {"ok": True, "patch": path, "entries": [], "provider": provider_id}
    for j in range(prov_idx + 1, len(raw)):
        if re.match(r"^ {6}\S", raw[j]):
            break
        if re.match(r"^ {8}models:\s*$", raw[j]):
            models_idx = j
            break
    if models_idx is not None:
        for line in raw[models_idx + 1:]:
            if not line.strip():
                continue
            if not re.match(r"^ {10,}", line):
                break
            m = re.match(r"^ {10,}- id: (.+)$", line)
            if m:
                entries.append({"id": m.group(1).strip().strip('"').strip("'"), "name": "", "extra": []})
            else:
                m = re.match(r"^ {12,}name: (.+)$", line)
                if m and entries:
                    entries[-1]["name"] = m.group(1).strip().strip('"').strip("'")
                elif entries and re.match(r"^ {12,}\S", line):
                    # keep capacity hints such as contextWindow / maxTokens
                    entries[-1].setdefault("extra", []).append(line.strip())
    return {"ok": True, "patch": path, "provider": provider_id, "entries": entries}

def _dsh_entry_is_vision(model):
    """Best-effort image capability for one sync entry.

    `find_local_models()` already resolves this from config.json; callers that pass
    bare repo ids get a second chance by inspecting the local snapshot before the
    name heuristic in `is_vision_model()`.
    """
    flag = model.get("is_vision")
    if flag is not None:
        return bool(flag)
    rid = (model.get("id") or "").strip()
    candidates = []
    if model.get("path"):
        candidates.append(model["path"])
    if rid:
        candidates.extend(_snapshot_dirs(rid))
    for cand in candidates:
        if cand and os.path.isdir(cand) and is_vision_model(cand):
            return True
    return is_vision_model(rid) if rid else False

def sync_dsh_models(provider_id=DSH_PROVIDER_ID, patch_path=None, models=None, dry_run=False):
    """Rewrite the provider's `models:` block with one entry per installed model.

    Vision models also get `input: [text, image]`. DSH treats a custom
    OpenAI-compatible provider as text-only unless the model declares image support,
    so without this line every attachment is rejected before it reaches the server.
    Hand-edited per-model lines (including `input`) are preserved.

    The DSH desktop profile loads this file at startup, so a restart is needed to see
    the result — which is exactly why regenerating it by hand is a chore worth automating.
    """
    path = patch_path or dsh_patch_path()
    if not os.path.exists(path):
        return {"ok": False, "error": "no DeepSeek Harness profile at " + path, "patch": path}
    if models is None:
        # name=None keeps any hand-written label already in the patch file
        models = [{
            "id": m.get("repo_id") or m.get("base_name") or m.get("name"),
            "name": None,
            "is_vision": m.get("is_vision"),
            "path": m.get("path"),
        } for m in find_local_models()]
    prior = {e["id"]: e for e in read_dsh_models(provider_id, path).get("entries", [])}
    existing = {k: (v.get("name") or "") for k, v in prior.items()}
    seen, entries = set(), []
    for m in models:
        rid = (m.get("id") or "").strip()
        if not rid or rid in seen:
            continue
        seen.add(rid)
        label = (m.get("name") or existing.get(rid) or dsh_model_label(rid)).strip()
        extra = list((prior.get(rid) or {}).get("extra") or [])
        have = {line.split(":")[0].strip() for line in extra}
        if _dsh_entry_is_vision(m) and "input" not in have:
            extra.insert(0, "input: [text, image]")
        cap = context_window_cap()
        ctx = cap if cap else (model_context_profile(rid).get("recommended") or model_context_window(rid))
        native = model_context_window(rid)
        if ctx and native:
            ctx = min(ctx, native)
        if ctx and "contextWindow" not in have:
            extra.append("contextWindow: %d" % ctx)
        if "maxTokens" not in have:
            try:
                configured = int(load_global_settings().get("default_max_tokens", 0) or 0)
            except Exception:
                configured = 0
            extra.append("maxTokens: %d" % output_token_budget(rid, configured))
        entries.append({"id": rid, "name": label, "extra": extra})
    entries.sort(key=lambda e: e["id"].lower())
    raw = open(path).read().splitlines()
    prov_re = re.compile(r"^ {6}" + re.escape(provider_id) + r":\s*$")
    prov_idx = next((i for i, l in enumerate(raw) if prov_re.match(l)), None)
    if prov_idx is None:
        return {"ok": False, "error": "provider '%s' is not in %s" % (provider_id, path), "patch": path}
    models_idx, block_end = None, None
    for j in range(prov_idx + 1, len(raw)):
        if re.match(r"^ {6}\S", raw[j]):
            block_end = j
            break
        if re.match(r"^ {8}models:\s*$", raw[j]) and models_idx is None:
            models_idx = j
    if block_end is None:
        block_end = len(raw)
    if models_idx is None:
        new_lines = ["        models:"] + _format_dsh_models(entries)
        out = raw[:prov_idx + 1] + new_lines + raw[prov_idx + 1:]
    else:
        end, trailing = models_idx + 1, []
        for k in range(models_idx + 1, block_end):
            if re.match(r"^ {10,}", raw[k]) and raw[k].strip():
                end = k + 1
            else:
                break
        for line in raw[models_idx + 1:end]:
            if line.strip().startswith("#"):
                trailing.append(line)
        out = raw[:models_idx + 1] + _format_dsh_models(entries) + trailing + raw[end:]
    before = read_dsh_models(provider_id, path).get("entries", [])
    changed = out != raw
    result = {
        "ok": True, "patch": path, "provider": provider_id, "changed": changed,
        "installed": len(entries), "before": [e["id"] for e in before],
        "entries": entries, "needs_restart": changed,
    }
    if dry_run or not changed:
        return result
    stamp = time.strftime("%Y%m%d-%H%M%S")
    try:
        import shutil as _shutil
        _shutil.copy2(path, path + ".bak-" + stamp)
        with open(path, "w") as f:
            f.write("\n".join(out) + "\n")
    except Exception as e:
        return {"ok": False, "error": str(e), "patch": path}
    result["backup"] = path + ".bak-" + stamp
    result["added"] = [e["id"] for e in entries if e["id"] not in result["before"]]
    result["removed"] = [i for i in result["before"] if i not in {e["id"] for e in entries}]
    return result

def repo_cache_dirname(repo_id):
    return "models--" + str(repo_id).replace("/", "--")

def model_context_window(model_name_or_path):
    """Read the model's native context length (tokens) from its own config.json.

    mlx-vlm accepts `--max-kv-size <tokens>`; without a declared capacity the dashboards
    and client apps fall back to guesses (DeepSeek Harness assumes 200k), so we read the
    real number instead of inventing one.
    """
    name = (model_name_or_path or "").strip().rstrip("/")
    if not name:
        return 0
    candidates = []
    if os.path.isdir(name):
        candidates.append(name)
    else:
        hub = hub_for_model(name)
        if hub and "/" in name:
            base = os.path.join(hub, repo_cache_dirname(name))
            refs = os.path.join(base, "refs", "main")
            rev = ""
            try:
                rev = open(refs).read().strip()
            except Exception:
                rev = ""
            for snap in ([os.path.join(base, "snapshots", rev)] if rev else []):
                candidates.append(snap)
            snaps = os.path.join(base, "snapshots")
            if os.path.isdir(snaps):
                for entry in sorted(os.listdir(snaps)):
                    candidates.append(os.path.join(snaps, entry))
    for cand in candidates:
        cfg_path = os.path.join(cand, "config.json")
        if not os.path.isfile(cfg_path):
            continue
        try:
            cfg = json.load(open(cfg_path))
        except Exception:
            continue
        for holder in (cfg.get("text_config") or {}, cfg):
            for key in ("max_position_embeddings", "max_seq_len", "seq_length",
                        "max_sequence_length", "n_positions", "model_max_length"):
                val = holder.get(key)
                if isinstance(val, int) and 1024 <= val <= 4_000_000:
                    return val
    return 0

def _snapshot_dirs(model_name_or_path):
    """Snapshot directories we can read config.json / weights from, locally only."""
    name = (model_name_or_path or "").strip().rstrip("/")
    dirs = []
    if not name:
        return dirs
    if os.path.isdir(name):
        dirs.append(name)
        return dirs
    hub = hub_for_model(name)
    if not hub or "/" not in name:
        return dirs
    base = os.path.join(hub, repo_cache_dirname(name))
    try:
        rev = open(os.path.join(base, "refs", "main")).read().strip()
        if rev:
            dirs.append(os.path.join(base, "snapshots", rev))
    except Exception:
        pass
    snaps = os.path.join(base, "snapshots")
    if os.path.isdir(snaps):
        for entry in sorted(os.listdir(snaps), reverse=True):
            path = os.path.join(snaps, entry)
            if path not in dirs:
                dirs.append(path)
    return dirs

def _weights_gb(snapshot_dir):
    total = 0
    try:
        for root, _dirs, files in os.walk(snapshot_dir):
            for fn in files:
                if fn.endswith((".safetensors", ".gguf", ".bin")):
                    try:
                        total += os.path.getsize(os.path.join(root, fn))
                    except OSError:
                        pass
    except Exception:
        return 0.0
    return round(total / 1e9, 2)

def _kv_bytes_per_token(cfg):
    """KV cache cost per token, honouring hybrid architectures (linear layers keep no KV)."""
    text = cfg.get("text_config") or cfg
    layers = text.get("num_hidden_layers") or cfg.get("num_hidden_layers") or 0
    kv_heads = text.get("num_key_value_heads") or text.get("num_attention_heads") or 0
    head_dim = text.get("head_dim")
    if not head_dim:
        hidden = text.get("hidden_size") or 0
        heads = text.get("num_attention_heads") or 1
        head_dim = hidden // heads if hidden and heads else 0
    layer_types = text.get("layer_types") or []
    full = sum(1 for t in layer_types if "full" in str(t) or "attention" == str(t))
    if layer_types:
        full_layers = full
    else:
        full_layers = layers
    if not (full_layers and kv_heads and head_dim):
        return 0
    return 2 * kv_heads * head_dim * 2 * full_layers  # K+V, fp16

def effective_context_window(model_name_or_path):
    """What a server/harness entry should use: the user cap if set, else the RAM-safe pick."""
    native = model_context_window(model_name_or_path)
    cap = context_window_cap()
    if cap:
        return min(cap, native) if native else cap
    profile = model_context_profile(model_name_or_path)
    return profile.get("recommended") or native

def model_context_profile(model_name_or_path):
    """Native window, KV cost, and a RAM-safe recommendation for this machine."""
    native = model_context_window(model_name_or_path)
    profile = {"native": native, "kv_kb_per_token": 0.0, "weights_gb": 0.0,
               "ram_budget_gb": 0.0, "ram_tokens": 0, "recommended": native,
               "limited_by": "native"}
    for snap in _snapshot_dirs(model_name_or_path):
        cfg_path = os.path.join(snap, "config.json")
        if not os.path.isfile(cfg_path):
            continue
        try:
            cfg = json.load(open(cfg_path))
        except Exception:
            continue
        kv = _kv_bytes_per_token(cfg)
        if not kv:
            break
        weights = _weights_gb(snap)
        stats = get_system_stats()
        total_ram = stats.get("total_ram_gb") or 0
        # keep 15 % of RAM for the OS, the app, and activation spikes
        budget = max(0.0, total_ram * 0.85 - max(weights, profile["weights_gb"]) - 3.0)
        ram_tokens = int(budget * 1e9 / kv) if budget else 0
        profile.update({
            "kv_kb_per_token": round(kv / 1024, 1),
            "weights_gb": weights,
            "ram_budget_gb": round(budget, 1),
            "ram_tokens": ram_tokens,
        })
        if native and ram_tokens and native > ram_tokens:
            step = 8192
            profile["recommended"] = max(step, (ram_tokens // step) * step)
            profile["limited_by"] = "ram"
        else:
            profile["recommended"] = native or ram_tokens
            profile["limited_by"] = "native" if native else "ram"
        break
    if not profile["recommended"]:
        profile["recommended"] = native or 0
    return profile

OUTPUT_TOKEN_FLOOR = 4096
OUTPUT_TOKEN_CEILING = 32768

def output_token_budget(model_name_or_path=None, requested=None):
    """Reply budget (tokens) that fits whichever model is loaded.

    A single constant is wrong in both directions: too small truncates a long tool
    call on a big model, too large asks a small-context model for more than it can
    hold. So the fallback scales with the model's effective context window — a
    quarter of it — floored at 4k tokens (enough for any agentic step) and capped
    at 32k (more than a turn ever needs, and already minutes of decoding locally).
    An explicit `requested` value from the dashboard or settings always wins, only
    clamped to what the model can actually hold.
    """
    try:
        want = int(requested or 0)
    except (TypeError, ValueError):
        want = 0
    ctx = 0
    if model_name_or_path:
        try:
            ctx = int(effective_context_window(model_name_or_path) or 0)
        except Exception:
            ctx = 0
    if want > 0:
        return min(want, ctx) if ctx and want > ctx else want
    if ctx <= 0:
        # Nothing known about the model: a middle-of-the-road budget beats a guess
        # that is either tiny or minutes long.
        return 8192
    return max(OUTPUT_TOKEN_FLOOR, min(OUTPUT_TOKEN_CEILING, ctx // 4))

def context_window_cap():
    """User cap from settings (0 = trust the model's own maximum)."""
    try:
        value = int(load_global_settings().get("default_context_window", 0) or 0)
    except Exception:
        value = 0
    return max(0, value)

def hub_for_model(model_name):
    """Find the library hub that already holds this repo, so the server can serve it offline.

    `mlx-vlm` resolves a request's `model` field through huggingface_hub, which defaults to
    ~/.cache/huggingface/hub — on a Mac with models on an external drive that means a silent
    re-download of tens of GB. Pointing HF_HUB_CACHE at the library avoids that.
    """
    name = (model_name or "").strip().rstrip("/")
    if not name:
        return ""
    candidates = []
    for hub in library_hub_paths():
        for repo in (name, name.replace("_", "/")):
            if "/" in repo:
                d = os.path.join(hub, repo_cache_dirname(repo))
                if os.path.isdir(d):
                    candidates.append(hub)
    # a path inside a library (…/models--org--repo/snapshots/<rev>) also identifies its hub
    if not candidates and os.sep in name:
        probe = name
        for _ in range(4):
            probe = os.path.dirname(probe)
            base = os.path.basename(probe)
            if base.startswith("models--"):
                candidates.append(os.path.dirname(probe))
                break
    return candidates[0] if candidates else ""

def get_download_cmd(repo_id, subfolder="", revision="", cache_dir="", include=None):
    """Finds available Hugging Face download CLI tool (hf, huggingface-cli, or uvx fallback)."""
    import shutil
    hf_path = shutil.which("hf")
    if not hf_path:
        for p in [os.path.expanduser("~/.local/bin/hf"), "/opt/homebrew/bin/hf", "/usr/local/bin/hf", os.path.expanduser("~/.cargo/bin/hf")]:
            if os.path.isfile(p) and os.access(p, os.X_OK):
                hf_path = p
                break

    if hf_path:
        cmd = [hf_path, "download", repo_id]
    elif shutil.which("huggingface-cli") or os.path.isfile(os.path.expanduser("~/.local/bin/huggingface-cli")):
        cmd = ["huggingface-cli", "download", repo_id]
    elif shutil.which("uvx") or os.path.isfile(os.path.expanduser("~/.local/bin/uvx")):
        uvx_bin = shutil.which("uvx") or os.path.expanduser("~/.local/bin/uvx")
        # NB: uvx ignores bracketed extras on --from; ask for the Xet engine directly.
        cmd = [uvx_bin, "--from", "huggingface_hub", "--with", "hf_xet", "hf", "download", repo_id]
    else:
        cmd = ["hf", "download", repo_id]

    if subfolder:
        cmd.extend(["--include", f"{subfolder}/*"])
    for pattern in (include or []):
        cmd.extend(["--include", pattern])
    if revision and revision != "main":
        cmd.extend(["--revision", revision])
    if cache_dir:
        cmd.extend(["--cache-dir", cache_dir])
    return cmd

def parse_hf_identifier(input_str, explicit_subfolder=""):
    """Parses any Hugging Face URL, repo ID with subfolder, or raw path into structured components."""
    s = (input_str or "").strip()
    subfolder = (explicit_subfolder or "").strip()
    revision = "main"
    s = re.sub(r"/+$", "", s)

    if "huggingface.co/" in s:
        path_part = s.split("huggingface.co/")[-1].strip("/")
        m_tree = re.match(r"^([^/]+/[^/]+)/tree/([^/]+)(?:/(.+))?$", path_part)
        m_blob = re.match(r"^([^/]+/[^/]+)/blob/([^/]+)(?:/(.+))?$", path_part)
        if m_tree:
            repo_id = m_tree.group(1)
            revision = m_tree.group(2)
            if m_tree.group(3):
                subfolder = m_tree.group(3).strip("/")
        elif m_blob:
            repo_id = m_blob.group(1)
            revision = m_blob.group(2)
            if m_blob.group(3):
                rest = m_blob.group(3).strip("/")
                if "/" in rest:
                    subfolder = os.path.dirname(rest)
                elif not any(rest.endswith(ext) for ext in [".json", ".safetensors", ".bin", ".md", ".jinja", ".txt"]):
                    subfolder = rest
        else:
            parts = path_part.split("/")
            if len(parts) >= 2:
                repo_id = f"{parts[0]}/{parts[1]}"
                if len(parts) > 2 and not subfolder:
                    subfolder = "/".join(parts[2:])
            else:
                repo_id = path_part
    elif ":" in s:
        repo_id, sf = s.split(":", 1)
        subfolder = sf.strip()
    else:
        parts = s.split("/")
        if len(parts) > 2:
            repo_id = f"{parts[0]}/{parts[1]}"
            if not subfolder:
                subfolder = "/".join(parts[2:])
        else:
            repo_id = s

    display_name = f"{repo_id} ({subfolder})" if subfolder else repo_id
    download_id = f"{repo_id}:{subfolder}" if subfolder else repo_id

    return {
        "repo_id": repo_id,
        "subfolder": subfolder,
        "revision": revision,
        "display_name": display_name,
        "download_id": download_id
    }

def with_gguf_projectors(repo_id, include, revision="main"):
    """Add the mmproj/projector files to a GGUF selection.

    A vision GGUF is useless without its projector, and the picker only lists the
    weights file the user clicked. Pulling the projector in automatically is what
    makes 'download a vision GGUF, press Import' work.
    """
    patterns = [str(p) for p in (include or [])]
    if not patterns or not any(p.lower().endswith(".gguf") for p in patterns):
        return patterns
    try:
        files = hf_repo_gguf_files(repo_id, revision=revision)
    except Exception:
        return patterns
    for f in files:
        path = f.get("path") or ""
        low = path.lower()
        if ("mmproj" in low or "projector" in low or os.path.basename(low).startswith("clip")) and path not in patterns:
            patterns.append(path)
    return patterns

def hf_repo_gguf_files(repo_id, revision="main", timeout=12):
    """List the .gguf files in a repo with their sizes, smallest quant names last.

    A GGUF repo usually holds half a dozen quants; downloading the whole thing would
    pull 100+ GB, so the UI asks which file to fetch and passes it through as an
    --include pattern.
    """
    repo_id = (repo_id or "").strip().strip("/")
    if not repo_id:
        return []
    url = (f"https://huggingface.co/api/models/{repo_id}/tree/"
           f"{urllib.parse.quote(revision or 'main')}?recursive=true")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    out = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return []
    for item in data or []:
        path = item.get("path") or ""
        if not path.lower().endswith(".gguf") or item.get("type") != "file":
            continue
        size = item.get("size") or (item.get("lfs") or {}).get("size") or 0
        out.append({
            "path": path,
            "size_gb": round(size / (1024 ** 3), 2),
            "quant": _detect_quant_label(path) or "",
        })
    out.sort(key=lambda f: (f["size_gb"], f["path"]))
    return out

def get_repo_files_size(repo_id, subfolder="", include=None):
    """Total bytes of the files a download will actually fetch.

    A GGUF repo can hold twenty quants of one model; without `include` this adds up
    every quant in the repo and the free-space guard then refuses a 44 GB download
    because the repo's other quants need 700 GB.
    """
    patterns = [str(p) for p in (include or []) if str(p).strip()]
    cache_key = f"{repo_id}:{subfolder}"
    if patterns:
        cache_key += ":include=" + "|".join(patterns)
    if not patterns and not subfolder:
        cache_key = repo_id
    if cache_key in _REPO_SIZE_CACHE:
        return _REPO_SIZE_CACHE[cache_key]
    try:
        url = f"https://huggingface.co/api/models/{urllib.parse.quote(repo_id)}/tree/main?recursive=true"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            files = json.loads(resp.read().decode())
            total = 0
            sf_prefix = f"{subfolder.strip('/')}/" if subfolder else ""
            for f in files:
                if isinstance(f, dict) and "size" in f:
                    path = f.get("path", "")
                    sz = f.get("size", 0)
                    if subfolder and not (path.startswith(sf_prefix) or path == subfolder):
                        continue
                    if patterns and not any(fnmatch.fnmatch(path, pat) for pat in patterns):
                        continue
                    total += sz
            if total > 0:
                _REPO_SIZE_CACHE[cache_key] = total
                return total
    except Exception:
        pass
    return 0

def get_repo_quant_variants(repo_id):
    """Detects if a repo has multiple quantization subfolders (e.g. 2-bit, 4-bit, 6-bit, 8-bit)."""
    try:
        url = f"https://huggingface.co/api/models/{urllib.parse.quote(repo_id)}/tree/main"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            items = json.loads(resp.read().decode())
            dirs = [it.get("path", "") for it in items if it.get("type") == "directory"]
            
            quant_dirs = []
            for d in dirs:
                dl = d.lower()
                if any(q in dl for q in ["bit", "quant", "bf16", "fp16", "q4", "q8", "int4", "int8", "awq", "gptq", "mlx"]):
                    quant_dirs.append(d)
                elif re.match(r"^\d+(-|_)?bit$", dl):
                    quant_dirs.append(d)
            
            # No quant-looking subfolder means this repo IS a single quantization:
            # a `manifests/` or `assets/` directory must never look like a choice.
            if not quant_dirs:
                return []

            variants = []
            for qd in quant_dirs:
                sz = get_repo_files_size(repo_id, subfolder=qd)
                if sz < 100 * 1024 * 1024:  # config/manifest folders are not variants
                    continue
                sz_gb = sz / (1024**3)
                sz_str = f"{sz_gb:.1f} GB" if sz_gb >= 1 else f"{sz/(1024**2):.1f} MB"
                label = qd
                if "4-bit" in qd.lower() or "4bit" in qd.lower():
                    label = f"{qd} (Recommended)"
                variants.append({
                    "subfolder": qd,
                    "label": label,
                    "size_bytes": sz,
                    "size_str": sz_str,
                    "size_gb": round(sz_gb, 2)
                })
            return variants
    except Exception:
        return []

def load_paused_downloads():
    if os.path.exists(PAUSED_FILE):
        try:
            with open(PAUSED_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_paused_downloads(data):
    try:
        with open(PAUSED_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass

def _read_json_file(path):
    try:
        with open(path, "r", errors="ignore") as f:
            return json.load(f)
    except Exception:
        return None

_VISION_MODEL_TOKENS = (
    "vision", "multimodal", "multi_mod", "vlm", "omni", "llava", "paligemma",
    "pixtral", "mllama", "idefics", "molmo", "smolvlm", "internvl", "minicpmv",
    "florence", "chameleon", "blip", "fuyu", "kosmos", "donut", "pix2struct",
    "qwen2vl", "qwen2_5_vl", "qwen3vl", "qwen_vl", "gemma3", "gemma4",
    "phi3_v", "phi3v", "mistral3",
)
_VISION_CONFIG_KEYS = (
    "vision_config", "vision_tower", "mm_vision_tower", "multi_modal_projector",
    "mm_projector_type", "projector_type", "image_token_id", "image_token_index",
    "vision_feature_layer", "vision_feature_select_strategy", "image_seq_length",
    "vision_soft_tokens_per_image",
)
_VISION_WEIGHT_HINTS = (
    "vision_tower", "vision_model", "vision_encoder", "visual.", "visual_encoder",
    "multi_modal_projector", "mm_projector", "image_encoder", "image_projection",
    "vision_projection",
)

def _model_type_is_vision(value):
    text = str(value or "").lower()
    if not text:
        return False
    if any(tok in text for tok in _VISION_MODEL_TOKENS):
        return True
    # Arch-independent conventions: Qwen2VL / InternVL / SmolVLM / ...
    return "_vl" in text or text.endswith("vl") or text.startswith("vl")

def _config_declares_vision(cfg):
    if not isinstance(cfg, dict):
        return False
    if _model_type_is_vision(cfg.get("model_type")):
        return True
    if _model_type_is_vision(" ".join(str(a) for a in (cfg.get("architectures") or []))):
        return True
    for key in _VISION_CONFIG_KEYS:
        val = cfg.get(key)
        if val not in (None, "", False, [], {}):
            return True
    return False

def _image_processor_declares_vision(cfg):
    """True when a preprocessor/processor JSON describes image (not audio) work."""
    if not isinstance(cfg, dict):
        return False
    if cfg.get("image_processor_type") or cfg.get("image_processor"):
        return True
    image_keys = ("image_mean", "image_std", "crop_size", "do_resize", "patch_size", "image_size")
    audio_keys = ("feature_extractor_type", "sampling_rate", "feature_size", "chunk_length", "n_fft", "hop_length")
    return any(k in cfg for k in image_keys) and not any(k in cfg for k in audio_keys)

def _safetensors_declare_vision(model_dir):
    """Read only the JSON headers: vision towers are visible in the weight names."""
    names = []
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    index = _read_json_file(index_path) if os.path.isfile(index_path) else None
    if isinstance(index, dict) and index.get("weight_map"):
        names = list(index["weight_map"].keys())
    else:
        try:
            shards = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))[:8]
        except Exception:
            shards = []
        for fn in shards:
            try:
                with open(os.path.join(model_dir, fn), "rb") as f:
                    size = int.from_bytes(f.read(8), "little")
                    if 0 < size <= 64 * 1024 * 1024:
                        header = json.loads(f.read(size))
                        names.extend(header.keys())
            except Exception:
                continue
    return any(hint in name for name in names for hint in _VISION_WEIGHT_HINTS)

def is_vision_model(model_path_or_id, config_dict=None):
    """Detect image input from model config, processor files, or weight names.

    Capability is read from the model itself, never from a fixed list of families:
    a `vision_config`/`image_token_id`, an image processor, or a `vision_tower` in
    the weights all count, so a new multimodal architecture is picked up without a
    code change. Name heuristics are only a fallback for remote repo ids.
    """
    path = str(model_path_or_id)
    name_check = path.lower()
    is_dir = os.path.isdir(path)

    cfg = config_dict
    if cfg is None and is_dir:
        cfg = _read_json_file(os.path.join(path, "config.json"))
    if _config_declares_vision(cfg):
        return True

    if is_dir:
        for fn in ("preprocessor_config.json", "processor_config.json",
                   "video_preprocessor_config.json", "image_processor_config.json"):
            proc = _read_json_file(os.path.join(path, fn))
            if _image_processor_declares_vision(proc):
                return True
        if _safetensors_declare_vision(path):
            return True

    # Last resort for remote repo ids: nothing local to inspect.
    vision_keywords = (
        "-vl-", "-vl", "_vl", "vl-", "vlm", "vision", "multimodal", "multi-modal",
        "llava", "paligemma", "pixtral", "mllama", "idefics", "smolvlm",
        "internvl", "minicpm-v", "florence", "molmo", "chameleon", "blip",
        "fuyu", "kosmos", "donut", "pix2struct", "gemma-3", "gemma-4",
        "qwen2-vl", "qwen2.5-vl", "qwen3-vl", "phi-3-vision", "mistral-small-3.1",
        "omni",
    )
    if any(k in name_check for k in vision_keywords):
        return True
    if any(k in name_check for k in ("code-reasoning", "r1-distill", "coder")):
        return False
    return False

OLLAMA_DSH_PROVIDER = "ollama-local"

def sync_dsh_ollama(provider_id=OLLAMA_DSH_PROVIDER, patch_path=None, dry_run=False, port=None):
    """Add or refresh an Ollama provider inside the DeepSeek Harness patch.

    A DSH provider carries one baseURL, so Ollama models cannot live under the MLX
    provider: the daemon answers on its own port. The block is rewritten in place,
    and only its models list changes on repeat runs.
    """
    path = patch_path or dsh_patch_path()
    if not os.path.exists(path):
        return {"ok": False, "error": "no DeepSeek Harness profile at " + path, "patch": path}
    cfg = ollama_settings()
    port = int(port or cfg["port"])
    models = []
    try:
        models = ollama_harness_models(port=port)
    except Exception:
        models = []
    raw = open(path).read().splitlines()
    body = ["        displayName: Ollama (this Mac)",
            "        api: openai-completions",
            "        baseURL: http://127.0.0.1:%d/v1" % port]
    if models:
        body.append("        models:")
        for m in models:
            body.append("          - id: %s" % m["id"])
            body.append("            name: %s" % (m["id"] + " (Ollama)"))
            body.append("            contextWindow: %d" % int(m["contextWindow"]))
            body.append("            maxTokens: %d" % int(m["maxTokens"]))
            if m.get("is_vision"):
                body.append("            input: [ text, image ]")
    else:
        body.append("        models: []")
    body.append("        apiKeyEnv: OLLAMA_LOCAL_API_KEY")

    start_re = re.compile(r"^ {6}" + re.escape(provider_id) + r":\s*$")
    start = next((i for i, l in enumerate(raw) if start_re.match(l)), None)
    if start is None:
        prov_idx = next((i for i, l in enumerate(raw) if l.strip() == "providers:"
                         and re.match(r"^ {4}providers:", l)), None)
        if prov_idx is None:
            return {"ok": False, "error": "no providers: block in " + path, "patch": path}
        ins = None
        for j in range(prov_idx + 1, len(raw)):
            if not raw[j].strip() or raw[j].startswith("#"):
                continue
            if re.match(r"^ {0,4}\S", raw[j]) or raw[j].startswith("- "):
                ins = j
                break
            ins = j + 1
        if ins is None:
            ins = len(raw)
        new = raw[:ins] + ["      %s:" % provider_id] + body + raw[ins:]
        entries_note = "inserted"
    else:
        end = len(raw)
        for j in range(start + 1, len(raw)):
            if raw[j].strip() and not raw[j].startswith(" "):
                end = j
                break
            if re.match(r"^ {6}\S", raw[j]):
                end = j
                break
        new = raw[:start] + ["      %s:" % provider_id] + body + raw[end:]
        entries_note = "updated"
    changed = new != raw
    result = {"ok": True, "patch": path, "provider": provider_id, "changed": changed,
              "models": [m["id"] for m in models], "action": entries_note,
              "base_url": "http://127.0.0.1:%d/v1" % port}
    if dry_run or not changed:
        return result
    stamp = time.strftime("%Y%m%d-%H%M%S")
    try:
        shutil.copy2(path, path + ".bak-" + stamp)
        with open(path, "w") as f:
            f.write("\n".join(new) + "\n")
    except Exception as exc:
        return {"ok": False, "error": str(exc), "patch": path}
    result["backup"] = path + ".bak-" + stamp
    return result

def sync_pi_ollama():
    """Write only the Ollama provider into Pi Coder and OpenCode, leaving MyMac alone."""
    cfg = ollama_settings()
    port = int(cfg["port"])
    models = ollama_harness_models(port=port)
    done = []

    pi_models_file = os.path.expanduser("~/.pi/agent/models.json")
    if os.path.exists(pi_models_file):
        try:
            with open(pi_models_file, "r") as f:
                md = json.load(f)
            md.setdefault("providers", {})
            if models:
                md["providers"]["Ollama"] = {
                    "baseUrl": f"http://127.0.0.1:{port}/v1",
                    "api": "openai-completions",
                    "apiKey": "local",
                    "name": "Ollama",
                    "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
                    "models": [{
                        "id": m["id"],
                        "name": f"{m['id']} (Ollama)",
                        "reasoning": False,
                        "input": ["text", "image"] if m.get("is_vision") else ["text"],
                        "contextWindow": m["contextWindow"],
                        "maxTokens": m["maxTokens"],
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    } for m in models],
                }
            else:
                md["providers"].pop("Ollama", None)
            with open(pi_models_file, "w") as f:
                json.dump(md, f, indent=2)
            done.append("pi")
        except Exception:
            pass

    for opf in (os.path.expanduser("~/.config/opencode/opencode.jsonc"),
                os.path.expanduser("~/opencode.jsonc")):
        if not os.path.exists(opf):
            continue
        try:
            with open(opf, "r") as f:
                opd = json.load(f)
            opd.setdefault("provider", {})
            if models:
                opd["provider"]["Ollama"] = {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Ollama",
                    "options": {"baseURL": f"http://127.0.0.1:{port}/v1"},
                    "models": {m["id"]: {"name": f"{m['id']} (Ollama)"} for m in models},
                }
            else:
                opd["provider"].pop("Ollama", None)
            with open(opf, "w") as f:
                json.dump(opd, f, indent=2)
            done.append("opencode")
        except Exception:
            pass
    return done

def sync_ollama_agents(port=None, patch_path=None):
    """Write Ollama into every agent config that can take it. Never raises."""
    cfg = ollama_settings()
    port = int(port or cfg["port"])
    out = {"ok": True, "port": port, "pi": None, "dsh": None}
    try:
        out["pi"] = ",".join(sync_pi_ollama()) or "no configs found"
    except Exception as exc:
        out["pi"] = "error: %s" % exc
    try:
        out["dsh"] = sync_dsh_ollama(patch_path=patch_path, port=port)
    except Exception as exc:
        out["dsh"] = {"ok": False, "error": str(exc)}
    return out

def sync_pi_and_opencode(active_model, port=9999, is_vision=False):
    """Automatically updates Pi Code and OpenCode configurations to auto-detect the active MLX model under provider 'MyMac'."""
    try:
        # 1. Update Pi settings.json
        pi_settings_file = os.path.expanduser("~/.pi/agent/settings.json")
        if os.path.exists(pi_settings_file):
            try:
                with open(pi_settings_file, "r") as f:
                    st = json.load(f)
                st["defaultProvider"] = "MyMac"
                st["defaultModel"] = "default_model"
                with open(pi_settings_file, "w") as f:
                    json.dump(st, f, indent=2)
            except Exception:
                pass

        # 2. Update Pi models.json
        pi_models_file = os.path.expanduser("~/.pi/agent/models.json")
        if os.path.exists(pi_models_file):
            try:
                with open(pi_models_file, "r") as f:
                    md = json.load(f)
                if "providers" not in md:
                    md["providers"] = {}
                
                inputs = ["text", "image"] if is_vision else ["text"]
                try:
                    model_ctx = int(effective_context_window(active_model) or 0)
                except Exception:
                    model_ctx = 0
                if not model_ctx:
                    model_ctx = 131072
                try:
                    configured = int(load_global_settings().get("default_max_tokens", 0) or 0)
                except Exception:
                    configured = 0
                model_budget = output_token_budget(active_model, configured)
                md["providers"]["MyMac"] = {
                    "baseUrl": f"http://127.0.0.1:{port}/v1",
                    "api": "openai-completions",
                    "apiKey": "local",
                    "name": "MyMac",
                    "compat": {
                        "supportsDeveloperRole": False,
                        "supportsReasoningEffort": False
                    },
                    "models": [
                        {
                            "id": "default_model",
                            "name": f"⚡ Active MLX Model ({'Vision' if is_vision else 'Text'} Auto-Detect)",
                            "reasoning": True,
                            "input": inputs,
                            "contextWindow": model_ctx,
                            "maxTokens": model_budget,
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
                        },
                        {
                            "id": active_model,
                            "name": f"{active_model} (Local)",
                            "reasoning": True,
                            "input": inputs,
                            "contextWindow": model_ctx,
                            "maxTokens": model_budget,
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
                        }
                    ]
                }
                try:
                    ol_models = ollama_harness_models()
                except Exception:
                    ol_models = []
                if ol_models:
                    ol_port = ollama_settings()["port"]
                    md["providers"]["Ollama"] = {
                        "baseUrl": f"http://127.0.0.1:{ol_port}/v1",
                        "api": "openai-completions",
                        "apiKey": "local",
                        "name": "Ollama",
                        "compat": {
                            "supportsDeveloperRole": False,
                            "supportsReasoningEffort": False
                        },
                        "models": [
                            {
                                "id": m["id"],
                                "name": f"{m['id']} (Ollama)",
                                "reasoning": False,
                                "input": ["text", "image"] if m.get("is_vision") else ["text"],
                                "contextWindow": m["contextWindow"],
                                "maxTokens": m["maxTokens"],
                                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
                            } for m in ol_models
                        ]
                    }
                else:
                    md["providers"].pop("Ollama", None)
                with open(pi_models_file, "w") as f:
                    json.dump(md, f, indent=2)
            except Exception:
                pass

        # 3. Update OpenCode configs
        opencode_files = [
            os.path.expanduser("~/.config/opencode/opencode.jsonc"),
            os.path.expanduser("~/opencode.jsonc")
        ]
        for opf in opencode_files:
            if os.path.exists(opf):
                try:
                    with open(opf, "r") as f:
                        opd = json.load(f)
                    opd["model"] = "MyMac/default_model"
                    if "provider" not in opd:
                        opd["provider"] = {}
                    
                    opd["provider"]["MyMac"] = {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "MyMac",
                        "options": {
                            "baseURL": f"http://127.0.0.1:{port}/v1"
                        },
                        "models": {
                            "default_model": {
                                "name": f"⚡ Active MLX Model ({'Vision' if is_vision else 'Text'} Auto-Detect)"
                            },
                            active_model: {
                                "name": active_model
                            }
                        }
                    }
                    with open(opf, "w") as f:
                        json.dump(opd, f, indent=2)
                except Exception:
                    pass
    except Exception:
        pass

def find_local_models(extra_roots=None):
    """Scans every storage library (internal cache + each mounted drive) for installed models."""
    search_paths = []
    for p in library_hub_paths():
        if p not in search_paths:
            search_paths.append(p)
    for p in [
        os.path.expanduser("~/MLXConsole/models/hub"),
        os.path.expanduser("~/.lmstudio/hub/models"),
        os.path.expanduser("~/.cache/lmstudio/models"),
        os.path.expanduser("~/mlx_models"),
    ]:
        if p not in search_paths:
            search_paths.append(p)
    for p in (extra_roots or []):
        if p and p not in search_paths:
            search_paths.append(p)
    models = []
    seen_paths = set()

    for spath in search_paths:
        if not os.path.exists(spath):
            continue
        for root, dirs, files in os.walk(spath):
            has_config = "config.json" in files
            has_gguf = any(f.endswith(".gguf") for f in files)

            if (has_config or has_gguf) and "1_Pooling" not in root:
                real_path = os.path.realpath(root)
                if real_path in seen_paths:
                    continue

                parent_hub_dir = root
                parts = root.split(os.sep)
                base_name = ""
                for p in parts:
                    if p.startswith("models--"):
                        base_name = p.replace("models--", "").replace("--", "/")
                        parent_hub_dir = root[:root.find(p) + len(p)]
                        break
                if not base_name:
                    base_name = os.path.basename(root)

                # Check if this model is inside a quantization subfolder (e.g. snapshots/<hash>/4-bit)
                subfolder = ""
                if "snapshots" in parts:
                    snap_idx = parts.index("snapshots")
                    if len(parts) > snap_idx + 2:
                        subfolder = "/".join(parts[snap_idx + 2:])

                display_name = f"{base_name} ({subfolder})" if subfolder else base_name

                model_type = "unknown"
                supported = True
                arch_tag = "MLX Compatible"
                is_vis = False

                if has_config:
                    try:
                        with open(os.path.join(root, "config.json"), "r", errors="ignore") as f:
                            cfg = json.load(f)
                            model_type = str(cfg.get("model_type", "unknown")).lower()
                            is_vis = is_vision_model(root, cfg)
                            
                            if model_type in ["bert", "xlm-roberta", "mpnet"]:
                                supported = False
                                arch_tag = f"Embedding Model ({model_type})"
                            elif is_vis:
                                supported = True
                                arch_tag = f"👁️ MLX Vision ({model_type})"
                            else:
                                arch_tag = f"⚡ MLX ({model_type})"
                    except Exception:
                        pass
                elif has_gguf:
                    supported = False
                    arch_tag = "GGUF Format"

                seen_paths.add(real_path)

                # Calculate accurate model size by resolving symlinks in this model directory
                size_bytes = 0
                for r, d, f in os.walk(root):
                    for file in f:
                        fp = os.path.join(r, file)
                        try:
                            size_bytes += os.stat(fp).st_size
                        except Exception:
                            pass
                
                # If size resolved to 0 (e.g. non-symlink blobs folder), fallback to walking parent
                if size_bytes == 0:
                    calc_dir = parent_hub_dir if "models--" in parent_hub_dir else root
                    for r, d, f in os.walk(calc_dir):
                        for file in f:
                            fp = os.path.join(r, file)
                            if not os.path.islink(fp):
                                try:
                                    size_bytes += os.path.getsize(fp)
                                except Exception:
                                    pass

                size_gb = size_bytes / (1024**3)
                size_str = f"{size_gb:.1f} GB" if size_gb >= 1 else f"{size_bytes/(1024**2):.1f} MB"

                target_stat_dir = parent_hub_dir if "models--" in parent_hub_dir else root
                try:
                    st_obj = os.stat(target_stat_dir)
                    model_ts = getattr(st_obj, "st_birthtime", st_obj.st_mtime)
                    if model_ts <= 0:
                        model_ts = st_obj.st_mtime
                except Exception:
                    model_ts = time.time()

                date_added_str = time.strftime("%b %d, %Y", time.localtime(model_ts))

                host_drive = drive_for_path(root)

                models.append({
                    "name": display_name,
                    "base_name": base_name,
                    "subfolder": subfolder,
                    "size": size_str,
                    "size_gb": round(size_gb, 2),
                    "path": root,
                    "delete_target": root if subfolder else (parent_hub_dir if "models--" in parent_hub_dir else root),
                    "supported": supported,
                    "arch_tag": arch_tag,
                    "model_type": model_type,
                    "is_vision": is_vis,
                    "engine": "mlx-vlm" if is_vis else "mlx-lm",
                    "created_ts": round(model_ts, 1),
                    "date_added": date_added_str,
                    "drive_id": (host_drive or {}).get("id", ""),
                    "drive_name": (host_drive or {}).get("name", "Internal cache" if os.path.realpath(root).startswith(os.path.expanduser("~/.cache/huggingface")) else ""),
                })
    return models

def get_running_servers():
    servers = []
    st = load_global_settings()
    default_p = str(st.get("default_port", 9999))
    try:
        output = subprocess.check_output(["ps", "aux"], text=True)
        for line in output.splitlines():
            if ("mlx_lm.server" in line or "mlx_vlm.server" in line or "mlx_vlm" in line) and "grep" not in line:
                parts = line.split()
                pid = parts[1]
                cmd_start = parts[10] if len(parts) > 10 else ""

                model_match = re.search(r"--model\s+([^\s]+)", line)
                port_match = re.search(r"--port\s+([^\s]+)", line)
                host_match = re.search(r"--host\s+([^\s]+)", line)

                model = model_match.group(1) if model_match else "unknown"
                port = port_match.group(1) if port_match else default_p
                host = host_match.group(1) if host_match else "127.0.0.1"

                if "models--" in model:
                    m = re.search(r"models--([^/]+)--(.*?)(?:/snapshots|$)", model)
                    if m:
                        model = f"{m.group(1)}/{m.group(2)}"

                is_vision = "mlx_vlm" in line
                engine = "mlx-vlm" if is_vision else "mlx-lm"

                status = "STARTING/UNRESPONSIVE"
                try:
                    url = f"http://{host}:{port}/v1/models"
                    req = urllib.request.Request(url)
                    with urllib.request.urlopen(req, timeout=1.2) as resp:
                        if resp.status == 200:
                            status = "READY"
                except Exception:
                    pass

                servers.append({
                    "pid": pid,
                    "model": model,
                    "host": host,
                    "port": port,
                    "status": status,
                    "engine": engine,
                    "is_vision": is_vision,
                    "is_python": "python" in cmd_start
                })
    except Exception:
        pass

    unique = {}
    for s in servers:
        key = (s["host"], s["port"])
        if key not in unique or s["is_python"]:
            unique[key] = s

    return list(unique.values())

def _pid_descendants(pid):
    """Every PID below `pid` (uvx wraps the real downloader in a child process)."""
    try:
        out = subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True)
    except Exception:
        return []
    children = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            c, p_ = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        children.setdefault(p_, []).append(c)
    found, stack = [], [pid]
    while stack:
        cur = stack.pop()
        for child in children.get(cur, []):
            found.append(child)
            stack.append(child)
    return found

def _kill_download_tree(pid):
    pids = [int(pid)] + _pid_descendants(int(pid))
    for p_ in reversed(pids):
        try:
            os.kill(p_, signal.SIGTERM)
        except Exception:
            pass
    time.sleep(2)
    for p_ in reversed(pids):
        try:
            os.kill(p_, signal.SIGKILL)
        except Exception:
            pass
    return pids

def _cmdline_for_pid(pid):
    try:
        out = subprocess.check_output(["ps", "-p", str(pid), "-o", "command="], text=True).strip()
        return shlex.split(out) if out else []
    except Exception:
        return []

def _relaunch_download(cmd, cache_dir=""):
    """Restart a stalled transfer; huggingface_hub resumes from .incomplete files."""
    if not cmd:
        return None
    env = get_mlx_env()
    try:
        os.makedirs(os.path.dirname(_DOWNLOAD_LOG), exist_ok=True)
        log = open(_DOWNLOAD_LOG, "a")
        log.write(f"\n--- auto-resume {time.strftime('%H:%M:%S')}: {' '.join(cmd)} ---\n")
        log.flush()
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
        return proc.pid
    except Exception:
        return None

RESUME_LOCK = os.path.expanduser("~/.mlx_pids/turbo/resume.lock")
_RESUME_LOCK_STALE = 180  # seconds before another process may steal it

def _claim_resume_lock():
    """Best-effort cross-process lock so two dashboards never relaunch the same transfer."""
    os.makedirs(os.path.dirname(RESUME_LOCK), exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(RESUME_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(RESUME_LOCK)
            except OSError:
                age = _RESUME_LOCK_STALE + 1
            if age > _RESUME_LOCK_STALE:
                try:
                    os.unlink(RESUME_LOCK)
                except OSError:
                    pass
                continue
            return False
        except OSError:
            return True  # never block recovery on lock trouble
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        return True
    return False

def _release_resume_lock():
    try:
        os.unlink(RESUME_LOCK)
    except OSError:
        pass

def _watchdog_tick():
    try:
        active = get_active_downloads()
    except Exception:
        return
    now = time.time()
    live_ids = set()
    for dl in active:
        did = dl.get("download_id") or dl.get("repo_id")
        live_ids.add(did)
        rec = _DL_WATCH.setdefault(did, {"bytes": 0, "ts": now, "attempts": 0,
                                         "stalled": False, "resuming": False})
        cur = int(dl.get("downloaded_bytes") or 0)
        finalising = (str(dl.get("phase") or "").lower() in ("hub", "finalising")
                      or dl.get("status") == "FINALISING")
        if finalising:
            # Linking/verifying moves no bytes; killing it would corrupt the cache.
            rec.update(bytes=cur, ts=now, stalled=False, resuming=False)
            continue
        if cur > rec["bytes"] + _STALL_MIN_BYTES:
            rec.update(bytes=cur, ts=now, stalled=False, resuming=False)
            continue
        rec["bytes"] = max(rec["bytes"], cur)
        quiet = now - rec["ts"]
        if quiet < _STALL_SECONDS or rec["stalled"]:
            continue
        pid = dl.get("pid")
        if not pid:
            continue
        if rec["attempts"] >= _STALL_MAX_RETRIES:
            rec.update(stalled=True, resuming=False)
            continue
        # Rebuild the command ourselves: the live process may be a uvx shim whose
        # argv[0] cannot be executed again outside its throwaway environment.
        repo = dl.get("repo_id") or ""
        if dl.get("turbo"):
            hub = dl.get("cache_dir") or ""
            cmd = ([sys.executable, os.path.abspath(__file__), "turbo_download", repo, hub,
                    dl.get("download_id") or repo] if repo and hub else [])
        else:
            hub = dl.get("cache_dir") or ""
            # `hf` prints repo-scoped paths; relaunch must target the library root.
            if os.path.basename(hub.rstrip("/")).startswith("models--"):
                hub = os.path.dirname(hub.rstrip("/"))
            cmd = (get_download_cmd(repo, subfolder=dl.get("subfolder") or "",
                                    revision=dl.get("revision") or "main",
                                    cache_dir=hub) if repo else [])
        if not _claim_resume_lock():
            # Another dashboard process is already restarting this transfer.
            rec["ts"] = time.time()
            continue
        try:
            rec["attempts"] += 1
            rec["resuming"] = True
            _kill_download_tree(pid)
            new_pid = _relaunch_download(cmd)
            rec.update(ts=time.time(), stalled=False, resuming=bool(new_pid), pid=new_pid or 0)
        finally:
            _release_resume_lock()
    for did in list(_DL_WATCH.keys()):
        if did not in live_ids:
            _DL_WATCH.pop(did, None)

def start_download_watchdog():
    """Background thread: kill 0 B/s transfers and resume them automatically."""
    global _WATCH_THREAD
    if _WATCH_THREAD and _WATCH_THREAD.is_alive():
        return _WATCH_THREAD
    def _loop():
        while True:
            try:
                _watchdog_tick()
            except Exception:
                pass
            time.sleep(10)
    _WATCH_THREAD = threading.Thread(target=_loop, name="mlx-download-watchdog", daemon=True)
    _WATCH_THREAD.start()
    return _WATCH_THREAD

def download_watch_state():
    return {k: dict(v) for k, v in _DL_WATCH.items()}

# ---------------------------------------------------------------------------
# Turbo: parallel range downloads for large LFS files (single streams are slow)
# ---------------------------------------------------------------------------
TURBO_MIN_BYTES = 256 * 1024 * 1024
TURBO_CONNS = 8
TURBO_DIR = os.path.expanduser("~/.mlx_pids/turbo")
_TURBO_LOCK = threading.Lock()

def _turbo_status_path(download_id):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", download_id or "download")
    return os.path.join(TURBO_DIR, safe + ".json")

def write_turbo_status(download_id, data):
    try:
        os.makedirs(TURBO_DIR, exist_ok=True)
        payload = dict(data)
        payload["updated"] = time.time()
        path = _turbo_status_path(download_id)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception:
        pass

def clear_turbo_status(download_id):
    try:
        os.remove(_turbo_status_path(download_id))
    except Exception:
        pass

def read_turbo_statuses():
    out = []
    if not os.path.isdir(TURBO_DIR):
        return out
    now = time.time()
    for name in os.listdir(TURBO_DIR):
        if not name.endswith(".json"):
            continue
        path = os.path.join(TURBO_DIR, name)
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        if now - float(data.get("updated") or 0) > 45:
            clear_turbo_status(data.get("download_id") or name[:-5])
            continue
        out.append(data)
    return out

def turbo_plan(repo_id, include=None):
    """Which files are worth parallelising, and how many bytes that is.

    `include` matters: without it a single-gguf download would be turbo-fetched as
    the whole repo (a 660 GB mistake in a GGUF repo with 24 quants).
    """
    token = get_hf_token()
    meta = fetch_repo_meta(repo_id, token=token)
    if not meta:
        return {"files": [], "bytes": 0, "reason": "repo metadata unavailable"}
    files = []
    for sib in meta.get("siblings") or []:
        lfs = sib.get("lfs") or {}
        size = sib.get("size") or lfs.get("size") or 0
        sha = lfs.get("sha256") or ""
        if sha and size >= TURBO_MIN_BYTES:
            path = sib.get("rfilename") or ""
            if include and not any(fnmatch.fnmatch(path, pat) for pat in include):
                continue
            files.append({"path": path, "size": int(size), "sha256": sha})
    files.sort(key=lambda f: -f["size"])
    return {"files": files, "bytes": sum(f["size"] for f in files)}

def fetch_repo_meta(repo_id, token=""):
    url = f"{HF_HOST}/api/models/{repo_id}?blobs=true"
    status, data = _http_json(url, token=token, timeout=30)
    if status == 200 and isinstance(data, dict):
        return data
    return {}

def _download_range(url, start, end, path, token="", retries=6):
    """Fetch [start, end] into `path`, resuming whatever is already on disk."""
    want = end - start + 1
    for attempt in range(retries):
        try:
            have = os.path.getsize(path) if os.path.exists(path) else 0
            if have >= want:
                return True
            headers = {"Range": f"bytes={start + have}-{end}", "User-Agent": "mlx-control-center"}
            if token:
                headers["Authorization"] = "Bearer " + token
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as r:
                if getattr(r, "status", 200) == 200:
                    return False  # server ignored the Range header
                with open(path, "ab") as f:
                    while True:
                        chunk = r.read(1024 * 256)
                        if not chunk:
                            break
                        f.write(chunk)
                        f.flush()
            if os.path.getsize(path) >= want:
                return True
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(min(1.5 ** attempt, 10) + (0.4 * (attempt % 3)))
    return os.path.getsize(path) >= want if os.path.exists(path) else False

def repo_blob_dir(hub_dir, repo_id):
    """Where huggingface_hub 2.x actually keeps blobs: models--org--repo/blobs."""
    return os.path.join(hub_dir, "models--" + repo_id.replace("/", "--"), "blobs")

def turbo_fetch_file(repo_id, rel_path, size, sha256, hub_dir, token="", conns=None,
                     on_progress=None, revision="main"):
    """Parallel download of one LFS file straight into the HF cache blob layout."""
    conns = max(2, min(int(conns or TURBO_CONNS), 16))
    blob = os.path.join(repo_blob_dir(hub_dir, repo_id), sha256)
    if os.path.exists(blob) and os.path.getsize(blob) == size:
        return {"ok": True, "skipped": True, "bytes": size}
    os.makedirs(os.path.dirname(blob), exist_ok=True)
    parts_dir = blob + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    url = f"{HF_HOST}/{repo_id}/resolve/{revision}/{urllib.parse.quote(rel_path)}"
    chunk = (size + conns - 1) // conns
    ranges = []
    for i in range(conns):
        start = i * chunk
        end = min(size - 1, start + chunk - 1)
        if start >= size:
            break
        ranges.append((i, start, end))

    def _part_path(i):
        return os.path.join(parts_dir, f"part{i:02d}")

    def _part_done(i, start, end):
        p = _part_path(i)
        return os.path.exists(p) and os.path.getsize(p) >= (end - start + 1)

    # Three passes: throttled CDNs drop connections; parts on disk make this cheap.
    last_error = "range download failed"
    for attempt in range(3):
        pending = [r for r in ranges if not _part_done(*r)]
        if not pending:
            break
        stop = {"flag": False}

        def _run(i, start, end, _stop=stop):
            if _stop["flag"]:
                return False
            ok = _download_range(url, start, end, _part_path(i), token=token, retries=8)
            if not ok and not _part_done(i, start, end):
                _stop["flag"] = True
                return False
            return True

        threads = [threading.Thread(target=_run, args=r, daemon=True) for r in pending]
        for t in threads:
            t.start()
        last_report = 0.0
        while any(t.is_alive() for t in threads):
            if on_progress:
                done = 0
                for i, _s, _e in ranges:
                    if os.path.exists(_part_path(i)):
                        done += os.path.getsize(_part_path(i))
                now = time.time()
                if now - last_report >= 1.0:
                    on_progress(done)
                    last_report = now
            time.sleep(0.4)
        if all(_part_done(*r) for r in ranges):
            break
        time.sleep(8 + 6 * attempt)

    if not all(_part_done(*r) for r in ranges):
        return {"ok": False, "error": last_error, "bytes": 0, "retryable": True}

    # assemble + verify before publishing the blob
    tmp_blob = blob + ".turbo"
    h = hashlib.sha256()
    with open(tmp_blob, "wb") as out:
        for i, _s, _e in ranges:
            p = os.path.join(parts_dir, f"part{i:02d}")
            with open(p, "rb") as f:
                while True:
                    buf = f.read(4 * 1024 * 1024)
                    if not buf:
                        break
                    h.update(buf)
                    out.write(buf)
    digest = h.hexdigest()
    if digest != sha256:
        try:
            os.remove(tmp_blob)
        except Exception:
            pass
        return {"ok": False, "error": f"sha256 mismatch for {rel_path}", "bytes": 0}
    os.replace(tmp_blob, blob)
    shutil.rmtree(parts_dir, ignore_errors=True)
    return {"ok": True, "skipped": False, "bytes": size}

def turbo_download(repo_id, hub_dir, subfolder="", conns=None, download_id="", revision="main", include=None):
    """Download every >256 MB LFS blob with N parallel connections, then let the
    regular hub client finish (snapshot links, small files, refs)."""
    token = get_hf_token()
    plan = turbo_plan(repo_id, include=include)
    files = plan.get("files") or []
    total = plan.get("bytes") or 0
    did = download_id or repo_id

    def _status(**kw):
        payload = {"download_id": did, "repo_id": repo_id, "hub": hub_dir,
                   "total_bytes": total, "conns": conns or TURBO_CONNS, "pid": os.getpid()}
        payload.update(kw)
        write_turbo_status(did, payload)

    if not files:
        _status(phase="hub", downloaded_bytes=0, note="no large LFS blobs")
        return {"ok": True, "files": 0, "bytes": 0}

    done_bytes = 0
    for f in files:
        # Ranges are sequential per file, so keep a per-file offset for progress.
        base = done_bytes
        _status(phase="turbo", current=f["path"], file_size=f["size"],
                downloaded_bytes=base, files_done=files.index(f), files_total=len(files))
        res = turbo_fetch_file(
            repo_id, f["path"], f["size"], f["sha256"], hub_dir,
            token=token, conns=conns, revision=revision,
            on_progress=lambda d, base=base: _status(
                phase="turbo", current=f["path"], file_size=f["size"],
                downloaded_bytes=base + d, files_done=files.index(f), files_total=len(files)),
        )
        if not res.get("ok"):
            _status(phase="error", error=res.get("error", "download failed"),
                    downloaded_bytes=done_bytes)
            return {"ok": False, "error": res.get("error"), "bytes": done_bytes}
        done_bytes += f["size"]
        _status(phase="turbo", current=f["path"], file_size=f["size"],
                downloaded_bytes=done_bytes, files_done=files.index(f) + 1, files_total=len(files))
    _status(phase="hub", downloaded_bytes=done_bytes, note="finalising snapshot links")
    return {"ok": True, "files": len(files), "bytes": done_bytes}

def get_active_downloads():
    downloads = []
    now = time.time()
    try:
        output = subprocess.check_output(["ps", "aux"], text=True)
        for line in output.splitlines():
            if ("hf download" in line or "huggingface-cli download" in line) and "grep" not in line:
                parts = line.split()
                pid = parts[1]
                repo_match = re.search(r"download\s+([^\s]+)", line)
                raw_repo = repo_match.group(1) if repo_match else "unknown"

                include_match = re.search(r"--include\s+[\"']?([^\"'\s]+)[\"']?", line)
                explicit_sf = include_match.group(1).replace("/*", "").strip() if include_match else ""

                parsed = parse_hf_identifier(raw_repo, explicit_sf)
                repo_id = parsed["repo_id"]
                subfolder = parsed["subfolder"]
                download_id = parsed["download_id"]
                display_name = parsed["display_name"]

                cache_dir = ""
                hub_root = ""
                cd_match = re.search(r"--cache-dir[=\s]+([^\s\"']+)", line)
                if cd_match:
                    hub_root = cd_match.group(1)
                if "/" in repo_id:
                    org, repo = repo_id.split("/", 1)
                    base_hub = hub_root or os.path.expanduser("~/.cache/huggingface/hub")
                    cache_dir = os.path.join(base_hub, f"models--{org}--{repo}")

                downloaded_bytes = 0
                if cache_dir and os.path.exists(cache_dir):
                    for r, d, f in os.walk(cache_dir):
                        for file in f:
                            fp = os.path.join(r, file)
                            if not os.path.islink(fp):
                                try:
                                    downloaded_bytes += os.path.getsize(fp)
                                except Exception:
                                    pass

                total_bytes = get_repo_files_size(repo_id, subfolder=subfolder)
                percent = round((downloaded_bytes / total_bytes) * 100, 1) if total_bytes > 0 else 0
                percent = min(100.0, percent)

                # Rolling 15 s window so a single quiet sample never reads as 0 MB/s.
                hist = _SPEED_HIST.setdefault(download_id, [])
                hist.append((now, downloaded_bytes))
                while len(hist) > 2 and now - hist[0][0] > 15:
                    hist.pop(0)
                speed_str = "Calculating..."
                eta_str = "--"
                if len(hist) >= 2 and now - hist[0][0] >= 1.5:
                    dbytes = downloaded_bytes - hist[0][1]
                    dt = now - hist[0][0]
                    speed_bps = max(0.0, dbytes / dt) if dt > 0 else 0.0
                    speed_mb = speed_bps / (1024**2)
                    if speed_mb > 0.05:
                        speed_str = f"{speed_mb:.1f} MB/s"
                        rem_bytes = max(0, total_bytes - downloaded_bytes)
                        if rem_bytes > 0 and speed_bps > 0:
                            rem_sec = int(rem_bytes / speed_bps)
                            eta_str = f"{rem_sec}s" if rem_sec < 60 else f"{rem_sec//60}m {rem_sec%60}s"
                    else:
                        speed_str = "0 MB/s"
                watch = _DL_WATCH.get(download_id) or {}
                stalled = bool(watch.get("stalled"))
                if stalled:
                    speed_str = "stalled"
                    eta_str = "resuming…" if watch.get("resuming") else "paused — retry"


                dl_gb = downloaded_bytes / (1024**3)
                dl_str = f"{dl_gb:.2f} GB" if dl_gb >= 1 else f"{downloaded_bytes/(1024**2):.1f} MB"

                tot_gb = total_bytes / (1024**3)
                tot_str = f"{tot_gb:.2f} GB" if tot_gb >= 1 else (f"{total_bytes/(1024**2):.1f} MB" if total_bytes > 0 else "Unknown")

                downloads.append({
                    "pid": pid,
                    "repo_id": repo_id,
                    "subfolder": subfolder,
                    "download_id": download_id,
                    "display_name": display_name,
                    "downloaded_bytes": downloaded_bytes,
                    "downloaded_size": dl_str,
                    "total_bytes": total_bytes,
                    "total_size": tot_str,
                    "percent": percent,
                    "speed": speed_str,
                    "eta": eta_str,
                    "status": "STALLED" if stalled else "DOWNLOADING",
                    "attempts": int(watch.get("attempts") or 0),
                    "cache_dir": cache_dir
                })
    except Exception:
        pass

    # Add Paused Downloads to list
    paused = load_paused_downloads()
    active_ids = {d["download_id"] for d in downloads}

    for dl_id, pinfo in list(paused.items()):
        if dl_id not in active_ids:
            if isinstance(pinfo, str):
                pinfo = {"repo_id": pinfo, "subfolder": "", "cache_dir": ""}
            repo_id = pinfo.get("repo_id", dl_id)
            subfolder = pinfo.get("subfolder", "")
            display_name = f"{repo_id} ({subfolder})" if subfolder else repo_id
            cache_dir = pinfo.get("cache_dir", "")
            downloaded_bytes = 0
            if cache_dir and os.path.exists(cache_dir):
                for r, d, f in os.walk(cache_dir):
                    for file in f:
                        fp = os.path.join(r, file)
                        if not os.path.islink(fp):
                            try:
                                downloaded_bytes += os.path.getsize(fp)
                            except Exception:
                                pass
            total_bytes = get_repo_files_size(repo_id, subfolder=subfolder)
            percent = round((downloaded_bytes / total_bytes) * 100, 1) if total_bytes > 0 else 0
            percent = min(100.0, percent)

            dl_gb = downloaded_bytes / (1024**3)
            dl_str = f"{dl_gb:.2f} GB" if dl_gb >= 1 else f"{downloaded_bytes/(1024**2):.1f} MB"

            tot_gb = total_bytes / (1024**3)
            tot_str = f"{tot_gb:.2f} GB" if tot_gb >= 1 else (f"{total_bytes/(1024**2):.1f} MB" if total_bytes > 0 else "Unknown")

            downloads.append({
                "pid": None,
                "repo_id": repo_id,
                "subfolder": subfolder,
                "download_id": dl_id,
                "display_name": display_name,
                "downloaded_bytes": downloaded_bytes,
                "downloaded_size": dl_str,
                "total_bytes": total_bytes,
                "total_size": tot_str,
                "percent": percent,
                "speed": "Paused",
                "eta": "Paused",
                "status": "PAUSED",
                "cache_dir": cache_dir
            })

    # uvx wraps the download in several processes; list each repo once.
    deduped = {}
    for d in downloads:
        key = d.get("download_id") or d.get("repo_id")
        prev = deduped.get(key)
        if prev is None or (d.get("pid") and not prev.get("pid")):
            deduped[key] = d
    downloads = list(deduped.values())

    # Turbo workers are not `hf download` processes; merge their own progress feed.
    known = {(d.get("download_id") or d.get("repo_id")) for d in downloads}
    for st in read_turbo_statuses():
        did = st.get("download_id") or st.get("repo_id")
        if did in known:
            continue
        got = int(st.get("downloaded_bytes") or 0)
        total = int(st.get("total_bytes") or 0)
        key = "turbo:" + str(did)
        hist = _SPEED_HIST.setdefault(key, [])
        hist.append((now, got))
        while len(hist) > 2 and now - hist[0][0] > 15:
            hist.pop(0)
        speed_str, eta_str = "Calculating...", "--"
        if len(hist) >= 2 and now - hist[0][0] >= 1.5:
            bps = max(0.0, (got - hist[0][1]) / (now - hist[0][0]))
            if bps / (1024 ** 2) > 0.05:
                speed_str = f"{bps / (1024**2):.1f} MB/s"
                if total > got > 0:
                    rem = int((total - got) / bps)
                    eta_str = f"{rem}s" if rem < 60 else f"{rem//60}m {rem%60}s"
            else:
                speed_str = "0 MB/s"
        phase = st.get("phase") or "turbo"
        status = "DOWNLOADING"
        if phase == "error":
            status = "STALLED"
            speed_str = st.get("error", "failed")
        elif phase == "hub":
            status = "FINALISING"
            speed_str = "linking files"
        dl_bytes = got / (1024 ** 3)
        tot_bytes = total / (1024 ** 3)
        downloads.append({
            "pid": st.get("pid"),
            "repo_id": st.get("repo_id"),
            "subfolder": "",
            "download_id": did,
            "display_name": st.get("repo_id"),
            "downloaded_bytes": got,
            "downloaded_size": f"{dl_bytes:.2f} GB" if got >= 1024**3 else f"{got/(1024**2):.1f} MB",
            "total_bytes": total,
            "total_size": f"{tot_bytes:.2f} GB" if total >= 1024**3 else (f"{total/(1024**2):.1f} MB" if total else "Unknown"),
            "percent": round(min(100.0, got / total * 100), 1) if total else 0,
            "speed": speed_str,
            "eta": eta_str,
            "status": status,
            "attempts": 0,
            "turbo": True,
            "phase": phase,
            "conns": st.get("conns"),
            "cache_dir": st.get("hub", ""),
        })

    return downloads

def get_system_stats():
    total_ram = 32 * 1024**3
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        total_ram = int(out.strip())
    except Exception:
        pass

    used_ram = 0
    free_ram = 0
    cached_ram = 0
    try:
        vm = subprocess.check_output(["vm_stat"], text=True)
        page_size = 16384
        vm_dict = {}
        for line in vm.splitlines():
            if "page size of" in line:
                m = re.search(r"page size of (\d+) bytes", line)
                if m:
                    page_size = int(m.group(1))
            elif ":" in line:
                parts = line.split(":")
                k = parts[0].strip()
                val_m = re.search(r"(\d+)", parts[1])
                if val_m:
                    vm_dict[k] = int(val_m.group(1))

        # Accurate Activity Monitor / HTOP calculation for macOS:
        # Memory Used = Anonymous pages (App memory) + Wired pages + Compressed pages
        anonymous = vm_dict.get("Anonymous pages", 0) * page_size
        wired = vm_dict.get("Pages wired down", 0) * page_size
        compressor = vm_dict.get("Pages occupied by compressor", 0) * page_size
        purgeable = vm_dict.get("Pages purgeable", 0) * page_size
        file_backed = vm_dict.get("File-backed pages", 0) * page_size

        used_ram = anonymous + wired + compressor
        cached_ram = file_backed + purgeable
        free_ram = max(0, total_ram - used_ram)
    except Exception:
        anonymous = int(total_ram * 0.4)
        wired = int(total_ram * 0.1)
        compressor = int(total_ram * 0.1)
        used_ram = int(total_ram * 0.6)
        cached_ram = int(total_ram * 0.2)
        free_ram = total_ram - used_ram

    mlx_ram = 0
    servers = get_running_servers()
    pids = [s["pid"] for s in servers]
    if pids:
        try:
            ps_out = subprocess.check_output(["ps", "-o", "pid,rss", "-p", ",".join(pids)], text=True)
            for line in ps_out.splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 2:
                    mlx_ram += int(parts[1]) * 1024
        except Exception:
            pass

    cpu_percent = 0.0
    try:
        out = subprocess.check_output(["ps", "-A", "-o", "%cpu"], text=True)
        total_cpu = sum(float(line.strip()) for line in out.splitlines()[1:] if line.strip())
        cores = os.cpu_count() or 8
        cpu_percent = min(100.0, round(total_cpu / cores, 1))
    except Exception:
        pass

    # RAM a model can realistically use: macOS keeps a wired/GPU reserve and compresses memory
    usable_ram = max(total_ram - max(4 * 1024**3, total_ram * 0.12), total_ram * 0.55)

    # Models held in memory by the Ollama engine count as model memory too; the
    # dashboard is engine-agnostic, so "0 GB" while a 72B is loaded would lie.
    try:
        ollama_loaded = ollama_ps()
    except Exception:
        ollama_loaded = []
    ollama_reported = sum(m["vram_gb"] or m["size_gb"] for m in ollama_loaded)
    ollama_resident = ollama_runner_rss_gb() if ollama_loaded else 0.0
    # Prefer the resident figure so it is comparable with "Mac System RAM".
    ollama_model_ram = ollama_resident or ollama_reported

    return {
        "total_ram_gb": round(total_ram / (1024**3), 1),
        "used_ram_gb": round(used_ram / (1024**3), 1),
        "free_ram_gb": round(free_ram / (1024**3), 1),
        "usable_ram_gb": round(usable_ram / (1024**3), 1),
        "cached_ram_gb": round(cached_ram / (1024**3), 1),
        "app_ram_gb": round(anonymous / (1024**3), 1),
        "wired_ram_gb": round(wired / (1024**3), 1),
        "compressed_ram_gb": round(compressor / (1024**3), 1),
        "ram_percent": round((used_ram / total_ram) * 100, 1) if total_ram else 0,
        "mlx_ram_gb": round(mlx_ram / (1024**3), 2),
        "ollama_ram_gb": round(ollama_model_ram, 2),
        "ollama_reported_gb": round(ollama_reported, 2),
        "model_ram_gb": round(mlx_ram / (1024**3) + ollama_model_ram, 2),
        "cpu_percent": cpu_percent,
        "active_models": len(servers) + len(ollama_loaded),
        "mlx_active_models": len(servers),
        "ollama_loaded": ollama_loaded
    }

def classify_process(pid, user, args, active_server_pids=None):
    if active_server_pids is None:
        active_server_pids = set()
    
    args_lower = args.lower()
    base = os.path.basename(args.split()[0]) if args.strip() else 'unknown'
    
    m_app = re.search(r'/([^/]+)\.app(?:/|$)', args)
    app_name = m_app.group(1) if m_app else None
    
    category = 'App'
    display_name = app_name or base
    is_system = False
    is_mlx = str(pid) in active_server_pids or pid in active_server_pids
    
    if is_mlx or 'mlx_lm.server' in args or 'mlx_lm' in args or 'mlx_vlm.server' in args or 'mlx_vlm' in args or 'mlx_gui_server' in args:
        category = 'AI Model'
        m_model = re.search(r'--model\s+([^\s]+)', args)
        if m_model:
            model_base = os.path.basename(m_model.group(1))
            display_name = f'MLX Vision: {model_base}' if 'mlx_vlm' in args else f'MLX: {model_base}'
        elif 'mlx_gui_server' in args:
            display_name = 'MLX Control Center GUI'
        else:
            display_name = f'MLX: {base}'
    elif any(k in args_lower for k in ['ollama', 'lmstudio', 'llama.cpp', 'vllm', 'text-generation']):
        category = 'AI Model'
        display_name = f'AI: {app_name or base}'
    elif any(k in args_lower for k in ['brave', 'chrome', 'firefox', 'safari', 'arc', 'edge', 'webkit']):
        category = 'Browser'
        if app_name:
            if 'helper' in args_lower or 'renderer' in args_lower:
                display_name = f'{app_name} (Tab)'
            elif 'gpu' in args_lower:
                display_name = f'{app_name} (GPU)'
            else:
                display_name = app_name
    elif any(k in args_lower for k in ['docker', 'code', 'cursor', 'node', 'python', 'git', 'xcode', 'iterm', 'terminal', 'agy']):
        category = 'Dev Tool'
        if 'agy' in args_lower:
            display_name = 'Antigravity CLI (agy)'
        elif app_name:
            display_name = app_name
        else:
            display_name = base
    elif user == 'root' or any(k in base.lower() for k in ['windowserver', 'kernel_task', 'launchd', 'loginwindow', 'spotlight', 'mediaanalysisd', 'mds', 'core', 'daemon']):
        category = 'macOS System'
        is_system = True
        display_name = base

    PROTECTED = {'windowserver', 'kernel_task', 'launchd', 'loginwindow', 'systemstats', 'opendirectoryd', 'syslogd', 'securityd', 'mds', 'cfprefsd', 'distnoted', 'fseventsd', 'trustd', 'runningboardd', 'diskarbitrationd', 'coreauthd'}
    if base.lower() in PROTECTED or pid in [0, 1] or user == 'root':
        is_system = True
        
    return display_name, category, is_system, is_mlx

def get_top_memory_processes(limit=35):
    try:
        servers = get_running_servers()
        server_pids = {str(s.get("pid", "")) for s in servers}
        
        out = subprocess.check_output(["ps", "-A", "-o", "pid,user,%mem,rss,args"], text=True)
        lines = out.strip().splitlines()
        procs = []
        for line in lines[1:]:
            parts = line.split(None, 4)
            if len(parts) < 5:
                continue
            pid_str, user, mem_pct_str, rss_kb_str, args = parts
            try:
                pid = int(pid_str)
                rss_kb = int(rss_kb_str)
                mem_pct = float(mem_pct_str)
            except ValueError:
                continue
                
            if rss_kb < 15360:  # Skip < 15 MB
                continue
                
            rss_bytes = rss_kb * 1024
            rss_gb = rss_bytes / (1024**3)
            rss_mb = rss_bytes / (1024**2)
            rss_str = f"{rss_gb:.2f} GB" if rss_gb >= 1.0 else f"{rss_mb:.0f} MB"
            
            dname, cat, is_sys, is_mlx = classify_process(pid, user, args, server_pids)
            
            procs.append({
                "pid": pid,
                "user": user,
                "name": dname,
                "command": args,
                "rss_bytes": rss_bytes,
                "rss_gb": round(rss_gb, 2),
                "rss_str": rss_str,
                "mem_pct": mem_pct,
                "category": cat,
                "is_system": is_sys,
                "is_mlx": is_mlx
            })
            
        procs.sort(key=lambda x: x["rss_bytes"], reverse=True)
        return procs[:limit]
    except Exception as e:
        return []

def kill_process_by_pid(pid):
    PROTECTED_PIDS = {0, 1}
    try:
        pid = int(pid)
        if pid in PROTECTED_PIDS or pid <= 0:
            return {"error": "Cannot terminate system PID"}
        
        try:
            comm = subprocess.check_output(["ps", "-p", str(pid), "-o", "comm="], text=True).strip()
            base = os.path.basename(comm).lower()
            if base in ['windowserver', 'kernel_task', 'launchd', 'loginwindow']:
                return {"error": f"Cannot terminate protected system process '{base}'"}
        except Exception:
            pass

        servers = get_running_servers()
        for s in servers:
            if str(s.get("pid")) == str(pid):
                try:
                    os.kill(pid, 9)
                except Exception:
                    pass
                return {"status": "ok", "pid": pid, "stopped_mlx": True}

        import signal
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.2)
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
            
        return {"status": "ok", "pid": pid}
    except Exception as e:
        return {"error": str(e)}

# Quantization markers that must never be mistaken for a parameter count (e.g. "4bit" != "4B").
_QUANT_NAME_RE = re.compile(
    r"(?<![0-9a-z])("
    r"(?:i?q[1-8])(?:[._-][a-z0-9]{1,5}){0,3}"   # q4, q4_k_m, iq3_xxs
    r"|f(?:16|32)|bf16|mxfp4"
    r"|[2-8][ ._-]?bit"
    r")(?![0-9a-z])",
    re.I
)

_PARAM_COUNT_RE = re.compile(r"(?<![0-9a-z.])(\d+(?:\.\d+)?)b(?![0-9a-z])", re.I)
_PARAM_MILLION_RE = re.compile(r"(?<![0-9a-z.])(\d+(?:\.\d+)?)m(?![0-9a-z])", re.I)
_PARAM_MOE_RE = re.compile(r"(?<![0-9a-z])(\d+)x(\d+(?:\.\d+)?)b(?![0-9a-z])", re.I)


def _parse_params_from_name(repo_id):
    """Parse parameter count (billions) from a repo id, ignoring quant labels like 4bit/8-bit/q4."""
    rid = (repo_id or "").lower()
    m = _PARAM_MOE_RE.search(rid)  # Mixtral-8x7B style
    if m:
        return round(int(m.group(1)) * float(m.group(2)), 2)
    m = _PARAM_COUNT_RE.search(rid)
    if m:
        val = float(m.group(1))
        if 0.05 <= val <= 2000:
            return val
    m = _PARAM_MILLION_RE.search(rid)  # granite-4.0-350m style
    if m:
        return round(float(m.group(1)) / 1000, 3)
    return None


def _detect_quant_label(repo_id, bits=None):
    """Best-effort quantization label such as '4-bit'."""
    if bits:
        try:
            return f"{int(bits)}-bit"
        except Exception:
            pass
    rid = (repo_id or "").lower()
    m = re.search(r"(?<![0-9])([2-8])[ ._-]?bit", rid)
    if not m:
        # GGUF naming: Q4_K_M, IQ3_XS, Q8_0, q5_k_s
        m = re.search(r"(?<![a-z0-9])i?q([2-8])(?![0-9])", rid)
    if m:
        return f"{m.group(1)}-bit"
    if "fp32" in rid:
        return "32-bit"
    if "fp16" in rid or "bf16" in rid:
        return "16-bit"
    return ""


def _load_size_info_cache():
    try:
        with open(SIZE_INFO_CACHE_FILE, "r") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_size_info_cache(cache):
    try:
        os.makedirs(os.path.dirname(SIZE_INFO_CACHE_FILE), exist_ok=True)
        tmp = SIZE_INFO_CACHE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cache, f)
        os.replace(tmp, SIZE_INFO_CACHE_FILE)
    except Exception:
        pass


def _best_weight_group(siblings):
    """Pick the weight files that represent downloading ONE quant of a repo.

    Handles: plain MLX repos (sharded safetensors at root), repos with per-quant
    subfolders (e.g. 4-bit/, 8-bit/) and GGUF repos holding several quants at root.
    Returns (bytes, subfolder, quant_label).
    """
    entries = []
    for s in siblings or []:
        fn = s.get("rfilename") or ""
        size = s.get("size") or 0
        if size <= 0 or not fn.lower().endswith(WEIGHT_FILE_EXTS):
            continue
        entries.append((fn, size))
    if not entries:
        return 0, None, None

    # Never double count duplicated formats (some repos ship .bin + .safetensors)
    for ext in (".safetensors", ".gguf"):
        subset = [e for e in entries if e[0].lower().endswith(ext)]
        if subset:
            entries = subset
            break

    dir_totals = {}
    for fn, size in entries:
        d = os.path.dirname(fn)
        dir_totals[d] = dir_totals.get(d, 0) + size
    best_dir, best_size = max(dir_totals.items(), key=lambda kv: kv[1])

    group = [(fn, size) for fn, size in entries if os.path.dirname(fn) == best_dir]
    # A single folder may still hold several quants (common for GGUF repos): split by quant token
    by_quant = {}
    for fn, size in group:
        m = _QUANT_NAME_RE.search(os.path.basename(fn))
        key = (m.group(1).lower().replace(" ", "-").replace("_", "-")) if m else ""
        by_quant[key] = by_quant.get(key, 0) + size

    if len(by_quant) > 1:
        four_bit = [k for k in by_quant if "4" in k]
        if four_bit:
            key = max(four_bit, key=lambda k: by_quant[k])
        else:
            key = min(by_quant, key=lambda k: by_quant[k])
        return by_quant[key], (best_dir or None), (key or None)

    return best_size, (best_dir or None), None


def estimate_ram_gb_from_info(weights_gb, params_b=0.0):
    """RAM needed to run a model: weights + runtime overhead + KV cache + context reserve."""
    if not weights_gb or weights_gb <= 0:
        return 0.0
    kv_gb = min(8.0, max(0.75, float(params_b or 0) * 0.02))
    return round(weights_gb * 1.06 + kv_gb + 1.5, 1)


def get_usable_ram_gb(stats=None):
    """RAM a model can realistically use on macOS after system/GPU reserves."""
    stats = stats if stats is not None else get_system_stats()
    total = float(stats.get("total_ram_gb") or 0)
    if total <= 0:
        return 16.0
    reserve = max(4.0, total * 0.12)
    return round(max(total - reserve, total * 0.55), 1)


def fetch_repo_size_info(repo_id):
    """Exact weight size + parameter count for a Hugging Face repo (disk-cached).

    Returns: weights_gb (size of one quant), params_b (billions), quant, est_ram_gb, source.
    """
    key = (repo_id or "").strip()
    if not key:
        return {"weights_gb": 0.0, "params_b": None, "quant": "", "est_ram_gb": 0.0, "source": "none"}

    with _SIZE_INFO_LOCK:
        cached = _SIZE_INFO_CACHE.get(key)
    if cached:
        return dict(cached)

    disk = _load_size_info_cache()
    entry = disk.get(key)
    if entry and (time.time() - float(entry.get("ts", 0))) < SIZE_INFO_TTL:
        info = {k: v for k, v in entry.items() if k != "ts"}
        with _SIZE_INFO_LOCK:
            _SIZE_INFO_CACHE[key] = info
        return dict(info)

    info = {
        "weights_gb": 0.0,
        "params_b": _parse_params_from_name(key),
        "quant": _detect_quant_label(key),
        "source": "name",
    }
    subfolder = None
    try:
        url = f"https://huggingface.co/api/models/{urllib.parse.quote(key, safe='/')}?blobs=true"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode())
        weights_bytes, subfolder, _label = _best_weight_group(data.get("siblings") or [])
        if weights_bytes > 0:
            info["weights_gb"] = round(weights_bytes / 1e9, 2)
            info["source"] = "hf"
        st = data.get("safetensors") or {}
        params_total = st.get("total") or (st.get("parameters") or {}).get("total")
        if params_total:
            info["params_b"] = round(float(params_total) / 1e9, 2)
        bits = ((data.get("config") or {}).get("quantization_config") or {}).get("bits")
        quant = _detect_quant_label(key, bits)
        if quant:
            info["quant"] = quant
    except Exception:
        pass

    if subfolder:
        info["subfolder"] = subfolder
    if info["weights_gb"]:
        info["est_ram_gb"] = estimate_ram_gb_from_info(info["weights_gb"], info["params_b"] or 0.0)
    elif info["params_b"]:
        # Weights unavailable (gated/nonexistent repo): fall back to 4-bit per-parameter size
        info["est_ram_gb"] = round(info["params_b"] * 0.62 + 2.0, 1)
    else:
        info["est_ram_gb"] = 0.0

    with _SIZE_INFO_LOCK:
        _SIZE_INFO_CACHE[key] = info
    disk[key] = dict(info)
    disk[key]["ts"] = time.time()
    _save_size_info_cache(disk)
    return dict(info)


def estimate_ram_requirement(repo_id):
    """Estimate RAM requirement in GB for running a HF model.

    Uses real weight sizes from the Hugging Face API when reachable; falls back to
    a parameter-count heuristic (quantization labels such as '4bit' are ignored).
    """
    info = fetch_repo_size_info(repo_id)
    if info.get("est_ram_gb"):
        return info["est_ram_gb"]
    params_b = info.get("params_b") or _parse_params_from_name(repo_id)
    if params_b:
        return round(params_b * 0.62 + 2.0, 1)
    return 6.0

def estimate_ram_from_name(repo_id):
    """Fast RAM estimate (GB) from the repo name only, no network calls."""
    params_b = _parse_params_from_name(repo_id)
    if params_b:
        return round(params_b * 0.62 + 2.0, 1)
    return 0.0


def get_repo_sizes_bulk(repo_ids, workers=12):
    """Fetch size/param info for many repos concurrently (cached after first run)."""
    ids = [r for r in dict.fromkeys(repo_ids or []) if r][:120]
    out = {}
    if not ids:
        return out
    from concurrent.futures import ThreadPoolExecutor

    def one(rid):
        return rid, fetch_repo_size_info(rid)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for rid, info in pool.map(one, ids):
                out[rid] = info
    except Exception:
        for rid in ids:
            try:
                out[rid] = fetch_repo_size_info(rid)
            except Exception:
                pass
    return out


def _enrich_search_results(results, workers=12, usable_ram_gb=None):
    """Attach real weight size, parameter count and RAM estimate to search hits."""
    if not results:
        return results
    sizes = get_repo_sizes_bulk([r.get("id", "") for r in results], workers=workers)
    for item in results:
        info = sizes.get(item.get("id", "")) or {}
        weights_gb = info.get("weights_gb") or 0.0
        est = info.get("est_ram_gb") or 0.0
        item["weights_gb"] = weights_gb
        item["params_b"] = info.get("params_b")
        item["quant"] = info.get("quant") or ""
        item["est_ram_gb"] = est or estimate_ram_from_name(item.get("id", ""))
        item["size_known"] = bool(weights_gb)
        if usable_ram_gb and item["est_ram_gb"]:
            if item["est_ram_gb"] <= usable_ram_gb * 0.8:
                item["fits"] = 0
            elif item["est_ram_gb"] <= usable_ram_gb:
                item["fits"] = 1
            else:
                item["fits"] = 2
    return results


def _repo_mentions_code(repo_id):
    return bool(re.search(r"code|coder|coding|sql|python-", (repo_id or "").lower()))


def search_hf_api(query, limit=50, sort="downloads", direction=-1, filter_tag="mlx",
                  filter_tags=None, quant=None, max_ram=None, min_ram=None,
                  usable_ram_gb=None, enrich=True):
    """Search Hugging Face with COMBINABLE filters.

    filter_tags: list/set of any of {"mlx", "vision", "code"} — all must match (AND).
    quant: one of "2","3","4","5","6","8","16" — keep repos whose quantization
           matches, based on the repo name first and the real config once enriched.
    """
    if filter_tags is None:
        filter_tags = [filter_tag] if filter_tag else []
    tags = {str(t).strip().lower() for t in filter_tags if t and str(t).strip().lower() != "all"}

    q = urllib.parse.quote(query)
    size_sort = sort in ("size_asc", "size_desc")
    api_sort = "downloads" if size_sort else sort
    quant = str(quant).strip() if quant else ""
    # Over-fetch when we post-filter (code / quant) so the page still fills up
    fetch_limit = limit
    if quant or "code" in tags:
        fetch_limit = min(limit * 4, 400)

    params = [f"search={q}", f"limit={fetch_limit}", f"sort={api_sort}", f"direction={direction}"]
    if "mlx" in tags:
        params.append("filter=mlx")
    if "gguf" in tags:
        params.append("filter=gguf")
    if "vision" in tags:
        params.append("pipeline_tag=image-text-to-text")
    url = "https://huggingface.co/api/models?" + "&".join(params)

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    results = []
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
            for item in data:
                repo_id = item.get("id", "")
                if "code" in tags and not _repo_mentions_code(repo_id):
                    continue
                item_tags = item.get("tags", [])
                pipeline_tag = item.get("pipeline_tag", "text-generation")
                is_vis = is_vision_model(repo_id) or pipeline_tag == "image-text-to-text" or "vision" in item_tags or "multimodal" in item_tags
                if "vision" in tags and not is_vis:
                    continue
                q_label = _detect_quant_label(repo_id)
                if quant and q_label and q_label != f"{quant}-bit":
                    continue
                results.append({
                    "id": repo_id,
                    "downloads": item.get("downloads", 0),
                    "likes": item.get("likes", 0),
                    "lastModified": item.get("lastModified", "")[:10] if item.get("lastModified") else "",
                    "tags": item_tags[:4],
                    "est_ram_gb": estimate_ram_from_name(repo_id),
                    "size_known": False,
                    "quant": q_label,
                    "pipeline_tag": pipeline_tag,
                    "is_vision": is_vis
                })
                if len(results) >= limit:
                    break
    except Exception:
        pass

    if results and enrich:
        results = _enrich_search_results(results, usable_ram_gb=usable_ram_gb)
        if quant:
            results = [r for r in results if not r.get("quant") or r["quant"] == f"{quant}-bit"]
        if max_ram:
            results = [r for r in results if not r.get("est_ram_gb") or r["est_ram_gb"] <= float(max_ram)]
        if min_ram:
            results = [r for r in results if not r.get("est_ram_gb") or r["est_ram_gb"] >= float(min_ram)]
    if size_sort and results:
        results.sort(key=lambda r: r.get("est_ram_gb") or 0.0, reverse=(sort == "size_desc"))
    return results

def compare_models_ai(repo_ids):
    """Generates expert technical AI comparison matrix between selected models for Apple Silicon Mac."""
    comparisons = []
    mac_stats = get_system_stats()
    free_ram = mac_stats.get("usable_ram_gb") or mac_stats.get("free_ram_gb", 16.0)
    total_ram = mac_stats.get("total_ram_gb", 0)

    for rid in repo_ids:
        rid_lower = rid.lower()
        
        quant = "Standard 4-Bit (Q4_K_M)"
        quant_desc = "Sweet spot for quality, speed, and memory size on Apple Silicon."
        if "2bit" in rid_lower or "2-bit" in rid_lower or "2bit-dq" in rid_lower:
            quant = "Ultra-Low 2-Bit (DQ)"
            quant_desc = "Extreme memory compression (saves 50%+ RAM). Ideal for fitting 32B+ models in smaller Mac RAM, but with slight accuracy tradeoff."
        elif "3bit" in rid_lower or "3-bit" in rid_lower:
            quant = "Compact 3-Bit (Q3_K_M)"
            quant_desc = "Saves ~25% RAM over 4-bit while maintaining 95% of standard quality."
        elif "8bit" in rid_lower or "8-bit" in rid_lower:
            quant = "High-Precision 8-Bit (Q8_0)"
            quant_desc = "Near lossless precision, requires ~2x RAM compared to 4-bit."
        elif "optiq" in rid_lower:
            quant = "OptiQ 4-Bit"
            quant_desc = "Quantized with per-layer optimization for higher reasoning benchmark scores."

        est_ram = estimate_ram_requirement(rid)
        
        if est_ram <= free_ram * 0.8:
            fit_status = f"🟢 Fits Smoothly ({est_ram} GB RAM needed)"
        elif est_ram <= free_ram:
            fit_status = f"🟡 High RAM Pressure ({est_ram} GB RAM needed)"
        else:
            fit_status = f"🔴 Swap Danger ({est_ram} GB RAM exceeds free {free_ram} GB)"

        is_vis = is_vision_model(rid)
        spec = "General Chat & Instruction"
        if is_vis:
            spec = "👁️ Vision & Multimodal Image Understanding"
        elif "code" in rid_lower or "coder" in rid_lower:
            spec = "💻 Coding Agent & Software Engineering"
        elif "reasoning" in rid_lower or "distilled" in rid_lower or "r1" in rid_lower:
            spec = "🧠 Chain-of-Thought Reasoning & Math"
        elif "flash" in rid_lower:
            spec = "⚡ High-Speed Low Latency Inference"

        if is_vis:
            verdict = "Top pick for multimodal tasks: image analysis, document vision, OCR, and reasoning."
        elif "2bit" in rid_lower:
            verdict = "Best choice if RAM is constrained. Allows running large models (32B) without memory paging."
        elif "code" in rid_lower or "coder" in rid_lower:
            verdict = "Top choice for Pi Code & OpenCode programming tasks."
        elif est_ram <= free_ram * 0.8:
            verdict = "🌟 Recommended Top Pick: Excellent quality and zero RAM paging."
        else:
            verdict = "Requires closing heavy background apps to prevent memory swapping."

        size_info = fetch_repo_size_info(rid)
        comparisons.append({
            "repo_id": rid,
            "quant": quant,
            "quant_desc": quant_desc,
            "weights_gb": size_info.get("weights_gb") or 0.0,
            "params_b": size_info.get("params_b"),
            "est_ram_gb": est_ram,
            "fit_status": fit_status,
            "fits": 0 if est_ram <= free_ram * 0.8 else (1 if est_ram <= free_ram else 2),
            "specialization": spec,
            "is_vision": is_vis,
            "verdict": verdict
        })

    return {"comparisons": comparisons, "free_ram_gb": free_ram, "total_ram_gb": total_ram}

def openai_target(base_url="", host="127.0.0.1", port=9999, model=""):
    """Resolve (chat-completions URL, model id) for any OpenAI-compatible engine.

    MLX servers are one process per port and ignore the model name; Ollama is one
    daemon for many models, so the id has to be the exact tag. Keeping this rule
    in one place is what makes the test panel engine-agnostic.
    """
    base = (base_url or "").strip().rstrip("/")
    # Collapse a doubled suffix: several callers hand us a URL that already ends
    # in /v1 (or /v1/v1 after an older config was saved). Exactly one /v1 is
    # always correct, so fix it here rather than trusting every caller.
    while base.endswith("/v1/v1"):
        base = base[:-3]
    if base:
        if not base.endswith("/v1"):
            base += "/v1"
        url = base + "/chat/completions"
    else:
        url = f"http://{host}:{port}/v1/chat/completions"
    return url, ((model or "").strip() or "default_model")

def run_model_benchmark(prompt, model, port=9999, host="127.0.0.1", base_url=""):
    """Benchmark any OpenAI-compatible endpoint — MLX servers and Ollama alike."""
    url, test_model_name = openai_target(base_url, host=host, port=port, model=model)

    req_data = json.dumps({
        "model": test_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256
    }).encode("utf-8")

    t0 = time.time()
    req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            t1 = time.time()
            res_data = json.loads(resp.read().decode("utf-8"))
            tot_sec = round(t1 - t0, 2)
            
            usage = res_data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            msg = res_data.get("choices", [{}])[0].get("message", {})
            content = msg.get("content", "")
            reasoning = msg.get("reasoning", "") or msg.get("reasoning_content", "")
            
            tok_per_sec = round(completion_tokens / tot_sec, 2) if tot_sec > 0 and completion_tokens > 0 else 0.0

            stats = get_system_stats()

            return {
                "status": "ok",
                "model": model,
                "port": port,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_sec": tot_sec,
                "tok_per_sec": tok_per_sec,
                "response_text": content or reasoning,
                "mlx_ram_gb": stats.get("mlx_ram_gb", 0.0)
            }
    except Exception as e:
        return {"error": str(e)}

def find_free_port(start_port=9999):
    port = int(start_port)
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    return start_port

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "find_models":
        print(json.dumps(find_local_models()))
    elif cmd == "running_servers":
        print(json.dumps(get_running_servers()))
    elif cmd == "active_downloads":
        print(json.dumps(get_active_downloads()))
    elif cmd == "system_stats":
        print(json.dumps(get_system_stats()))
    elif cmd == "sync_configs":
        mname = sys.argv[2] if len(sys.argv) > 2 else "default_model"
        sync_pi_and_opencode(mname)
        print(f"Synced configs for {mname}")
    elif cmd == "search_hf":
        q = sys.argv[2] if len(sys.argv) > 2 else "mlx"
        mr = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
        tags_arg = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else "mlx"
        qt = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None
        print(json.dumps(search_hf_api(q, filter_tags=[t for t in tags_arg.split(",") if t],
                                       quant=qt, max_ram=mr)))
    elif cmd == "repo_size":
        print(json.dumps(fetch_repo_size_info(sys.argv[2] if len(sys.argv) > 2 else "")))
    elif cmd == "turbo_download":
        # argv: turbo_download <repo> <hub> <download_id> [subfolder] [revision]
        repo = sys.argv[2] if len(sys.argv) > 2 else ""
        hub = sys.argv[3] if len(sys.argv) > 3 else ""
        did = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else repo
        sub = sys.argv[5] if len(sys.argv) > 5 else ""
        rev = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] else "main"
        inc = json.loads(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[7] else None
        try:
            res = turbo_download(repo, hub, subfolder=sub, download_id=did, revision=rev, include=inc)
        except Exception as e:
            res = {"ok": False, "error": str(e)}
        if not res.get("ok"):
            write_turbo_status(did, {"download_id": did, "repo_id": repo, "phase": "error",
                                     "error": res.get("error", "download failed"),
                                     "downloaded_bytes": res.get("bytes", 0), "pid": os.getpid()})
            print(f"TURBO FAILED: {res.get('error')} — falling back to the standard downloader")
        # Phase 2: normal hub client finishes links/refs and any remaining small files.
        write_turbo_status(did, {"download_id": did, "repo_id": repo, "phase": "hub",
                                 "hub": hub, "downloaded_bytes": res.get("bytes", 0),
                                 "total_bytes": res.get("bytes", 0), "pid": os.getpid()})
        final = get_download_cmd(repo, subfolder=sub, revision=rev, cache_dir=hub)
        env = get_mlx_env()
        if res.get("ok"):
            # Xet keeps its own chunk store and would re-download what we just placed;
            # the classic client links our verified blobs instead.
            env["HF_HUB_DISABLE_XET"] = "1"
        try:
            proc = subprocess.Popen(final, env=env)
            proc.wait()
        finally:
            clear_turbo_status(did)
    elif cmd == "turbo_plan":
        print(json.dumps(turbo_plan(sys.argv[2] if len(sys.argv) > 2 else "")))
    elif cmd == "hf_token":
        print(get_hf_token())
    elif cmd == "hf_auth":
        print(json.dumps(hf_auth_state()))
    elif cmd == "hf_login":
        st = hf_device_start()
        if not st.get("ok"):
            print(json.dumps({"ok": False, "error": st.get("error")}))
            sys.exit(1)
        print(f"Open {st['verification_uri']} and approve the code: {st['user_code']}")
        deadline = time.time() + int(st.get("expires_in", 300))
        interval = max(3, int(st.get("interval", 5)))
        while time.time() < deadline:
            time.sleep(interval)
            poll = hf_device_poll(st["device_code"])
            if poll.get("status") == "accepted":
                print(f"✓ Signed in as {poll.get('user') or 'unknown'}")
                break
            if poll.get("status") in ("expired", "denied", "error"):
                print(json.dumps({"ok": False, "status": poll.get("status")}))
                sys.exit(1)
    elif cmd == "hf_logout":
        clear_hf_token()
        print("Signed out of Hugging Face (token removed from this Mac).")
    elif cmd == "drives":
        print(json.dumps({
            "drives": list_storage_drives(force=True),
            "default": (get_default_drive() or {}).get("id", ""),
        }))
    elif cmd == "set_drive":
        target = sys.argv[2] if len(sys.argv) > 2 else ""
        d = set_default_drive(target)
        print(json.dumps({"default": (d or {}).get("name", "")}))
    elif cmd == "free_port":
        p = sys.argv[2] if len(sys.argv) > 2 else "9999"
        print(find_free_port(p))
