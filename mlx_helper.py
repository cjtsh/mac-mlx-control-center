#!/usr/bin/env python3
import os, sys, json, subprocess, re, urllib.request, urllib.parse, socket, time, threading

_REPO_SIZE_CACHE = {}
_SPEED_CACHE = {}
PAUSED_FILE = os.path.expanduser("~/.mlx_pids/paused_downloads.json")
SETTINGS_FILE = os.path.expanduser("~/.mlx_pids/settings.json")
SIZE_INFO_CACHE_FILE = os.path.expanduser("~/.mlx_pids/repo_sizes.json")
SIZE_INFO_TTL = 14 * 86400  # re-check repo sizes every 2 weeks
_SIZE_INFO_CACHE = {}
_SIZE_INFO_LOCK = threading.Lock()
WEIGHT_FILE_EXTS = (".safetensors", ".gguf", ".bin", ".pt", ".pth", ".npz", ".h5")

def load_global_settings():
    default_settings = {
        "default_port": 9999,
        "default_host": "127.0.0.1",
        "default_temp": 0.0,
        "default_max_tokens": 4096,
        "auto_sync_agents": True
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
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    env["HF_XET_HIGH_PERFORMANCE"] = "1"
    return env

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

def get_download_cmd(repo_id, subfolder="", revision="", cache_dir=""):
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
        cmd = [uvx_bin, "--from", "huggingface_hub[hf_transfer]", "hf", "download", repo_id]
    else:
        cmd = ["hf", "download", repo_id]

    if subfolder:
        cmd.extend(["--include", f"{subfolder}/*"])
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

def get_repo_files_size(repo_id, subfolder=""):
    cache_key = f"{repo_id}:{subfolder}" if subfolder else repo_id
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
                    if subfolder:
                        if path.startswith(sf_prefix) or path == subfolder:
                            total += sz
                    else:
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
            
            variants = []
            for qd in (quant_dirs if quant_dirs else dirs):
                sz = get_repo_files_size(repo_id, subfolder=qd)
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

def is_vision_model(model_path_or_id, config_dict=None):
    """Accurately detects whether a local path or Hugging Face model ID is a Vision / Multimodal model."""
    name_check = str(model_path_or_id).lower()
    
    # 1. Inspect config dictionary or config.json if available
    cfg = config_dict
    if cfg is None and os.path.exists(str(model_path_or_id)):
        cfg_file = os.path.join(model_path_or_id, "config.json")
        if os.path.exists(cfg_file):
            try:
                with open(cfg_file, "r", errors="ignore") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = None
                
    if isinstance(cfg, dict):
        model_type = str(cfg.get("model_type", "")).lower()
        archs = [str(a).lower() for a in cfg.get("architectures", []) if a]
        
        vision_model_types = {
            "gemma4", "gemma4_unified", "gemma4_unified_audio", "gemma3", "gemma3_vl",
            "qwen2_vl", "qwen2_5_vl", "qwen_vl", "mllama", "paligemma", "paligemma2",
            "pixtral", "llava", "llava_next", "llava_hound", "internvl_chat", "smolvlm",
            "florence2", "phi3_v", "chameleon", "minicpmv", "molmo", "idefics2", "idefics3",
            "multi_modality", "blip-2", "vision-encoder-decoder", "mistral3"
        }
        if model_type in vision_model_types:
            return True
            
        if isinstance(cfg.get("vision_config"), dict) and len(cfg["vision_config"]) > 0:
            return True
            
        for a in archs:
            if any(k in a for k in ["qwen2vl", "qwen2_5_vl", "qwenvl", "gemma4", "paligemma", "pixtral", "mllama", "llava", "molmo", "smolvlm", "vision", "multimodality"]):
                return True

    # 2. Check if preprocessor or processor config exists in local directory
    if os.path.isdir(str(model_path_or_id)):
        if os.path.exists(os.path.join(model_path_or_id, "preprocessor_config.json")) or \
           os.path.exists(os.path.join(model_path_or_id, "processor_config.json")):
            return True

    # 3. Explicitly exclude code reasoning / distilled text LLMs that lack vision towers
    if any(k in name_check for k in ["code-reasoning", "deepseek-v4", "r1-distill", "coder"]):
        return False

    # 4. Specific Keyword / Substring heuristics on model ID / path
    vision_keywords = ["-vl-", "-vl", "qwen2-vl", "qwen2.5-vl", "qwen3-vl", "gemma-4", "gemma-3", "paligemma", "pixtral", "llava", "smolvlm", "minicpm-v", "florence", "internvl", "molmo", "vision"]
    if any(k in name_check for k in vision_keywords):
        return True

    return False

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
                            "contextWindow": 131072,
                            "maxTokens": 8192,
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
                        },
                        {
                            "id": active_model,
                            "name": f"{active_model} (Local)",
                            "reasoning": True,
                            "input": inputs,
                            "contextWindow": 131072,
                            "maxTokens": 8192,
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
                        }
                    ]
                }
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

                speed_str = "Calculating..."
                eta_str = "--"
                if download_id in _SPEED_CACHE:
                    prev_bytes, prev_time = _SPEED_CACHE[download_id]
                    dt = now - prev_time
                    if dt > 0.5:
                        dbytes = downloaded_bytes - prev_bytes
                        speed_bps = dbytes / dt
                        speed_mb = speed_bps / (1024**2)
                        if speed_mb > 0.05:
                            speed_str = f"{speed_mb:.1f} MB/s"
                            rem_bytes = max(0, total_bytes - downloaded_bytes)
                            if rem_bytes > 0 and speed_bps > 0:
                                rem_sec = int(rem_bytes / speed_bps)
                                if rem_sec < 60:
                                    eta_str = f"{rem_sec}s"
                                else:
                                    eta_str = f"{rem_sec//60}m {rem_sec%60}s"
                        else:
                            speed_str = "0 MB/s"
                _SPEED_CACHE[download_id] = (downloaded_bytes, now)

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
                    "status": "DOWNLOADING",
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
        "cpu_percent": cpu_percent,
        "active_models": len(servers)
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

def run_model_benchmark(prompt, model, port=9999, host="127.0.0.1"):
    url = f"http://{host}:{port}/v1/chat/completions"
    test_model_name = "default_model"
    if model and "/" in model:
        test_model_name = model

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
