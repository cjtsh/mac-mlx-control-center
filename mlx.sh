#!/bin/bash
#
# mlx.sh - Complete MLX Model Manager for MacBook
# Start, stop, list, search, download, delete, and test MLX LM models.
#

export PATH="$HOME/.local/bin:$HOME/bin:$PATH"

MODEL_DIR="${HOME}/mlx_models"
PID_DIR="${HOME}/.mlx_pids"
LOG_DIR="${HOME}/.mlx_logs"
DEFAULT_PORT=9999
DEFAULT_HOST="127.0.0.1"

mkdir -p "$MODEL_DIR" "$PID_DIR" "$LOG_DIR"

# Prefer the helper shipped next to this script (keeps CLI in sync with the GUI),
# falling back to the copy installed in $PID_DIR for standalone installs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/mlx_helper.py" ]; then
    HELPER_PY="$SCRIPT_DIR/mlx_helper.py"
else
    HELPER_PY="$PID_DIR/mlx_helper.py"
fi

# Ensure Python helper exists
if [ ! -f "$HELPER_PY" ]; then
    cat << 'EOF' > "$HELPER_PY"
#!/usr/bin/env python3
import os, sys, json, subprocess, re, urllib.request, urllib.parse, socket

def find_local_models():
    search_paths = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/MLXConsole/models/hub"),
        os.path.expanduser("~/.lmstudio/hub/models"),
        os.path.expanduser("~/.cache/lmstudio/models"),
        os.path.expanduser("~/mlx_models"),
    ]
    models = []
    seen_paths = set()

    for spath in search_paths:
        if not os.path.exists(spath):
            continue
        for root, dirs, files in os.walk(spath):
            if "config.json" in files and "1_Pooling" not in root:
                real_path = os.path.realpath(root)
                if real_path in seen_paths:
                    continue
                seen_paths.add(real_path)

                parent_hub_dir = root
                parts = root.split(os.sep)
                name = ""
                for p in parts:
                    if p.startswith("models--"):
                        name = p.replace("models--", "").replace("--", "/")
                        parent_hub_dir = root[:root.find(p) + len(p)]
                        break
                if not name:
                    name = os.path.basename(root)

                size_bytes = 0
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

                models.append({
                    "name": name,
                    "size": size_str,
                    "path": root,
                    "delete_target": calc_dir
                })
    return models

def get_running_servers():
    servers = []
    try:
        output = subprocess.check_output(["ps", "aux"], text=True)
        for line in output.splitlines():
            if "mlx_lm.server" in line and "grep" not in line:
                parts = line.split()
                pid = parts[1]
                cmd_start = parts[10] if len(parts) > 10 else ""

                model_match = re.search(r"--model\s+([^\s]+)", line)
                port_match = re.search(r"--port\s+([^\s]+)", line)
                host_match = re.search(r"--host\s+([^\s]+)", line)

                model = model_match.group(1) if model_match else "unknown"
                port = port_match.group(1) if port_match else "9999"
                host = host_match.group(1) if host_match else "127.0.0.1"

                if "models--" in model:
                    m = re.search(r"models--([^/]+)--(.*?)(?:/snapshots|$)", model)
                    if m:
                        model = f"{m.group(1)}/{m.group(2)}"

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

def _params_from_name(rid):
    rid = (rid or "").lower()
    m = re.search(r"(?<![0-9a-z])(\d+)x(\d+(?:\.\d+)?)b(?![0-9a-z])", rid)
    if m:
        return int(m.group(1)) * float(m.group(2))
    m = re.search(r"(?<![0-9a-z.])(\d+(?:\.\d+)?)b(?![0-9a-z])", rid)
    return float(m.group(1)) if m else None

def repo_size_info(repo_id):
    """Exact weight size + param count from the HF API (blobs=true)."""
    exts = (".safetensors", ".gguf", ".bin", ".pt", ".pth")
    try:
        url = f"https://huggingface.co/api/models/{urllib.parse.quote(repo_id, safe='/')}?blobs=true"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode())
        groups = {}
        for s in data.get("siblings") or []:
            fn = s.get("rfilename") or ""
            sz = s.get("size") or 0
            if sz > 0 and fn.lower().endswith(exts):
                d = os.path.dirname(fn)
                groups[d] = groups.get(d, 0) + sz
        wb = max(groups.values()) if groups else 0
        st = data.get("safetensors") or {}
        total = st.get("total") or (st.get("parameters") or {}).get("total")
        pb = (total / 1e9) if total else _params_from_name(repo_id)
        if wb:
            est = round(wb / 1e9 * 1.06 + min(8.0, max(0.75, (pb or 0) * 0.02)) + 1.5, 1)
        elif pb:
            est = round(pb * 0.62 + 2.0, 1)
        else:
            est = 0.0
        return {"weights_gb": round(wb / 1e9, 2), "params_b": pb, "est_ram_gb": est}
    except Exception:
        return {"weights_gb": 0.0, "params_b": _params_from_name(repo_id), "est_ram_gb": 0.0}

def search_hf_api(query, limit=15, max_ram=None, filter_tags=None, quant=None):
    q = urllib.parse.quote(query)
    tags = {str(t).strip().lower() for t in (filter_tags or []) if t and str(t).strip().lower() != "all"}
    params = [f"search={q}", f"limit={limit if not (quant or 'code' in tags) else limit * 4}", "sort=downloads", "direction=-1"]
    if "mlx" in tags:
        params.append("filter=mlx")
    if "vision" in tags:
        params.append("pipeline_tag=image-text-to-text")
    url = "https://huggingface.co/api/models?" + "&".join(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    results = []
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            for item in data:
                rid = item.get("id", "")
                if "code" in tags and not __import__("re").search(r"code|coder|coding", rid.lower()):
                    continue
                ql = ""
                m = __import__("re").search(r"(?<![0-9])([2-8])[ ._-]?bit", rid.lower())
                if m:
                    ql = m.group(1) + "-bit"
                if quant and ql and ql != f"{quant}-bit":
                    continue
                results.append({
                    "id": rid,
                    "downloads": item.get("downloads", 0),
                    "likes": item.get("likes", 0),
                    "quant": ql
                })
                if len(results) >= limit:
                    break
    except Exception:
        pass
    if results:
        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=12) as pool:
                infos = list(pool.map(lambda r: repo_size_info(r["id"]), results))
            for r, info in zip(results, infos):
                r.update(info)
        except Exception:
            for r in results:
                r.update({"weights_gb": 0.0, "params_b": None, "est_ram_gb": 0.0})
    if max_ram:
        results = [r for r in results if not r.get("est_ram_gb") or r["est_ram_gb"] <= float(max_ram)]
    return results

def find_free_port(start_port=9999):
    port = int(start_port)
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    return start_port

# ── Storage drives (compact fallback: discovery + library layout) ────
STORAGE_FILE = os.path.expanduser("~/.mlx_pids/storage.json")
LIBRARY_DIRNAME = "MLX-Models"
MIN_DRIVE_GB = 8

def load_storage_config():
    try:
        with open(STORAGE_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def save_storage_config(cfg):
    try:
        os.makedirs(os.path.dirname(STORAGE_FILE), exist_ok=True)
        tmp = STORAGE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, STORAGE_FILE)
    except Exception:
        pass

def tm_exclusion_state(path):
    probe = path
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe.rstrip("/"))
        if parent == probe:
            break
        probe = parent
    try:
        out = subprocess.run(["tmutil", "isexcluded", probe or path], capture_output=True, text=True, timeout=8)
        return "[Excluded]" in (out.stdout or "")
    except Exception:
        return False

def exclude_from_time_machine(path):
    if not path or not os.path.exists(path):
        return False
    if tm_exclusion_state(path):
        return True
    try:
        subprocess.run(["tmutil", "addexclusion", path], capture_output=True, text=True, timeout=15)
        return tm_exclusion_state(path)
    except Exception:
        return False

def protect_model_storage():
    done = []
    for t in [os.path.expanduser("~/MLX-Models"), os.path.expanduser("~/.cache/huggingface")]:
        if os.path.isdir(t) and not tm_exclusion_state(t) and exclude_from_time_machine(t):
            done.append(t)
    for d in list_storage_drives(force=True):
        lib = d.get("library") or ""
        parent = os.path.dirname(lib) if lib else ""
        if parent and os.path.isdir(parent) and not tm_exclusion_state(parent):
            if exclude_from_time_machine(parent):
                done.append(parent)
    return done

def _vol_free_gb(path):
    try:
        st = os.statvfs(path)
        return round(st.f_bavail * st.f_frsize / (1024 ** 3), 1)
    except Exception:
        return 0.0

def list_storage_drives(force=False, include_known=True):
    drives = []
    mounts = [("/", "Macintosh HD")]
    try:
        for name in sorted(os.listdir("/Volumes")):
            mp = os.path.join("/Volumes", name)
            if os.path.ismount(mp):
                mounts.append((mp, name))
    except Exception:
        pass
    seen = set()
    for mount, name in mounts:
        probe = os.path.expanduser("~") if mount == "/" else mount
        try:
            if not os.access(probe, os.W_OK):
                continue
            st = os.statvfs(probe)
            total_gb = st.f_blocks * st.f_frsize / (1024 ** 3)
        except Exception:
            continue
        if total_gb < MIN_DRIVE_GB:
            continue
        key = os.path.realpath(mount)
        if key in seen:
            continue
        seen.add(key)
        internal = mount == "/"
        library = (os.path.expanduser("~/MLX-Models") if internal else os.path.join(mount, LIBRARY_DIRNAME)) + "/hub"
        drives.append({
            "id": key,
            "name": name,
            "mount": mount,
            "internal": internal,
            "free_gb": _vol_free_gb(probe),
            "total_gb": round(total_gb, 1),
            "library": library,
            "library_exists": os.path.isdir(library),
            "models": len([d for d in os.listdir(library) if d.startswith("models--")]) if os.path.isdir(library) else 0,
            "is_time_machine": False,
            "tm_excluded": tm_exclusion_state(library),
            "connected": True,
            "disabled": False,
        })
    cfg = load_storage_config()
    default_id = cfg.get("default_drive", "")
    for d in drives:
        d["is_default"] = d["id"] == default_id
    return sorted(drives, key=lambda d: (d["internal"], -d["free_gb"]))

def get_default_drive(drives=None):
    drives = drives if drives is not None else list_storage_drives()
    for d in drives:
        if d.get("is_default"):
            return d
    external = [d for d in drives if not d["internal"] and not d["is_time_machine"]]
    return (external or drives or [None])[0]

def set_default_drive(drive_id):
    cfg = load_storage_config()
    cfg["default_drive"] = drive_id
    save_storage_config(cfg)

def ensure_library_hub(drive_id=None):
    drives = list_storage_drives()
    target = None
    if drive_id:
        for d in drives:
            if drive_id in (d["id"], d["name"], d["mount"]):
                target = d
                break
    if not target:
        target = get_default_drive(drives)
    if not target:
        return "", None
    hub = target["library"]
    os.makedirs(hub, exist_ok=True)
    exclude_from_time_machine(os.path.dirname(hub))
    return hub, target

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "find_models":
        print(json.dumps(find_local_models()))
    elif cmd == "running_servers":
        print(json.dumps(get_running_servers()))
    elif cmd == "search_hf":
        q = sys.argv[2] if len(sys.argv) > 2 else "mlx"
        mr = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
        tg = [t for t in (sys.argv[4] if len(sys.argv) > 4 else "mlx").split(",") if t]
        qt = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None
        print(json.dumps(search_hf_api(q, max_ram=mr, filter_tags=tg, quant=qt)))
    elif cmd == "free_port":
        p = sys.argv[2] if len(sys.argv) > 2 else "9999"
        print(find_free_port(p))
    elif cmd == "drives":
        print(json.dumps(list_storage_drives(force=True)))
EOF
    chmod +x "$HELPER_PY"
fi

# Check navigation input (b = Back, e = Exit)
check_nav() {
    local val="$1"
    local input_lower
    input_lower=$(echo "$val" | tr '[:upper:]' '[:lower:]')
    if [ "$input_lower" = "e" ]; then
        echo ""
        echo "Exiting MLX Model Manager. Goodbye!"
        exit 0
    elif [ "$input_lower" = "b" ]; then
        return 1
    fi
    return 0
}

# Display running models header
show_running_header() {
    echo ""
    echo "=== Currently Running MLX Models ==="
    local running_json
    running_json=$(python3 "$HELPER_PY" running_servers)
    
    local count
    count=$(echo "$running_json" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)

    if [ "$count" -eq 0 ]; then
        echo "  (none running)"
    else
        echo "$running_json" | python3 -c "
import sys, json
servers = json.load(sys.stdin)
unresponsive = False
for i, s in enumerate(servers, 1):
    tag = f'[{s[\"status\"]}]'
    if 'UNRESPONSIVE' in s['status']:
        unresponsive = True
    print(f'  {i}. {tag:<22} {s[\"model\"]} (PID: {s[\"pid\"]}, Host: {s[\"host\"]}, Port: {s[\"port\"]})')
if unresponsive:
    print('  ⚡ Tip: Type \"k\" or \"k <#>\" (e.g. k 2) to KILL any process immediately!')
"
    fi
    echo "======================================"
    python3 - "$HELPER_PY" <<'PYCODE' 2>/dev/null || true
import sys
sys.path.insert(0, sys.argv[1].rsplit("/", 1)[0])
try:
    import mlx_helper as h
    st = h.stream_guard_status()
except Exception:
    raise SystemExit(0)
if not st.get("available"):
    print("  🛡️  Stream guard: not installed (mlx_stream_guard/ missing)")
elif not st.get("enabled"):
    print("  🛡️  Stream guard: disabled by MLX_STREAM_GUARD=0")
elif not st.get("active"):
    print("  🛡️  Stream guard: armed — applies to the next server you start")
else:
    bits = []
    if st.get("keepalives"):
        bits.append("%d keepalive(s)" % st["keepalives"])
    if st.get("synthetic_done"):
        bits.append("%d late [DONE]" % st["synthetic_done"])
    if st.get("errors"):
        bits.append("%d error(s)" % st["errors"])
    print("  🛡️  Stream guard: on · %d stream(s)%s" % (
        st.get("streams", 0), (" · " + " · ".join(bits)) if bits else ""))
PYCODE
    echo ""
}

# ── 1. List Models ───────────────────────────────────────────────────
list_models() {
    local interactive="${1:-true}"
    echo ""
    echo "=== Downloaded MLX Models ==="
    local local_json
    local_json=$(python3 "$HELPER_PY" find_models)

    local count
    count=$(echo "$local_json" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)

    if [ "$count" -eq 0 ]; then
        echo "  (no downloaded models found)"
    else
        echo "$local_json" | python3 -c "
import sys, json
models = json.load(sys.stdin)
for i, m in enumerate(models, 1):
    print(f'  {i:2d}. {m[\"name\"]:<50} ({m[\"size\"]})')
    print(f'      Path: {m[\"path\"]}')
"
    fi
    echo ""
    if [ "$interactive" = "false" ]; then
        return
    fi

    echo "Options: Enter model number [1-$count] to Start, [b] Back, [e] Exit"
    echo -n "Select option: "
    read -r choice
    check_nav "$choice" || return

    if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "$count" ]; then
        local selected_name
        selected_name=$(echo "$local_json" | python3 -c "import sys, json; print(json.load(sys.stdin)[$choice-1]['name'])")
        start_model "$selected_name"
    fi
}

# ── 2. Search Hugging Face ───────────────────────────────────────────
search_hf() {
    local query="${1:-}"
    local max_ram="${2:-}"
    local tags="${3:-}"
    local quant="${4:-}"
    if [ -z "$query" ]; then
        echo ""
        echo -n "Enter search term for Hugging Face (e.g. mlx, qwen, llama) [default: mlx] (b: Back, e: Exit): "
        read -r query
        check_nav "$query" || return
        [ -z "$query" ] && query="mlx"

        echo -n "Filters, combinable (mlx,vision,code | all) [Enter = mlx]: "
        read -r tags
        check_nav "$tags" || return
        [ -z "$tags" ] && tags="mlx"

        echo -n "Quantization (2,4,6,8,16 | Enter = any): "
        read -r quant
        check_nav "$quant" || return
    fi
    case "$tags" in
        all|"") tags="" ;;
    esac
    case "$quant" in
        any|all) quant="" ;;
    esac

    # How much unified memory can a model actually use on this Mac?
    local usable_ram
    usable_ram=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.1f", $1/1073741824*0.88}')
    [ -z "$usable_ram" ] && usable_ram="0"

    if [ -z "$max_ram" ]; then
        echo ""
        echo -n "RAM budget in GB — only show models that fit [Enter = this Mac's usable ${usable_ram} GB, 'all' = no limit]: "
        read -r max_ram
        check_nav "$max_ram" || return
    fi
    if [ "$max_ram" = "all" ] || [ "$max_ram" = "0" ]; then
        max_ram=""
    fi

    echo ""
    if [ -n "$max_ram" ]; then
        echo "Searching Hugging Face for '$query' (filters: ${tags:-ALL}${quant:+ | ${quant}-bit} | RAM budget: ${max_ram} GB)..."
    else
        echo "Searching Hugging Face for '$query' (filters: ${tags:-ALL}${quant:+ | ${quant}-bit} | no RAM budget)..."
    fi
    echo "---------------------------------------------------------"

    local results_json
    results_json=$(python3 "$HELPER_PY" search_hf "$query" "$max_ram" "${tags:-mlx}" "$quant")

    local count
    count=$(echo "$results_json" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)

    if [ "$count" -eq 0 ]; then
        echo "No results found on Hugging Face${max_ram:+ within ${max_ram} GB RAM}."
        echo ""
        read -r -p "Press Enter to return..."
        return
    fi

    echo "$results_json" | python3 -c "
import sys, json
results = json.load(sys.stdin)
for i, r in enumerate(results, 1):
    est = r.get('est_ram_gb') or 0
    weights = r.get('weights_gb') or 0
    params = r.get('params_b')
    size = f'{est:5.1f} GB RAM' if est else '       ? GB'
    pub = f'({params:.1f}B)' if params else ''
    print(f'  {i:2d}. {r[\"id\"]:<52} {size} {weights:6.1f} GB weights {pub:<12} Downloads: {r[\"downloads\"]:>7,}')
"
    echo "---------------------------------------------------------"
    echo "  Fit status is relative to ${max_ram:-unlimited} GB RAM budget."
    echo ""
    echo -n "Enter number [1-$count] to Download, [s] New Search, [b] Back, [e] Exit: "
    read -r choice
    check_nav "$choice" || return

    if [ "$choice" = "s" ] || [ "$choice" = "S" ]; then
        search_hf ""
    elif [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "$count" ]; then
        local selected_id
        selected_id=$(echo "$results_json" | python3 -c "import sys, json; print(json.load(sys.stdin)[$choice-1]['id'])")
        download_model "$selected_id"
    fi
}

# ── Storage Drives ───────────────────────────────────────────────────
# Every writable APFS/HFS+ volume is a candidate library; nothing is
# configured by hand.  Each drive keeps models in <drive>/MLX-Models/hub.
cmd_sync_harness() {
    python3 - "$SCRIPT_DIR" "$PID_DIR" <<'PYCODE'
import sys
for p in (sys.argv[2], sys.argv[1]):
    if p not in sys.path:
        sys.path.insert(0, p)
import mlx_helper as h
res = h.sync_dsh_models()
if not res.get("ok"):
    print("x " + (res.get("error") or "sync failed"))
    sys.exit(1)
if not res.get("changed"):
    print("OK DeepSeek Harness already lists all %d installed model(s)." % res["installed"])
else:
    for rid in res.get("added") or []:
        print("  + " + rid)
    for rid in res.get("removed") or []:
        print("  - " + rid)
    print("OK Synced %d model(s) into %s" % (res["installed"], res["patch"]))
    print("   Restart DeepSeek Harness to load the new list (desktop profile reads it at startup).")
PYCODE
}

list_drives_table() {
    python3 -c "
import sys
sys.path.insert(0, '$PID_DIR')
sys.path.insert(0, '$SCRIPT_DIR')
import mlx_helper as h

drives = h.list_storage_drives(force=True)
default = h.get_default_drive(drives)
default_id = default['id'] if default else ''
usable = [d for d in drives if d['connected'] and not d['disabled']]

print('💾 MODEL LIBRARY DRIVES')
print('-' * 78)
for i, d in enumerate(usable, 1):
    mark = '*' if d['id'] == default_id else ' '
    kind = 'INT' if d['internal'] else 'EXT'
    count = d.get('models', 0)
    tm = '🚫 TM off' if d.get('tm_excluded') else '⚠️  TM ON'
    print(f\"{mark} {i}. {d['name']:<24} {kind} {d['free_gb']:>7.1f} GB free  {count:>2} model(s)  {tm}\")
    print(f\"      {d['library']}\")

off = [d for d in drives if not d['connected'] and d.get('models')]
if off:
    print('-' * 78)
    print('⚠️  Not connected right now:')
    for d in off:
        print(f\"   {d['name']:<24} {d['mount']}  ({d['models']} model(s) offline)\")

print('-' * 78)
if default:
    print(f\"Default download target: {default['name']} ({default['free_gb']:.1f} GB free)\")
print('* = default.  Change it: ./mlx.sh use-drive <name>')
"
}

cmd_drives() {
    list_drives_table
}

cmd_login() {
    echo ""
    echo "🤗 Hugging Face sign-in (device flow — no password leaves this Mac)"
    echo ""
    python3 "$HELPER_PY" hf_login
    echo ""
}

cmd_logout() {
    python3 "$HELPER_PY" hf_logout
}

cmd_whoami() {
    python3 "$HELPER_PY" hf_auth | python3 -c "
import sys, json
d = json.load(sys.stdin)
if d.get('signed_in'):
    print(f\"🤗 Signed in as {d.get('user') or 'unknown'} ({d.get('kind','token')}) — stored in {d.get('storage','keychain')}\")
else:
    print('Not signed in — downloads run anonymously (public repos only, lower rate limits)')
    print('Run ./mlx.sh login to sign in with your own account.')
"
}

cmd_protect_models() {
    python3 -c "
import sys
sys.path.insert(0, '$PID_DIR')
sys.path.insert(0, '$SCRIPT_DIR')
import mlx_helper as h

done = h.protect_model_storage()
if done:
    print('🚫 Excluded from Time Machine:')
    for p in done:
        print(f'   {p}')
else:
    print('✓ Every model library is already excluded from Time Machine.')
print()
for d in h.list_storage_drives(force=True):
    state = 'excluded' if d.get('tm_excluded') else 'BACKED UP'
    print(f\"   {d['name']:<24} {d['library']} — {state}\")
"
}

cmd_use_drive() {
    local sel="${1:-}"
    if [ -z "$sel" ]; then
        list_drives_table
        echo ""
        echo -n "Set default download drive (name or number): "
        read -r sel
    fi
    [ -z "$sel" ] && return
    python3 -c "
import sys
sys.path.insert(0, '$PID_DIR')
sys.path.insert(0, '$SCRIPT_DIR')
import mlx_helper as h

drives = h.list_storage_drives(force=True)
usable = [d for d in drives if d['connected'] and not d['disabled']]
sel = '''$sel'''.strip()
target = None
if sel.isdigit() and 1 <= int(sel) <= len(usable):
    target = usable[int(sel) - 1]
else:
    for d in usable:
        if sel.lower() in (d['name'].lower(), d['id'].lower(), d['mount'].lower()):
            target = d
            break
if not target:
    print(f\"✗ No drive matches '{sel}'\")
    sys.exit(1)
h.set_default_drive(target['id'])
print(f\"✓ Default download drive is now {target['name']} ({target['library']})\")
"
}

# Prints "<drive name>|<hub path>" for the drive the user picked, and
# prompts with the drive list when nothing was passed on the command line.
resolve_download_drive() {
    local sel="${1:-}"
    if [ -z "$sel" ]; then
        list_drives_table
        echo ""
        echo -n "Download to which drive? [Enter = default, name or number]: "
        read -r sel
    fi
    python3 -c "
import sys
sys.path.insert(0, '$PID_DIR')
sys.path.insert(0, '$SCRIPT_DIR')
import mlx_helper as h

drives = h.list_storage_drives(force=True)
usable = [d for d in drives if d['connected'] and not d['disabled']]
sel = '''$sel'''.strip()
target = None
if not sel:
    target = h.get_default_drive(drives)
elif sel.isdigit() and 1 <= int(sel) <= len(usable):
    target = usable[int(sel) - 1]
else:
    for d in usable:
        if sel.lower() in (d['name'].lower(), d['id'].lower(), d['mount'].lower()):
            target = d
            break
if not target:
    print(f\"✗ No such drive: '{sel}'\", file=sys.stderr)
    sys.exit(1)
hub, target = h.ensure_library_hub(target['id'])
print(f\"{target['name']}|{hub}\")
"
}

# ── 3. Download / Add Model ──────────────────────────────────────────
download_model() {
    local model_input="${1:-}"
    local drive_sel=""
    shift || true
    while [ $# -gt 0 ]; do
        case "$1" in
            --to) drive_sel="${2:-}"; shift 2 2>/dev/null || shift ;;
            *) drive_sel="$1"; shift ;;
        esac
    done
    if [ -z "$model_input" ]; then
        echo ""
        echo -n "Enter Hugging Face Repo ID or URL (e.g. mlx-community/Qwen2.5-Coder-7B-Instruct-4bit or https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-MLX/tree/main/4-bit) [b: Back, e: Exit]: "
        read -r model_input
        check_nav "$model_input" || return
    fi
    [ -z "$model_input" ] && return

    local parsed_json
    parsed_json=$(python3 -c "
import sys, json, os
sys.path.insert(0, '$PID_DIR')
sys.path.insert(0, '$(pwd)')
import mlx_helper
print(json.dumps(mlx_helper.parse_hf_identifier('$model_input')))
")

    local repo_id
    local subfolder
    local revision
    local display_name
    repo_id=$(echo "$parsed_json" | python3 -c "import sys, json; print(json.load(sys.stdin).get('repo_id', ''))")
    subfolder=$(echo "$parsed_json" | python3 -c "import sys, json; print(json.load(sys.stdin).get('subfolder', ''))")
    revision=$(echo "$parsed_json" | python3 -c "import sys, json; print(json.load(sys.stdin).get('revision', 'main'))")
    display_name=$(echo "$parsed_json" | python3 -c "import sys, json; print(json.load(sys.stdin).get('display_name', ''))")

    # If no subfolder was provided in URL, check if the repo has multiple quantization variants
    if [ -z "$subfolder" ]; then
        local variants_json
        variants_json=$(python3 -c "
import sys, json, os
sys.path.insert(0, '$PID_DIR')
sys.path.insert(0, '$(pwd)')
import mlx_helper
print(json.dumps(mlx_helper.get_repo_quant_variants('$repo_id')))
")
        local v_count
        v_count=$(echo "$variants_json" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)

        if [ "$v_count" -gt 0 ]; then
            echo ""
            echo "⚡ Repository '$repo_id' has multiple quantization subfolders:"
            echo "$variants_json" | python3 -c "
import sys, json
variants = json.load(sys.stdin)
for i, v in enumerate(variants, 1):
    rec = ' 🌟 [Recommended]' if 'Recommended' in v['label'] else ''
    print(f\"  {i}. {v['subfolder']:<10} (Size: {v['size_str']}){rec}\")
print(f\"  {len(variants)+1}. Download Full Repository (All Quantizations)\")
"
            echo -n "Select option [1-$((v_count+1)), b: Back]: "
            local opt
            read -r opt
            check_nav "$opt" || return
            if [ -n "$opt" ] && [ "$opt" -ge 1 ] 2>/dev/null && [ "$opt" -le "$v_count" ] 2>/dev/null; then
                subfolder=$(echo "$variants_json" | python3 -c "import sys, json; print(json.load(sys.stdin)[$opt-1]['subfolder'])")
                display_name="$repo_id ($subfolder)"
            fi
        fi
    fi

    echo ""
    echo "Downloading '$display_name' using High Speed Hugging Face CLI ('hf')..."
    echo "---------------------------------------------------------"

    # ── Pick the drive that will hold the weights ────────────────────
    local drive_resolved
    drive_resolved=$(resolve_download_drive "$drive_sel") || return
    local drive_name="${drive_resolved%%|*}"
    local hub_dir="${drive_resolved##*|}"
    echo "💾 Target drive: $drive_name"
    echo "📁 Model library: $hub_dir"
    echo ""

    # Free-space guard: ask the HF API for the weight size before downloading.
    local space_json
    space_json=$(python3 -c "
import sys, json, os
sys.path.insert(0, '$PID_DIR')
sys.path.insert(0, '$SCRIPT_DIR')
import mlx_helper as h
info = h.fetch_repo_size_info('$repo_id')
need = 0.0
try:
    if info.get('weights_gb'):
        need = round(info['weights_gb'] * 1.05, 1)
    elif info.get('est_ram_gb'):
        need = round(info['est_ram_gb'] * 0.85, 1)
except Exception:
    pass
free = 0.0
try:
    st = os.statvfs('$hub_dir')
    free = round(st.f_bavail * st.f_frsize / (1024**3), 1)
except Exception:
    pass
print(json.dumps({'need': need, 'free': free}))
" 2>/dev/null)

    local need_gb free_gb
    need_gb=$(echo "$space_json" | python3 -c "import sys, json; print(json.load(sys.stdin).get('need', 0))" 2>/dev/null || echo 0)
    free_gb=$(echo "$space_json" | python3 -c "import sys, json; print(json.load(sys.stdin).get('free', 0))" 2>/dev/null || echo 0)

    if awk "BEGIN{exit !($need_gb > 0 && $free_gb > 0 && $need_gb > $free_gb)}"; then
        echo "⚠️  '$drive_name' has $free_gb GB free but this model needs about $need_gb GB."
        echo -n "   Pick another drive? [Enter = abort, or drive name/number]: "
        local alt
        read -r alt
        if [ -z "$alt" ]; then
            echo "✗ Download cancelled — no space on $drive_name."
            return
        fi
        drive_resolved=$(resolve_download_drive "$alt") || return
        drive_name="${drive_resolved%%|*}"
        hub_dir="${drive_resolved##*|}"
        echo "💾 Target drive: $drive_name"
        echo "📁 Model library: $hub_dir"
        echo ""
    fi

    export HF_XET_HIGH_PERFORMANCE=1
    export HF_HUB_DOWNLOAD_TIMEOUT=30

    # Signed in? Then the transfer carries a token (higher limits, gated repos).
    local hf_token
    hf_token=$(python3 "$HELPER_PY" hf_token 2>/dev/null || true)

    # Big LFS blobs: parallel range download first, then let the hub link them up.
    local use_turbo="${MLXCTL_TURBO:-auto}"
    if [ "$use_turbo" = "auto" ]; then
        if [ -n "$(python3 "$HELPER_PY" turbo_plan "$repo_id" 2>/dev/null | python3 -c "import sys,json; print('yes' if json.load(sys.stdin).get('files') else '')" 2>/dev/null)" ]; then
            use_turbo="yes"
        else
            use_turbo="no"
        fi
    fi

    if [ "$use_turbo" = "yes" ]; then
        local tplan
        tplan=$(python3 "$HELPER_PY" turbo_plan "$repo_id" 2>/dev/null || echo '{}')
        local tcount tgb
        tcount=$(echo "$tplan" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('files',[])))" 2>/dev/null || echo 0)
        tgb=$(echo "$tplan" | python3 -c "import sys,json; print(round(json.load(sys.stdin).get('bytes',0)/1e9,1))" 2>/dev/null || echo 0)
        echo "⚡ Turbo: $tcount large file(s), ${tgb} GB — 8 parallel connections"
        HF_TOKEN="$hf_token" python3 "$HELPER_PY" turbo_download "$repo_id" "$hub_dir" "$repo_id" "$subfolder" "${revision:-main}"
        echo ""
        echo "✓ Download process completed for '$display_name'."
        echo ""
        read -r -p "Press Enter to continue..."
        return
    fi

    local dl_args=("$repo_id")
    if [ -n "$subfolder" ]; then
        dl_args+=(--include "$subfolder/*")
    fi
    if [ -n "$revision" ] && [ "$revision" != "main" ]; then
        dl_args+=(--revision "$revision")
    fi
    dl_args+=(--cache-dir "$hub_dir")

    if command -v hf &>/dev/null; then
        HF_TOKEN="$hf_token" hf download "${dl_args[@]}"
    elif command -v huggingface-cli &>/dev/null; then
        HF_TOKEN="$hf_token" huggingface-cli download "${dl_args[@]}"
    elif command -v uvx &>/dev/null; then
        # No global hf CLI: run it out of a throwaway uv environment.
        HF_TOKEN="$hf_token" uvx --from huggingface_hub --with hf_xet hf download "${dl_args[@]}"
    else
        HF_TOKEN="$hf_token" python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$repo_id', allow_patterns=['$subfolder/*'] if '$subfolder' else None, cache_dir='$hub_dir')"
    fi

    echo ""
    echo "✓ Download process completed for '$display_name'."
    echo ""
    read -r -p "Press Enter to continue..."
}

# ── 4. Start Model (Fast Swap & Concurrent Options) ──────────────────
start_model() {
    local model_arg="${1:-}"
    local model_name=""
    local model_path=""

    local local_json
    local_json=$(python3 "$HELPER_PY" find_models)

    local count
    count=$(echo "$local_json" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)

    if [ -n "$model_arg" ]; then
        model_name="$model_arg"
        model_path=$(echo "$local_json" | python3 -c "
import sys, json
models = json.load(sys.stdin)
target = '$model_name'
found = ''
for m in models:
    if m['name'] == target or target in m['path']:
        found = m['path']
        break
print(found)
")
        [ -z "$model_path" ] && model_path="$model_name"
    else
        if [ "$count" -eq 0 ]; then
            echo ""
            echo "No local models found."
            echo -n "Enter a Hugging Face Repo ID to download and start [b: Back, e: Exit]: "
            read -r model_arg
            check_nav "$model_arg" || return
            [ -z "$model_arg" ] && return
            download_model "$model_arg"
            model_name="$model_arg"
            model_path="$model_arg"
        else
            echo ""
            echo "=== Select Model to Start ==="
            echo "$local_json" | python3 -c "
import sys, json
models = json.load(sys.stdin)
for i, m in enumerate(models, 1):
    print(f'  {i:2d}. {m[\"name\"]} ({m[\"size\"]})')
"
            echo ""
            echo -n "Select model number [1-$count] or enter Repo ID [b: Back, e: Exit]: "
            read -r choice
            check_nav "$choice" || return
            [ -z "$choice" ] && return

            if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "$count" ]; then
                model_name=$(echo "$local_json" | python3 -c "import sys, json; print(json.load(sys.stdin)[$choice-1]['name'])")
                model_path=$(echo "$local_json" | python3 -c "import sys, json; print(json.load(sys.stdin)[$choice-1]['path'])")
            else
                model_name="$choice"
                model_path="$choice"
            fi
        fi
    fi

    local running_json
    running_json=$(python3 "$HELPER_PY" running_servers)
    local run_count
    run_count=$(echo "$running_json" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)

    local target_port="$DEFAULT_PORT"
    local host="$DEFAULT_HOST"
    local temp="0.0"
    local max_tokens="4096"
    local extra_args=""

    if [ "$run_count" -gt 0 ]; then
        echo ""
        echo "---------------------------------------------------------"
        echo "⚠️ Active Model(s) Currently Running:"
        echo "$running_json" | python3 -c "
import sys, json
servers = json.load(sys.stdin)
for i, s in enumerate(servers, 1):
    print(f'   - {s[\"model\"]} (PID {s[\"pid\"]} on port {s[\"port\"]})')
"
        echo ""
        echo "Target Model to Launch: '$model_name'"
        echo "---------------------------------------------------------"
        echo "  1. ⚡ Fast Swap (Stop active model & launch '$model_name' on port $DEFAULT_PORT) [Default]"
        echo "  2. 🔀 Run Concurrently (Keep active model & launch '$model_name' on a new port)"
        echo "  3. ⚙️ Custom Parameters (Select custom port, host, temperature, thinking mode)"
        echo ""
        echo -n "Select option [1-3, default 1] (or [b] Back, [e] Exit): "
        read -r swap_choice
        check_nav "$swap_choice" || return
        [ -z "$swap_choice" ] && swap_choice="1"

        if [ "$swap_choice" = "1" ]; then
            echo ""
            echo "⚡ Fast Swap: Stopping current model server(s) on port $DEFAULT_PORT..."
            local check_port_pid
            check_port_pid=$(lsof -t -i :"$DEFAULT_PORT" 2>/dev/null)
            if [ -n "$check_port_pid" ]; then
                kill -9 "$check_port_pid" 2>/dev/null
                sleep 1
            fi
            target_port="$DEFAULT_PORT"
        elif [ "$swap_choice" = "2" ]; then
            target_port=$(python3 "$HELPER_PY" free_port 10000)
            echo "🔀 Running Concurrently: Launching on port $target_port..."
        elif [ "$swap_choice" = "3" ]; then
            echo ""
            echo "--- Custom Launch Parameters ---"
            echo -n "Host [$DEFAULT_HOST] (b: Back, e: Exit): "
            read -r custom_host
            check_nav "$custom_host" || return
            [ -n "$custom_host" ] && host="$custom_host"

            echo -n "Port [$DEFAULT_PORT] (b: Back, e: Exit): "
            read -r custom_port
            check_nav "$custom_port" || return
            [ -n "$custom_port" ] && target_port="$custom_port"

            echo -n "Temperature [$temp] (b: Back, e: Exit): "
            read -r custom_temp
            check_nav "$custom_temp" || return
            [ -n "$custom_temp" ] && temp="$custom_temp"

            echo -n "Max Tokens [$max_tokens] (b: Back, e: Exit): "
            read -r custom_tokens
            check_nav "$custom_tokens" || return
            [ -n "$custom_tokens" ] && max_tokens="$custom_tokens"

            echo -n "Enable Thinking Tokens? (y/n, default y) [b: Back, e: Exit]: "
            read -r custom_thinking
            check_nav "$custom_thinking" || return
            if [ "$custom_thinking" = "n" ] || [ "$custom_thinking" = "N" ]; then
                extra_args="$extra_args --chat-template-args '{\"enable_thinking\":false}'"
            fi
        fi
    fi

    local check_port_pid
    check_port_pid=$(lsof -t -i :"$target_port" 2>/dev/null)
    if [ -n "$check_port_pid" ]; then
        kill -9 "$check_port_pid" 2>/dev/null
        sleep 1
    fi

    local safe_name
    safe_name=$(echo "$model_name" | tr '/' '_' | tr -cd '[:alnum:]_.-')
    local log_file="$LOG_DIR/${safe_name}_${target_port}.log"

    local is_vis
    is_vis=$(python3 -c "
import sys, os
sys.path.insert(0, '$HOME/mac-mlx-control-center')
try:
    import mlx_helper
    print('yes' if mlx_helper.is_vision_model('$model_path') else 'no')
except Exception:
    print('no')
" 2>/dev/null || echo "no")

    echo ""
    echo "Launching MLX Model Server..."
    echo "  Model:  $model_name"
    echo "  Path:   $model_path"
    echo "  Engine: $([ "$is_vis" = "yes" ] && echo "👁️ mlx-vlm (Vision & Multimodal)" || echo "⚡ mlx-lm (Text LLM)")"
    echo "  Host:   $host"
    echo "  Port:   $target_port"
    echo "  Log:    $log_file"
    echo "---------------------------------------------------------"

    # Requests arrive by repo id, not by the absolute path above; point the hub at the
    # library that already holds this model so nothing re-downloads into ~/.cache.
    local hub_for_model
    hub_for_model=$(python3 - "$SCRIPT_DIR" "$PID_DIR" "$model_name" "$model_path" <<'PYCODE' 2>/dev/null || true
import sys
for p in (sys.argv[2], sys.argv[1]):
    if p not in sys.path:
        sys.path.insert(0, p)
import mlx_helper as h
print(h.hub_for_model(sys.argv[3]) or h.hub_for_model(sys.argv[4]) or "")
PYCODE
)
    [ -n "$hub_for_model" ] && export HF_HUB_CACHE="$hub_for_model"

    # The guard is imported by the server process itself (PYTHONPATH + the
    # sitecustomize hook), so the command line, port, pid and log stay identical.
    local guard_dir="$SCRIPT_DIR/mlx_stream_guard"
    if [ -f "$guard_dir/sitecustomize.py" ] && [ "${MLX_STREAM_GUARD:-1}" != "0" ]; then
        export PYTHONPATH="$guard_dir${PYTHONPATH:+:$PYTHONPATH}"
        export MLX_STREAM_GUARD=1
        export MLX_GUARD_KEEPALIVE_SECS="${MLX_GUARD_KEEPALIVE_SECS:-5}"
        export MLX_GUARD_LOG="${MLX_GUARD_LOG:-$LOG_DIR/stream_guard.log}"
        export MLX_GUARD_STATE="${MLX_GUARD_STATE:-$PID_DIR/stream_guard.json}"
    fi

    if [ "$is_vis" = "yes" ]; then
        nohup uvx --from mlx-vlm --with jinja2 python -m mlx_vlm.server \
            --model "$model_path" \
            --host "$host" \
            --port "$target_port" \
            --max-tokens "$max_tokens" \
            --trust-remote-code $extra_args > "$log_file" 2>&1 &
    else
        nohup uvx --from mlx-lm mlx_lm.server \
            --model "$model_path" \
            --host "$host" \
            --port "$target_port" \
            --temp "$temp" \
            --max-tokens "$max_tokens" $extra_args > "$log_file" 2>&1 &
    fi

    local server_pid=$!
    echo "$server_pid" > "$PID_DIR/${safe_name}_${target_port}.pid"

    python3 "$HELPER_PY" sync_configs "$model_name" 2>/dev/null || true

    echo -n "Waiting for server startup..."
    local attempts=0
    local ready=0
    while [ $attempts -lt 12 ]; do
        sleep 1
        echo -n "."
        if curl -s -m 1 "http://$host:$target_port/v1/models" | grep -q "data" 2>/dev/null; then
            ready=1
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            break
        fi
        attempts=$((attempts + 1))
    done
    echo ""

    if [ "$ready" -eq 1 ]; then
        echo "✓ Model server successfully started and READY!"
        echo "  Endpoint: http://$host:$target_port/v1"
        echo "  Curl test:"
        echo "  curl http://$host:$target_port/v1/chat/completions \\"
        echo "    -H \"Content-Type: application/json\" \\"
        echo "    -d '{\"model\": \"$model_name\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}], \"max_tokens\": 50}'"
    else
        if kill -0 "$server_pid" 2>/dev/null; then
            echo "⏳ Process PID $server_pid is running and loading model weights into memory."
            echo "   Check server log with option 8: tail -f $log_file"
        else
            echo "❌ Server failed to start. View log file: $log_file"
        fi
    fi
    echo ""
    read -r -p "Press Enter to continue..."
}

# ── 5. KILL SWITCH / Stop Model ──────────────────────────────────────
stop_model() {
    local direct_target="${1:-}"
    local running_json
    running_json=$(python3 "$HELPER_PY" running_servers)

    local count
    count=$(echo "$running_json" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)

    if [ "$count" -eq 0 ]; then
        echo ""
        echo "No running MLX model servers detected."
        echo ""
        read -r -p "Press Enter to return..."
        return
    fi

    # Handle direct invocation e.g. "k 2" or "stop 2" or "stop 38961"
    if [ -n "$direct_target" ]; then
        if [[ "$direct_target" =~ ^[0-9]+$ ]] && [ "$direct_target" -ge 1 ] && [ "$direct_target" -le "$count" ]; then
            local pid
            pid=$(echo "$running_json" | python3 -c "import sys, json; print(json.load(sys.stdin)[$direct_target-1]['pid'])")
            local mname
            mname=$(echo "$running_json" | python3 -c "import sys, json; print(json.load(sys.stdin)[$direct_target-1]['model'])")
            local port
            port=$(echo "$running_json" | python3 -c "import sys, json; print(json.load(sys.stdin)[$direct_target-1]['port'])")
            echo ""
            echo "⚡ KILL SWITCH: Force killing #$direct_target: '$mname' (PID $pid on port $port)..."
            kill -9 "$pid" 2>/dev/null
            rm -f "$PID_DIR"/*_"${port}".pid 2>/dev/null
            echo "✓ Process PID $pid force-killed."
            echo ""
            sleep 1
            return
        elif [[ "$direct_target" =~ ^[0-9]+$ ]]; then
            echo ""
            echo "⚡ KILL SWITCH: Force killing PID $direct_target..."
            kill -9 "$direct_target" 2>/dev/null
            rm -f "$PID_DIR"/*.pid 2>/dev/null
            echo "✓ PID $direct_target killed."
            echo ""
            sleep 1
            return
        fi
    fi

    echo ""
    echo "⚡ KILL SWITCH: Select Process to Terminate Immediately"
    echo "---------------------------------------------------------"
    echo "$running_json" | python3 -c "
import sys, json
servers = json.load(sys.stdin)
for i, s in enumerate(servers, 1):
    print(f'  {i:2d}. [{s[\"status\"]}] {s[\"model\"]} (PID: {s[\"pid\"]}, Host: {s[\"host\"]}, Port: {s[\"port\"]})')
"
    echo "---------------------------------------------------------"
    echo ""
    echo -n "Enter number [1-$count] to KILL (-9), [u] Kill Unresponsive, [a] Kill ALL, [b] Back, [e] Exit: "
    read -r choice
    check_nav "$choice" || return

    if [ "$choice" = "u" ] || [ "$choice" = "U" ]; then
        echo ""
        echo "⚡ KILL SWITCH: Terminating all unresponsive/stale processes..."
        echo "$running_json" | python3 -c "
import sys, json, os, signal
servers = json.load(sys.stdin)
cleaned = 0
for s in servers:
    if 'UNRESPONSIVE' in s['status']:
        try:
            os.kill(int(s['pid']), signal.SIGKILL)
            print(f'  ⚡ Killed unresponsive PID {s[\"pid\"]} ({s[\"model\"]} on port {s[\"port\"]})')
            cleaned += 1
        except Exception as e:
            print(f'  Could not kill PID {s[\"pid\"]}: {e}')
if cleaned == 0:
    print('  No unresponsive processes found.')
"
        rm -f "$PID_DIR"/*.pid 2>/dev/null
    elif [ "$choice" = "a" ] || [ "$choice" = "A" ] || [ "$choice" = "all" ]; then
        echo ""
        echo "⚡ KILL SWITCH: Terminating ALL running MLX model servers..."
        echo "$running_json" | python3 -c "
import sys, json, os, signal
servers = json.load(sys.stdin)
for s in servers:
    try:
        os.kill(int(s['pid']), signal.SIGKILL)
        print(f'  ⚡ Force-killed PID {s[\"pid\"]} ({s[\"model\"]})')
    except Exception as e:
        print(f'  Could not kill PID {s[\"pid\"]}: {e}')
"
        rm -f "$PID_DIR"/*.pid 2>/dev/null
        echo "✓ All servers stopped."
    elif [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "$count" ]; then
        local pid
        pid=$(echo "$running_json" | python3 -c "import sys, json; print(json.load(sys.stdin)[$choice-1]['pid'])")
        local mname
        mname=$(echo "$running_json" | python3 -c "import sys, json; print(json.load(sys.stdin)[$choice-1]['model'])")
        local port
        port=$(echo "$running_json" | python3 -c "import sys, json; print(json.load(sys.stdin)[$choice-1]['port'])")
        echo ""
        echo "⚡ KILL SWITCH: Force killing model '$mname' (PID $pid on port $port)..."
        kill -9 "$pid" 2>/dev/null
        rm -f "$PID_DIR"/*_"${port}".pid 2>/dev/null
        echo "✓ Process PID $pid force-killed."
    else
        echo "Invalid selection."
    fi
    echo ""
    read -r -p "Press Enter to continue..."
}

# ── 6. Delete Model ──────────────────────────────────────────────────
delete_model() {
    local local_json
    local_json=$(python3 "$HELPER_PY" find_models)

    local count
    count=$(echo "$local_json" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)

    if [ "$count" -eq 0 ]; then
        echo ""
        echo "No downloaded models found."
        echo ""
        read -r -p "Press Enter to return..."
        return
    fi

    echo ""
    echo "=== Delete a Downloaded Model ==="
    echo "$local_json" | python3 -c "
import sys, json
models = json.load(sys.stdin)
for i, m in enumerate(models, 1):
    print(f'  {i:2d}. {m[\"name\"]} ({m[\"size\"]})')
    print(f'      Path: {m[\"delete_target\"]}')
"
    echo ""
    echo -n "Select model number to delete [1-$count] (or [b] Back, [e] Exit): "
    read -r choice
    check_nav "$choice" || return

    if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "$count" ]; then
        local target_name
        target_name=$(echo "$local_json" | python3 -c "import sys, json; print(json.load(sys.stdin)[$choice-1]['name'])")
        local target_dir
        target_dir=$(echo "$local_json" | python3 -c "import sys, json; print(json.load(sys.stdin)[$choice-1]['delete_target'])")
        local target_size
        target_size=$(echo "$local_json" | python3 -c "import sys, json; print(json.load(sys.stdin)[$choice-1]['size'])")

        echo ""
        echo "⚠️ WARNING: You are about to delete '$target_name' ($target_size)."
        echo "   Target directory: $target_dir"
        echo -n "Are you sure you want to permanently delete this model? (y/N) [b: Back, e: Exit]: "
        read -r confirm
        check_nav "$confirm" || return

        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            echo "Deleting $target_dir..."
            rm -rf "$target_dir"
            echo "✓ Model '$target_name' deleted."
        else
            echo "Deletion cancelled."
        fi
    else
        echo "Invalid selection."
    fi
    echo ""
    read -r -p "Press Enter to continue..."
}

# ── 7. View Running Models & Health ──────────────────────────────────
view_health() {
    show_running_header
    echo "API Endpoints status:"
    curl -s http://127.0.0.1:9999/v1/models 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print('  http://127.0.0.1:9999/v1 -> Active Models:')
    for m in data.get('data', []):
        print(f'    - {m.get(\"id\")}')
except Exception:
    print('  http://127.0.0.1:9999/v1 -> (No active HTTP response)')
"
    echo ""
    read -r -p "Press Enter to return..."
}

# ── 8. View Server Logs ──────────────────────────────────────────────
view_logs() {
    echo ""
    echo "=== Available Server Logs ==="
    local log_files=("$LOG_DIR"/*.log)
    if [ ! -e "${log_files[0]}" ]; then
        echo "  (no log files found in $LOG_DIR)"
        echo ""
        read -r -p "Press Enter to return..."
        return
    fi

    local i=1
    for lf in "${log_files[@]}"; do
        local fname
        fname=$(basename "$lf")
        local fsize
        fsize=$(du -sh "$lf" 2>/dev/null | cut -f1)
        echo "  $i. $fname ($fsize)"
        i=$((i + 1))
    done

    echo ""
    echo -n "Select log number to view [1-$((i-1))] (or [b] Back, [e] Exit): "
    read -r choice
    check_nav "$choice" || return

    if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -lt "$i" ]; then
        local selected_log="${log_files[$((choice-1))]}"
        echo ""
        echo "Tailing log file (Press Ctrl+C to stop)..."
        echo "---------------------------------------------------------"
        tail -f "$selected_log"
    fi
}

# ── 9. Quick Curl Test ───────────────────────────────────────────────
test_curl() {
    local running_json
    running_json=$(python3 "$HELPER_PY" running_servers)

    local count
    count=$(echo "$running_json" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)

    if [ "$count" -eq 0 ]; then
        echo ""
        echo "No running MLX model servers to test."
        echo ""
        read -r -p "Press Enter to return..."
        return
    fi

    echo ""
    echo "=== Quick Curl Test ==="
    echo "$running_json" | python3 -c "
import sys, json
servers = json.load(sys.stdin)
for i, s in enumerate(servers, 1):
    print(f'  {i:2d}. [{s[\"status\"]}] {s[\"model\"]} (Host: {s[\"host\"]}, Port: {s[\"port\"]})')
"
    echo ""
    echo -n "Select server number [1-$count] (or [b] Back, [e] Exit): "
    read -r choice
    check_nav "$choice" || return

    if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "$count" ]; then
        local host
        host=$(echo "$running_json" | python3 -c "import sys, json; print(json.load(sys.stdin)[$choice-1]['host'])")
        local port
        port=$(echo "$running_json" | python3 -c "import sys, json; print(json.load(sys.stdin)[$choice-1]['port'])")
        local model
        model=$(echo "$running_json" | python3 -c "import sys, json; print(json.load(sys.stdin)[$choice-1]['model'])")

        echo ""
        echo -n "Enter test prompt [default: Hello!]: "
        read -r prompt_text
        check_nav "$prompt_text" || return
        [ -z "$prompt_text" ] && prompt_text="Hello!"

        echo ""
        echo "Sending curl request to http://$host:$port/v1/chat/completions ..."
        echo "---------------------------------------------------------"
        curl -s "http://$host:$port/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"$model\",
                \"messages\": [{\"role\": \"user\", \"content\": \"$prompt_text\"}],
                \"max_tokens\": 100
            }" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    content = data['choices'][0]['message']['content']
    print('\nModel Response:\n' + content + '\n')
except Exception as e:
    print('Error parsing response:', e)
"
    fi
    echo ""
    read -r -p "Press Enter to return..."
}

# ── Main Menu ────────────────────────────────────────────────────────
main_menu() {
    while true; do
        clear 2>/dev/null || true
        echo "╔════════════════════════════════════════════════════════════╗"
        echo "║     MLX Model Control Center v0.10.1 (MLX Controller)       ║"
        echo "╚════════════════════════════════════════════════════════════╝"
        show_running_header
        echo "  1. List Downloaded Models (Select # to Launch)"
        echo "  2. Search Hugging Face for MLX Models"
        echo "  3. Download / Add Model by HF Repo ID"
        echo "  4. Start a Model (Fast Swap or Custom Parameters)"
        echo "  5. ⚡ KILL SWITCH: Stop / Terminate a Running Process"
        echo "  6. Delete a Downloaded Model"
        echo "  7. View Running Models & Test API Health"
        echo "  8. View Server Logs"
        echo "  9. Quick Curl Test Prompt"
        echo ""
        echo "  [k <#>] Instant Kill process #    [b] Refresh / Back    [e] Exit"
        echo "--------------------------------------------------------------"
        echo -n "Select option [1-9, k, b, e]: "
        read -r choice

        case "$(echo "$choice" | tr '[:upper:]' '[:lower:]')" in
            1) list_models ;;
            2) search_hf ;;
            3) download_model ;;
            4) start_model ;;
            5|k|kill) stop_model ;;
            k*)
                local target_num
                target_num=$(echo "$choice" | sed -E 's/^[kK][[:space:]]*//')
                stop_model "$target_num"
                ;;
            6) delete_model ;;
            7) view_health ;;
            8) view_logs ;;
            9) test_curl ;;
            b) continue ;;
            e)
                echo ""
                echo "Exiting MLX Model Manager. Goodbye!"
                exit 0
                ;;
            *)
                echo "Invalid option."
                sleep 1
                ;;
        esac
    done
}

# CLI direct invocation support
case "${1:-}" in
    start) start_model "${2:-}" ;;
    stop|kill) stop_model "${2:-}" ;;
    list) list_models false ;;
    search) search_hf "${2:-}" "${3:-}" "${4:-}" "${5:-}" ;;
    download) shift; download_model "$@" ;;
    sync-harness|harness) cmd_sync_harness ;;
    drives|storage) cmd_drives ;;
    protect-models|tm-off) cmd_protect_models ;;
    login|hf-login) cmd_login ;;
    logout|hf-logout) cmd_logout ;;
    whoami|hf-whoami) cmd_whoami ;;
    use-drive|drive) cmd_use_drive "${2:-}" ;;
    delete) delete_model ;;
    status|running) view_health ;;
    logs) view_logs ;;
    test) test_curl ;;
    *) main_menu ;;
esac
