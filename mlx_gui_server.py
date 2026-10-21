#!/usr/bin/env python3
#
# mlx_gui_server.py - REST API & Web Dashboard Server for MLX Models
#

import os
import sys
import json
import urllib.request
import urllib.parse
import subprocess
import re
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

PID_DIR = os.path.expanduser("~/.mlx_pids")
LOG_DIR = os.path.expanduser("~/.mlx_logs")
MODEL_DIR = os.path.expanduser("~/mlx_models")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(1, PID_DIR)
import mlx_helper

FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0a0e1a"/>
      <stop offset="100%" stop-color="#1e1b4b"/>
    </linearGradient>
    <linearGradient id="neonGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#00f2fe"/>
      <stop offset="50%" stop-color="#4facfe"/>
      <stop offset="100%" stop-color="#a855f7"/>
    </linearGradient>
    <linearGradient id="sparkGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8"/>
      <stop offset="100%" stop-color="#ec4899"/>
    </linearGradient>
  </defs>
  <rect x="2" y="2" width="60" height="60" rx="16" fill="url(#bgGrad)" stroke="#6366f1" stroke-width="2" stroke-opacity="0.4"/>
  <line x1="20" y1="2" x2="20" y2="0" stroke="#00f2fe" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="32" y1="2" x2="32" y2="0" stroke="#4facfe" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="44" y1="2" x2="44" y2="0" stroke="#a855f7" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="20" y1="64" x2="20" y2="62" stroke="#00f2fe" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="32" y1="64" x2="32" y2="62" stroke="#4facfe" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="44" y1="64" x2="44" y2="62" stroke="#a855f7" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="2" y1="20" x2="0" y2="20" stroke="#00f2fe" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="2" y1="32" x2="0" y2="32" stroke="#4facfe" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="2" y1="44" x2="0" y2="44" stroke="#a855f7" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="64" y1="20" x2="62" y2="20" stroke="#00f2fe" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="64" y1="32" x2="62" y2="32" stroke="#4facfe" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="64" y1="44" x2="62" y2="44" stroke="#a855f7" stroke-width="2.5" stroke-linecap="round"/>
  <path d="M 32 14 C 20 14 14 20 14 28 C 14 34 18 38 20 42 C 22 46 25 50 31 51" fill="none" stroke="url(#neonGrad)" stroke-width="3" stroke-linecap="round"/>
  <path d="M 32 14 C 44 14 50 20 50 28 C 50 34 46 38 44 42 C 42 46 39 50 33 51" fill="none" stroke="url(#neonGrad)" stroke-width="3" stroke-linecap="round"/>
  <path d="M 22 25 L 32 30 L 42 25" fill="none" stroke="#00f2fe" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M 20 37 L 32 32 L 44 37" fill="none" stroke="#a855f7" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M 32 14 L 32 50" fill="none" stroke="url(#sparkGrad)" stroke-width="2.5" stroke-linecap="round"/>
  <circle cx="22" cy="25" r="3" fill="#00f2fe"/>
  <circle cx="42" cy="25" r="3" fill="#a855f7"/>
  <circle cx="20" cy="37" r="3" fill="#00f2fe"/>
  <circle cx="44" cy="37" r="3" fill="#a855f7"/>
  <circle cx="32" cy="32" r="4.5" fill="#ffffff"/>
  <circle cx="32" cy="32" r="3" fill="#6366f1"/>
</svg>"""

PORT = 9998

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class MLXGuiHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_json(self, data, status_code=200):
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        try:
            self.wfile.write(json.dumps(data).encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            pass  # the tab navigated away; nothing to do

    def send_html(self, html_content, status_code=200):
        self.send_response(status_code)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html_content.encode("utf-8"))

    def send_svg(self, svg_content, status_code=200):
        self.send_response(status_code)
        self.send_header("Content-type", "image/svg+xml")
        self.send_header("Cache-Control", "public, max-age=86400")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(svg_content.encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.send_html(INDEX_HTML)
        elif path == "/favicon.ico" or path == "/favicon.svg":
            self.send_svg(FAVICON_SVG)
        elif path == "/api/harness":
            listed = mlx_helper.read_dsh_models()
            installed = []
            for m in mlx_helper.find_local_models():
                rid = m.get("repo_id") or m.get("name")
                if rid and rid not in installed:
                    installed.append(rid)  # same repo can sit on two drives
            listed_ids = [e["id"] for e in listed.get("entries", [])]
            self.send_json({
                "patch": listed.get("patch"),
                "provider": listed.get("provider"),
                "listed": listed_ids,
                "installed": installed,
                "missing": [i for i in installed if i not in listed_ids],
                "stale": [i for i in listed_ids if i not in installed],
                "in_sync": set(listed_ids) == set(installed),
            })
        elif path == "/api/hf/auth":
            self.send_json(mlx_helper.hf_auth_state())
        elif path == "/api/status":
            servers = mlx_helper.get_running_servers()
            stats = mlx_helper.get_system_stats()
            downloads = mlx_helper.get_active_downloads()
            config = mlx_helper.load_global_settings()
            self.send_json({"servers": servers, "stats": stats, "downloads": downloads, "config": config})
        elif path == "/api/config":
            config = mlx_helper.load_global_settings()
            self.send_json(config)
        elif path == "/api/models":
            models = mlx_helper.find_local_models()
            for m in models:
                m["context_window"] = mlx_helper.model_context_window(m.get("path") or m.get("name") or "")
            self.send_json({"models": models, "context_cap": mlx_helper.context_window_cap()})
        elif path == "/api/memory_processes":
            stats = mlx_helper.get_system_stats()
            procs = mlx_helper.get_top_memory_processes(limit=40)
            self.send_json({"stats": stats, "processes": procs})
        elif path == "/api/repo_variants":
            raw_repo = query.get("repo_id", [""])[0]
            parsed = mlx_helper.parse_hf_identifier(raw_repo)
            variants = mlx_helper.get_repo_quant_variants(parsed["repo_id"])
            self.send_json({"repo_id": parsed["repo_id"], "variants": variants})
        elif path == "/api/search":
            q = query.get("q", ["mlx"])[0]
            limit = int(query.get("limit", ["50"])[0])
            sort = query.get("sort", ["downloads"])[0]
            filter_tag = query.get("filter", ["mlx"])[0]
            tags_raw = query.get("tags", [filter_tag])[0]
            filter_tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
            quant = query.get("quant", [""])[0].strip()
            max_ram = float(query.get("max_ram", ["0"])[0] or 0) or None
            usable = mlx_helper.get_usable_ram_gb()
            results = mlx_helper.search_hf_api(q, limit=limit, sort=sort, filter_tags=filter_tags,
                                               quant=quant or None,
                                               max_ram=max_ram, usable_ram_gb=usable, enrich=False)
            stats = mlx_helper.get_system_stats()
            self.send_json({"results": results, "free_ram_gb": stats.get("free_ram_gb", 16.0),
                            "usable_ram_gb": usable, "total_ram_gb": stats.get("total_ram_gb", 0),
                            "tags": filter_tags, "quant": quant})
        elif path == "/api/repo_sizes":
            ids = query.get("ids", [""])[0]
            repo_ids = [i.strip() for i in ids.split(",") if i.strip()][:80]
            sizes = mlx_helper.get_repo_sizes_bulk(repo_ids) if repo_ids else {}
            self.send_json({"sizes": sizes})
        elif path == "/api/drives":
            force = query.get("force", ["0"])[0].lower() in ("1", "true", "yes")
            if force:
                mlx_helper.protect_model_storage()
            drives = mlx_helper.list_storage_drives(force=force)
            default = mlx_helper.get_default_drive(drives)
            self.send_json({
                "drives": drives,
                "default": (default or {}).get("id", ""),
                "default_name": (default or {}).get("name", ""),
            })
        elif path == "/api/logs":
            log_files = []
            if os.path.exists(LOG_DIR):
                for f in sorted(os.listdir(LOG_DIR), reverse=True):
                    if f.endswith(".log"):
                        fp = os.path.join(LOG_DIR, f)
                        size_bytes = os.path.getsize(fp)
                        log_files.append({"name": f, "size": f"{size_bytes/1024:.1f} KB"})
            
            selected_log = query.get("name", [""])[0]
            hide_polling = query.get("hide_polling", ["true"])[0].lower() in ["true", "1", "yes"]
            content = ""
            errors = []
            if selected_log:
                safe_name = os.path.basename(selected_log)
                target_path = os.path.join(LOG_DIR, safe_name)
                if os.path.exists(target_path):
                    try:
                        with open(target_path, "r", errors="ignore") as lf:
                            lines = lf.readlines()
                            if hide_polling:
                                filtered_lines = [l for l in lines if "GET /v1/models" not in l]
                            else:
                                filtered_lines = lines
                            content = "".join(filtered_lines[-250:])
                            
                            full_text = "".join(lines)
                            err_matches = re.findall(r"(Traceback[\s\S]*?(?:ValueError|ModuleNotFoundError|Exception|Error|OSError):[^\n]+)", full_text)
                            if err_matches:
                                errors = err_matches[-5:]
                            else:
                                for line in lines:
                                    if any(k in line for k in ["Error:", "Exception:", "ValueError:", "ModuleNotFoundError:"]):
                                        errors.append(line.strip())
                    except Exception as e:
                        content = f"Error reading log: {e}"

            self.send_json({"logs": log_files, "content": content, "errors": errors})
        else:
            self.send_json({"error": "Not Found"}, 404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if path == "/api/config":
            st = mlx_helper.load_global_settings()
            if "default_port" in payload: st["default_port"] = int(payload["default_port"])
            if "default_host" in payload: st["default_host"] = str(payload["default_host"])
            if "default_temp" in payload: st["default_temp"] = float(payload["default_temp"])
            if "default_max_tokens" in payload: st["default_max_tokens"] = int(payload["default_max_tokens"])
            if "default_context_window" in payload: st["default_context_window"] = int(payload["default_context_window"])
            if "auto_sync_agents" in payload: st["auto_sync_agents"] = bool(payload["auto_sync_agents"])
            
            mlx_helper.save_global_settings(st)
            self.send_json({"status": "ok", "config": st})

        elif path == "/api/harness/sync":
            try:
                res = mlx_helper.sync_dsh_models(provider_id=(payload.get("provider") or "").strip()
                                                 or mlx_helper.DSH_PROVIDER_ID)
                self.send_json(res)
            except Exception as e:
                self.send_json({"ok": False, "error": str(e)}, 500)
        elif path == "/api/hf/device/start":
            try:
                self.send_json(mlx_helper.hf_device_start())
            except Exception as e:
                self.send_json({"ok": False, "error": str(e)}, 500)

        elif path == "/api/hf/device/poll":
            code = (payload.get("device_code") or "").strip()
            if not code:
                self.send_json({"status": "error", "error": "missing device_code"}, 400)
                return
            self.send_json(mlx_helper.hf_device_poll(code))

        elif path == "/api/hf/token":
            token = (payload.get("token") or "").strip()
            if not token:
                self.send_json({"ok": False, "error": "missing token"}, 400)
                return
            who = mlx_helper.hf_whoami(token)
            if not who.get("ok"):
                self.send_json({"ok": False, "error": who.get("error", "invalid token")}, 401)
                return
            mlx_helper.set_hf_token(token, "token", who.get("user", ""))
            self.send_json({"ok": True, "user": who.get("user", ""), "auth": mlx_helper.hf_auth_state()})

        elif path == "/api/hf/logout":
            mlx_helper.clear_hf_token()
            self.send_json({"ok": True, "auth": mlx_helper.hf_auth_state()})

        elif path == "/api/drives/default":
            target = (payload.get("drive") or "").strip()
            chosen = mlx_helper.set_default_drive(target)
            self.send_json({
                "status": "ok",
                "default": (chosen or {}).get("id", ""),
                "default_name": (chosen or {}).get("name", ""),
                "drives": mlx_helper.list_storage_drives(force=True),
            })

        elif path == "/api/drives/exclude":
            target = (payload.get("drive") or "").strip()
            drive = mlx_helper.get_drive(target) if target else mlx_helper.get_default_drive()
            if not drive:
                self.send_json({"error": "Unknown drive", "code": "unknown_drive"}, 404)
                return
            library = drive.get("library") or ""
            ok = mlx_helper.exclude_from_time_machine(os.path.dirname(library) or library)
            self.send_json({
                "status": "ok" if ok else "failed",
                "drive": drive.get("name", ""),
                "tm_excluded": mlx_helper.tm_exclusion_state(os.path.dirname(library) or library),
                "drives": mlx_helper.list_storage_drives(force=True),
            })

        elif path == "/api/start":
            st = mlx_helper.load_global_settings()
            default_p = st.get("default_port", 9999)
            default_h = st.get("default_host", "127.0.0.1")

            model_name = payload.get("model", "")
            mode = payload.get("mode", "swap")
            port = payload.get("port")
            host = payload.get("host", default_h)
            temp = str(payload.get("temp", st.get("default_temp", 0.0)))
            max_tokens = str(payload.get("max_tokens", st.get("default_max_tokens", 4096)))
            thinking = payload.get("thinking", True)
            draft_model = payload.get("draft", "")
            engine_pref = payload.get("engine", "auto")

            if not model_name:
                self.send_json({"error": "No model name provided"}, 400)
                return

            if mode == "concurrent":
                port = mlx_helper.find_free_port(10000)
            elif not port:
                # an explicit port from the Launch dialog wins; otherwise use the default
                port = default_p

            try:
                out = subprocess.check_output(["lsof", "-t", "-i", f":{port}"], text=True)
                for pid_str in out.splitlines():
                    pid_str = pid_str.strip()
                    if pid_str:
                        os.kill(int(pid_str), 9)
            except Exception:
                pass

            local_models = mlx_helper.find_local_models()
            model_path = model_name
            is_vis = False
            for m in local_models:
                if m["name"] == model_name or m.get("base_name") == model_name or model_name in m["path"]:
                    model_path = m["path"]
                    is_vis = m.get("is_vision", False)
                    break

            if engine_pref == "mlx-vlm":
                use_engine = "mlx-vlm"
                is_vis = True
            elif engine_pref == "mlx-lm":
                use_engine = "mlx-lm"
                is_vis = False
            else: # "auto"
                if not is_vis:
                    is_vis = mlx_helper.is_vision_model(model_path)
                use_engine = "mlx-vlm" if is_vis else "mlx-lm"

            # Clients address models by repo id. `mlx-vlm` reloads whenever the requested id
            # differs from the loaded one, so launching by absolute path means every first
            # repo-id request pays a full reload (~26 s for a 70 GB model).
            launch_hub = mlx_helper.hub_for_model(model_name) or mlx_helper.hub_for_model(model_path)
            launch_target = (model_name if (launch_hub and "/" in model_name
                                            and not os.path.isabs(model_name)) else model_path)

            extra_args = []
            ctx_cap = mlx_helper.context_window_cap()
            native_ctx = mlx_helper.model_context_window(model_name) or mlx_helper.model_context_window(model_path)
            if ctx_cap and native_ctx:
                ctx_cap = min(ctx_cap, native_ctx)
            elif ctx_cap and not native_ctx:
                ctx_cap = 0  # unknown native size: do not guess a hard limit
            if use_engine == "mlx-vlm":
                if ctx_cap:
                    extra_args.extend(["--max-kv-size", str(ctx_cap)])
                if thinking:
                    extra_args.append("--enable-thinking")
                if draft_model:
                    extra_args.extend(["--draft-model", draft_model])
                cmd = [
                    "uvx", "--from", "mlx-vlm", "--with", "jinja2", "python", "-m", "mlx_vlm.server",
                    "--model", launch_target,
                    "--host", str(host),
                    "--port", str(port),
                    "--max-tokens", str(max_tokens),
                    "--trust-remote-code"
                ] + extra_args
            else:
                if not thinking:
                    extra_args.extend(["--chat-template-args", '{"enable_thinking":false}'])
                if draft_model:
                    extra_args.extend(["--draft-model", draft_model, "--num-draft-tokens", "3"])
                cmd = [
                    "uvx", "--from", "mlx-lm", "mlx_lm.server",
                    "--model", launch_target,
                    "--host", str(host),
                    "--port", str(port),
                    "--temp", str(temp),
                    "--max-tokens", str(max_tokens)
                ] + extra_args

            safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", model_name)
            log_file = os.path.join(LOG_DIR, f"{safe_name}_{port}.log")

            env = mlx_helper.get_mlx_env()
            # Requests reach the server by repo id ("sh0wie/...-4bit"), not by the absolute
            # snapshot path we launch with. Without this the hub re-downloads the repo into
            # ~/.cache instead of reusing the copy on the selected drive.
            if launch_hub:
                env["HF_HUB_CACHE"] = launch_hub

            with open(log_file, "a") as lf:
                proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, env=env)

            pid_file = os.path.join(PID_DIR, f"{safe_name}_{port}.pid")
            with open(pid_file, "w") as pf:
                pf.write(str(proc.pid))

            if st.get("auto_sync_agents", True):
                mlx_helper.sync_pi_and_opencode(model_name, port=port, is_vision=is_vis)

            self.send_json({
                "status": "ok",
                "message": f"Started {model_name} on port {port} ({use_engine})",
                "pid": proc.pid,
                "port": port,
                "host": host,
                "engine": use_engine,
                "is_vision": is_vis
            })

        elif path == "/api/stop":
            pid = payload.get("pid")
            port = payload.get("port")
            stop_all = payload.get("all", False)
            stop_unresponsive = payload.get("unresponsive", False)

            servers = mlx_helper.get_running_servers()
            stopped = []

            if stop_all:
                for s in servers:
                    try:
                        os.kill(int(s["pid"]), 9)
                        stopped.append(s["pid"])
                    except Exception:
                        pass
                try:
                    out = subprocess.check_output(["lsof", "-t", "-i", ":9999"], text=True)
                    for pid_str in out.splitlines():
                        pid_str = pid_str.strip()
                        if pid_str:
                            os.kill(int(pid_str), 9)
                except Exception:
                    pass
            elif stop_unresponsive:
                for s in servers:
                    if "UNRESPONSIVE" in s["status"]:
                        try:
                            os.kill(int(s["pid"]), 9)
                            stopped.append(s["pid"])
                        except Exception:
                            pass
            elif pid:
                try:
                    os.kill(int(pid), 9)
                    stopped.append(pid)
                except Exception:
                    pass
            elif port:
                try:
                    out = subprocess.check_output(["lsof", "-t", "-i", f":{port}"], text=True)
                    for pid_str in out.splitlines():
                        pid_str = pid_str.strip()
                        if pid_str:
                            os.kill(int(pid_str), 9)
                            stopped.append(pid_str)
                except Exception:
                    pass

            self.send_json({"status": "ok", "stopped": stopped})

        elif path == "/api/clear_logs":
            target_log = payload.get("name", "")
            clear_all = payload.get("all", False)
            if clear_all:
                if os.path.exists(LOG_DIR):
                    for f in os.listdir(LOG_DIR):
                        if f.endswith(".log"):
                            fp = os.path.join(LOG_DIR, f)
                            with open(fp, "w") as lf:
                                lf.write("")
                self.send_json({"status": "ok", "cleared": "all"})
            elif target_log:
                safe_name = os.path.basename(target_log)
                target_path = os.path.join(LOG_DIR, safe_name)
                if os.path.exists(target_path):
                    with open(target_path, "w") as lf:
                        lf.write("")
                self.send_json({"status": "ok", "cleared": safe_name})
            else:
                self.send_json({"error": "No log target specified"}, 400)

        elif path == "/api/pause_download":
            pid = payload.get("pid")
            raw_repo = payload.get("repo_id", "")
            explicit_subfolder = payload.get("subfolder", "")
            parsed = mlx_helper.parse_hf_identifier(raw_repo, explicit_subfolder)
            repo_id = parsed["repo_id"]
            subfolder = parsed["subfolder"]
            download_id = parsed["download_id"]

            if pid:
                try:
                    os.kill(int(pid), 9)
                except Exception:
                    pass
            if repo_id:
                cache_dir = ""
                if "/" in repo_id:
                    org, repo = repo_id.split("/", 1)
                    cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{org}--{repo}")
                paused = mlx_helper.load_paused_downloads()
                paused[download_id] = {
                    "repo_id": repo_id,
                    "subfolder": subfolder,
                    "cache_dir": cache_dir,
                    "paused_at": os.path.getmtime(cache_dir) if os.path.exists(cache_dir) else 0
                }
                mlx_helper.save_paused_downloads(paused)
                self.send_json({"status": "paused", "repo_id": repo_id, "subfolder": subfolder, "download_id": download_id})
            else:
                self.send_json({"error": "No repo_id provided"}, 400)

        elif path == "/api/cancel_download":
            pid = payload.get("pid")
            raw_repo = payload.get("repo_id", "")
            explicit_subfolder = payload.get("subfolder", "")
            parsed = mlx_helper.parse_hf_identifier(raw_repo, explicit_subfolder)
            repo_id = parsed["repo_id"]
            download_id = parsed["download_id"]

            if pid:
                try:
                    os.kill(int(pid), 9)
                except Exception:
                    pass
            if repo_id:
                paused = mlx_helper.load_paused_downloads()
                if download_id in paused:
                    del paused[download_id]
                    mlx_helper.save_paused_downloads(paused)
                elif repo_id in paused:
                    del paused[repo_id]
                    mlx_helper.save_paused_downloads(paused)
            self.send_json({"status": "ok", "cancelled": download_id or pid})

        elif path == "/api/restart_download" or path == "/api/resume_download":
            pid = payload.get("pid")
            raw_repo = payload.get("repo_id", "")
            explicit_subfolder = payload.get("subfolder", "")
            if pid:
                try:
                    os.kill(int(pid), 9)
                except Exception:
                    pass
            if raw_repo:
                parsed = mlx_helper.parse_hf_identifier(raw_repo, explicit_subfolder)
                repo_id = parsed["repo_id"]
                subfolder = parsed["subfolder"]
                revision = parsed["revision"]
                download_id = parsed["download_id"]
                display_name = parsed["display_name"]

                paused = mlx_helper.load_paused_downloads()
                if download_id in paused:
                    del paused[download_id]
                    mlx_helper.save_paused_downloads(paused)
                elif repo_id in paused:
                    del paused[repo_id]
                    mlx_helper.save_paused_downloads(paused)

                env = mlx_helper.get_mlx_env()
                hub, drive = mlx_helper.ensure_library_hub(payload.get("drive") or None)
                cmd = mlx_helper.get_download_cmd(repo_id, subfolder=subfolder, revision=revision,
                                                 cache_dir=hub if drive and drive.get("connected") else "")

                proc = subprocess.Popen(cmd, env=env)
                self.send_json({
                    "status": "resumed",
                    "repo_id": repo_id,
                    "subfolder": subfolder,
                    "download_id": download_id,
                    "display_name": display_name,
                    "drive": (drive or {}).get("name", ""),
                    "new_pid": proc.pid
                })
            else:
                self.send_json({"error": "No repo_id provided"}, 400)

        elif path == "/api/delete":
            delete_target = payload.get("delete_target")
            if delete_target and os.path.exists(delete_target):
                subprocess.run(["rm", "-rf", delete_target])
                self.send_json({"status": "ok", "deleted": delete_target})
            else:
                self.send_json({"error": "Target path not found"}, 400)

        elif path == "/api/download":
            raw_repo = payload.get("repo_id", "")
            explicit_subfolder = payload.get("subfolder", "")
            requested_drive = (payload.get("drive") or "").strip()
            if not raw_repo:
                self.send_json({"error": "No repo_id specified"}, 400)
                return

            parsed = mlx_helper.parse_hf_identifier(raw_repo, explicit_subfolder)
            repo_id = parsed["repo_id"]
            subfolder = parsed["subfolder"]
            revision = parsed["revision"]
            download_id = parsed["download_id"]
            display_name = parsed["display_name"]

            hub, drive = mlx_helper.ensure_library_hub(requested_drive or None)
            if drive and not drive.get("connected"):
                self.send_json({
                    "error": f"{drive.get('name', 'That drive')} is not connected",
                    "code": "drive_disconnected",
                    "drives": mlx_helper.list_storage_drives(force=True),
                }, 409)
                return

            # Free-space guard: refuse to start a download that cannot fit.
            needed_gb = 0.0
            try:
                total_bytes = mlx_helper.get_repo_files_size(repo_id, subfolder=subfolder)
                if total_bytes:
                    needed_gb = round((total_bytes / (1024 ** 3)) * 1.1, 1)
            except Exception:
                needed_gb = 0.0
            if drive and needed_gb and needed_gb > float(drive.get("free_gb") or 0):
                self.send_json({
                    "error": (f"{drive.get('name')} has {drive.get('free_gb')} GB free but "
                              f"{repo_id} needs about {needed_gb} GB"),
                    "code": "insufficient_space",
                    "needed_gb": needed_gb,
                    "drives": mlx_helper.list_storage_drives(force=True),
                }, 409)
                return

            paused = mlx_helper.load_paused_downloads()
            if download_id in paused:
                del paused[download_id]
                mlx_helper.save_paused_downloads(paused)
            elif repo_id in paused:
                del paused[repo_id]
                mlx_helper.save_paused_downloads(paused)

            env = mlx_helper.get_mlx_env()
            hub_dir = hub if drive and drive.get("connected") else ""
            cmd = mlx_helper.get_download_cmd(repo_id, subfolder=subfolder, revision=revision,
                                             cache_dir=hub_dir)

            # Single LFS streams run at ~75 KB/s here; parallelise big blobs first.
            cfg = mlx_helper.load_global_settings()
            turbo_used = False
            if cfg.get("turbo_downloads", True) and hub_dir:
                try:
                    plan = mlx_helper.turbo_plan(repo_id)
                except Exception:
                    plan = {"files": []}
                if plan.get("files"):
                    cmd = [sys.executable, mlx_helper.__file__, "turbo_download", repo_id,
                           hub_dir, download_id, subfolder, revision]
                    turbo_used = True

            proc = subprocess.Popen(cmd, env=env)
            self.send_json({
                "status": "started",
                "repo_id": repo_id,
                "subfolder": subfolder,
                "download_id": download_id,
                "display_name": display_name,
                "drive": (drive or {}).get("name", ""),
                "drive_id": (drive or {}).get("id", ""),
                "hub": hub,
                "needed_gb": needed_gb,
                "pid": proc.pid,
                "turbo": turbo_used
            })

        elif path == "/api/compare_models":
            repos = payload.get("repos", [])
            if not repos:
                self.send_json({"error": "No model repositories selected for comparison"}, 400)
                return
            res = mlx_helper.compare_models_ai(repos)
            self.send_json(res)

        elif path == "/api/test":
            st = mlx_helper.load_global_settings()
            prompt = payload.get("prompt", "Hello!")
            model = payload.get("model", "default_model")
            port = payload.get("port", st.get("default_port", 9999))
            host = payload.get("host", "127.0.0.1")
            image_data = payload.get("image", "")

            url = f"http://{host}:{port}/v1/chat/completions"
            test_model_name = "default_model"
            if model and "/" in model:
                test_model_name = model

            if image_data:
                user_msg = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data}}
                    ]
                }
            else:
                user_msg = {"role": "user", "content": prompt}

            req_data = json.dumps({
                "model": test_model_name,
                "messages": [user_msg],
                "max_tokens": 512
            }).encode("utf-8")

            req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    msg = res_data.get("choices", [{}])[0].get("message", {})
                    content = msg.get("content", "")
                    reasoning = msg.get("reasoning", "") or msg.get("reasoning_content", "")

                    full_output = ""
                    if reasoning:
                        full_output += f"🧠 THINKING PROCESS:\n{reasoning}\n\n"
                    if content:
                        full_output += f"💬 RESPONSE:\n{content}"
                    elif not full_output:
                        full_output = json.dumps(msg, indent=2)

                    self.send_json({"status": "ok", "response": full_output})
            except Exception as e:
                self.send_json({"error": f"API request failed on http://{host}:{port}: {e}"}, 500)

        elif path == "/api/benchmark":
            st = mlx_helper.load_global_settings()
            prompt = payload.get("prompt", "Write a Python script to calculate Fibonacci sequence up to 100 elements.")
            model = payload.get("model", "default_model")
            port = payload.get("port", st.get("default_port", 9999))
            host = payload.get("host", "127.0.0.1")

            res = mlx_helper.run_model_benchmark(prompt, model, port=port, host=host)
            self.send_json(res)
        elif path == "/api/kill_process":
            pid = payload.get("pid")
            res = mlx_helper.kill_process_by_pid(pid)
            self.send_json(res)
        else:
            self.send_json({"error": "Not Found"}, 404)

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>macOS MLX Control Center v0.8.7 (Auto Context)</title>
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <link rel="alternate icon" href="/favicon.ico">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: #151c2c;
      --card-border: rgba(255, 255, 255, 0.08);
      --accent: #6366f1;
      --accent-hover: #4f46e5;
      --accent-glow: rgba(99, 102, 241, 0.35);
      --success: #10b981;
      --success-glow: rgba(16, 185, 129, 0.25);
      --warning: #f59e0b;
      --danger: #ef4444;
      --danger-glow: rgba(239, 68, 68, 0.3);
      --text: #f3f4f6;
      --text-muted: #9ca3af;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', system-ui, -apple-system, sans-serif; }
    body { background: var(--bg); color: var(--text); padding: 20px; line-height: 1.5; min-height: 100vh; padding-bottom: 90px; }

    /* GLOBAL DARK THEME INPUT CONTROLS */
    select, input[type="text"], input[type="number"], textarea {
      background: rgba(11, 15, 25, 0.8) !important;
      border: 1px solid rgba(255, 255, 255, 0.12) !important;
      border-radius: 8px !important;
      color: #f3f4f6 !important;
      padding: 12px 16px !important;
      font-size: 14px !important;
      outline: none !important;
      transition: all 0.2s ease !important;
      box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
    }
    select:focus, input[type="text"]:focus, input[type="number"]:focus, textarea:focus {
      border-color: var(--accent) !important;
      box-shadow: 0 0 12px var(--accent-glow), inset 0 2px 4px rgba(0, 0, 0, 0.4) !important;
    }
    select option {
      background: #151c2c !important;
      color: #f3f4f6 !important;
      padding: 10px !important;
    }
    
    header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 16px 24px; background: var(--card-bg); border: 1px solid var(--card-border);
      border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand-icon { width: 36px; height: 36px; background: linear-gradient(135deg, #6366f1, #a855f7); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 18px; color: #fff; }
    .brand-title { font-size: 20px; font-weight: 700; background: linear-gradient(90deg, #fff, #93c5fd); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    
    .status-pill {
      display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px;
      border-radius: 20px; font-size: 13px; font-weight: 600; border: 1px solid var(--card-border);
    }
    .status-pill.ready { background: var(--success-glow); color: #34d399; border-color: rgba(16, 185, 129, 0.4); }
    .status-pill.downloading { background: rgba(99, 102, 241, 0.2); color: #818cf8; border-color: rgba(99, 102, 241, 0.4); }
    .status-pill.paused { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border-color: rgba(245, 158, 11, 0.4); }
    .status-pill.none { background: rgba(156, 163, 175, 0.1); color: var(--text-muted); }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
    .status-dot.pulse { animation: pulseAnim 1.5s infinite ease-in-out; }
    @keyframes pulseAnim { 0%, 100% { opacity: 0.3; } 50% { opacity: 1; } }

    /* HTOP RESOURCE MONITOR BAR */
    .htop-bar {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px;
      background: rgba(21, 28, 44, 0.7); border: 1px solid var(--card-border); border-radius: 12px;
      padding: 16px 20px; margin-bottom: 20px; backdrop-filter: blur(8px);
    }
    .htop-item { display: flex; flex-direction: column; gap: 6px; }
    .htop-item.htop-clickable {
      cursor: pointer;
      border-radius: 8px;
      padding: 6px 8px;
      margin: -6px -8px;
      transition: background 0.2s ease, transform 0.2s ease;
    }
    .htop-item.htop-clickable:hover {
      background: rgba(99, 102, 241, 0.12);
      transform: translateY(-1px);
    }
    .manage-ram-pill {
      font-size: 10px;
      font-weight: 700;
      color: #818cf8;
      background: rgba(99, 102, 241, 0.2);
      border: 1px solid rgba(99, 102, 241, 0.4);
      border-radius: 12px;
      padding: 2px 8px;
      letter-spacing: 0.3px;
      text-transform: none;
      transition: all 0.2s ease;
    }
    .htop-item.htop-clickable:hover .manage-ram-pill {
      background: var(--accent);
      color: #fff;
      box-shadow: 0 0 10px var(--accent-glow);
    }
    .htop-label { font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
    .htop-val { font-size: 18px; font-weight: 700; color: #fff; }
    .htop-progress { width: 100%; height: 8px; background: rgba(255,255,255,0.08); border-radius: 4px; overflow: hidden; }
    .htop-fill { height: 100%; background: linear-gradient(90deg, #6366f1, #34d399); transition: width 0.3s ease; }

    /* MEMORY MANAGER BREAKDOWN METER & PROCESS TABLE */
    .mem-breakdown-bar {
      display: flex;
      width: 100%;
      height: 14px;
      border-radius: 7px;
      overflow: hidden;
      background: rgba(255, 255, 255, 0.08);
      margin: 12px 0 16px 0;
    }
    .mem-segment {
      height: 100%;
      transition: width 0.3s ease;
    }
    .mem-seg-app { background: #3b82f6; }
    .mem-seg-wired { background: #f97316; }
    .mem-seg-comp { background: #a855f7; }
    .mem-seg-cache { background: #10b981; }

    .proc-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }
    .proc-table th {
      padding: 10px 14px;
      background: rgba(0, 0, 0, 0.3);
      color: var(--text-muted);
      font-weight: 600;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      border-bottom: 1px solid var(--card-border);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    .proc-table td {
      padding: 10px 14px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      vertical-align: middle;
    }
    .proc-table tr:hover td {
      background: rgba(255, 255, 255, 0.03);
    }
    .cat-badge {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      font-size: 11px;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 6px;
    }
    .cat-ai { background: rgba(99, 102, 241, 0.2); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.4); }
    .cat-browser { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .cat-dev { background: rgba(59, 130, 246, 0.15); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.3); }
    .cat-app { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .cat-system { background: rgba(156, 163, 175, 0.1); color: var(--text-muted); border: 1px solid rgba(156, 163, 175, 0.2); }

    nav { display: flex; gap: 8px; margin-bottom: 24px; background: rgba(21, 28, 44, 0.6); padding: 6px; border-radius: 10px; border: 1px solid var(--card-border); }
    .nav-btn {
      flex: 1; padding: 10px 16px; border: none; background: transparent; color: var(--text-muted);
      font-weight: 600; font-size: 14px; border-radius: 8px; cursor: pointer; transition: all 0.2s ease;
    }
    .nav-btn:hover { color: #fff; background: rgba(255,255,255,0.05); }
    .nav-btn.active { color: #fff; background: var(--accent); box-shadow: 0 0 12px var(--accent-glow); }
    
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    
    /* VIEW MODE TOGGLE BUTTONS (CARDS / LIST) */
    .view-toggle-group {
      display: inline-flex;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 3px;
      gap: 3px;
    }
    .view-toggle-btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 12px;
      font-size: 12px;
      font-weight: 600;
      border-radius: 6px;
      border: none;
      background: transparent;
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.2s ease;
      user-select: none;
    }
    .view-toggle-btn:hover {
      color: #fff;
      background: rgba(255, 255, 255, 0.08);
    }
    .view-toggle-btn.active {
      background: var(--accent);
      color: #fff;
      box-shadow: 0 0 10px var(--accent-glow);
    }

    /* CARD STRUCTURE FOR BOTH MODES */
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }
    .card {
      background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px;
      padding: 18px 20px; transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease; position: relative;
    }
    .card:hover { border-color: rgba(99, 102, 241, 0.5); transform: translateY(-2px); }
    .card-title { font-size: 15px; font-weight: 600; word-break: break-all; color: #fff; }
    .card-meta { font-size: 13px; color: var(--text-muted); }

    /* CARD GRID MODE SPECIFIC STYLES */
    .grid:not(.list-view) .card {
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    .grid:not(.list-view) .card-header-row {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 8px;
      gap: 8px;
      flex-wrap: wrap;
    }
    .grid:not(.list-view) .card-meta {
      margin-bottom: 12px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .grid:not(.list-view) .path-meta {
      font-size: 11px;
      word-break: break-all;
      color: var(--text-muted);
      opacity: 0.8;
      font-family: monospace;
    }
    .grid:not(.list-view) .card-actions {
      display: flex;
      gap: 8px;
      margin-top: 12px;
      align-items: center;
      flex-wrap: wrap;
    }
    .grid:not(.list-view) .card-actions .btn {
      flex: 1;
    }

    /* LIST VIEW MODE (VERTICAL LIST WITH HORIZONTAL CARDS) */
    .grid.list-view {
      display: flex !important;
      flex-direction: column !important;
      gap: 10px !important;
    }
    .grid.list-view .card {
      display: flex !important;
      flex-direction: row !important;
      align-items: center !important;
      justify-content: space-between !important;
      padding: 14px 20px !important;
      gap: 16px !important;
      border-radius: 10px !important;
    }
    .grid.list-view .card:hover {
      border-color: rgba(99, 102, 241, 0.5);
      background: rgba(25, 34, 54, 0.95);
      transform: translateX(3px) !important;
    }
    .grid.list-view .card-main {
      flex: 1 1 auto;
      min-width: 0;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .grid.list-view .card-header-row {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 2px;
    }
    .grid.list-view .card-title {
      font-size: 14px;
      margin-bottom: 0;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .grid.list-view .card-meta {
      display: flex;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
      font-size: 12px;
      margin-bottom: 0;
    }
    .grid.list-view .path-meta {
      max-width: 420px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 11px;
      opacity: 0.8;
      font-family: monospace;
    }
    .grid.list-view .ram-badge {
      margin-bottom: 0 !important;
      padding: 2px 8px !important;
      font-size: 11px !important;
    }
    .grid.list-view .card-actions {
      display: flex !important;
      align-items: center !important;
      gap: 8px !important;
      flex-shrink: 0 !important;
      margin-top: 0 !important;
    }
    .grid.list-view .card-actions .btn {
      padding: 8px 14px;
      font-size: 12px;
      white-space: nowrap;
    }
    .grid.list-view .card-actions .compare-label {
      margin-right: 6px;
    }

    @media (max-width: 768px) {
      .grid.list-view .card {
        flex-direction: column !important;
        align-items: flex-start !important;
      }
      .grid.list-view .card-actions {
        width: 100% !important;
        justify-content: flex-start !important;
        margin-top: 8px !important;
      }
    }
    
    .btn {
      display: inline-flex; align-items: center; justify-content: center; gap: 8px; padding: 9px 16px;
      font-size: 13px; font-weight: 600; border-radius: 8px; border: none; cursor: pointer; transition: all 0.2s ease;
      text-decoration: none;
    }
    .btn-primary { background: var(--accent); color: #fff; }
    .btn-primary:hover { background: var(--accent-hover); box-shadow: 0 0 12px var(--accent-glow); }
    .btn-warning { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .btn-warning:hover { background: var(--warning); color: #fff; }
    .btn-danger { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .btn-danger:hover { background: var(--danger); color: #fff; box-shadow: 0 0 12px var(--danger-glow); }
    .btn-secondary { background: rgba(255, 255, 255, 0.08); color: var(--text); }
    .btn-secondary:hover { background: rgba(255, 255, 255, 0.15); }
    
    /* FEATURE CHIPS & SEARCH FILTER BAR */
    .filter-bar { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
    .chip {
      padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 600;
      background: rgba(255,255,255,0.05); color: var(--text-muted); border: 1px solid var(--card-border);
      cursor: pointer; transition: all 0.2s ease;
    }
    .chip:hover { color: #fff; border-color: rgba(255,255,255,0.2); }
    .chip.active { background: rgba(99, 102, 241, 0.2); color: #818cf8; border-color: var(--accent); }

    /* CAN I RUN IT RAM BADGES */
    .ram-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; margin-bottom: 8px; }
    .ram-badge.fit { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .ram-badge.warn { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .ram-badge.heavy { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }

    /* STICKY BOTTOM AI COMPARISON BAR */
    .compare-bar {
      position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%);
      background: rgba(21, 28, 44, 0.95); border: 1px solid var(--accent);
      box-shadow: 0 8px 32px rgba(0,0,0,0.6), 0 0 20px var(--accent-glow);
      border-radius: 40px; padding: 12px 24px; display: flex; align-items: center; gap: 16px;
      z-index: 999; backdrop-filter: blur(12px); animation: slideUp 0.3s ease;
    }
    @keyframes slideUp { from { transform: translate(-50%, 40px); opacity: 0; } to { transform: translate(-50%, 0); opacity: 1; } }

    /* WIDER HF SEARCH INPUT BOX */
    .search-box { display: flex; gap: 12px; margin-bottom: 16px; width: 100%; max-width: 980px; }
    .search-input-wide { flex: 1; min-width: 420px; font-size: 15px !important; }
    
    .chat-box { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 20px; margin-top: 16px; }
    .chat-output { background: rgba(0,0,0,0.3); border: 1px solid var(--card-border); border-radius: 8px; padding: 16px; min-height: 140px; font-family: monospace; white-space: pre-wrap; margin-top: 12px; color: #a7f3d0; font-size: 13px; }
    .log-output { background: #05070f; border: 1px solid var(--card-border); border-radius: 8px; padding: 16px; height: 350px; overflow-y: auto; font-family: monospace; font-size: 12px; color: #94a3b8; }
    .error-box { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; color: #fca5a5; font-family: monospace; font-size: 12px; white-space: pre-wrap; }

    /* MODAL OVERLAY */
    .modal-overlay {
      display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0, 0, 0, 0.75); backdrop-filter: blur(4px);
      z-index: 1000; align-items: center; justify-content: center;
    }
    .modal-overlay.active { display: flex; }
    .modal {
      background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 16px;
      padding: 24px; width: 100%; max-width: 960px; max-height: 90vh; overflow-y: auto; box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .modal-header { font-size: 18px; font-weight: 700; margin-bottom: 16px; color: #fff; }
    .strategy-option {
      display: flex; align-items: flex-start; gap: 12px; padding: 14px;
      border: 1px solid var(--card-border); border-radius: 10px; margin-bottom: 12px;
      cursor: pointer; transition: all 0.2s ease; background: rgba(255,255,255,0.02);
    }
    .strategy-option:hover { border-color: var(--accent); background: rgba(99,102,241,0.08); }
    .strategy-option input[type="radio"] { margin-top: 4px; accent-color: var(--accent); }
    .strategy-title { font-weight: 600; font-size: 14px; color: #fff; }
    .strategy-desc { font-size: 12px; color: var(--text-muted); margin-top: 2px; }

    /* PRESET BUTTONS */
    .preset-btn {
      display: flex; flex-direction: column; gap: 4px; padding: 10px 14px; border-radius: 8px;
      background: rgba(255,255,255,0.04); border: 1px solid var(--card-border); cursor: pointer; text-align: left;
      transition: all 0.2s ease; flex: 1;
    }
    .preset-btn:hover { border-color: var(--accent); background: rgba(99,102,241,0.12); }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-icon">⚡</div>
      <div>
        <div class="brand-title" style="display: flex; align-items: center; gap: 8px;">
          macOS MLX Control Center
          <span style="font-size: 11px; padding: 2px 8px; border-radius: 6px; background: rgba(99, 102, 241, 0.25); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.4); font-weight: 600; letter-spacing: 0.5px;">v0.8.7 (Auto Context)</span>
        </div>
        <div style="font-size: 12px; color: var(--text-muted);">Apple Silicon Local Multimodal & LLM Powerstation</div>
      </div>
    </div>
    <div style="display: flex; align-items: center; gap: 12px;">
      <button class="btn btn-secondary" id="hfAuthBtn" style="padding: 8px 14px; font-size: 13px;" onclick="openHfModal()">🤗 Sign in</button>
      <button class="btn btn-secondary" style="padding: 8px 14px; font-size: 13px;" onclick="openConfigModal()">⚙️ API Settings</button>
      <div id="statusHeader">
        <div class="status-pill none"><span class="status-dot"></span> Checking Status...</div>
      </div>
    </div>
  </header>

  <!-- HTOP RESOURCE MONITOR BAR -->
  <div class="htop-bar">
    <div class="htop-item htop-clickable" onclick="openMemoryModal()" title="Click to view memory breakdown and free RAM">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="htop-label">Mac System RAM</div>
        <span class="manage-ram-pill">🔍 Manage RAM</span>
      </div>
      <div class="htop-val" id="htopRamVal">-- / -- GB</div>
      <div class="htop-progress"><div class="htop-fill" id="htopRamFill" style="width: 0%;"></div></div>
      <div style="font-size: 12px; color: var(--text-muted);" id="htopRamMeta">-- GB Available</div>
    </div>
    <div class="htop-item">
      <div class="htop-label">MLX Models Memory</div>
      <div class="htop-val" id="htopMlxVal">-- GB</div>
      <div style="font-size: 12px; color: var(--text-muted);" id="htopMlxMeta">0 Active Models</div>
    </div>
    <div class="htop-item">
      <div class="htop-label">CPU Utilization</div>
      <div class="htop-val" id="htopCpuVal">-- %</div>
      <div class="htop-progress"><div class="htop-fill" id="htopCpuFill" style="width: 0%; background: #a855f7;"></div></div>
    </div>
  </div>

  <nav>
    <button class="nav-btn active" onclick="switchTab('dashboard')">Dashboard</button>
    <button class="nav-btn" onclick="switchTab('models')">Downloaded Models</button>
    <button class="nav-btn" onclick="switchTab('search')">Search Hugging Face</button>
    <button class="nav-btn" onclick="switchTab('test')">API & Chat Test</button>
    <button class="nav-btn" onclick="switchTab('bench')">📊 Speed Benchmark</button>
    <button class="nav-btn" onclick="switchTab('logs')">Server Logs</button>
  </nav>

  <!-- LIVE DOWNLOAD QUEUE BANNER -->
  <div id="liveDownloadBanner" style="display: none; margin-bottom: 24px;"></div>

  <!-- TAB 1: DASHBOARD -->
  <div id="tab-dashboard" class="tab-content active">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; flex-wrap: wrap; gap: 10px;">
      <h2 style="font-size: 18px; font-weight: 600;">Active MLX Model Servers</h2>
      <button class="btn btn-danger" onclick="killAllServers()">⚡ KILL SWITCH (Stop All)</button>
    </div>
    <div id="activeServerContainer" class="grid" style="margin-bottom: 24px;"></div>

    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; flex-wrap: wrap; gap: 10px;">
      <h2 style="font-size: 18px; font-weight: 600;">Launch a Model</h2>
      <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
        <span style="font-size: 12px; color: var(--text-muted);">Sort:</span>
        <select id="dashboardModelSortSelect" onchange="handleModelSortChange(this.value)" style="padding: 6px 12px !important; font-size: 12px !important;">
          <option value="name-asc">🔤 Name (A → Z)</option>
          <option value="name-desc">🔤 Name (Z → A)</option>
          <option value="size-desc">📦 Size (Largest)</option>
          <option value="size-asc">📦 Size (Smallest)</option>
          <option value="date-desc">📅 Date Added (Newest)</option>
          <option value="date-asc">📅 Date Added (Oldest)</option>
        </select>
        <div class="view-toggle-group">
          <button class="view-toggle-btn active" data-view="grid" onclick="setViewMode('grid')" title="Card Grid View">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M3 3h8v8H3V3zm10 0h8v8h-8V3zM3 13h8v8H3v-8zm10 0h8v8h-8v-8z"/></svg> Cards
          </button>
          <button class="view-toggle-btn" data-view="list" onclick="setViewMode('list')" title="Horizontal List View">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M3 4h18v3H3V4zm0 7h18v3H3v-3zm0 7h18v3H3v-3z"/></svg> List
          </button>
        </div>
      </div>
    </div>
    <div id="fastSwapGrid" class="grid"></div>
  </div>

  <!-- TAB 2: DOWNLOADED MODELS -->
  <div id="tab-models" class="tab-content">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; flex-wrap: wrap; gap: 10px;">
      <h2 style="font-size: 18px; font-weight: 600;">Downloaded Models</h2>
      <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
        <span style="font-size: 12px; color: var(--text-muted);">Sort:</span>
        <select id="modelSortSelect" onchange="handleModelSortChange(this.value)" style="padding: 6px 12px !important; font-size: 12px !important;">
          <option value="name-asc">🔤 Name (A → Z)</option>
          <option value="name-desc">🔤 Name (Z → A)</option>
          <option value="size-desc">📦 Size (Largest)</option>
          <option value="size-asc">📦 Size (Smallest)</option>
          <option value="date-desc">📅 Date Added (Newest)</option>
          <option value="date-asc">📅 Date Added (Oldest)</option>
        </select>
        <div class="view-toggle-group">
          <button class="view-toggle-btn active" data-view="grid" onclick="setViewMode('grid')" title="Card Grid View">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M3 3h8v8H3V3zm10 0h8v8h-8V3zM3 13h8v8H3v-8zm10 0h8v8h-8v-8z"/></svg> Cards
          </button>
          <button class="view-toggle-btn" data-view="list" onclick="setViewMode('list')" title="Horizontal List View">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M3 4h18v3H3V4zm0 7h18v3H3v-3zm0 7h18v3H3v-3z"/></svg> List
          </button>
        </div>
        <button class="btn btn-warning" onclick="clearCompareSelection()">🧹 Clear Checkmarks</button>
        <button class="btn btn-secondary" onclick="loadModels()">🔄 Refresh Models</button>
      </div>
    </div>
    <div id="storageSummary" style="margin-bottom: 14px;"></div>
    <div id="modelsGrid" class="grid"></div>
  </div>

  <!-- TAB 3: SEARCH HF (SUPERCHARGED SEARCH WITH FILTERS & RAM ESTIMATOR) -->
  <div id="tab-search" class="tab-content">
    <div class="search-box">
      <input type="text" id="searchInput" class="search-input-wide" placeholder="Search Hugging Face (e.g. gemma 4, qwen 2.5 coder, llama 3)..." value="gemma 4" onkeyup="if(event.key==='Enter') doSearch()">
      <button class="btn btn-primary" style="padding: 14px 24px; font-size: 15px;" onclick="doSearch()">Search HF</button>
      <button class="btn btn-warning" style="padding: 14px 18px; font-size: 14px;" onclick="clearCompareSelection()">🧹 Clear Checkmarks</button>
    </div>

    <!-- FILTER & SORT BAR (filters combine with AND — e.g. MLX + Vision + 8-bit + Code) -->
    <div class="filter-bar">
      <div class="chip active" id="chip-mlx" onclick="toggleTag('mlx')" title="Click to combine with the other filters">⚡ MLX Models</div>
      <div class="chip" id="chip-vision" onclick="toggleTag('vision')" title="Combine freely: MLX + Vision + 8-bit + Code">👁️ Vision & Multimodal</div>
      <div class="chip" id="chip-code" onclick="toggleTag('code')" title="Combine freely with MLX, Vision and Quant">💻 Code Models</div>
      <div class="chip" id="chip-all" onclick="clearTags()" title="Clear every filter and search all of Hugging Face">🌐 All HF Repos</div>
      <span id="filterHint" style="font-size: 11px; color: var(--text-muted); align-self: center; margin-left: 2px;"></span>

      <div style="margin-left: auto; display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
        <span style="font-size: 12px; color: var(--text-muted);">Sort By:</span>
        <select id="sortSelect" onchange="doSearch()" style="padding: 6px 12px !important; font-size: 12px !important;">
          <option value="downloads">Most Downloads</option>
          <option value="likes">Most Liked</option>
          <option value="lastModified">Recently Updated</option>
          <option value="size_asc">Smallest RAM First</option>
          <option value="size_desc">Largest RAM First</option>
        </select>
        <select id="limitSelect" onchange="doSearch()" style="padding: 6px 12px !important; font-size: 12px !important;">
          <option value="50">50 Results</option>
          <option value="100">100 Results</option>
        </select>
        <span style="font-size: 12px; color: var(--text-muted); margin-left: 4px;">Quant:</span>
        <select id="quantSelect" onchange="doSearch()" title="Filter by quantization level"
                style="padding: 6px 12px !important; font-size: 12px !important;">
          <option value="">Any</option>
          <option value="2">2-bit</option>
          <option value="3">3-bit</option>
          <option value="4">4-bit</option>
          <option value="5">5-bit</option>
          <option value="6">6-bit</option>
          <option value="8">8-bit</option>
          <option value="16">16-bit</option>
        </select>
        <span style="font-size: 12px; color: var(--text-muted); margin-left: 4px;">RAM Budget:</span>
        <select id="ramSelect" onchange="renderSearchCards(currentSearchQuery)" title="Only show models that fit within this much RAM"
                style="padding: 6px 12px !important; font-size: 12px !important; border-color: rgba(99, 102, 241, 0.55); background: rgba(99, 102, 241, 0.12); font-weight: 600;">
          <option value="auto">🖥️ Fits My Mac (auto)</option>
          <option value="0">Any Size</option>
          <option value="8">≤ 8 GB</option>
          <option value="16">≤ 16 GB</option>
          <option value="24">≤ 24 GB</option>
          <option value="32">≤ 32 GB</option>
          <option value="48">≤ 48 GB</option>
          <option value="64">≤ 64 GB</option>
          <option value="96">≤ 96 GB</option>
          <option value="128">≤ 128 GB</option>
          <option value="192">≤ 192 GB</option>
        </select>
        <span style="font-size: 12px; color: var(--text-muted); margin-left: 4px;">Download to:</span>
        <select id="driveSelect" onchange="onDriveChange(this.value)" title="Where new models are saved — every mounted drive is listed automatically"
                style="padding: 6px 12px !important; font-size: 12px !important; border-color: rgba(52, 211, 153, 0.55); background: rgba(52, 211, 153, 0.12); font-weight: 600; max-width: 250px;">
          <option value="">Loading drives…</option>
        </select>
        <button class="btn btn-secondary" style="padding: 6px 9px; font-size: 12px;" onclick="loadDrives(true)" title="Re-scan mounted drives">⟳</button>
        <div class="view-toggle-group">
          <button class="view-toggle-btn active" data-view="grid" onclick="setViewMode('grid')" title="Card Grid View">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M3 3h8v8H3V3zm10 0h8v8h-8V3zM3 13h8v8H3v-8zm10 0h8v8h-8v-8z"/></svg> Cards
          </button>
          <button class="view-toggle-btn" data-view="list" onclick="setViewMode('list')" title="Horizontal List View">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M3 4h18v3H3V4zm0 7h18v3H3v-3zm0 7h18v3H3v-3z"/></svg> List
          </button>
        </div>
      </div>
    </div>

    <div id="searchMetaHeader" style="font-size: 13px; color: var(--text-muted); margin-bottom: 16px;"></div>
    <div id="searchResultsGrid" class="grid"></div>
  </div>

  <!-- TAB 4: API & CHAT TEST (WITH VISION & IMAGE SUPPORT) -->
  <div id="tab-test" class="tab-content">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
      <h2 style="font-size: 18px; font-weight: 600;">Test Active MLX Model Server (Text & Vision)</h2>
      <button class="btn btn-secondary" onclick="loadTestServers()">🔄 Refresh Server List</button>
    </div>
    <div class="chat-box">
      <div style="margin-bottom: 16px;">
        <label style="display: block; font-size: 13px; font-weight: 600; margin-bottom: 8px; color: var(--text-muted);">
          Select Target Model Server:
        </label>
        <select id="testServerSelect" style="width: 100%; font-weight: 500;"></select>
      </div>

      <!-- IMAGE ATTACHMENT FOR VISION TESTING -->
      <div style="background: rgba(0,0,0,0.25); border: 1px dashed rgba(99, 102, 241, 0.4); border-radius: 10px; padding: 14px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
          <div style="font-size: 13px; font-weight: 600; color: #a5b4fc; display: flex; align-items: center; gap: 6px;">
            👁️ Image / Vision Attachment (Optional for Multimodal Models)
          </div>
          <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 11px;" onclick="document.getElementById('testImageFile').click()">📁 Choose Image</button>
        </div>
        <input type="file" id="testImageFile" accept="image/*" onchange="handleImageSelected(this)" style="display: none;">
        
        <div id="testImagePreviewRow" style="display: none; align-items: center; gap: 14px; margin-top: 10px;">
          <img id="testImagePreview" style="max-height: 90px; max-width: 140px; border-radius: 6px; border: 1px solid var(--accent); object-fit: cover;">
          <div>
            <div id="testImageName" style="font-size: 12px; font-weight: 600; color: #f3f4f6;">image.png</div>
            <div id="testImageSize" style="font-size: 11px; color: var(--text-muted);">0 KB</div>
            <button class="btn btn-danger" style="padding: 3px 8px; font-size: 11px; margin-top: 6px;" onclick="removeTestImage()">✕ Remove Image</button>
          </div>
        </div>

        <div id="testImagePromptShortcuts" style="display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px;">
          <span style="font-size: 11px; color: var(--text-muted); align-self: center;">Quick Prompts:</span>
          <button class="btn btn-secondary" style="padding: 4px 8px; font-size: 11px;" onclick="setPrompt('Describe what you see in this image in detail.')">🖼️ Describe Image</button>
          <button class="btn btn-secondary" style="padding: 4px 8px; font-size: 11px;" onclick="setPrompt('Extract and transcribe all text from this image (OCR).')">🔍 OCR Text</button>
          <button class="btn btn-secondary" style="padding: 4px 8px; font-size: 11px;" onclick="setPrompt('Explain the code and diagram shown in this screenshot.')">💻 Explain Screenshot</button>
        </div>
      </div>

      <div style="margin-bottom: 16px;">
        <label style="display: block; font-size: 13px; font-weight: 600; margin-bottom: 8px; color: var(--text-muted);">
          Test Prompt:
        </label>
        <input type="text" id="promptInput" style="width: 100%; font-weight: 500;" placeholder="Enter prompt to test..." value="Hello! State your model name and what you can do.">
      </div>
      <button class="btn btn-primary" style="padding: 12px 24px; font-size: 14px;" onclick="sendTestPrompt()">🚀 Send Test Prompt</button>
      <div id="chatOutput" class="chat-output">Response will appear here...</div>
    </div>
  </div>

  <!-- TAB 5: SPEED BENCHMARK & COMPARISON -->
  <div id="tab-bench" class="tab-content">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
      <h2 style="font-size: 18px; font-weight: 600;">📊 MLX Tokens/Second Benchmark</h2>
    </div>
    <div class="chat-box">
      <div style="margin-bottom: 16px;">
        <label style="display: block; font-size: 13px; font-weight: 600; margin-bottom: 8px; color: var(--text-muted);">
          Benchmark Test Prompt:
        </label>
        <input type="text" id="benchPromptInput" style="width: 100%; font-weight: 500;" value="Write a Python script to calculate Fibonacci sequence up to 100 elements with memoization.">
      </div>
      <button class="btn btn-primary" style="padding: 12px 24px; font-size: 14px;" onclick="runBenchmark()">⚡ Run Speed Benchmark Test</button>
      
      <div id="benchMetricsCard" style="display: none; margin-top: 20px;" class="card">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 16px;">
          <div>
            <div style="font-size: 12px; color: var(--text-muted);">Generation Speed</div>
            <div style="font-size: 24px; font-weight: 700; color: #34d399;" id="benchTokSec">-- tok/s</div>
          </div>
          <div>
            <div style="font-size: 12px; color: var(--text-muted);">Total Execution Time</div>
            <div style="font-size: 24px; font-weight: 700; color: #60a5fa;" id="benchTotSec">-- s</div>
          </div>
          <div>
            <div style="font-size: 12px; color: var(--text-muted);">Tokens (Prompt / Gen)</div>
            <div style="font-size: 18px; font-weight: 700; color: #f3f4f6;" id="benchTokensMeta">-- / --</div>
          </div>
          <div>
            <div style="font-size: 12px; color: var(--text-muted);">MLX GPU RAM Used</div>
            <div style="font-size: 18px; font-weight: 700; color: #a855f7;" id="benchRamVal">-- GB</div>
          </div>
        </div>
        <div id="benchTextOutput" style="background: rgba(0,0,0,0.3); border: 1px solid var(--card-border); border-radius: 8px; padding: 14px; font-family: monospace; font-size: 12px; max-height: 250px; overflow-y: auto; color: #93c5fd; white-space: pre-wrap;"></div>
      </div>
    </div>
  </div>

  <!-- TAB 6: LOGS -->
  <div id="tab-logs" class="tab-content">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
      <h2 style="font-size: 18px; font-weight: 600;">Server Logs Inspector</h2>
      <div style="display: flex; gap: 12px; align-items: center;">
        <label style="font-size: 12px; color: var(--text-muted); cursor: pointer;">
          <input type="checkbox" id="hidePollingCheck" checked onchange="loadLogContent()"> Hide GET /v1/models polling
        </label>
        <select id="logSelect" onchange="loadLogContent()"></select>
        <button class="btn btn-warning" onclick="clearCurrentLog()">🗑️ Clear Selected Log</button>
        <button class="btn btn-danger" onclick="clearAllLogs()">🧹 Clear All Logs</button>
      </div>
    </div>
    <div id="errorHighlightContainer" style="display: none;" class="error-box"></div>
    <div id="logContent" class="log-output">Loading log output...</div>
  </div>

  <!-- FLOATING STICKY AI COMPARISON BAR -->
  <div id="floatingCompareBar" class="compare-bar" style="display: none;">
    <div style="font-size: 13px; font-weight: 600; color: #fff;">
      <span id="compareCountLabel">Selected Models (0):</span>
    </div>
    <button class="btn btn-primary" style="padding: 8px 18px; font-size: 13px;" onclick="runAiComparison()">🤖 Compare Selected Models with AI</button>
    <button class="btn btn-secondary" style="padding: 8px 14px; font-size: 12px;" onclick="clearCompareSelection()">🧹 Clear Checkmarks</button>
  </div>

  <!-- QUANTIZATION SELECTION MODAL FOR MULTI-QUANT REPOSITORIES -->
  <div id="quantModal" class="modal-overlay">
    <div class="modal" style="max-width: 600px;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
        <div class="modal-header" style="margin-bottom: 0;">⚡ Select Quantization / Bit Precision</div>
        <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 12px;" onclick="closeQuantModal()">✕ Close</button>
      </div>
      <div id="quantModalSubtitle" style="font-size: 13px; color: var(--text-muted); margin-bottom: 16px; line-height: 1.5;"></div>
      <div id="quantModalVariants" style="display: flex; flex-direction: column; gap: 10px; margin-bottom: 18px; max-height: 380px; overflow-y: auto;"></div>
      <div style="display: flex; justify-content: flex-end; gap: 8px; border-top: 1px solid var(--card-border); padding-top: 14px;">
        <button class="btn btn-secondary" onclick="closeQuantModal()">Cancel</button>
        <button class="btn btn-warning" id="quantDownloadAllBtn">Download Full Repo (All Quantizations)</button>
      </div>
    </div>
  </div>

  <!-- DRIVE PICKER MODAL -->
  <div id="driveModal" class="modal-overlay">
    <div class="modal" style="max-width: 620px;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
        <div class="modal-header" style="margin-bottom: 0;">💾 Choose a Drive</div>
        <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 12px;" onclick="closeDriveModal()">✕ Close</button>
      </div>
      <div id="driveModalSubtitle" style="font-size: 13px; color: var(--text-muted); margin-bottom: 16px; line-height: 1.5;"></div>
      <div id="driveModalList" style="display: flex; flex-direction: column; gap: 10px; margin-bottom: 16px; max-height: 380px; overflow-y: auto;"></div>
      <div id="driveConfirmBar" style="display: none; margin-bottom: 16px;"></div>
      <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid var(--card-border); padding-top: 14px;">
        <span id="driveModalHint" style="font-size: 11px; color: var(--text-muted);">Drives are detected automatically — plug in a new one and it appears here.</span>
        <button class="btn btn-secondary" onclick="loadDrives(true); renderDriveModalList();">⟳ Refresh Drives</button>
      </div>
    </div>
  </div>

  <!-- AI MODEL COMPARISON MATRIX MODAL -->
  <div id="compareModal" class="modal-overlay">
    <div class="modal">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
        <div class="modal-header" style="margin-bottom: 0;">🤖 Antigravity AI Model Comparison Breakdown</div>
        <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 12px;" onclick="closeCompareModal()">✕ Close</button>
      </div>

      <div id="compareModalContent">Loading comparison matrix...</div>
    </div>
  </div>

  <!-- GLOBAL CONFIGURATION MODAL WITH RAM & TOKEN GUIDE -->
  <div id="hfModal" class="modal-overlay">
    <div class="modal" style="max-width: 560px;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
        <div class="modal-header" style="margin-bottom: 0;">🤗 Hugging Face Account</div>
        <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 12px;" onclick="closeHfModal()">✕ Close</button>
      </div>

      <div id="hfSignedOut">
        <div style="font-size: 13px; color: var(--text-muted); line-height: 1.6; margin-bottom: 14px;">
          Signing in removes anonymous rate limits, unlocks gated and private repositories, and lets the Hub attribute your
          downloads. The token is stored in your <strong>macOS Keychain</strong> — never in this repository, never in a config file
          that gets committed.
        </div>

        <div id="hfDeviceBox" style="display: none; background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.4); border-radius: 10px; padding: 16px; margin-bottom: 14px;">
          <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 6px;">1. Open this page and approve:</div>
          <a id="hfDeviceLink" href="#" target="_blank" style="color: #60a5fa; font-size: 13px; word-break: break-all;"></a>
          <div style="font-size: 12px; color: var(--text-muted); margin: 12px 0 6px;">2. Enter this code:</div>
          <div style="display: flex; align-items: center; gap: 10px;">
            <code id="hfUserCode" style="font-size: 24px; letter-spacing: 3px; font-weight: 700; color: #a5b4fc;"></code>
            <button class="btn btn-secondary" style="padding: 6px 10px; font-size: 12px;" onclick="copyHfCode()">Copy</button>
          </div>
          <div id="hfDeviceStatus" style="font-size: 12px; color: var(--text-muted); margin-top: 12px;">Waiting for approval…</div>
        </div>

        <div style="display: flex; gap: 8px; margin-bottom: 16px;">
          <button class="btn btn-primary" id="hfLoginBtn" onclick="startHfLogin()">🤗 Sign in with Hugging Face</button>
          <button class="btn btn-secondary" onclick="toggleHfTokenForm()">Paste a token instead</button>
        </div>

        <div id="hfTokenForm" style="display: none; border-top: 1px solid var(--card-border); padding-top: 14px;">
          <label style="display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; color: var(--text-muted);">
            User Access Token
          </label>
          <input type="password" id="hfTokenInput" placeholder="hf_…" style="width: 100%;" autocomplete="off" spellcheck="false">
          <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">
            A <strong>Read</strong> token from
            <a href="https://huggingface.co/settings/tokens/new?tokenType=read" target="_blank" style="color: #60a5fa;">huggingface.co/settings/tokens</a>
            is enough. It never leaves this Mac.
          </div>
          <button class="btn btn-primary" style="margin-top: 10px;" onclick="submitHfToken()">Save &amp; verify</button>
        </div>

        <div id="hfError" style="display: none; color: #f87171; font-size: 12px; margin-top: 12px;"></div>
      </div>

      <div id="hfSignedIn" style="display: none;">
        <div style="font-size: 14px; margin-bottom: 6px;">Signed in as <strong id="hfUserName" style="color: #34d399;"></strong></div>
        <div id="hfUserMeta" style="font-size: 12px; color: var(--text-muted); margin-bottom: 16px;"></div>
        <button class="btn btn-secondary" onclick="logoutHf()">Sign out</button>
      </div>
    </div>
  </div>

  <div id="configModal" class="modal-overlay">
    <div class="modal" style="max-width: 560px;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
        <div class="modal-header" style="margin-bottom: 0;">⚙️ Global API & Server Settings</div>
        <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 12px;" onclick="closeConfigModal()">✕ Close</button>
      </div>

      <div style="margin-bottom: 14px;">
        <label style="display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; color: var(--text-muted);">
          Default MLX Server Port:
        </label>
        <input type="number" id="cfgPortInput" style="width: 100%;" value="9999">
        <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Default port for OpenAI-compatible endpoint (e.g. 9999, 8888, 8080).</div>
      </div>

      <div style="margin-bottom: 14px;">
        <label style="display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; color: var(--text-muted);">
          Server Host Binding:
        </label>
        <select id="cfgHostSelect" style="width: 100%;">
          <option value="127.0.0.1">127.0.0.1 (Localhost Only - Secure)</option>
          <option value="0.0.0.0">0.0.0.0 (LAN Access - Allow Home Devices)</option>
        </select>
        <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Select 0.0.0.0 if you want other PCs or phones on Wi-Fi to query your Mac.</div>
      </div>

      <div style="display: flex; gap: 12px; margin-bottom: 14px;">
        <div style="flex: 1;">
          <label style="display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; color: var(--text-muted);">Default Temp:</label>
          <input type="text" id="cfgTempInput" style="width: 100%;" value="0.0">
        </div>
        <div style="flex: 1;">
          <label style="display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; color: var(--text-muted);">Max Output Tokens:</label>
          <input type="number" id="cfgMaxTokensInput" style="width: 100%;" value="4096">
        </div>
      </div>

      <div style="margin-bottom: 16px;">
        <label style="display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; color: var(--text-muted);">Context Window (KV cache cap):</label>
        <select id="cfgContextSelect" style="width: 100%;" onchange="renderContextNote()">
          <option value="0">Auto — use each model's native maximum</option>
          <option value="32768">32K tokens</option>
          <option value="65536">64K tokens</option>
          <option value="131072">128K tokens</option>
          <option value="262144">256K tokens</option>
        </select>
        <div id="cfgContextNote" style="font-size: 10px; color: #64748b; margin-top: 6px; line-height: 1.5;"></div>
      </div>

      <!-- RICH TOOLTIP & RAM GUIDANCE CARD -->
      <div style="background: rgba(0,0,0,0.35); border: 1px solid var(--card-border); border-radius: 10px; padding: 14px; margin-bottom: 16px; font-size: 12px; color: #94a3b8; line-height: 1.5;">
        <strong style="color: #60a5fa; font-size: 13px;">💡 What is "Max Output Tokens"?</strong><br>
        This specifies the maximum number of <strong>NEW tokens (generated response + thinking tokens)</strong> the server will produce in a single request. It does <em>not</em> limit your input prompt size.<br><br>
        <strong style="color: #34d399;">🧠 Recommended token settings — tuned for <span id="tokenGuideMac">this Mac</span>:</strong>
        <table style="width: 100%; margin-top: 6px; border-collapse: collapse; font-size: 11px; color: #cbd5e1;">
          <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
            <th style="text-align: left; padding: 4px;">Model Size</th>
            <th style="text-align: left; padding: 4px;">Rec. Max Tokens</th>
            <th style="text-align: left; padding: 4px;">KV Cache RAM Impact</th>
          </tr>
          <tbody id="tokenGuideBody"></tbody>
        </table>
        <div id="tokenGuideNote" style="margin-top: 6px; font-size: 10px; color: #64748b; line-height: 1.45;"></div>
      </div>

      <div style="margin-bottom: 16px;">
        <label style="font-size: 13px; color: #fff; cursor: pointer; display: flex; align-items: center; gap: 8px;">
          <input type="checkbox" id="cfgAutoSyncCheck" checked> Auto-sync Pi Code & OpenCode under provider 'MyMac'
        </label>
      </div>

      <div style="background: rgba(255,255,255,0.03); border: 1px solid #2a2f3d; border-radius: 10px; padding: 12px 14px; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; gap: 14px;">
        <div style="min-width: 0;">
          <div style="font-size: 13px; color: #fff; margin-bottom: 3px;">🤗 DeepSeek Harness model list</div>
          <div id="harnessSyncNote" style="font-size: 11px; color: #94a3b8; line-height: 1.5;">Checking…</div>
        </div>
        <button id="harnessSyncBtn" class="btn btn-secondary" style="padding: 8px 12px; font-size: 12px; white-space: nowrap;" onclick="syncHarnessModels()">🔄 Sync now</button>
      </div>

      <div style="display: flex; justify-content: flex-end; gap: 12px; margin-top: 20px;">
        <button class="btn btn-secondary" onclick="closeConfigModal()">Cancel</button>
        <button class="btn btn-primary" onclick="saveConfigModal()">💾 Save & Apply Settings</button>
      </div>
    </div>
  </div>

  <!-- MACOS MEMORY & PROCESS MANAGER MODAL -->
  <div id="memoryModal" class="modal-overlay">
    <div class="modal" style="max-width: 880px;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
        <div class="modal-header" style="margin-bottom: 0; display: flex; align-items: center; gap: 8px;">
          <span>🧠 macOS Memory & Process Manager</span>
        </div>
        <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 12px;" onclick="closeMemoryModal()">✕ Close</button>
      </div>

      <!-- RAM BREAKDOWN SECTION -->
      <div style="background: rgba(0,0,0,0.3); border: 1px solid var(--card-border); border-radius: 12px; padding: 16px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
          <strong style="font-size: 14px; color: #fff;">Unified Memory Allocation</strong>
          <span style="font-size: 13px; font-weight: 700; color: #60a5fa;" id="memModalTotalVal">-- / -- GB Used</span>
        </div>

        <div class="mem-breakdown-bar">
          <div class="mem-segment mem-seg-app" id="memSegApp" style="width: 45%;" title="App Memory"></div>
          <div class="mem-segment mem-seg-wired" id="memSegWired" style="width: 10%;" title="Wired Memory"></div>
          <div class="mem-segment mem-seg-comp" id="memSegComp" style="width: 8%;" title="Compressed Memory"></div>
          <div class="mem-segment mem-seg-cache" id="memSegCache" style="width: 37%;" title="Available & Cache"></div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; font-size: 12px;">
          <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 8px; padding: 8px 12px;">
            <div style="color: #93c5fd; font-weight: 600;">📱 App Memory</div>
            <div style="font-size: 16px; font-weight: 700; color: #fff;" id="memModalAppVal">-- GB</div>
          </div>
          <div style="background: rgba(249, 115, 22, 0.1); border: 1px solid rgba(249, 115, 22, 0.25); border-radius: 8px; padding: 8px 12px;">
            <div style="color: #fdba74; font-weight: 600;">🔌 Wired Memory</div>
            <div style="font-size: 16px; font-weight: 700; color: #fff;" id="memModalWiredVal">-- GB</div>
          </div>
          <div style="background: rgba(168, 85, 247, 0.1); border: 1px solid rgba(168, 85, 247, 0.25); border-radius: 8px; padding: 8px 12px;">
            <div style="color: #d8b4fe; font-weight: 600;">🗜️ Compressed</div>
            <div style="font-size: 16px; font-weight: 700; color: #fff;" id="memModalCompVal">-- GB</div>
          </div>
          <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 8px 12px;">
            <div style="color: #6ee7b7; font-weight: 600;">🟢 Available for MLX</div>
            <div style="font-size: 16px; font-weight: 700; color: #34d399;" id="memModalAvailVal">-- GB</div>
          </div>
        </div>
      </div>

      <!-- PROCESS SEARCH & FILTER CONTROLS -->
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; gap: 10px; flex-wrap: wrap;">
        <div style="display: flex; gap: 6px; flex-wrap: wrap;" id="procFilterChips">
          <div class="chip active" onclick="filterMemoryProcesses('all', this)">All Processes</div>
          <div class="chip" onclick="filterMemoryProcesses('AI Model', this)">🤖 AI Models</div>
          <div class="chip" onclick="filterMemoryProcesses('Browser', this)">🌐 Browsers</div>
          <div class="chip" onclick="filterMemoryProcesses('Dev Tool', this)">🛠️ Dev Tools</div>
          <div class="chip" onclick="filterMemoryProcesses('App', this)">💬 Apps</div>
        </div>
        <div style="display: flex; gap: 8px; align-items: center; margin-left: auto;">
          <input type="text" id="procSearchInput" placeholder="Filter by name / PID..." style="padding: 6px 12px !important; font-size: 12px !important; width: 180px;" oninput="filterMemoryProcesses()">
          <button class="btn btn-secondary" style="padding: 7px 12px; font-size: 12px;" onclick="loadMemoryProcesses()">🔄 Refresh</button>
        </div>
      </div>

      <!-- PROCESS TABLE CONTAINER -->
      <div style="max-height: 380px; overflow-y: auto; border: 1px solid var(--card-border); border-radius: 10px; background: rgba(0,0,0,0.2);">
        <table class="proc-table">
          <thead>
            <tr>
              <th>Process / App</th>
              <th>Category</th>
              <th>PID</th>
              <th>RAM Usage</th>
              <th style="text-align: right;">Action</th>
            </tr>
          </thead>
          <tbody id="procTableBody">
            <tr>
              <td colspan="5" style="text-align: center; color: var(--text-muted); padding: 24px;">Scanning active processes...</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 16px; font-size: 12px; color: var(--text-muted);">
        <span>💡 Terminating heavy background apps (e.g. browsers, background dev servers) frees memory for larger MLX models.</span>
        <button class="btn btn-secondary" onclick="closeMemoryModal()">Close</button>
      </div>
    </div>
  </div>

  <!-- LAUNCH MODAL WITH PRESET PROFILES -->
  <div id="launchModal" class="modal-overlay">
    <div class="modal" style="max-width: 560px;">
      <div class="modal-header" id="modalModelTitle">Start Model</div>
      
      <!-- PRESET PROFILES BAR -->
      <div style="font-size: 13px; font-weight: 600; margin-bottom: 8px; color: #fff;">1-Click Preset Profiles:</div>
      <div style="display: flex; gap: 8px; margin-bottom: 16px;">
        <div class="preset-btn" onclick="applyPreset('vision')">
          <div style="font-weight:600; font-size:12px; color:#a855f7;">👁️ Vision / Multimodal</div>
          <div style="font-size:11px; color:var(--text-muted);">mlx-vlm | Image & Text</div>
        </div>
        <div class="preset-btn" onclick="applyPreset('code')">
          <div style="font-weight:600; font-size:12px; color:#60a5fa;">💻 Coding Agent</div>
          <div style="font-size:11px; color:var(--text-muted);">Temp 0.0 | Max 8192</div>
        </div>
        <div class="preset-btn" onclick="applyPreset('reason')">
          <div style="font-weight:600; font-size:12px; color:#f59e0b;">🧠 Deep Reasoning</div>
          <div style="font-size:11px; color:var(--text-muted);">Temp 0.6 | Thinking ON</div>
        </div>
        <div class="preset-btn" onclick="applyPreset('fast')">
          <div style="font-weight:600; font-size:12px; color:#34d399;">⚡ Fast Chat</div>
          <div style="font-size:11px; color:var(--text-muted);">Temp 0.7 | Thinking OFF</div>
        </div>
      </div>

      <!-- INFERENCE ENGINE SELECTION -->
      <div style="margin-bottom: 14px;">
        <label style="display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; color: var(--text-muted);">
          Inference Server Engine:
        </label>
        <select id="launchEngineSelect" style="width: 100%;">
          <option value="auto">⚡ Auto-Detect Engine (Uses mlx-vlm for Vision, mlx-lm for Text)</option>
          <option value="mlx-vlm">👁️ mlx-vlm (Vision & Multimodal Model Engine)</option>
          <option value="mlx-lm">🤖 mlx-lm (Standard Text LLM Engine)</option>
        </select>
      </div>

      <div style="margin-bottom: 16px; font-size: 13px; color: var(--text-muted);">
        Select launch mode:
      </div>

      <label class="strategy-option">
        <input type="radio" name="launchStrategy" value="swap" checked onclick="toggleCustomFields(false)">
        <div>
          <div class="strategy-title">⚡ Fast Swap (Port 9999)</div>
          <div class="strategy-desc">Stops active model on port 9999 and launches this model immediately. (Recommended)</div>
        </div>
      </label>

      <label class="strategy-option">
        <input type="radio" name="launchStrategy" value="concurrent" onclick="toggleCustomFields(false)">
        <div>
          <div class="strategy-title">🔀 Run Concurrently</div>
          <div class="strategy-desc">Keeps existing model(s) running and launches on new free port.</div>
        </div>
      </label>

      <label class="strategy-option">
        <input type="radio" name="launchStrategy" value="custom" onclick="toggleCustomFields(true)">
        <div>
          <div class="strategy-title">⚙️ Custom Parameters</div>
          <div class="strategy-desc">Configure custom port, temperature, max tokens, and thinking tokens.</div>
        </div>
      </label>

      <div id="customFields" style="display: none; padding: 12px; background: rgba(0,0,0,0.2); border-radius: 8px; margin-bottom: 16px;">
        <div style="display: flex; gap: 12px; margin-bottom: 8px;">
          <input type="number" id="customPort" placeholder="Port" value="9999" style="flex:1;">
          <input type="text" id="customTemp" placeholder="Temp (0.0)" value="0.0" style="flex:1;">
        </div>
        <div style="display: flex; gap: 12px; align-items: center;">
          <input type="number" id="customMaxTokens" placeholder="Max Tokens" value="4096" style="flex:1;">
          <label style="font-size: 12px; color: #fff; cursor: pointer;">
            <input type="checkbox" id="customThinking" checked> Thinking Tokens
          </label>
        </div>
      </div>

      <div style="display: flex; justify-content: flex-end; gap: 12px; margin-top: 20px;">
        <button class="btn btn-secondary" onclick="closeLaunchModal()">Cancel</button>
        <button class="btn btn-primary" onclick="confirmLaunch()">Confirm Launch</button>
      </div>
    </div>
  </div>

  <script>
    let activeServers = [];
    let activeDownloads = [];
    let targetModelForLaunch = '';
    // Filters combine with AND: any subset of {mlx, vision, code} plus one quant level
    let activeTags = new Set(['mlx']);
    let currentFreeRam = 16.0;
    let currentUsableRam = 16.0;
    let currentTotalRam = 0;
    let searchResults = [];
    let searchSeq = 0;
    let currentSearchQuery = '';
    let selectedForCompare = new Set();
    let mlxModelsCache = [];
    let globalConfig = { default_context_window: 0, default_port: 9999, default_host: '127.0.0.1', default_temp: 0.0, default_max_tokens: 4096, auto_sync_agents: true };
    let currentViewMode = localStorage.getItem('mlx_view_mode') || 'grid';

    // ---- Model storage: drives are discovered from the mount table, never configured ----
    let drivesCache = [];
    let currentDriveId = localStorage.getItem('mlx_drive') || '';
    let pendingDownload = null;   // {repoId, subfolder} awaiting a drive choice

    async function loadDrives(force = false) {
      try {
        const res = await fetch('/api/drives' + (force ? '?force=1' : ''));
        const data = await res.json();
        drivesCache = data.drives || [];
        const usable = drivesCache.filter(d => d.connected && !d.disabled);
        if (!currentDriveId || !usable.find(d => d.id === currentDriveId)) {
          currentDriveId = data.default || (usable[0] ? usable[0].id : '');
          if (currentDriveId) localStorage.setItem('mlx_drive', currentDriveId);
        }
        renderDriveSelect();
        renderStorageSummary();
        if (document.getElementById('driveModal').classList.contains('active')) renderDriveModalList();
      } catch (e) {
        console.warn('Could not load drives:', e);
      }
    }

    function driveLabel(d) {
      const icon = d.internal ? '💻' : '💾';
      const warn = d.is_time_machine ? ' ⚠️ Time Machine' : '';
      return `${icon} ${d.name} — ${d.free_gb} GB free${warn}`;
    }

    function renderDriveSelect() {
      const sel = document.getElementById('driveSelect');
      if (!sel) return;
      const usable = drivesCache.filter(d => d.connected && !d.disabled);
      let html = usable.map(d =>
        `<option value="${d.id}" ${d.id === currentDriveId ? 'selected' : ''}>${driveLabel(d)}</option>`
      ).join('');
      html += drivesCache.filter(d => !d.connected)
        .map(d => `<option value="${d.id}" disabled>⏏ ${d.name} — not connected</option>`).join('');
      sel.innerHTML = html || '<option value="">No drive available</option>';
    }

    async function onDriveChange(driveId) {
      currentDriveId = driveId;
      localStorage.setItem('mlx_drive', driveId);
      try {
        await fetch('/api/drives/default', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ drive: driveId })
        });
      } catch (e) { console.warn(e); }
      renderDriveSelect();
      await loadDrives();
    }

    async function excludeDriveBackups(driveId) {
      try {
        const res = await fetch('/api/drives/exclude', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ drive: driveId })
        });
        const data = await res.json();
        if (data && data.drives) drivesCache = data.drives;
        renderStorageSummary();
        renderDriveModalList();
        showToast(data && data.tm_excluded
          ? '🚫 Time Machine now skips model weights on that drive'
          : 'Could not change the Time Machine exclusion');
      } catch (e) {
        showToast('Could not change the Time Machine exclusion');
      }
    }

    function renderStorageSummary() {
      const el = document.getElementById('storageSummary');
      if (!el) return;
      const connected = drivesCache.filter(d => d.connected);
      const totalModels = connected.reduce((n, d) => n + (d.models || 0), 0);
      const off = drivesCache.filter(d => !d.connected && d.models);
      let text = connected.map(d => {
        const bits = `${d.models || 0} model${(d.models || 0) === 1 ? '' : 's'} · ${d.free_gb} GB free`;
        const tm = d.tm_excluded
          ? `<span title="Time Machine skips model weights on this drive">🚫</span>`
          : `<span style="color:#fbbf24; cursor:pointer;" title="This library is inside your Time Machine backup set — click to exclude it" onclick="excludeDriveBackups('${d.id}')">⚠️ backup on</span>`;
        return `${d.internal ? '💻' : '💾'} ${d.name}: ${bits} ${tm}`;
      }).join('   ');
      if (off.length) text += `   ⏏ ${off.map(d => `${d.name} (${d.models})`).join(', ')} not connected`;
      el.innerHTML = `<span style="font-size:12px; color: var(--text-muted);">${text || 'No drives detected'} · ${totalModels} model${totalModels === 1 ? '' : 's'} installed</span>`;
    }

    function renderDriveModalList() {
      const list = document.getElementById('driveModalList');
      if (!list) return;
      const usable = drivesCache.filter(d => d.connected && !d.disabled);
      const chosen = pendingDownload && pendingDownload.chosenId
        ? usable.find(d => d.id === pendingDownload.chosenId)
        : null;
      renderDriveConfirmBar(chosen);
      list.innerHTML = usable.map(d => {
        const sizeNote = pendingDownload && pendingDownload.neededGb
          ? (d.free_gb >= pendingDownload.neededGb
              ? `<span style="color:#34d399;">fits ${pendingDownload.neededGb} GB download</span>`
              : `<span style="color:#f87171;">too small for ${pendingDownload.neededGb} GB</span>`)
          : `${d.free_gb} GB free of ${d.total_gb} GB`;
        const isChosen = chosen && chosen.id === d.id;
        return `
          <div style="display:flex; justify-content:space-between; align-items:center; padding:12px 16px; border-radius:8px; background:${isChosen ? 'rgba(99,102,241,0.18)' : 'rgba(255,255,255,0.04)'}; border:1px solid ${isChosen ? '#6366f1' : 'rgba(255,255,255,0.08)'};">
            <div>
              <strong style="font-size:14px;">${d.internal ? '💻' : '💾'} ${d.name}</strong>
              <div style="font-size:12px; color:var(--text-muted); margin-top:2px;">
                ${d.internal ? 'Internal — keep this free if you can' : d.mount} · ${sizeNote}
                ${d.is_time_machine ? ' · ⚠️ Time Machine destination' : ''}
                ${d.tm_excluded
                  ? ' · <span style="color:#34d399; white-space:nowrap;" title="Time Machine skips this library">🚫 TM off</span>'
                  : ` · <span style="color:#fbbf24; cursor:pointer; white-space:nowrap;" title="This library is inside your Time Machine backup set — click to exclude it" onclick="excludeDriveBackups('${d.id}')">⚠️ TM ON</span>`}
              </div>
            </div>
            <button class="btn ${isChosen ? 'btn-secondary' : 'btn-primary'}" onclick="chooseDrive('${d.id}')">${isChosen ? '✓ Selected' : 'Select'}</button>
          </div>`;
      }).join('') || '<div style="color:var(--text-muted); font-size:13px;">No writable APFS/HFS+ drives detected.</div>';
    }

    function showDrivePicker(repoId, subfolder, info) {
      pendingDownload = {
        repoId,
        subfolder,
        neededGb: (info && info.needed_gb) || 0,
        chosenId: null
      };
      document.getElementById('driveModalSubtitle').innerHTML =
        (info && info.error ? `<span style="color:#f87171;">${info.error}</span><br>` : '') +
        `1. Pick the drive that should hold <strong style="color:#60a5fa;">${repoId}</strong>` +
        `<br>2. Confirm — nothing downloads until you do.`;
      document.getElementById('driveModal').classList.add('active');
      loadDrives().then(renderDriveModalList);
      renderDriveModalList();
    }

    function closeDriveModal() {
      document.getElementById('driveModal').classList.remove('active');
      document.getElementById('driveConfirmBar').style.display = 'none';
      pendingDownload = null;
    }

    function chooseDrive(driveId) {
      if (!pendingDownload) return;
      pendingDownload.chosenId = driveId;
      renderDriveModalList();
      const bar = document.getElementById('driveConfirmBar');
      if (bar) bar.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }

    function renderDriveConfirmBar(chosen) {
      const bar = document.getElementById('driveConfirmBar');
      if (!bar) return;
      if (!chosen || !pendingDownload) {
        bar.style.display = 'none';
        bar.innerHTML = '';
        return;
      }
      const size = pendingDownload.neededGb ? ` (~${pendingDownload.neededGb} GB)` : '';
      const fits = !pendingDownload.neededGb || chosen.free_gb >= pendingDownload.neededGb;
      bar.style.display = 'block';
      bar.innerHTML = `
        <div style="padding: 14px 16px; border-radius: 10px; background: rgba(52,211,153,0.10); border: 1px solid rgba(52,211,153,0.45);">
          <div style="font-size: 13px; color: #f3f4f6; margin-bottom: 10px;">
            Download <strong style="color:#60a5fa;">${pendingDownload.repoId}</strong>${size} to
            <strong style="color:#34d399;">${chosen.internal ? '💻' : '💾'} ${chosen.name}</strong>?
            <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">
              ${fits
                ? `${chosen.free_gb} GB free on that drive`
                : `<span style="color:#f87171;">only ${chosen.free_gb} GB free — the download will be refused</span>`}
              · ${chosen.tm_excluded ? '🚫 not in Time Machine backups' : '⚠️ inside Time Machine backups'}
            </div>
          </div>
          <div style="display: flex; gap: 8px;">
            <button class="btn btn-primary" onclick="confirmDownload()">✓ Confirm download</button>
            <button class="btn btn-secondary" onclick="cancelDriveConfirm()">✕ Cancel</button>
          </div>
        </div>`;
    }

    function cancelDriveConfirm() {
      if (pendingDownload) pendingDownload.chosenId = null;
      renderDriveModalList();
    }

    function confirmDownload() {
      const target = pendingDownload;
      if (!target || !target.chosenId) return;
      closeDriveModal();
      executeDownload(target.repoId, target.subfolder, target.chosenId);
    }


    function setViewMode(mode) {
      currentViewMode = mode;
      localStorage.setItem('mlx_view_mode', mode);
      applyViewMode();
    }

    function applyViewMode() {
      const isList = currentViewMode === 'list';
      const gridIds = ['activeServerContainer', 'fastSwapGrid', 'modelsGrid', 'searchResultsGrid'];
      gridIds.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
          if (isList) el.classList.add('list-view');
          else el.classList.remove('list-view');
        }
      });

      document.querySelectorAll('.view-toggle-btn').forEach(btn => {
        if (btn.getAttribute('data-view') === currentViewMode) {
          btn.classList.add('active');
        } else {
          btn.classList.remove('active');
        }
      });
    }

    let currentModelSort = localStorage.getItem('mlx_model_sort') || 'name-asc';
    let cachedDownloadedModels = [];

    function handleModelSortChange(sortVal) {
      currentModelSort = sortVal;
      localStorage.setItem('mlx_model_sort', sortVal);
      syncSortDropdowns();
      renderModelGrids();
    }

    function syncSortDropdowns() {
      const d1 = document.getElementById('modelSortSelect');
      const d2 = document.getElementById('dashboardModelSortSelect');
      if (d1) d1.value = currentModelSort;
      if (d2) d2.value = currentModelSort;
    }

    function sortModels(models, sortKey) {
      const sorted = [...models];
      switch (sortKey) {
        case 'name-asc':
          sorted.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }));
          break;
        case 'name-desc':
          sorted.sort((a, b) => b.name.localeCompare(a.name, undefined, { sensitivity: 'base' }));
          break;
        case 'size-desc':
          sorted.sort((a, b) => (b.size_gb || 0) - (a.size_gb || 0));
          break;
        case 'size-asc':
          sorted.sort((a, b) => (a.size_gb || 0) - (b.size_gb || 0));
          break;
        case 'date-desc':
          sorted.sort((a, b) => (b.created_ts || 0) - (a.created_ts || 0));
          break;
        case 'date-asc':
          sorted.sort((a, b) => (a.created_ts || 0) - (b.created_ts || 0));
          break;
        default:
          sorted.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }));
      }
      return sorted;
    }

    let currentProcCategoryFilter = 'all';
    let cachedMemoryProcesses = [];

    function openMemoryModal() {
      document.getElementById('memoryModal').classList.add('active');
      loadMemoryProcesses();
    }

    function closeMemoryModal() {
      document.getElementById('memoryModal').classList.remove('active');
    }

    async function loadMemoryProcesses() {
      const tbody = document.getElementById('procTableBody');
      if (!tbody) return;
      tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 24px;">Scanning active processes...</td></tr>';
      
      try {
        const res = await fetch('/api/memory_processes');
        const data = await res.json();
        const stats = data.stats || {};
        cachedMemoryProcesses = data.processes || [];

        if (stats.total_ram_gb) {
          document.getElementById('memModalTotalVal').innerText = `${stats.used_ram_gb} / ${stats.total_ram_gb} GB (${stats.ram_percent}%)`;
          document.getElementById('memModalAppVal').innerText = `${stats.app_ram_gb || '--'} GB`;
          document.getElementById('memModalWiredVal').innerText = `${stats.wired_ram_gb || '--'} GB`;
          document.getElementById('memModalCompVal').innerText = `${stats.compressed_ram_gb || '--'} GB`;
          document.getElementById('memModalAvailVal').innerText = `${stats.free_ram_gb || '--'} GB`;

          const appPct = (stats.app_ram_gb / stats.total_ram_gb) * 100;
          const wiredPct = (stats.wired_ram_gb / stats.total_ram_gb) * 100;
          const compPct = (stats.compressed_ram_gb / stats.total_ram_gb) * 100;
          const availPct = Math.max(0, 100 - appPct - wiredPct - compPct);

          document.getElementById('memSegApp').style.width = `${appPct}%`;
          document.getElementById('memSegWired').style.width = `${wiredPct}%`;
          document.getElementById('memSegComp').style.width = `${compPct}%`;
          document.getElementById('memSegCache').style.width = `${availPct}%`;
        }

        filterMemoryProcesses();
      } catch (e) {
        tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: #f87171; padding: 24px;">Error loading processes: ${e}</td></tr>`;
      }
    }

    function filterMemoryProcesses(category, chipEl) {
      if (category !== undefined) {
        currentProcCategoryFilter = category;
        if (chipEl) {
          document.querySelectorAll('#procFilterChips .chip').forEach(c => c.classList.remove('active'));
          chipEl.classList.add('active');
        }
      }

      const search = (document.getElementById('procSearchInput')?.value || '').toLowerCase().trim();
      const tbody = document.getElementById('procTableBody');
      if (!tbody) return;

      const filtered = cachedMemoryProcesses.filter(p => {
        const matchCat = currentProcCategoryFilter === 'all' || p.category === currentProcCategoryFilter;
        const matchSearch = !search || p.name.toLowerCase().includes(search) || String(p.pid).includes(search) || (p.command && p.command.toLowerCase().includes(search));
        return matchCat && matchSearch;
      });

      if (filtered.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 24px;">No matching processes found.</td></tr>';
        return;
      }

      tbody.innerHTML = '';
      filtered.forEach(p => {
        let catClass = 'cat-app';
        if (p.category === 'AI Model') catClass = 'cat-ai';
        else if (p.category === 'Browser') catClass = 'cat-browser';
        else if (p.category === 'Dev Tool') catClass = 'cat-dev';
        else if (p.category === 'macOS System') catClass = 'cat-system';

        let actionBtn = '';
        if (p.is_mlx) {
          actionBtn = `<button class="btn btn-danger" style="padding: 5px 10px; font-size: 11px;" onclick="killProcess(${p.pid}, '${p.name.replace(/'/g, "\\'")}', true)">🛑 Stop Server</button>`;
        } else if (p.is_system) {
          actionBtn = `<button class="btn btn-secondary" style="padding: 5px 10px; font-size: 11px; opacity: 0.5; cursor: not-allowed;" disabled title="Protected macOS system process">🔒 Protected</button>`;
        } else {
          actionBtn = `<button class="btn btn-danger" style="padding: 5px 10px; font-size: 11px; background: rgba(239, 68, 68, 0.2); border: 1px solid rgba(239, 68, 68, 0.4); color: #fca5a5;" onclick="killProcess(${p.pid}, '${p.name.replace(/'/g, "\\'")}', false)">❌ End Process</button>`;
        }

        tbody.innerHTML += `
          <tr>
            <td>
              <strong style="color: #fff; font-size: 13px;">${p.name}</strong>
              <div style="font-size: 11px; color: var(--text-muted); max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${p.command}">${p.command}</div>
            </td>
            <td><span class="cat-badge ${catClass}">${p.category}</span></td>
            <td><code style="color: #94a3b8; font-size: 12px;">${p.pid}</code></td>
            <td><strong style="color: #34d399;">${p.rss_str}</strong> <span style="font-size: 11px; color: var(--text-muted);">(${p.mem_pct}%)</span></td>
            <td style="text-align: right;">${actionBtn}</td>
          </tr>`;
      });
    }

    async function killProcess(pid, name, isMlx) {
      const msg = isMlx 
        ? `Stop MLX Server for '${name}' (PID ${pid})?`
        : `Terminate process '${name}' (PID ${pid}) to free memory?`;

      if (!confirm(msg)) return;

      try {
        const res = await fetch('/api/kill_process', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ pid: pid })
        });
        const data = await res.json();
        if (data.error) {
          alert('Error: ' + data.error);
        } else {
          await loadMemoryProcesses();
          updateStatus();
        }
      } catch (e) {
        alert('Failed to kill process: ' + e);
      }
    }

    function switchTab(tabId) {
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
      document.getElementById('tab-' + tabId).classList.add('active');
      
      const activeNav = Array.from(document.querySelectorAll('.nav-btn')).find(b => b.getAttribute('onclick').includes(tabId));
      if (activeNav) activeNav.classList.add('active');

      if (tabId === 'models' || tabId === 'dashboard') loadModels();
      if (tabId === 'logs') loadLogs();
      if (tabId === 'test') loadTestServers();
      if (tabId === 'search') doSearch();
    }

    // ── Hugging Face account (device flow or pasted read token) ──────────────
    let hfAuth = { signed_in: false };
    let hfPollTimer = null;
    let hfDeviceCode = null;

    async function loadHfAuth() {
      try {
        const res = await fetch('/api/hf/auth');
        hfAuth = await res.json();
      } catch (e) { hfAuth = { signed_in: false }; }
      renderHfAuth();
      return hfAuth;
    }

    function renderHfAuth() {
      const btn = document.getElementById('hfAuthBtn');
      if (!btn) return;
      if (hfAuth && hfAuth.signed_in) {
        btn.innerHTML = `🤗 ${hfAuth.user || 'Signed in'}`;
        btn.style.borderColor = 'rgba(52, 211, 153, 0.5)';
        btn.title = `Signed in to Hugging Face as ${hfAuth.user} (secret in ${hfAuth.storage})`;
      } else {
        btn.innerHTML = '🤗 Sign in';
        btn.style.borderColor = '';
        btn.title = 'Sign in to Hugging Face for higher rate limits and gated repos';
      }
      const signedOut = document.getElementById('hfSignedOut');
      const signedIn = document.getElementById('hfSignedIn');
      if (signedOut && signedIn) {
        const on = !!(hfAuth && hfAuth.signed_in);
        signedOut.style.display = on ? 'none' : 'block';
        signedIn.style.display = on ? 'block' : 'none';
        if (on) {
          document.getElementById('hfUserName').textContent = hfAuth.user || 'unknown';
          document.getElementById('hfUserMeta').textContent =
            `${hfAuth.kind === 'oauth' ? 'OAuth device login' : 'access token'} · secret stored in ${hfAuth.storage === 'keychain' ? 'macOS Keychain' : 'a 0600 file'}`;
        }
      }
    }

    function openHfModal() {
      document.getElementById('hfModal').classList.add('active');
      stopHfPolling();
      loadHfAuth();
    }

    function closeHfModal() {
      document.getElementById('hfModal').classList.remove('active');
      stopHfPolling();
    }

    function toggleHfTokenForm() {
      const form = document.getElementById('hfTokenForm');
      form.style.display = form.style.display === 'none' ? 'block' : 'none';
    }

    function hfShowError(msg) {
      const el = document.getElementById('hfError');
      if (!el) return;
      if (!msg) { el.style.display = 'none'; return; }
      el.textContent = msg;
      el.style.display = 'block';
    }

    function copyHfCode() {
      const code = document.getElementById('hfUserCode').textContent.trim();
      if (!code) return;
      navigator.clipboard.writeText(code).then(() => showToast(`Copied ${code}`));
    }

    function stopHfPolling() {
      if (hfPollTimer) { clearTimeout(hfPollTimer); hfPollTimer = null; }
    }

    async function startHfLogin() {
      hfShowError('');
      const btn = document.getElementById('hfLoginBtn');
      btn.disabled = true;
      btn.textContent = 'Starting…';
      try {
        const res = await fetch('/api/hf/device/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
        const data = await res.json();
        if (!data.ok) throw new Error(data.error || 'could not start the login');
        hfDeviceCode = data.device_code;
        document.getElementById('hfDeviceBox').style.display = 'block';
        const link = document.getElementById('hfDeviceLink');
        link.href = `${data.verification_uri}?user_code=${encodeURIComponent(data.user_code)}`;
        link.textContent = link.href;
        document.getElementById('hfUserCode').textContent = data.user_code;
        document.getElementById('hfDeviceStatus').textContent = 'Waiting for approval…';
        btn.textContent = 'Waiting…';
        pollHfLogin(Math.max(3, data.interval || 5) * 1000, Date.now() + (data.expires_in || 300) * 1000);
      } catch (e) {
        hfShowError(e.message || String(e));
        btn.disabled = false;
        btn.textContent = '🤗 Sign in with Hugging Face';
      }
    }

    function pollHfLogin(interval, deadline) {
      stopHfPolling();
      hfPollTimer = setTimeout(async () => {
        if (Date.now() > deadline) {
          document.getElementById('hfDeviceStatus').textContent = 'Code expired — start again.';
          const btn = document.getElementById('hfLoginBtn');
          btn.disabled = false; btn.textContent = '🤗 Sign in with Hugging Face';
          return;
        }
        try {
          const res = await fetch('/api/hf/device/poll', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ device_code: hfDeviceCode })
          });
          const data = await res.json();
          if (data.status === 'accepted') {
            document.getElementById('hfDeviceStatus').textContent = 'Approved ✓';
            await loadHfAuth();
            showToast(`Signed in to Hugging Face as ${data.user || 'you'}`);
            const btn = document.getElementById('hfLoginBtn');
            btn.disabled = false; btn.textContent = '🤗 Sign in with Hugging Face';
            setTimeout(closeHfModal, 600);
            return;
          }
          if (data.status === 'expired') {
            document.getElementById('hfDeviceStatus').textContent = 'Code expired — start again.';
            const btn = document.getElementById('hfLoginBtn');
            btn.disabled = false; btn.textContent = '🤗 Sign in with Hugging Face';
            return;
          }
          if (data.status === 'denied') {
            document.getElementById('hfDeviceStatus').textContent = 'Access denied.';
            const btn = document.getElementById('hfLoginBtn');
            btn.disabled = false; btn.textContent = '🤗 Sign in with Hugging Face';
            return;
          }
          if (data.status === 'error') hfShowError(data.error || 'login failed');
          document.getElementById('hfDeviceStatus').textContent = 'Waiting for approval…';
          pollHfLogin(data.status === 'slow_down' ? interval + 5000 : interval, deadline);
        } catch (e) {
          pollHfLogin(interval, deadline);
        }
      }, interval);
    }

    async function submitHfToken() {
      const input = document.getElementById('hfTokenInput');
      const token = (input.value || '').trim();
      if (!token) { hfShowError('Paste a token first.'); return; }
      hfShowError('');
      const res = await fetch('/api/hf/token', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token })
      });
      const data = await res.json();
      if (!data.ok) { hfShowError(data.error || 'that token was rejected'); return; }
      input.value = '';
      document.getElementById('hfTokenForm').style.display = 'none';
      await loadHfAuth();
      showToast(`Signed in to Hugging Face as ${data.user}`);
      setTimeout(closeHfModal, 500);
    }

    async function logoutHf() {
      if (!confirm('Sign out of Hugging Face and remove the stored token from this Mac?')) return;
      await fetch('/api/hf/logout', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
      await loadHfAuth();
      showToast('Signed out of Hugging Face');
    }

    function renderTokenGuide() {
      const total = Math.round(currentTotalRam || 0);
      const macEl = document.getElementById('tokenGuideMac');
      const body = document.getElementById('tokenGuideBody');
      const note = document.getElementById('tokenGuideNote');
      if (!body) return;

      let rows, tierNote;
      if (!total) {
        rows = [
          ['2B – 8B', '4096 – 8192', 'Light (~0.5 – 1 GB)', '#34d399'],
          ['14B – 35B', '4096', 'Moderate (~1.5 – 3 GB)', '#fbbf24'],
          ['70B+', '2048 – 8192', 'High (~4 – 10 GB)', '#f87171'],
        ];
        tierNote = 'Machine RAM unknown — using conservative defaults.';
        if (macEl) macEl.innerText = 'your Mac';
      } else if (total >= 96) {
        rows = [
          ['2B – 8B', '8192 – 16384', 'Light (~1 GB)', '#34d399'],
          ['14B – 35B', '8192', 'Moderate (~3 GB)', '#34d399'],
          ['70B+', '4096 – 8192', 'High (~6 – 10 GB)', '#fbbf24'],
        ];
        tierNote = `Plenty of headroom at ${total} GB — long agentic runs are fine, but budget ~1 GB of KV per 8k tokens on 70B+ models.`;
        if (macEl) macEl.innerText = `your ${total} GB Mac`;
      } else if (total >= 48) {
        rows = [
          ['2B – 8B', '8192', 'Light (~1 GB)', '#34d399'],
          ['14B – 35B', '4096 – 8192', 'Moderate (~2 – 4 GB)', '#fbbf24'],
          ['70B+', '2048 – 4096', 'High (~8 GB+)', '#f87171'],
        ];
        tierNote = `At ${total} GB a 70B model leaves little room — keep the context short and close other apps.`;
        if (macEl) macEl.innerText = `your ${total} GB Mac`;
      } else if (total >= 24) {
        rows = [
          ['2B – 8B', '4096 – 8192', 'Light (~0.5 – 1 GB)', '#34d399'],
          ['14B – 35B', '2048 – 4096', 'Moderate (~2 – 3 GB)', '#fbbf24'],
          ['70B+', 'will not fit', 'Exceeds unified memory', '#f87171'],
        ];
        tierNote = `At ${total} GB stick to 8B and below for long contexts; 14B+ only at short context.`;
        if (macEl) macEl.innerText = `your ${total} GB Mac`;
      } else {
        rows = [
          ['2B – 8B', '4096', 'Light (~0.5 GB)', '#34d399'],
          ['14B – 35B', '2048', 'Moderate (~1.5 – 2 GB)', '#fbbf24'],
          ['70B+', 'will not fit', 'Exceeds unified memory', '#f87171'],
        ];
        tierNote = `At ${total} GB prefer 4-bit models at 8B or less, and keep context modest.`;
        if (macEl) macEl.innerText = `your ${total} GB Mac`;
      }

      body.innerHTML = rows.map(([size, toks, kv, color], i) => `
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);${i === rows.length - 1 ? 'border-bottom: none;' : ''}">
          <td style="padding: 4px;"><strong>${size}</strong></td>
          <td style="padding: 4px; color: ${color};">${toks}</td>
          <td style="padding: 4px;">${kv}</td>
        </tr>`).join('');

      if (note) {
        note.innerHTML = `${tierNote} KV cache scales with context length × layers × KV heads, assuming 4-bit weights and fp16 KV: roughly 1 GB per 8k tokens on an 8B model, 2–3 GB per 8k on 30B, 4–6 GB per 8k on 70B+.`;
      }
    }

    function renderContextNote() {
      const note = document.getElementById('cfgContextNote');
      if (!note) return;
      const big = mlxModelsCache
        .map(m => ({ name: (m.repo_id || m.name || '').split('/').pop(), ctx: m.context_window || 0 }))
        .filter(x => x.ctx)
        .sort((a, b) => b.ctx - a.ctx);
      if (!big.length) {
        note.innerText = 'Each model\\'s native context is read from its config.json when it starts.';
        return;
      }
      const top = big[0];
      const sel = parseInt(document.getElementById('cfgContextSelect')?.value || '0') || 0;
      const eff = sel ? Math.min(sel, top.ctx) : top.ctx;
      note.innerText = `Largest installed model (${top.name}) supports ${top.ctx.toLocaleString()} tokens — ` +
        (sel ? `this cap starts servers with ${eff.toLocaleString()} and hands the same number to DeepSeek Harness.`
             : 'no cap, so servers and Harness use that number.');
    }

    async function loadHarnessState() {
      const note = document.getElementById('harnessSyncNote');
      try {
        const res = await fetch('/api/harness');
        const d = await res.json();
        const listed = (d.listed || []).length, installed = (d.installed || []).length;
        let text = `${listed} model${listed === 1 ? '' : 's'} listed · ${installed} installed`;
        if ((d.missing || []).length) text += ` · ${d.missing.length} downloaded but not listed`;
        if ((d.stale || []).length) text += ` · ${d.stale.length} listed but no longer installed`;
        if (d.in_sync) text += ' · in sync ✓';
        note.innerText = text;
        note.style.color = d.in_sync ? '#34d399' : '#fbbf24';
        const btn = document.getElementById('harnessSyncBtn');
        if (btn) btn.innerText = d.in_sync ? '✓ Synced' : '🔄 Sync now';
      } catch (e) {
        note.innerText = 'Could not read the DeepSeek Harness profile.';
        note.style.color = '#f87171';
      }
    }

    async function syncHarnessModels() {
      const btn = document.getElementById('harnessSyncBtn');
      const note = document.getElementById('harnessSyncNote');
      if (btn) { btn.disabled = true; btn.innerText = '⏳ Syncing…'; }
      try {
        const res = await fetch('/api/harness/sync', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
        const d = await res.json();
        if (d.ok && !d.changed) {
          showToast('✓ Harness model list is already up to date');
        } else if (d.ok) {
          const added = (d.added || []).length, removed = (d.removed || []).length;
          showToast(`✓ Synced ${d.installed} model${d.installed === 1 ? '' : 's'} into DeepSeek Harness${added ? ` (+${added})` : ''}${removed ? ` (-${removed})` : ''} — restart Harness to see them`);
        } else {
          showToast('⚠️ ' + (d.error || 'sync failed'));
        }
      } catch (e) {
        showToast('⚠️ Could not write the Harness profile');
      }
      if (btn) btn.disabled = false;
      loadHarnessState();
    }

    async function openConfigModal() {
      const res = await fetch('/api/config');
      globalConfig = await res.json();
      document.getElementById('cfgPortInput').value = globalConfig.default_port || 9999;
      document.getElementById('cfgHostSelect').value = globalConfig.default_host || '127.0.0.1';
      document.getElementById('cfgTempInput').value = globalConfig.default_temp || 0.0;
      document.getElementById('cfgMaxTokensInput').value = globalConfig.default_max_tokens || 4096;
      const ctxSel = document.getElementById('cfgContextSelect');
      if (ctxSel) ctxSel.value = String(globalConfig.default_context_window || 0);
      renderContextNote();
      document.getElementById('cfgAutoSyncCheck').checked = globalConfig.auto_sync_agents !== false;
      renderTokenGuide();
      loadHarnessState();
      document.getElementById('configModal').classList.add('active');
    }

    function closeConfigModal() {
      document.getElementById('configModal').classList.remove('active');
    }

    async function saveConfigModal() {
      const port = parseInt(document.getElementById('cfgPortInput').value) || 9999;
      const host = document.getElementById('cfgHostSelect').value;
      const temp = parseFloat(document.getElementById('cfgTempInput').value) || 0.0;
      const maxTokens = parseInt(document.getElementById('cfgMaxTokensInput').value) || 4096;
      const autoSync = document.getElementById('cfgAutoSyncCheck').checked;

      await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ default_port: port, default_host: host, default_temp: temp, default_max_tokens: maxTokens, default_context_window: parseInt(document.getElementById('cfgContextSelect')?.value || '0') || 0, auto_sync_agents: autoSync })
      });

      closeConfigModal();
      alert(`Settings saved! Default port set to ${port} on ${host}.`);
      updateStatus();
    }

    function toggleCompareSelection(repoId) {
      if (selectedForCompare.has(repoId)) {
        selectedForCompare.delete(repoId);
      } else {
        selectedForCompare.add(repoId);
      }
      updateCompareBar();
    }

    function clearCompareSelection() {
      selectedForCompare.clear();
      document.querySelectorAll('.compare-check').forEach(el => el.checked = false);
      updateCompareBar();
    }

    function updateCompareBar() {
      const bar = document.getElementById('floatingCompareBar');
      const label = document.getElementById('compareCountLabel');
      if (selectedForCompare.size > 0) {
        bar.style.display = 'flex';
        label.innerText = `Selected Models (${selectedForCompare.size}):`;
      } else {
        bar.style.display = 'none';
      }
    }

    async function runAiComparison() {
      if (selectedForCompare.size === 0) return;
      const repos = Array.from(selectedForCompare);
      
      document.getElementById('compareModal').classList.add('active');
      const content = document.getElementById('compareModalContent');
      content.innerHTML = '<div style="color: var(--text-muted); font-size: 14px;">🤖 Antigravity AI is evaluating model architecture, quantization precision, and RAM requirements...</div>';

      try {
        const res = await fetch('/api/compare_models', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ repos: repos })
        });
        const data = await res.json();
        const comparisons = data.comparisons || [];
        
        let gridHtml = `
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 16px; margin-top: 16px;">`;

        comparisons.forEach(c => {
          const hfUrl = `https://huggingface.co/${c.repo_id}`;
          gridHtml += `
            <div class="card" style="border-color: rgba(99, 102, 241, 0.4); background: rgba(15, 23, 42, 0.8);">
              <div class="card-title" style="color: #60a5fa; font-size: 15px;">${c.repo_id}</div>
              
              <div style="margin-bottom: 12px;">
                <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">Quantization & Precision</div>
                <div style="font-weight: 700; font-size: 13px; color: #f3f4f6;">${c.quant}</div>
                <div style="font-size: 11px; color: #94a3b8; margin-top: 2px;">${c.quant_desc}</div>
              </div>

              <div style="margin-bottom: 12px;">
                <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">Specialization</div>
                <div style="font-weight: 600; font-size: 13px; color: #a7f3d0;">${c.specialization}</div>
              </div>

              <div style="margin-bottom: 12px;">
                <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">Mac RAM Compatibility</div>
                <div>${c.fit_status}</div>
              </div>

              <div style="padding: 10px; background: rgba(0,0,0,0.3); border-radius: 8px; border: 1px solid var(--card-border); margin-bottom: 14px;">
                <div style="font-size: 11px; font-weight: 700; color: #fbbf24; margin-bottom: 2px;">🌟 AI VERDICT:</div>
                <div style="font-size: 12px; color: #e2e8f0;">${c.verdict}</div>
              </div>

              <div style="display: flex; gap: 8px;">
                <button class="btn btn-primary" style="flex: 1;" onclick="closeCompareModal(); downloadRepo('${c.repo_id}')">💾 Download…</button>
                <a href="${hfUrl}" target="_blank" class="btn btn-secondary" style="padding: 9px 12px; font-size: 12px;">🔗 HF Page</a>
              </div>
            </div>`;
        });

        gridHtml += `</div>`;
        content.innerHTML = gridHtml;

      } catch (e) {
        content.innerHTML = `<div style="color: #f87171;">Error running AI comparison: ${e}</div>`;
      }
    }

    function closeCompareModal() {
      document.getElementById('compareModal').classList.remove('active');
    }

    async function updateStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        activeServers = data.servers || [];
        activeDownloads = data.downloads || [];
        globalConfig = data.config || globalConfig;
        
        // Render Rich Download Progress Tracker Banner with Pause / Resume / Kick / Cancel Controls
        const dlBanner = document.getElementById('liveDownloadBanner');
        if (activeDownloads.length > 0) {
          dlBanner.style.display = 'block';
          dlBanner.innerHTML = '';
          activeDownloads.forEach(d => {
            const isPaused = d.status === 'PAUSED';
            const displayName = d.display_name || d.repo_id;
            const subfolder = d.subfolder || '';
            dlBanner.innerHTML += `
              <div class="card" style="border-color: ${isPaused ? '#f59e0b' : '#6366f1'}; background: ${isPaused ? 'rgba(245, 158, 11, 0.1)' : 'rgba(99, 102, 241, 0.12)'};">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                  <div style="flex: 1; margin-right: 16px;">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                      <span class="status-pill ${isPaused ? 'paused' : 'downloading'}">
                        <span class="status-dot ${isPaused ? '' : 'pulse'}"></span> ${isPaused ? '⏸️ DOWNLOAD PAUSED (SAVED)' : '⚡ RUST ACCELERATED HF DOWNLOAD'}
                      </span>
                      <strong style="font-size: 15px; color: #fff;">${displayName}</strong>
                    </div>
                    <div style="font-size: 13px; color: var(--text-muted); margin-bottom: 8px;">
                      ${isPaused ? 'Saved on Disk' : 'PID: ' + d.pid} | Downloaded: <strong style="color: #34d399;">${d.downloaded_size}</strong> of <strong>${d.total_size}</strong> (<strong style="color: #93c5fd;">${d.percent}%</strong>)
                      | Speed: <strong style="color: ${isPaused ? '#fbbf24' : '#60a5fa'};">${d.speed}</strong> | ETA: <strong style="color: ${isPaused ? '#fbbf24' : '#f472b6'};">${d.eta}</strong>
                    </div>
                    <div style="width: 100%; height: 10px; background: rgba(255,255,255,0.1); border-radius: 5px; overflow: hidden;">
                      <div style="height: 100%; width: ${d.percent}%; background: ${isPaused ? 'linear-gradient(90deg, #f59e0b, #fbbf24)' : 'linear-gradient(90deg, #6366f1, #34d399)'}; transition: width 0.4s ease;"></div>
                    </div>
                  </div>
                  <div style="display: flex; gap: 8px; flex-shrink: 0;">
                    ${isPaused ? `
                      <button class="btn btn-primary" onclick="restartDownload('${d.pid}', '${d.repo_id}', '${subfolder}')">▶️ Resume Download</button>
                    ` : `
                      <button class="btn btn-warning" onclick="pauseDownload('${d.pid}', '${d.repo_id}', '${subfolder}')">⏸️ Pause</button>
                      <button class="btn btn-primary" onclick="restartDownload('${d.pid}', '${d.repo_id}', '${subfolder}')" title="Restarts connection with Rust multi-threaded acceleration">🔄 Kick / Re-connect</button>
                    `}
                    <button class="btn btn-danger" onclick="cancelDownload('${d.pid}', '${d.repo_id}', '${subfolder}')">🛑 Cancel</button>
                  </div>
                </div>
              </div>`;
          });
        } else {
          dlBanner.style.display = 'none';
          dlBanner.innerHTML = '';
        }

        const stats = data.stats || {};
        if (stats.free_ram_gb) currentFreeRam = stats.free_ram_gb;
        if (stats.usable_ram_gb) {
          currentUsableRam = stats.usable_ram_gb;
          currentTotalRam = stats.total_ram_gb || currentTotalRam;
          syncRamBudgetLabel();
        }

        if (stats.total_ram_gb) {
          document.getElementById('htopRamVal').innerText = stats.used_ram_gb + ' / ' + stats.total_ram_gb + ' GB (' + stats.ram_percent + '%)';
          document.getElementById('htopRamFill').style.width = stats.ram_percent + '%';
          if (document.getElementById('htopRamMeta')) {
            document.getElementById('htopRamMeta').innerText = stats.free_ram_gb + ' GB Available (' + (stats.cached_ram_gb || 0) + ' GB Cache)';
          }
          document.getElementById('htopMlxVal').innerText = stats.mlx_ram_gb + ' GB';
          document.getElementById('htopMlxMeta').innerText = stats.active_models + ' Active Model(s)';
          document.getElementById('htopCpuVal').innerText = stats.cpu_percent + '%';
          document.getElementById('htopCpuFill').style.width = Math.min(stats.cpu_percent, 100) + '%';
        }

        const statusHeader = document.getElementById('statusHeader');
        const serverContainer = document.getElementById('activeServerContainer');

        if (activeServers.length > 0) {
          const readyCount = activeServers.filter(s => s.status === 'READY').length;
          statusHeader.innerHTML = `
            <div class="status-pill ${readyCount > 0 ? 'ready' : 'none'}">
              <span class="status-dot"></span> ${activeServers.length} Active MLX Server(s) (${readyCount} Ready)
            </div>`;
          
          serverContainer.innerHTML = '';
          activeServers.forEach(s => {
            const isReady = s.status === 'READY';
            const isVisEngine = s.engine === 'mlx-vlm' || s.is_vision;
            const engineBadge = isVisEngine 
              ? `<span style="font-size: 11px; padding: 2px 8px; border-radius: 6px; background: rgba(168, 85, 247, 0.25); color: #d8b4fe; border: 1px solid rgba(168, 85, 247, 0.4); font-weight: 600;">👁️ mlx-vlm (Vision)</span>`
              : `<span style="font-size: 11px; padding: 2px 8px; border-radius: 6px; background: rgba(59, 130, 246, 0.2); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.3); font-weight: 600;">⚡ mlx-lm (Text)</span>`;

            serverContainer.innerHTML += `
              <div class="card">
                <div class="card-main">
                  <div class="card-header-row">
                    <div class="card-title">${s.model}</div>
                    <div style="display: flex; gap: 6px; align-items: center; flex-wrap: wrap;">
                      ${engineBadge}
                      <span class="status-pill ${isReady ? 'ready' : 'none'}">
                        <span class="status-dot ${isReady ? '' : 'pulse'}"></span> ${s.status}
                      </span>
                    </div>
                  </div>
                  <div class="card-meta">
                    <span>Port: <strong style="color: #f3f4f6;">${s.port}</strong></span>
                    <span>Host: <strong style="color: #f3f4f6;">${s.host}</strong></span>
                    <span>PID: <strong style="color: #f3f4f6;">${s.pid}</strong></span>
                    <span>Endpoint: <code style="color: #60a5fa;">http://${s.host}:${s.port}/v1</code></span>
                  </div>
                </div>
                <div class="card-actions">
                  <button class="btn btn-danger" onclick="stopServer('${s.pid}')">🛑 Stop Server</button>
                  <button class="btn btn-secondary" onclick="switchTab('test'); loadTestServers();">🧪 Test API</button>
                </div>
              </div>`;
          });
        } else {
          statusHeader.innerHTML = `
            <div class="status-pill none">
              <span class="status-dot"></span> 0 Active MLX Servers
            </div>`;
          serverContainer.innerHTML = `<div class="card" style="grid-column: 1 / -1; width: 100%; text-align: center; color: var(--text-muted); padding: 18px;">No model currently running. Click "Start Model" on any downloaded model below!</div>`;
        }

        applyViewMode();
        loadTestServers();
      } catch (e) {
        console.error(e);
      }
    }

    async function loadTestServers() {
      const select = document.getElementById('testServerSelect');
      if (!select) return;

      const currentVal = select.value;
      select.innerHTML = '';

      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const servers = data.servers || [];

        if (servers.length === 0) {
          select.innerHTML = '<option value="">(No active MLX servers running)</option>';
          return;
        }

        servers.forEach(s => {
          const opt = document.createElement('option');
          opt.value = s.port;
          opt.setAttribute('data-model', s.model);
          opt.setAttribute('data-host', s.host || '127.0.0.1');
          const engTag = (s.engine === 'mlx-vlm' || s.is_vision) ? '👁️ Vision' : '⚡ Text';
          opt.innerText = `[Port ${s.port} - ${engTag}] ${s.model} (${s.status})`;
          if (String(s.port) === String(currentVal)) {
            opt.selected = true;
          }
          select.appendChild(opt);
        });
      } catch (e) {
        console.error(e);
      }
    }

    function getRamBadge(estGb, budgetGb, sizeKnown, sizePending) {
      const budget = Number(budgetGb) || Number(currentUsableRam) || 0;
      if (!estGb) {
        const txt = sizePending ? '⏳ Checking size…' : '❔ Size unknown';
        return `<span class="ram-badge" style="background: rgba(255,255,255,0.06); color: var(--text-muted); border-color: rgba(255,255,255,0.15);">${txt}</span>`;
      }
      const est = Number(estGb).toFixed(1);
      const approx = sizeKnown ? '' : ' approx';
      if (budget && estGb <= budget * 0.8) {
        return `<span class="ram-badge fit">🟢 Fits Smoothly (${est} GB RAM${approx})</span>`;
      } else if (budget && estGb <= budget) {
        return `<span class="ram-badge warn">🟡 Tight Fit (${est} GB RAM${approx})</span>`;
      } else {
        return `<span class="ram-badge heavy">🔴 Too Large (${est} GB > ${budget} GB usable${approx})</span>`;
      }
    }

    function renderModelGrids() {
      const fastSwapGrid = document.getElementById('fastSwapGrid');
      const modelsGrid = document.getElementById('modelsGrid');
      if (!fastSwapGrid || !modelsGrid) return;

      fastSwapGrid.innerHTML = '';
      modelsGrid.innerHTML = '';

      if (cachedDownloadedModels.length === 0) {
        const emptyHtml = `<div class="card" style="grid-column: 1 / -1; width: 100%; text-align: center; color: var(--text-muted); padding: 24px;">No models downloaded yet. Use the "Search Hugging Face" tab to download models!</div>`;
        fastSwapGrid.innerHTML = emptyHtml;
        modelsGrid.innerHTML = emptyHtml;
        applyViewMode();
        return;
      }

      const models = sortModels(cachedDownloadedModels, currentModelSort);

      models.forEach(m => {
        const isRunning = activeServers.some(s => s.model === m.name);
        const runningServer = activeServers.find(s => s.model === m.name);

        let badgeColor = m.is_vision ? '#a855f7' : (m.supported ? '#6366f1' : '#f59e0b');
        const tagBadge = m.arch_tag ? `<span style="font-size:11px; padding: 2px 8px; border-radius: 12px; background: rgba(255,255,255,0.08); color: ${badgeColor}; margin-left: 6px;">${m.arch_tag}</span>` : '';
        const visionBadge = m.is_vision ? `<span class="ram-badge fit" style="background: rgba(168, 85, 247, 0.2); color: #d8b4fe; border-color: rgba(168, 85, 247, 0.4);">👁️ Vision Ready</span>` : '';
        const ramBadge = getRamBadge(m.size_gb ? (m.size_gb * 1.08 + 2.0) : 0, currentUsableRam, true);
        const isChecked = selectedForCompare.has(m.name) ? 'checked' : '';
        const hfUrl = m.base_name ? `https://huggingface.co/${m.base_name}${m.subfolder ? '/tree/main/' + m.subfolder : ''}` : `https://huggingface.co/${m.name}`;

        const fastSwapCard = `
          <div class="card">
            <div class="card-main">
              <div class="card-header-row">
                <div class="card-title">${m.name} ${tagBadge}</div>
                <div style="display: flex; gap: 6px; align-items: center; flex-wrap: wrap;">
                  ${visionBadge}
                  ${ramBadge}
                </div>
                <label class="compare-label" style="font-size: 11px; cursor: pointer; color: var(--text-muted); display: inline-flex; align-items: center; gap: 4px; margin-left: 6px;">
                  <input type="checkbox" class="compare-check" ${isChecked} onchange="toggleCompareSelection('${m.name}')"> Compare
                </label>
              </div>
              <div class="card-meta">
                <span>📦 Size: <strong style="color: #f3f4f6;">${m.size}</strong></span>
                ${m.context_window ? `<span>🧠 Context: <strong style="color: #f3f4f6;">${Math.round(m.context_window / 1024)}K tokens</strong></span>` : ''}
                ${m.date_added ? `<span>📅 Added: <strong style="color: #f3f4f6;">${m.date_added}</strong></span>` : ''}
              </div>
            </div>
            <div class="card-actions">
              <button class="btn ${isRunning ? 'btn-secondary' : 'btn-primary'}" onclick="openLaunchModal('${m.name}')">
                ${isRunning ? `✓ Active on Port ${runningServer.port}` : '🚀 Start Model'}
              </button>
              <a href="${hfUrl}" target="_blank" class="btn btn-secondary" style="padding: 9px 12px; font-size: 12px;">🔗 HF Page</a>
            </div>
          </div>`;
        fastSwapGrid.innerHTML += fastSwapCard;

        const modelCard = `
          <div class="card">
            <div class="card-main">
              <div class="card-header-row">
                <div class="card-title">${m.name} ${tagBadge}</div>
                <div style="display: flex; gap: 6px; align-items: center; flex-wrap: wrap;">
                  ${visionBadge}
                  ${ramBadge}
                </div>
                <label class="compare-label" style="font-size: 11px; cursor: pointer; color: var(--text-muted); display: inline-flex; align-items: center; gap: 4px; margin-left: 6px;">
                  <input type="checkbox" class="compare-check" ${isChecked} onchange="toggleCompareSelection('${m.name}')"> Compare
                </label>
              </div>
              <div class="card-meta">
                <span>📦 Size: <strong style="color: #f3f4f6;">${m.size}</strong></span>
                ${m.context_window ? `<span>🧠 Context: <strong style="color: #f3f4f6;">${Math.round(m.context_window / 1024)}K tokens</strong></span>` : ''}
                ${m.drive_name ? `<span>${m.drive_id ? '💾' : '🧠'} Drive: <strong style="color: #34d399;">${m.drive_name}</strong></span>` : ''}
                ${m.date_added ? `<span>📅 Added: <strong style="color: #f3f4f6;">${m.date_added}</strong></span>` : ''}
                <span class="path-meta" title="${m.path}">📁 <span style="opacity: 0.85;">${m.path}</span></span>
              </div>
            </div>
            <div class="card-actions">
              <button class="btn btn-primary" onclick="openLaunchModal('${m.name}')">🚀 Start Model</button>
              <a href="${hfUrl}" target="_blank" class="btn btn-secondary" style="padding: 9px 12px; font-size: 12px;">🔗 HF Page</a>
              <button class="btn btn-danger" onclick="deleteModel('${m.delete_target}')">🗑️ Delete</button>
            </div>
          </div>`;
        modelsGrid.innerHTML += modelCard;
      });
      applyViewMode();
    }

    async function loadModels() {
      try {
        const res = await fetch('/api/models');
        const data = await res.json();
        cachedDownloadedModels = data.models || [];
        mlxModelsCache = cachedDownloadedModels;
        syncSortDropdowns();
        renderModelGrids();
      } catch (e) {
        console.error(e);
      }
    }

    function toggleTag(tag) {
      if (activeTags.has(tag)) {
        activeTags.delete(tag);
      } else {
        activeTags.add(tag);
      }
      syncChipState();
      doSearch();
    }

    function clearTags() {
      activeTags.clear();
      syncChipState();
      doSearch();
    }

    function syncChipState() {
      ['mlx', 'vision', 'code'].forEach(t => {
        const el = document.getElementById('chip-' + t);
        if (el) el.classList.toggle('active', activeTags.has(t));
      });
      const allChip = document.getElementById('chip-all');
      if (allChip) allChip.classList.toggle('active', activeTags.size === 0);
      const hint = document.getElementById('filterHint');
      if (hint) {
        const parts = [...activeTags].map(t => t === 'mlx' ? 'MLX' : t === 'vision' ? 'Vision' : 'Code');
        const q = document.getElementById('quantSelect') ? document.getElementById('quantSelect').value : '';
        if (q) parts.push(q + '-bit');
        hint.innerText = parts.length ? 'AND: ' + parts.join(' + ') : '';
      }
    }

    function quantFilterValue() {
      const sel = document.getElementById('quantSelect');
      return sel ? sel.value : '';
    }

    // Does the repo name advertise this bit width? (used before real sizes arrive)
    function nameSuggestsQuant(repoId, bits) {
      const rid = (repoId || '').toLowerCase();
      if (new RegExp(`(^|[^0-9])${bits}[ ._-]?bit`).test(rid)) return true;
      if (new RegExp(`(^|[^a-z0-9])q${bits}([^0-9]|$)`).test(rid)) return true;
      return false;
    }

    function matchesQuant(r, bits) {
      if (!bits) return true;
      if (r.quant) return r.quant === bits + '-bit';
      if (r.size_known) return false; // real quant known and it differs
      return nameSuggestsQuant(r.id, bits);
    }

    async function doSearch() {
      syncChipState();
      const query = (document.getElementById('searchInput').value || 'gemma 4').trim();
      const sort = document.getElementById('sortSelect').value || 'downloads';
      const limit = document.getElementById('limitSelect').value || '50';
      const resultsGrid = document.getElementById('searchResultsGrid');
      const metaHeader = document.getElementById('searchMetaHeader');
      currentSearchQuery = query;

      resultsGrid.innerHTML = '<div style="color: var(--text-muted);">Searching Hugging Face repositories...</div>';

      // Check if user entered a direct Hugging Face URL with subfolder/tree
      if (query.includes('huggingface.co/') || query.includes('/tree/') || query.includes('/blob/')) {
        let repoPart = query.split('huggingface.co/').pop().replace(/^[/]+|[/]+$/g, '');
        let subfolder = '';
        let repoId = repoPart;
        if (repoPart.includes('/tree/')) {
          const m = repoPart.match(/^([^/]+[/][^/]+)[/]tree[/][^/]+(?:[/](.+))?$/);
          if (m) {
            repoId = m[1];
            subfolder = m[2] || '';
          }
        } else if (repoPart.includes('/blob/')) {
          const m = repoPart.match(/^([^/]+[/][^/]+)[/]blob[/][^/]+(?:[/](.+))?$/);
          if (m) {
            repoId = m[1];
            let rest = m[2] || '';
            subfolder = rest.includes('/') ? rest.substring(0, rest.lastIndexOf('/')) : rest;
          }
        }
        
        const display = subfolder ? `${repoId} (${subfolder})` : repoId;
        const hfDirectUrl = `https://huggingface.co/${repoId}${subfolder ? '/tree/main/' + subfolder : ''}`;
        
        resultsGrid.innerHTML = `
          <div class="card" style="grid-column: 1 / -1; border-color: #6366f1; background: rgba(99, 102, 241, 0.1);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
              <div>
                <span class="status-pill downloading" style="margin-bottom: 6px; display: inline-block;">
                  ⚡ DIRECT HUGGING FACE MODEL DETECTED
                </span>
                <div class="card-title" style="font-size: 17px; color: #fff;">${display}</div>
                <div class="card-meta" style="margin-top: 4px;">
                  Repo ID: <strong>${repoId}</strong> ${subfolder ? `| Target Subfolder: <strong style="color: #a5b4fc;">${subfolder}</strong>` : ''}
                </div>
              </div>
              <div style="display: flex; gap: 8px;">
                <button class="btn btn-primary" style="padding: 12px 24px; font-size: 14px;" onclick="executeDownload('${repoId}', '${subfolder}')">
                  ⚡ Download ${subfolder ? subfolder : 'Model'}
                </button>
                <a href="${hfDirectUrl}" target="_blank" class="btn btn-secondary" style="padding: 12px 16px; font-size: 13px;">🔗 Open on HF</a>
              </div>
            </div>
          </div>
        `;
        metaHeader.innerText = `Detected direct Hugging Face URL for '${display}'`;
        return;
      }

      const quant = quantFilterValue();
      const tagsParam = [...activeTags].join(',');
      const res = await fetch(`/api/search?q=${encodeURIComponent(query)}&sort=${sort}&limit=${limit}&tags=${encodeURIComponent(tagsParam)}&quant=${encodeURIComponent(quant)}`);
      const data = await res.json();
      currentFreeRam = data.free_ram_gb || currentFreeRam;
      if (data.usable_ram_gb) {
        currentUsableRam = data.usable_ram_gb;
        currentTotalRam = data.total_ram_gb || currentTotalRam;
        syncRamBudgetLabel();
      }
      searchResults = data.results || [];
      searchSeq += 1;
      const mySeq = searchSeq;

      renderSearchCards(query);

      // Phase 2: real Hugging Face weight sizes + exact parameter counts (cached after first run)
      const unknown = searchResults.filter(r => !r.size_known).map(r => r.id);
      if (!unknown.length) return;
      try {
        const sizeRes = await fetch(`/api/repo_sizes?ids=${encodeURIComponent(unknown.join(','))}`);
        const sizeData = await sizeRes.json();
        if (mySeq !== searchSeq) return; // a newer search superseded this one
        const sizes = sizeData.sizes || {};
        searchResults.forEach(r => {
          const info = sizes[r.id];
          if (!info) return;
          r.weights_gb = info.weights_gb || 0;
          r.params_b = info.params_b;
          r.quant = info.quant || '';
          r.est_ram_gb = info.est_ram_gb || r.est_ram_gb;
          r.size_known = !!info.weights_gb;
        });
        renderSearchCards(query);
      } catch (e) {
        console.error(e);
        if (mySeq === searchSeq) {
          searchResults.forEach(r => { if (!r.size_known) r.size_failed = true; });
          renderSearchCards(query);
        }
      }
    }

    function syncRamBudgetLabel() {
      const sel = document.getElementById('ramSelect');
      if (!sel) return;
      const auto = sel.querySelector('option[value="auto"]');
      if (auto && currentUsableRam) {
        auto.innerText = `🖥️ Fits My Mac (${currentUsableRam} GB usable)`;
      }
    }

    function getRamBudget() {
      const sel = document.getElementById('ramSelect');
      if (!sel) return 0;
      if (sel.value === 'auto') return currentUsableRam || 0;
      return parseFloat(sel.value) || 0;
    }

    function renderSearchCards(query) {
      const resultsGrid = document.getElementById('searchResultsGrid');
      const metaHeader = document.getElementById('searchMetaHeader');
      const budget = getRamBudget();

      const qBits = quantFilterValue();
      let shown = searchResults.filter(r => (!budget || !r.est_ram_gb || r.est_ram_gb <= budget) && matchesQuant(r, qBits));
      const sortMode = document.getElementById('sortSelect').value;
      if (sortMode === 'size_asc') shown = shown.slice().sort((a, b) => (a.est_ram_gb || 1e9) - (b.est_ram_gb || 1e9));
      if (sortMode === 'size_desc') shown = shown.slice().sort((a, b) => (b.est_ram_gb || 0) - (a.est_ram_gb || 0));

      const budgetTxt = budget ? `RAM budget: ${budget} GB` : 'RAM budget: any';
      const filterTxt = activeTags.size ? [...activeTags].map(t => t.toUpperCase()).join(' + ') : 'ALL HF';
      const qTxt = quantFilterValue() ? ` + ${quantFilterValue()}-bit` : '';
      metaHeader.innerText = `Showing ${shown.length} of ${searchResults.length} repos for '${query}' (Filter: ${filterTxt}${qTxt} | ${budgetTxt}${currentTotalRam ? ' of ' + currentTotalRam + ' GB total' : ''})`;
      resultsGrid.innerHTML = '';

      if (shown.length === 0) {
        resultsGrid.innerHTML = `<div class="card" style="grid-column: 1 / -1; text-align: center; color: var(--text-muted); padding: 24px;">No models match '${query}' within ${budget} GB RAM. Try increasing the RAM budget or a different search.</div>`;
        applyViewMode();
        return;
      }

      shown.forEach(r => {
        const ramBadge = getRamBadge(r.est_ram_gb, budget || currentUsableRam, r.size_known, !r.size_known && !r.size_failed);
        const tagsHtml = (r.tags || []).map(t => `<span style="font-size:10px; padding:2px 6px; border-radius:4px; background:rgba(255,255,255,0.06); color:var(--text-muted); margin-right:4px;">${t}</span>`).join('');
        const isChecked = selectedForCompare.has(r.id) ? 'checked' : '';
        const hfUrl = `https://huggingface.co/${r.id}`;

        const sizeBits = [];
        if (r.weights_gb) {
          const w = r.weights_gb >= 100 ? Math.round(r.weights_gb) : Math.round(r.weights_gb * 10) / 10;
          sizeBits.push(`💾 ${w} GB weights`);
        }
        if (r.params_b) {
          const pb = r.params_b >= 10 ? Math.round(r.params_b) : Math.round(r.params_b * 10) / 10;
          sizeBits.push(`🧮 ${pb}B params`);
        }
        if (r.quant) sizeBits.push(`🎯 ${r.quant}`);
        const sizeHtml = sizeBits.length
          ? sizeBits.map(s => `<span>${s}</span>`).join('')
          : `<span style="opacity:0.6;">💾 size unknown</span>`;

        resultsGrid.innerHTML += `
          <div class="card">
            <div class="card-main">
              <div class="card-header-row">
                <div class="card-title">${r.id}</div>
                ${ramBadge}
                <label class="compare-label" style="font-size: 11px; cursor: pointer; color: var(--text-muted); display: inline-flex; align-items: center; gap: 4px; margin-left: 6px;">
                  <input type="checkbox" class="compare-check" ${isChecked} onchange="toggleCompareSelection('${r.id}')"> Compare
                </label>
              </div>
              <div class="card-meta">
                <div style="display: inline-flex; flex-wrap: wrap; gap: 10px; width: 100%;">${sizeHtml}</div>
                <span>⬇️ <strong style="color: #f3f4f6;">${r.downloads.toLocaleString()}</strong></span>
                <span>❤️ <strong style="color: #f3f4f6;">${r.likes}</strong></span>
                <span>🕒 <strong style="color: #f3f4f6;">${r.lastModified || 'Recent'}</strong></span>
                <div style="display: inline-flex; flex-wrap: wrap; gap: 4px;">${tagsHtml}</div>
              </div>
            </div>
            <div class="card-actions">
              <button class="btn btn-primary" style="flex: 1;" title="Choose a drive, confirm, then download"
                      onclick="downloadRepo('${r.id}', '', { needed_gb: ${r.est_ram_gb ? Math.round(r.est_ram_gb * 0.8) : 0} })">💾 Download…</button>
              <a href="${hfUrl}" target="_blank" class="btn btn-secondary" style="padding: 9px 12px; font-size: 12px;">🔗 HF Page</a>
            </div>
          </div>`;
      });
      applyViewMode();
    }

    function applyPreset(type) {
      document.querySelector('input[name="launchStrategy"][value="custom"]').checked = true;
      toggleCustomFields(true);

      if (type === 'vision') {
        document.getElementById('launchEngineSelect').value = 'mlx-vlm';
        document.getElementById('customTemp').value = '0.0';
        document.getElementById('customMaxTokens').value = '4096';
        document.getElementById('customThinking').checked = true;
      } else if (type === 'code') {
        document.getElementById('customTemp').value = '0.0';
        document.getElementById('customMaxTokens').value = '8192';
        document.getElementById('customThinking').checked = true;
      } else if (type === 'reason') {
        document.getElementById('customTemp').value = '0.6';
        document.getElementById('customMaxTokens').value = '4096';
        document.getElementById('customThinking').checked = true;
      } else if (type === 'fast') {
        document.getElementById('customTemp').value = '0.7';
        document.getElementById('customMaxTokens').value = '2048';
        document.getElementById('customThinking').checked = false;
      }
    }

    function openLaunchModal(modelName) {
      targetModelForLaunch = modelName;
      document.getElementById('modalModelTitle').innerText = `Start '${modelName}'`;
      
      const targetModelObj = cachedDownloadedModels.find(m => m.name === modelName || m.base_name === modelName);
      const engineSelect = document.getElementById('launchEngineSelect');
      if (engineSelect) {
        if (targetModelObj && targetModelObj.is_vision) {
          engineSelect.value = 'mlx-vlm';
        } else {
          engineSelect.value = 'auto';
        }
      }
      
      document.getElementById('launchModal').classList.add('active');
    }

    function closeLaunchModal() {
      document.getElementById('launchModal').classList.remove('active');
    }

    function toggleCustomFields(show) {
      document.getElementById('customFields').style.display = show ? 'block' : 'none';
    }

    async function confirmLaunch() {
      const strategy = document.querySelector('input[name="launchStrategy"]:checked').value;
      const engine = document.getElementById('launchEngineSelect') ? document.getElementById('launchEngineSelect').value : 'auto';
      closeLaunchModal();

      let payload = { model: targetModelForLaunch, mode: strategy, engine: engine };

      if (strategy === 'custom') {
        payload.port = parseInt(document.getElementById('customPort').value) || globalConfig.default_port || 9999;
        payload.temp = parseFloat(document.getElementById('customTemp').value) || globalConfig.default_temp || 0.0;
        payload.max_tokens = parseInt(document.getElementById('customMaxTokens').value) || globalConfig.default_max_tokens || 4096;
        payload.thinking = document.getElementById('customThinking').checked;
      }

      alert(`Launching ${targetModelForLaunch} with ${engine.toUpperCase()} engine...`);

      await fetch('/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      setTimeout(() => { updateStatus(); loadModels(); }, 2000);
    }

    let attachedTestImageBase64 = '';

    function handleImageSelected(input) {
      if (input.files && input.files[0]) {
        const file = input.files[0];
        const reader = new FileReader();
        reader.onload = function(e) {
          attachedTestImageBase64 = e.target.result;
          document.getElementById('testImagePreview').src = attachedTestImageBase64;
          document.getElementById('testImageName').innerText = file.name;
          document.getElementById('testImageSize').innerText = Math.round(file.size / 1024) + ' KB';
          document.getElementById('testImagePreviewRow').style.display = 'flex';
        };
        reader.readAsDataURL(file);
      }
    }

    function removeTestImage() {
      attachedTestImageBase64 = '';
      const fileInput = document.getElementById('testImageFile');
      if (fileInput) fileInput.value = '';
      const previewRow = document.getElementById('testImagePreviewRow');
      if (previewRow) previewRow.style.display = 'none';
    }

    function setPrompt(text) {
      const promptInput = document.getElementById('promptInput');
      if (promptInput) promptInput.value = text;
    }

    async function sendTestPrompt() {
      const prompt = document.getElementById('promptInput').value;
      const select = document.getElementById('testServerSelect');
      if (!select || !select.value) {
        alert("Please select a running MLX model server from the dropdown to test.");
        return;
      }
      const port = parseInt(select.value);
      const selectedOption = select.options[select.selectedIndex];
      const model = selectedOption ? selectedOption.getAttribute('data-model') : '';
      const host = selectedOption ? selectedOption.getAttribute('data-host') : '127.0.0.1';

      const chatOutput = document.getElementById('chatOutput');
      const hasImage = !!attachedTestImageBase64;
      chatOutput.innerText = `⏳ Sending ${hasImage ? 'multimodal image + text' : 'text'} query to '${model}' on port ${port}... (Generating thinking & response tokens)`;

      try {
        const res = await fetch('/api/test', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            prompt: prompt, 
            model: model, 
            port: port, 
            host: host,
            image: attachedTestImageBase64 
          })
        });
        const data = await res.json();
        if (data.response) {
          chatOutput.innerText = `[Model: ${model} | Port: ${port}${hasImage ? ' | Multimodal Vision 👁️' : ''}]\n\n` + data.response;
        } else {
          chatOutput.innerText = "Error: " + (data.error || "No response");
        }
      } catch (e) {
        chatOutput.innerText = "Error requesting server: " + e;
      }
    }

    async function runBenchmark() {
      const prompt = document.getElementById('benchPromptInput').value;
      const select = document.getElementById('testServerSelect');
      if (!select || !select.value) {
        alert("Please start an active MLX model server first to run benchmark.");
        return;
      }
      const port = parseInt(select.value);
      const selectedOption = select.options[select.selectedIndex];
      const model = selectedOption ? selectedOption.getAttribute('data-model') : '';
      const host = selectedOption ? selectedOption.getAttribute('data-host') : '127.0.0.1';

      const metricsCard = document.getElementById('benchMetricsCard');
      metricsCard.style.display = 'block';
      document.getElementById('benchTokSec').innerText = 'Benchmarking...';
      document.getElementById('benchTotSec').innerText = '-- s';
      document.getElementById('benchTokensMeta').innerText = '-- / --';
      document.getElementById('benchRamVal').innerText = '-- GB';
      document.getElementById('benchTextOutput').innerText = '⏳ Benchmarking model prompt processing and generation speed on Apple Silicon GPU...';

      try {
        const res = await fetch('/api/benchmark', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: prompt, model: model, port: port, host: host })
        });
        const data = await res.json();
        
        if (data.tok_per_sec !== undefined) {
          document.getElementById('benchTokSec').innerText = `${data.tok_per_sec} tok/s`;
          document.getElementById('benchTotSec').innerText = `${data.total_sec} s`;
          document.getElementById('benchTokensMeta').innerText = `${data.prompt_tokens} / ${data.completion_tokens}`;
          document.getElementById('benchRamVal').innerText = `${data.mlx_ram_gb} GB`;
          document.getElementById('benchTextOutput').innerText = `[Model: ${model}]\n\n` + data.response_text;
        } else {
          document.getElementById('benchTextOutput').innerText = "Benchmark Error: " + (data.error || "Failed");
        }
      } catch (e) {
        document.getElementById('benchTextOutput').innerText = "Benchmark Error: " + e;
      }
    }

    async function stopServer(pid) {
      await fetch('/api/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pid: pid })
      });
      updateStatus(); loadModels();
    }

    function closeQuantModal() {
      document.getElementById('quantModal').classList.remove('active');
    }

    function showQuantModal(repoId, variants, info = {}) {
      document.getElementById('quantModalSubtitle').innerHTML = `
        Repository <strong style="color: #60a5fa;">${repoId}</strong> bundles multiple subfolder quantizations. Select a specific bit precision below to download only that model (saving dozens of GBs!):
      `;
      const container = document.getElementById('quantModalVariants');
      container.innerHTML = '';

      variants.forEach(v => {
        const isRec = v.label.includes('Recommended') || v.subfolder.includes('4-bit') || v.subfolder.includes('4bit');
        container.innerHTML += `
          <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; border-radius: 8px; background: ${isRec ? 'rgba(99, 102, 241, 0.15)' : 'rgba(255, 255, 255, 0.04)'}; border: 1px solid ${isRec ? '#6366f1' : 'rgba(255, 255, 255, 0.08)'};">
            <div>
              <strong style="font-size: 14px; color: ${isRec ? '#a5b4fc' : '#f3f4f6'};">${v.label}</strong>
              <div style="font-size: 12px; color: var(--text-muted); margin-top: 2px;">Subfolder: <code>${v.subfolder}</code> | Size: <strong style="color: #34d399;">${v.size_str}</strong></div>
            </div>
            <button class="btn ${isRec ? 'btn-primary' : 'btn-secondary'}" onclick="closeQuantModal(); downloadRepo('${repoId}', '${v.subfolder}', ${JSON.stringify(info)})">
              💾 Download ${v.subfolder}
            </button>
          </div>
        `;
      });

      document.getElementById('quantDownloadAllBtn').onclick = () => {
        closeQuantModal();
        showDrivePicker(repoId, '', info);
      };

      document.getElementById('quantModal').classList.add('active');
    }

    async function executeDownload(repoId, subfolder = '', driveId = null) {
      const targetDrive = driveId || currentDriveId || '';
      const res = await fetch('/api/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_id: repoId, subfolder: subfolder, drive: targetDrive })
      });
      let data = {};
      try { data = await res.json(); } catch (e) { data = {}; }

      if (res.status === 409 && (data.code === 'insufficient_space' || data.code === 'drive_disconnected')) {
        // Let the user pick another drive instead of failing the download outright.
        drivesCache = data.drives || drivesCache;
        showDrivePicker(repoId, subfolder, data);
        return;
      }
      if (data.error) {
        alert(data.error);
        return;
      }
      const where = data.drive ? ` → ${data.drive}` : '';
      showToast(`Downloading ${subfolder ? repoId + ' (' + subfolder + ')' : repoId}${where}`);
      setTimeout(updateStatus, 800);
      setTimeout(loadDrives, 2500);
    }

    function showToast(message) {
      let el = document.getElementById('mlxToast');
      if (!el) {
        el = document.createElement('div');
        el.id = 'mlxToast';
        el.style.cssText = 'position:fixed;bottom:24px;right:24px;z-index:9999;background:rgba(17,24,39,0.96);color:#f9fafb;border:1px solid rgba(99,102,241,0.6);border-radius:10px;padding:12px 18px;font-size:13px;box-shadow:0 10px 30px rgba(0,0,0,0.45);max-width:420px;';
        document.body.appendChild(el);
      }
      el.textContent = message;
      el.style.display = 'block';
      clearTimeout(el._timer);
      el._timer = setTimeout(() => { el.style.display = 'none'; }, 4200);
    }

    // Every download starts here: pick a drive, confirm, then transfer.
    async function downloadRepo(repoId, subfolder = '', info = {}) {
      if (subfolder || repoId.includes('/tree/') || repoId.includes(':') || repoId.includes('/blob/')) {
        showDrivePicker(repoId, subfolder, info);
        return;
      }

      // Check if repo has multiple quant variants
      try {
        const res = await fetch(`/api/repo_variants?repo_id=${encodeURIComponent(repoId)}`);
        const data = await res.json();
        // One variant is the model itself — only offer a choice when there is one.
        if (data.variants && data.variants.length > 1) {
          showQuantModal(repoId, data.variants, info);
          return;
        }
      } catch (e) {
        console.warn("Could not check repo variants:", e);
      }

      showDrivePicker(repoId, subfolder, info);
    }

    async function pauseDownload(pid, repoId, subfolder = '') {
      await fetch('/api/pause_download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pid: pid, repo_id: repoId, subfolder: subfolder })
      });
      updateStatus();
    }

    async function restartDownload(pid, repoId, subfolder = '') {
      const desc = subfolder ? `${repoId} (${subfolder})` : repoId;
      alert(`Resuming download for ${desc} with Rust acceleration... Resuming from saved downloaded cache!`);
      await fetch('/api/restart_download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pid: pid, repo_id: repoId, subfolder: subfolder })
      });
      setTimeout(updateStatus, 1000);
    }

    async function cancelDownload(pid, repoId, subfolder = '') {
      if (confirm("Cancel and stop tracking this Hugging Face download?")) {
        await fetch('/api/cancel_download', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ pid: pid, repo_id: repoId, subfolder: subfolder })
        });
        updateStatus();
      }
    }

    async function killAllServers() {
      if (confirm("Kill all running MLX servers?")) {
        await fetch('/api/stop', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ all: true })
        });
        updateStatus(); loadModels();
      }
    }

    async function deleteModel(deleteTarget) {
      if (confirm(`Permanently delete model at ${deleteTarget}?`)) {
        await fetch('/api/delete', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ delete_target: deleteTarget })
        });
        loadModels();
      }
    }

    async function loadLogs() {
      const res = await fetch('/api/logs');
      const data = await res.json();
      const select = document.getElementById('logSelect');
      select.innerHTML = '';
      (data.logs || []).forEach(l => {
        select.innerHTML += `<option value="${l.name}">${l.name} (${l.size})</option>`;
      });
      loadLogContent();
    }

    async function loadLogContent() {
      const select = document.getElementById('logSelect');
      const logName = select.value || '';
      const hidePolling = document.getElementById('hidePollingCheck').checked;
      const errBox = document.getElementById('errorHighlightContainer');

      const res = await fetch(`/api/logs?name=${encodeURIComponent(logName)}&hide_polling=${hidePolling}`);
      const data = await res.json();
      
      if (data.errors && data.errors.length > 0) {
        errBox.style.display = 'block';
        errBox.innerText = "⚠️ ERROR TRACEBACK FOUND IN THIS LOG:\\n\\n" + data.errors.join("\\n---\\n");
      } else {
        errBox.style.display = 'none';
        errBox.innerText = '';
      }

      document.getElementById('logContent').innerText = data.content || "(empty log)";
    }

    async function clearCurrentLog() {
      const select = document.getElementById('logSelect');
      const logName = select.value || '';
      if (!logName) return;
      if (confirm(`Clear content of '${logName}'?`)) {
        await fetch('/api/clear_logs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: logName })
        });
        loadLogContent();
      }
    }

    async function clearAllLogs() {
      if (confirm("Clear all server logs?")) {
        await fetch('/api/clear_logs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ all: true })
        });
        loadLogs();
      }
    }

    applyViewMode();
    syncSortDropdowns();
    setInterval(updateStatus, 3000);
    updateStatus();
    loadModels();
    loadDrives();
    setInterval(loadDrives, 30000);
    loadHfAuth();
    setInterval(loadHfAuth, 60000);
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    # Model weights are re-downloadable: keep every library out of Time Machine.
    try:
        mlx_helper.protect_model_storage()
    except Exception:
        pass
    # Watchdog: a stalled transfer must not sit at 0 B/s forever.
    try:
        mlx_helper.start_download_watchdog()
    except Exception as e:
        print(f"Download watchdog not started: {e}")
    server = ThreadedHTTPServer(("127.0.0.1", PORT), MLXGuiHandler)
    print(f"macOS MLX Control Center GUI Server running at http://127.0.0.1:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down GUI Server...")
        server.shutdown()
