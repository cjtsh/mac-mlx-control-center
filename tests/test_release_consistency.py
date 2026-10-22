#!/usr/bin/env python3
"""Guards that would have caught the last handful of release mistakes.

- the version string agrees across README, mlx.sh, the dashboard and Info.plist
- every onclick="fn(" in the dashboard has a matching function definition
- no engine-specific label leaks back into the engine-agnostic UI
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mlx_gui_server as g  # noqa: E402

failures = []


def html_count(text, needle):
    return text.count(needle)


def check(name, ok, detail=None):
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"  <- {detail}"))
    if not ok:
        failures.append(name)


def version_of(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        text = f.read()
    # 127.0.0.1 is not a version: require a v prefix or an XML version string
    found = re.findall(r"\bv(\d+\.\d+\.\d+)\b", text)
    if not found:
        found = re.findall(r"<(?:string|key)>?(\d+\.\d+\.\d+)<", text)
    return found


versions = {
    "README.md": version_of("README.md")[0],
    "mlx.sh": version_of("mlx.sh")[0],
    "mlx_gui_server.py": g.APP_VERSION,
    "desktop/Info.plist": version_of("desktop/Info.plist")[0],
}
# the banner, the title and the bundle must carry the same release
check("version agrees everywhere", len(set(versions.values())) == 1, versions)
# The badge and the title once drifted: v0.11.6 in the header, v0.11.9 served.
indexed = sorted(set(re.findall(r"v(\d+\.\d+\.\d+)", g.INDEX_HTML)))
check("the dashboard shows only the current version", indexed == [g.APP_VERSION], indexed)
stamp = "v" + g.APP_VERSION
check("title and badge both carry that version", g.INDEX_HTML.count(stamp) >= 2,
      g.INDEX_HTML.count(stamp))

html = g.INDEX_HTML
import inspect
check("dashboard is not cached by the app",
      "no-store" in inspect.getsource(g.MLXGuiHandler.send_html))

# Dead buttons: a handler that does not exist is invisible until someone clicks.
handlers = set(re.findall(r'onclick="\s*([A-Za-z_$][\w$]*)\s*\(', html))
handlers |= set(re.findall(r"onclick='\s*([A-Za-z_$][\w$]*)\s*\(", html))
defined = set(re.findall(r"function\s+([A-Za-z_$][\w$]*)\s*\(", html))
assigned = set(re.findall(r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(", html))
missing = sorted(h for h in handlers if h not in defined and h not in assigned)
check("every onclick handler exists", not missing, missing)

# The UI is engine-agnostic now: these labels are regressions.
stale = [t for t in ("Active MLX Model Servers", "MLX Models Memory") if t in html]
check("no MLX-only labels in the shell", not stale, stale)

# Ollama base URLs already carry /v1; the UI must not append another.
check("dashboard never appends a second /v1", "/v1' ) + '/v1'" not in html and "+ '/v1/v1'" not in html)

# The label was renamed to LLM in 1.0.0; the identity must not drift with it.
plist = open(os.path.join(ROOT, "desktop/Info.plist"), encoding="utf-8").read()
check("bundle id is unchanged by the rename",
      "com.cjtsh.mlxcontroller" in plist, "com.cjtsh.mlxcontroller")
check("the app is labelled LLM Controller",
      "<string>LLM Controller</string>" in plist, "LLM Controller")
support_dir = "Application Support/MLX Controller"
main_swift = open(os.path.join(ROOT, "desktop/main.swift"), encoding="utf-8").read()
install_sh = open(os.path.join(ROOT, "install.sh"), encoding="utf-8").read()
check("config path is unchanged by the rename",
      support_dir in main_swift and support_dir in install_sh, support_dir)
check("the dashboard no longer calls itself MLX-only",
      "macOS MLX Control Center" not in g.INDEX_HTML, "macOS MLX Control Center")

print()
if failures:
    print(f"{len(failures)} failure(s): " + ", ".join(failures))
    sys.exit(1)
print("All release-consistency checks passed.")
