#!/usr/bin/env python3
"""The dashboard ships its JavaScript inside a Python string literal.

An unescaped newline silently turns a string into a syntax error, and the page
then renders as an empty shell with no data — which is exactly what a broken
build looks like to a user. This parses the served script with Node when it is
available, so that class of bug cannot ship again.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

failures = 0


def check(name, ok, detail=""):
    global failures
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"  <- {detail}"))
    if not ok:
        failures += 1


def main():
    import mlx_gui_server

    html = mlx_gui_server.INDEX_HTML
    check("index page is a single document", html.lstrip().startswith("<!DOCTYPE html>"))

    blocks = re.findall(r"<script>(.*?)</script>", html, re.S)
    check("page has inline script", bool(blocks), len(blocks))
    script = "\n".join(blocks)
    check("script is not truncated", script.rstrip().endswith("}") or "loadModels" in script)

    # A real newline inside a quoted JS string is the bug this test exists for.
    bad = re.findall(r"alert\('[^'\n]*$", script, re.M)
    check("no unterminated alert() strings", not bad, bad[:2])

    node = shutil.which("node")
    if not node:
        print("SKIP  node not installed, syntax check skipped")
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(script)
            path = fh.name
        try:
            proc = subprocess.run([node, "--check", path], capture_output=True, text=True)
            detail = (proc.stderr or "").strip().splitlines()[:3]
            check("dashboard JavaScript parses", proc.returncode == 0, detail)
        finally:
            os.unlink(path)

    # The pages that carry model state must still be reachable from the script.
    for fn in ("loadModels", "openOllamaModal", "importGguf", "refreshOllamaModal"):
        check(f"script defines {fn}", f"function {fn}" in script or f"{fn} =" in script)

    print(f"\n{'All dashboard checks passed.' if not failures else f'{failures} failure(s)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
