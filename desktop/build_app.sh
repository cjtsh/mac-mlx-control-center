#!/bin/bash
# Build (and optionally install) the LLM Controller desktop app.
#
#   ./desktop/build_app.sh                 # build into desktop/build/
#   ./desktop/build_app.sh --install       # build + install to /Applications + Desktop alias
#   ./desktop/build_app.sh --install --open
#   ./desktop/build_app.sh --portable      # build without machine-specific paths (release zip)
#
# The app is a thin controller: it launches mlx_gui_server.py from this repo.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BUILD_DIR="$HERE/build"
APP="$BUILD_DIR/LLMController.app"
INSTALL_TARGET="/Applications/LLMController.app"
CONFIG_DIR="$HOME/Library/Application Support/MLX Controller"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
REPO_PATH="${MLX_REPO_PATH:-$ROOT}"
DO_INSTALL=0
DO_OPEN=0
DO_PORTABLE=0

for arg in "$@"; do
    case "$arg" in
        --install) DO_INSTALL=1 ;;
        --open) DO_OPEN=1 ;;
        --portable) DO_PORTABLE=1 ;;
        --help|-h)
            sed -n '2,10p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 2
            ;;
    esac
done

if [ -z "$PYTHON_BIN" ]; then
    echo "✗ Python 3 is required." >&2
    exit 1
fi
if ! command -v swiftc >/dev/null 2>&1; then
    echo "✗ swiftc not found. Install Xcode Command Line Tools: xcode-select --install" >&2
    exit 1
fi
if [ ! -f "$HERE/AppIcon.icns" ]; then
    echo "✗ desktop/AppIcon.icns is missing. Run desktop/generate_icon.py first." >&2
    exit 1
fi

echo "🔨 Building LLM Controller…"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

swiftc -swift-version 5 -O -framework Cocoa "$HERE/main.swift" -o "$APP/Contents/MacOS/LLMController"
cp "$HERE/Info.plist" "$APP/Contents/Info.plist"
cp "$HERE/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"

if [ "$DO_PORTABLE" -eq 0 ]; then
    /usr/libexec/PlistBuddy -c "Add :MLXRepoPath string $REPO_PATH" "$APP/Contents/Info.plist" >/dev/null
    /usr/libexec/PlistBuddy -c "Add :MLXPythonPath string $PYTHON_BIN" "$APP/Contents/Info.plist" >/dev/null
fi

xattr -cr "$APP" 2>/dev/null || true
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || true
echo "✓ Built $APP"

if [ "$DO_INSTALL" -eq 1 ]; then
    echo "📦 Installing to ${INSTALL_TARGET}…"
    if pgrep -x LLMController >/dev/null 2>&1; then
        osascript -e 'tell application id "com.cjtsh.mlxcontroller" to quit' >/dev/null 2>&1 || true
        for _ in {1..20}; do
            pgrep -x LLMController >/dev/null 2>&1 || break
            sleep 0.1
        done
    fi
    rm -rf "$INSTALL_TARGET"
    ditto "$APP" "$INSTALL_TARGET"

    mkdir -p "$CONFIG_DIR"
    MLX_REPO_PATH="$REPO_PATH" MLX_PYTHON_PATH="$PYTHON_BIN" MLX_PORT="${MLX_PORT:-9998}" \
    "$PYTHON_BIN" - <<'PY'
import json
import os
import pathlib

path = pathlib.Path.home() / "Library/Application Support/MLX Controller/config.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({
    "repoPath": os.environ["MLX_REPO_PATH"],
    "pythonPath": os.environ["MLX_PYTHON_PATH"],
    "port": int(os.environ.get("MLX_PORT", "9998")),
    "autoStart": True,
    "stopOnQuit": True,
}, indent=2) + "\n")
print("✓ Wrote", path)
PY

    # Drop the pre-1.0.1 alias so there is only one controller on the Desktop.
    rm -f "$HOME/Desktop/MLX Controller"
    osascript <<'APPLESCRIPT' >/dev/null 2>&1 || {
tell application "Finder"
    if exists alias file "LLM Controller" of desktop then
        delete alias file "LLM Controller" of desktop
    end if
    set appAlias to make alias file to POSIX file "/Applications/LLMController.app" at desktop
    set name of appAlias to "LLM Controller"
end tell
APPLESCRIPT
        ln -sfn "$INSTALL_TARGET" "$HOME/Desktop/LLM Controller"
    }
    echo "✓ Desktop alias: $HOME/Desktop/LLM Controller"

    if [ "$DO_OPEN" -eq 1 ]; then
        open "$INSTALL_TARGET"
    fi
fi
