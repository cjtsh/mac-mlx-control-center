#!/bin/bash
# ⚡ macOS MLX Control Center - 1-Click Installer & Launcher
set -e

export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:$HOME/.local/bin:$HOME/.cargo/bin:$HOME/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

echo "========================================================"
echo "⚡ macOS MLX Control Center - 1-Click Launcher & Setup"
echo "========================================================"

TARGET_DIR="$HOME/mac-mlx-control-center"
INSTALL_APP="/Applications/MLXController.app"
CONFIG_DIR="$HOME/Library/Application Support/MLX Controller"
LOG_DIR="$HOME/.mlx_logs"
PID_DIR="$HOME/.mlx_pids"
MODEL_DIR="$HOME/mlx_models"

mkdir -p "$LOG_DIR" "$PID_DIR" "$MODEL_DIR"

# 1. Check Python 3
PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
elif [ -f "/opt/homebrew/bin/python3" ]; then
    PYTHON_BIN="/opt/homebrew/bin/python3"
elif [ -f "/usr/local/bin/python3" ]; then
    PYTHON_BIN="/usr/local/bin/python3"
elif [ -f "/usr/bin/python3" ]; then
    PYTHON_BIN="/usr/bin/python3"
else
    echo "❌ Error: Python 3 is required. Please install Python 3 (via https://brew.sh or Xcode Tools)."
    exit 1
fi

# 2. Check / Install uv (ultra-fast zero-config MLX runner)
if ! command -v uv >/dev/null 2>&1 && ! command -v uvx >/dev/null 2>&1; then
    echo "⚡ Setting up uv runner for Apple Silicon MLX..."
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null || true
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# 3. Clone or Update Repository
if [ ! -d "$TARGET_DIR" ]; then
    echo "📦 Cloning macOS MLX Control Center to $TARGET_DIR..."
    git clone https://github.com/cjtsh/mac-mlx-control-center.git "$TARGET_DIR"
else
    echo "🔄 Updating repository in $TARGET_DIR..."
    cd "$TARGET_DIR" && git pull origin main 2>/dev/null || true
fi

cd "$TARGET_DIR"

# 4. Set Execution Permissions & Sync Files
chmod +x "$TARGET_DIR"/*.sh "$TARGET_DIR"/*.command "$TARGET_DIR"/mlx_gui_server.py "$TARGET_DIR"/desktop/build_app.sh 2>/dev/null || true
cp "$TARGET_DIR/mlx_gui_server.py" "$HOME/mlx_gui_server.py" 2>/dev/null || true
cp "$TARGET_DIR/mlx_helper.py" "$HOME/mlx_helper.py" 2>/dev/null || true
cp "$TARGET_DIR/mlx.sh" "$HOME/mlx.sh" 2>/dev/null || true
cp "$TARGET_DIR/mlx_helper.py" "$PID_DIR/mlx_helper.py" 2>/dev/null || true

# 5. Quit the old controller (if any) and stop the GUI gracefully
quit_controller() {
    if pgrep -x MLXController >/dev/null 2>&1; then
        osascript -e 'tell application id "com.cjtsh.mlxcontroller" to quit' >/dev/null 2>&1 || true
        for _ in {1..30}; do
            pgrep -x MLXController >/dev/null 2>&1 || break
            sleep 0.1
        done
    fi
}

stop_existing_gui() {
    quit_controller
    local pid cmd
    pid="$(/usr/sbin/lsof -nP -iTCP:9998 -sTCP:LISTEN -t 2>/dev/null | head -1 || true)"
    [ -z "$pid" ] && return 0
    cmd="$(/bin/ps -p "$pid" -o command= 2>/dev/null || true)"
    case "$cmd" in
        *mlx_gui_server.py*) ;;
        *)
            echo "❌ Port 9998 is in use by something other than the control center:"
            echo "   $cmd"
            echo "   Quit that app (or free the port) and run this installer again."
            exit 1
            ;;
    esac
    echo "🛑 Stopping the existing control center (PID $pid)…"
    kill -TERM "$pid" 2>/dev/null || true
    for _ in {1..30}; do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 0.2
    done
    kill -KILL "$pid" 2>/dev/null || true
    sleep 0.2
}

write_config() {
    mkdir -p "$CONFIG_DIR"
    MLX_REPO_PATH="$TARGET_DIR" MLX_PYTHON_PATH="$PYTHON_BIN" "$PYTHON_BIN" - <<'PY'
import json
import os
import pathlib

path = pathlib.Path.home() / "Library/Application Support/MLX Controller/config.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({
    "repoPath": os.environ["MLX_REPO_PATH"],
    "pythonPath": os.environ["MLX_PYTHON_PATH"],
    "port": 9998,
    "autoStart": True,
    "stopOnQuit": True,
}, indent=2) + "\n")
PY
}

make_alias() {
    osascript <<'APPLESCRIPT' >/dev/null 2>&1 || true
tell application "Finder"
    if exists alias file "MLX Controller" of desktop then
        delete alias file "MLX Controller" of desktop
    end if
    set appAlias to make alias file to POSIX file "/Applications/MLXController.app" at desktop
    set name of appAlias to "MLX Controller"
end tell
APPLESCRIPT
    if [ ! -e "$HOME/Desktop/MLX Controller" ]; then
        ln -sfn "$INSTALL_APP" "$HOME/Desktop/MLX Controller"
    fi
}

stop_existing_gui

# 6. Build and install the desktop controller
if [ -x "$TARGET_DIR/desktop/build_app.sh" ] && command -v swiftc >/dev/null 2>&1; then
    echo "🖥️  Building MLX Controller…"
    MLX_REPO_PATH="$TARGET_DIR" PYTHON_BIN="$PYTHON_BIN" "$TARGET_DIR/desktop/build_app.sh" --install
else
    echo "🖥️  Downloading the prebuilt MLX Controller…"
    TMP_DIR="$(mktemp -d)"
    trap 'rm -rf "$TMP_DIR"' EXIT
    URL="https://github.com/cjtsh/mac-mlx-control-center/releases/latest/download/MLXController.app.zip"
    if ! curl -fL --retry 2 -o "$TMP_DIR/MLXController.app.zip" "$URL"; then
        echo "❌ Could not download MLX Controller from $URL"
        echo "   Install Xcode Command Line Tools (xcode-select --install) and run this installer again."
        exit 1
    fi
    ditto -x -k "$TMP_DIR/MLXController.app.zip" "$TMP_DIR"
    quit_controller
    rm -rf "$INSTALL_APP"
    mv "$TMP_DIR/MLXController.app" "$INSTALL_APP"
    xattr -dr com.apple.quarantine "$INSTALL_APP" 2>/dev/null || true
    codesign --force --deep --sign - "$INSTALL_APP" >/dev/null 2>&1 || true
    write_config
    make_alias
fi

# 7. Launch the controller and wait for the dashboard
echo "🚀 Starting MLX Controller…"
open "$INSTALL_APP"

for _ in {1..80}; do
    if curl -sf --max-time 1 http://127.0.0.1:9998/api/status >/dev/null 2>&1; then
        break
    fi
    sleep 0.25
done

if ! curl -sf --max-time 2 http://127.0.0.1:9998/api/status >/dev/null 2>&1; then
    echo "❌ The dashboard did not start. Check:"
    echo "   $HOME/Library/Logs/MLX Controller/server.log"
    exit 1
fi

open "http://127.0.0.1:9998"

echo ""
echo "--------------------------------------------------------"
echo "🎉 MLX Control Center is running at http://127.0.0.1:9998"
echo "📁 Location: $TARGET_DIR"
echo "🖥️  Desktop app: $INSTALL_APP"
echo "💡 Double-click “MLX Controller” on the Desktop anytime."
echo "--------------------------------------------------------"
