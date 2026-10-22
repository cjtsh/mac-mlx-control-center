#!/bin/bash
# ⚡ Double-Click Launcher for macOS MLX Control Center
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:$HOME/.local/bin:$HOME/.cargo/bin:$HOME/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Prefer the desktop controller: it starts the server, shows status, and stops it on quit.
APP="/Applications/MLXController.app"
if [ ! -d "$APP" ]; then
    APP="$DIR/desktop/build/MLXController.app"
fi
if [ -d "$APP" ]; then
    open "$APP"
    exit 0
fi

echo "⚡ MLX Controller is not installed. Run ./install.sh first."
open "https://github.com/cjtsh/mac-mlx-control-center/releases/latest"
