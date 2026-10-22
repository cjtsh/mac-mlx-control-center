#!/bin/bash
# ⚡ Double-click installer for macOS MLX Control Center + MLX Controller app
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
exec ./install.sh
