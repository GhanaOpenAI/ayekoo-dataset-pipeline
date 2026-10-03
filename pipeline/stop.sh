#!/usr/bin/env bash
set -euo pipefail
P="${AYEKOO_PROJECT_DIR:-/mnt/volume_d2wey28/projects/ayekoo-videos}"

echo "Stopping downloader and extractor..."
tmux kill-session -t "ayekoo_dl" 2>/dev/null || true
tmux kill-session -t "ayekoo_extract" 2>/dev/null || true

if [[ -f "$P/download.pid" ]]; then
  pid=$(cat "$P/download.pid")
  kill -TERM "$pid" 2>/dev/null || true
  rm -f "$P/download.pid"
fi

pkill -f "extract_frames.py" 2>/dev/null || true
echo "All processes stopped."
