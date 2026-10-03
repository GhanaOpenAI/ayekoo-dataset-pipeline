#!/usr/bin/env bash
set -euo pipefail
P="${AYEKOO_PROJECT_DIR:-/mnt/volume_d2wey28/projects/ayekoo-videos}"
SESSION="ayekoo_extract"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Extractor session '$SESSION' is already running in tmux."
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

echo "Starting extractor in background tmux '$SESSION' (watch mode)..."
tmux new-session -d -s "$SESSION" -c "$P" "$P/.venv/bin/python3 extract_frames.py --watch --interval 20 2>&1 | tee -a logs/extract.log"
sleep 2

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Extractor started successfully in session '$SESSION'."
else
  echo "Failed to start extractor. Check logs/extract.log."
  exit 1
fi
