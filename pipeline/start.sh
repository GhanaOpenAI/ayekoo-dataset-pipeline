#!/usr/bin/env bash
# Start the downloader and frame extractor in tmux (idempotent).
set -euo pipefail

P="${AYEKOO_PROJECT_DIR:-/mnt/volume_d2wey28/projects/ayekoo-videos}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DL_SESSION="ayekoo_dl"
EXT_SESSION="ayekoo_extract"
mkdir -p "$P/logs"

# Start Downloader
if tmux has-session -t "$DL_SESSION" 2>/dev/null; then
  echo "Downloader session '$DL_SESSION' is already running."
else
  echo "Starting downloader in background tmux '$DL_SESSION'..."
  tmux new-session -d -s "$DL_SESSION" -c "$P" \
    "bash $REPO/pipeline/download/download.sh 2>&1 | tee -a $P/logs/latest.log"
fi

# Start Extractor
if tmux has-session -t "$EXT_SESSION" 2>/dev/null; then
  echo "Extractor session '$EXT_SESSION' is already running."
else
  echo "Starting extractor in background tmux '$EXT_SESSION'..."
  tmux new-session -d -s "$EXT_SESSION" -c "$P" \
    "$P/.venv/bin/python3 -u $REPO/pipeline/extract/extract_frames.py --watch --interval 20 2>&1 | tee -a $P/logs/extract.log"
fi

echo "Both downloader and extractor are active."
echo "Run $REPO/pipeline/status.sh to check live progress."
