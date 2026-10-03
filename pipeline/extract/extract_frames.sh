#!/usr/bin/env bash
set -euo pipefail
P="${AYEKOO_PROJECT_DIR:-/mnt/volume_d2wey28/projects/ayekoo-videos}"
source "$P/.venv/bin/activate"

echo "Running video frame extraction (1 frame every 2 seconds)..."
python3 "$P/extract_frames.py" --project-dir "$P" --fps 0.5 "$@"
