#!/usr/bin/env bash
set -euo pipefail
P="${AYEKOO_PROJECT_DIR:-/mnt/volume_d2wey28/projects/ayekoo-videos}"
source "$P/.venv/bin/activate"

export HF_HOME="/mnt/volume_d2wey28/hf_cache"
echo "Running Qwen2.5-VL image filtering..."
python3 "$P/filter_images_qwen.py" --project-dir "$P" "$@"
