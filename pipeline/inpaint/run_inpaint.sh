#!/usr/bin/env bash
# Run LaMa (via IOPaint) over the masked frames.
set -uo pipefail

P="${1:-${AYEKOO_PROJECT_DIR:-/mnt/volume_d2wey28/projects/ayekoo-videos}}"
LOG="$P/logs/inpaint.log"
mkdir -p "$P/logs"
exec >>"$LOG" 2>&1

echo "[$(date '+%F %T')] inpaint start"
source "$P/.venv/bin/activate"
cd "$P"
iopaint run --model=lama --device=cuda \
    --image "$P/inpaint_input" --mask "$P/masks" --output "$P/inpainted"
echo "[$(date '+%F %T')] inpaint done exit=$?"
