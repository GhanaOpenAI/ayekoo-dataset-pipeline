#!/usr/bin/env bash
# Wait for consolidated frames, then for enough free GPU memory, then run the
# Qwen2.5-VL people filter over every frame. Retries on OOM (resumes from
# people_filtered.csv).
#
# Usage: run_people_qwen.sh [PROJECT_DIR] [NEED_MIB] [BATCH] [MAX_PIXELS] [MODEL]
set -uo pipefail

P="${1:-${AYEKOO_PROJECT_DIR:-/mnt/volume_d2wey28/projects/ayekoo-videos}}"
NEED="${2:-22000}"
BATCH="${3:-24}"
MAXPIX="${4:-401408}"
MODEL="${5:-Qwen/Qwen2.5-VL-7B-Instruct}"
LOG="$P/logs/people_qwen.log"

mkdir -p "$P/logs"
exec >>"$LOG" 2>&1

log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for consolidated frames ($P/frames/metadata.csv)"
while [ ! -f "$P/frames/metadata.csv" ]; do
    sleep 60
done
log "frames ready: $(find "$P/frames" -maxdepth 1 -type f -name '*.jpg' | wc -l) jpgs"

log "waiting for GPU free >= ${NEED} MiB"
while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    log "gpu free=${free} MiB"
    [ "$free" -ge "$NEED" ] && break
    sleep 120
done

cd "$P"
source "$P/.venv/bin/activate"

attempt=0
while :; do
    attempt=$((attempt + 1))
    log "starting Qwen people filter (attempt $attempt)"
    if python filter_people_qwen.py \
        --project-dir "$P" --src frames --action move \
        --model "$MODEL" \
        --batch-size "$BATCH" --max-pixels "$MAXPIX" --save-every 1000; then
        log "filter finished OK"
        break
    fi
    log "attempt $attempt failed; retrying in 180s (resumes from people_filtered.csv)"
    sleep 180
done

log "done: no_people=$(find "$P/no_people" -maxdepth 1 -type f -name '*.jpg' | wc -l) people=$(find "$P/people" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
