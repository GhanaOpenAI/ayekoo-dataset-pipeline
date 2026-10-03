#!/usr/bin/env bash
# Periodically delete source videos once their frames are extracted, until all
# playlist videos have been processed. Frees disk during extraction.
set -uo pipefail

P="${1:-${AYEKOO_PROJECT_DIR:-/mnt/volume_d2wey28/projects/ayekoo-videos}}"
LOG="$P/logs/prune.log"
mkdir -p "$P/logs"
exec >>"$LOG" 2>&1

log() { echo "[$(date '+%F %T')] $*"; }

N=$(wc -l < "$P/urls.txt")
log "auto-prune started (target $N videos)"

while :; do
    bash "$P/prune_processed_videos.sh" "$P" 0
    done=$(find "$P/images" -name .done | wc -l)
    failed=$(find "$P/images" -name .failed | wc -l)
    free=$(df -h /mnt/volume_d2wey28 | awk 'NR==2 {print $4}')
    log "cycle: done=$done failed=$failed / $N | disk free=$free"
    [ "$((done + failed))" -ge "$N" ] && break
    sleep 300
done

bash "$P/prune_processed_videos.sh" "$P" 0
log "all videos processed; auto-prune finished"
