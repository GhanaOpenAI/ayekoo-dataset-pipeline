#!/usr/bin/env bash
# Local monitor: wait for the H200 pipeline to finish (remote DONE marker),
# then pull the final report and audit sample into the local project.
set -uo pipefail

REMOTE="${AYEKOO_REMOTE_DIR:-/mnt/volume_d2wey28/projects/ayekoo-videos}"
LOCAL="${AYEKOO_LOCAL_DIR:-/media/owusus/Godstestimo/NLP-Projects/ayekoo-videos}"
LOG=/tmp/opencode/ayekoo_finalize_local.log

echo "[$(date '+%F %T')] local finalize monitor started" >>"$LOG"
while :; do
    if ssh -o ConnectTimeout=20 -o BatchMode=yes h200 "test -f $REMOTE/DONE" 2>/dev/null; then
        echo "[$(date '+%F %T')] remote DONE detected; pulling results" >>"$LOG"
        rsync -az "h200:$REMOTE/FINAL_REPORT.txt" "$LOCAL/FINAL_REPORT.txt" >>"$LOG" 2>&1 || true
        rsync -az "h200:$REMOTE/final_audit/" "$LOCAL/final_audit/" >>"$LOG" 2>&1 || true
        echo "[$(date '+%F %T')] pulled report + audit sample to $LOCAL" >>"$LOG"
        break
    fi
    sleep 300
done
echo "[$(date '+%F %T')] local monitor exiting" >>"$LOG"
