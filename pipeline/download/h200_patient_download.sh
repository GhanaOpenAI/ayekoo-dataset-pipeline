#!/usr/bin/env bash
# Patient H200 downloader: probes a known playlist video; only downloads once the
# YouTube bot-block on this datacenter IP lifts. Then it uses H200 fast bandwidth
# (100 MB/s) to drain the queue. Runs in a loop, sleeping between probe rounds.
P="${AYEKOO_PROJECT_DIR:-/mnt/volume_d2wey28/projects/ayekoo-videos}"
VID="$P/.venv/bin/yt-dlp"
ARCHIVE="$P/archive.txt"
LOGDIR="$P/logs"
LOG="$LOGDIR/h200_patient-$(date +%Y%m%d-%H%M%S).log"
PROBE_ID="99HRIhveEuo"   # first playlist video; must be unblocked to proceed
PROBE_SLEEP=300          # seconds between probe rounds

mkdir -p "$LOGDIR"; touch "$ARCHIVE"
exec >>"$LOG" 2>&1
log(){ echo "[$(date -u '+%F %T')] $*"; }

log "=== H200 patient downloader started (probing every ${PROBE_SLEEP}s) ==="

while true; do
  # already finished everything?
  total=$(wc -l < "$P/urls.txt")
  done_n=$(grep -c . "$ARCHIVE" 2>/dev/null || echo 0)
  if [ "$done_n" -ge "$((total-1))" ]; then
    log "All done ($done_n/$total). Patient loop exiting."
    break
  fi

  # Probe: can this IP fetch a playlist video right now?
  probe=$("$VID" --simulate -f "bv*" --print "UNBLOCKED" "https://www.youtube.com/watch?v=$PROBE_ID" 2>&1 | grep -c "UNBLOCKED")
  if [ "$probe" -eq 0 ]; then
    log "Still blocked. Sleeping ${PROBE_SLEEP}s..."
    sleep "$PROBE_SLEEP"
    continue
  fi

  log "BLOCK LIFTED (or intermittent). Starting full download pass now."
  # Delegate to the main download.sh which does the real work with pacing.
  bash "$P/download.sh"
  log "download.sh returned; re-probing."
  sleep 60
done
