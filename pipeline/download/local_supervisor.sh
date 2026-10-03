#!/usr/bin/env bash
# Keeps local_download.sh running continuously so the residential IP
# keeps grinding even if a pass exits early.
STAGE="${AYEKOO_STAGE_DIR:-/media/owusus/Godstestimo/ayekoo_stage}"
LOGDIR="$STAGE/logs"
LOG="$LOGDIR/supervisor-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$LOGDIR"
exec >>"$LOG" 2>&1

echo "[$(date -u '+%F %T')] local supervisor started (PID $$)"
LOCK="/tmp/opencode/ayekoo_local_supervisor.lock"
# Acquire the lock ONCE for the whole supervisor lifetime.
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "[$(date -u '+%F %T')] Another supervisor holds the lock; exiting."
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
while true; do
  # Stop if everything is done.
  total=$(wc -l < "$STAGE/urls.txt" 2>/dev/null || echo 0)
  arch=$(grep -c . "$STAGE/archive.txt" 2>/dev/null || echo 0)
  fail=$(grep -c . "$STAGE/failed.txt" 2>/dev/null || echo 0)
  if [ "$((arch+fail))" -ge "$total" ]; then
    echo "[$(date -u '+%F %T')] All $total videos accounted for. Supervisor exiting."
    break
  fi
  echo "[$(date -u '+%F %T')] Starting a pass (done=$arch failed=$fail total=$total)"
  bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/local_download.sh"
  echo "[$(date -u '+%F %T')] Pass ended; sleeping 30s before next pass"
  sleep 30
done
