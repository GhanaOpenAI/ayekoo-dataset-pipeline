#!/usr/bin/env bash
RDIR="${AYEKOO_REMOTE_DIR:-/mnt/volume_d2wey28/projects/ayekoo-videos}"
# Local residential-IP URL resolver.
# Never downloads media: it only asks yt-dlp for the direct googlevideo URL for
# each pending playlist video and ships the URL to the H200 (.urlqueue/), where
# aria2c pulls the bytes at ~30 MB/s. This evades YouTube's datacenter-IP block.
set -u
STAGE="${AYEKOO_STAGE_DIR:-/media/owusus/Godstestimo/ayekoo_stage}"
URLS="$STAGE/urls.txt"
QD="$STAGE/urlqueue"
SENT="$STAGE/urlqueue_sent.txt"
ATT="$STAGE/attempts"
ARCH_LOCAL="$STAGE/archive.txt"
LOGDIR="$STAGE/logs"
LOG="$LOGDIR/resolve-$(date +%Y%m%d-%H%M%S).log"
REMOTE="${AYEKOO_REMOTE_DIR:-$RDIR}"
SSH="ssh -o ServerAliveInterval=20 -o ServerAliveCountMax=10 -o ConnectTimeout=30"

mkdir -p "$QD" "$ATT" "$LOGDIR" "$STAGE/failed_pull"
touch "$SENT"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

BATCH=40          # max new resolutions per pass
HWM=80            # keep the remote unresolved queue under this many
TOTAL=$(wc -l < "$URLS")
log "local resolver started (total=$TOTAL, batch=$BATCH, hwm=$HWM)"

while true; do
  # 1) pull H200 archive + failed-URL list
  scp -q h200:"$REMOTE/archive.txt" "$STAGE/archive_h200.txt" 2>/dev/null || true
  rsync -a -e "$SSH" h200:"$REMOTE/.urlqueue_failed/" "$STAGE/failed_pull/" >/dev/null 2>&1 || true
  for f in "$STAGE/failed_pull"/*.url; do
    [ -e "$f" ] || continue
    id=$(basename "$f" .url)
    grep -vx "$id" "$SENT" > "$SENT.tmp" 2>/dev/null || true
    mv "$SENT.tmp" "$SENT"
    rm -f "$ATT/$id"
    $SSH "rm -f '$REMOTE/.urlqueue_failed/$id.url'" >/dev/null 2>&1 || true
    log "requeued failed id $id"
  done

  DONE_IDS=$(cat "$ARCH_LOCAL" "$STAGE/archive_h200.txt" 2>/dev/null | awk '{print $NF}' | sort -u)
  N_DONE=$(printf '%s\n' "$DONE_IDS" | grep -c .)
  RQ=$($SSH "ls -1 $REMOTE/.urlqueue/*.url 2>/dev/null | wc -l" 2>/dev/null || echo 0)
  # prune local queue for ids the H200 has already completed
  for f in "$QD"/*.url; do
    [ -e "$f" ] || continue
    id=$(basename "$f" .url)
    printf '%s\n' "$DONE_IDS" | grep -qx "$id" && rm -f "$f"
  done
  log "pass start: done=$N_DONE/$TOTAL remote_queue=$RQ"

  if [ "$N_DONE" -ge "$TOTAL" ] && [ "$RQ" -eq 0 ]; then
    log "ALL DONE ($N_DONE/$TOTAL)"
    break
  fi

  resolved=0
  while IFS= read -r url <&3; do
    [ "$resolved" -ge "$BATCH" ] && break
    [ "$RQ" -ge "$HWM" ] && break
    id=$(printf '%s' "$url" | sed -E 's/.*[?&]v=([A-Za-z0-9_-]{11}).*/\1/')
    [ -z "$id" ] && continue
    printf '%s\n' "$DONE_IDS" | grep -qx "$id" && continue
    grep -qx "$id" "$SENT" && continue

    u=$(yt-dlp -f "bv*[protocol=https]" -g --no-warnings --no-playlist "$url" 2>/dev/null | head -1)
    if [ -n "$u" ]; then
      printf '%s' "$u" > "$QD/$id.url"
      echo "$id" >> "$SENT"
      rm -f "$ATT/$id"
      resolved=$((resolved+1)); RQ=$((RQ+1))
      log "resolved $id"
    else
      n=$(( $(cat "$ATT/$id" 2>/dev/null || echo 0) + 1 ))
      echo "$n" > "$ATT/$id"
      if [ "$n" -ge 3 ]; then
        echo "$id" >> "$SENT"
        echo "$id" >> "$STAGE/resolve_failed.txt"
        log "GIVE UP resolving $id after $n tries"
      else
        log "resolve failed $id (attempt $n)"
      fi
    fi
    sleep $((2 + RANDOM % 3))
  done 3< "$URLS"

  rsync -a -e "$SSH" "$QD/" h200:"$REMOTE/.urlqueue/" >/dev/null 2>&1 || true
  log "pass end: resolved=$resolved"
  sleep 20
done
log "resolver exiting"
