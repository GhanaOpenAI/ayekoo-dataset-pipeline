#!/usr/bin/env bash
# Polite, resumable YouTube downloader for Ayekoo video corpus.
# High-resolution video-only, human-like pacing, anti-bot protections.
set -uo pipefail

P="${AYEKOO_PROJECT_DIR:-/mnt/volume_d2wey28/projects/ayekoo-videos}"
cd "$P"
source "$P/.venv/bin/activate"

URLS="$P/urls.txt"
OUT="$P/videos"
ARCHIVE="$P/archive.txt"
FAILED="$P/failed.txt"
PIDFILE="$P/download.pid"
LOGDIR="$P/logs"
LOG="$LOGDIR/run-$(date +%Y%m%d-%H%M%S).log"
MIN_FREE_GB=40

mkdir -p "$OUT" "$LOGDIR" "$OUT/.part"
touch "$ARCHIVE" "$FAILED"

echo $$ > "$PIDFILE"
cleanup() {
  rm -f "$PIDFILE"
  echo "[$(date '+%F %T')] Downloader process exited." | tee -a "$LOG"
}
trap cleanup EXIT INT TERM

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

total=$(grep -cve '^\s*$' "$URLS" || echo 0)
n=0; ok=0; skipped=0; perm=0
cooldown=300

log "========================================================"
log "Starting Ayekoo video-only downloader"
log "Target: $total URLs -> $OUT"
log "Resolution: Best video-only (bv*) remuxed to MP4"
log "PID: $$"
log "========================================================"

while IFS= read -r url; do
  [[ -z "${url// }" || "$url" == \#* ]] && continue
  n=$((n+1))

  # Extract ID
  id=$(sed -E 's|.*[?&]v=([^&]+).*|\1|; s|.*youtu\.be/([^?]+).*|\1|' <<<"$url")

  # 1. Archive check (already downloaded)
  if grep -qx "youtube $id" "$ARCHIVE" 2>/dev/null; then
    skipped=$((skipped+1))
    continue
  fi

  # 2. Known permanently failed check
  if grep -qx "$id" "$FAILED" 2>/dev/null; then
    perm=$((perm+1))
    continue
  fi

  # 3. Disk space guard
  free_gb=$(df -BG --output=avail /mnt/volume_d2wey28 | tail -1 | tr -dc '0-9')
  if (( free_gb < MIN_FREE_GB )); then
    log "STOPPING: Only ${free_gb}GB free on /mnt/volume_d2wey28 (safety limit: ${MIN_FREE_GB}GB)"
    break
  fi

  log "[$n/$total] Downloading ID: $id"

  # Attempt loop
  success=0
  for attempt in 1 2 3; do
    err=$(mktemp)

    yt-dlp \
      --download-archive "$ARCHIVE" \
      --paths "home:$OUT" \
      --paths "temp:$OUT/.part" \
      -o '%(id)s - %(title).150B.%(ext)s' \
      --restrict-filenames --trim-filenames 200 \
      -f "bv*" \
      --remux-video mp4 \
      --sleep-requests 2 \
      --sleep-interval 10 --max-sleep-interval 25 \
      --limit-rate 8M \
      --concurrent-fragments 2 \
      --retries 8 --fragment-retries 12 \
      --retry-sleep 'http:exp=5:180' \
      --socket-timeout 30 \
      --no-playlist --no-warnings \
      --write-info-json \
      --no-overwrites --continue \
      "$url" >>"$LOG" 2>"$err"

    rc=$?
    msg=$(cat "$err")
    cat "$err" >>"$LOG"
    rm -f "$err"

    if (( rc == 0 )); then
      ok=$((ok+1))
      success=1
      log "[$n/$total] SUCCESS: $id"
      cooldown=300
      break
    fi

    # Permanent failures (private/removed/copyright/deleted)
    if grep -qiE 'private video|video unavailable|has been removed|removed by the uploader|account associated|terminated|does not exist|members-only|Join this channel|age-restricted|copyright' <<<"$msg"; then
      echo "$id" >>"$FAILED"
      perm=$((perm+1))
      log "[$n/$total] PERMANENT FAILURE: $id (recorded in failed.txt)"
      break
    fi

    # Throttled / bot-checked / 429
    if grep -qiE '429|Too Many Requests|Sign in to confirm|not a bot|blocked it in your country|failed to extract any player response|The following content is not available' <<<"$msg"; then
      log "[$n/$total] WARNING: Throttled on $id -- cooling down for ${cooldown}s"
      sleep "$cooldown"
      cooldown=$(( cooldown < 1800 ? cooldown * 2 : 1800 ))
      continue
    fi

    # Transient error: wait and retry
    log "[$n/$total] RETRY attempt $attempt/3 for $id (rc=$rc)"
    sleep $(( 25 + RANDOM % 35 ))
  done

  # Polite human-like jitter delay between videos
  if (( success == 1 )); then
    jitter=$(( 15 + RANDOM % 25 ))
    log "Sleeping ${jitter}s before next video..."
    sleep "$jitter"
  fi

  # Every 20 videos, take an 8-minute rest
  if (( n % 20 == 0 )); then
    log "--- Periodic break: 8 minutes pause after $n videos processed ($ok downloaded, $skipped skipped) ---"
    sleep 480
  fi

done < "$URLS"

log "========================================================"
log "DONE: $ok downloaded, $skipped already existed, $perm unavailable, out of $total total."
du -sh "$OUT" | tee -a "$LOG"
log "Starting automated frame extraction (1 frame every 2s)..."
bash "$P/extract_frames.sh" >> "$LOG" 2>&1 || log "Frame extraction returned non-zero."
log "========================================================"
