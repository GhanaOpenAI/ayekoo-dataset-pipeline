#!/usr/bin/env bash
RDIR="${AYEKOO_REMOTE_DIR:-/mnt/volume_d2wey28/projects/ayekoo-videos}"
# Ayekoo video downloader - runs on the LOCAL residential IP (H200's datacenter IP is bot-blocked).
# Downloads one video at a time, rsyncs it to H200 atomically, then deletes the local copy.

STAGE="${AYEKOO_STAGE_DIR:-/media/owusus/Godstestimo/ayekoo_stage}"
VIDEOS="$STAGE/videos"
PART="$STAGE/.part"
ARCHIVE="$STAGE/archive.txt"
FAILED="$STAGE/failed.txt"
URLS="$STAGE/urls.txt"
LOGDIR="$STAGE/logs"
LOG="$LOGDIR/run-$(date +%Y%m%d-%H%M%S).log"

REMOTE="h200:${AYEKOO_REMOTE_DIR:-$RDIR}"
MAX_STAGE_GB=40
MIN_REMOTE_FREE_GB=40

mkdir -p "$VIDEOS" "$PART" "$LOGDIR"
touch "$ARCHIVE" "$FAILED"
exec >>"$LOG" 2>&1

log() { echo "[$(date '+%F %T')] $*"; }

# Merge H200's archive/failed with ours (UNION, so we never lose entries either
# side recorded). A plain rsync overwrite would clobber entries H200 hasn't seen yet.
log "Merging archive/failed state with H200 (union)..."
for f in archive.txt failed.txt urls.txt; do
  rsync -a "$REMOTE/$f" "$STAGE/.remote_$f" 2>/dev/null || true
done
if [ -s "$STAGE/.remote_archive.txt" ]; then
  cat "$STAGE/.remote_archive.txt" "$ARCHIVE" 2>/dev/null | sort -u > "$STAGE/.arch.tmp"
  mv "$STAGE/.arch.tmp" "$ARCHIVE"
fi
if [ -s "$STAGE/.remote_failed.txt" ]; then
  cat "$STAGE/.remote_failed.txt" "$FAILED" 2>/dev/null | sort -u > "$STAGE/.fail.tmp"
  mv "$STAGE/.fail.tmp" "$FAILED"
fi
rm -f "$STAGE"/.remote_*.txt
[ -s "$URLS" ] || { rsync -a "$REMOTE/urls.txt" "$STAGE/"; }

# --- Reconcile pending transfers -------------------------------------------
# The archive can be ahead of reality: a video may be marked done locally while
# H200 never actually received the file (e.g. an interrupted upload). Re-check
# every locally-downloaded mp4 against H200 and re-ship anything missing.
log "Checking for videos downloaded locally but not yet on H200..."
pending=()
for f in "$VIDEOS"/*.mp4; do
  [ -e "$f" ] || continue
  b=$(basename "$f"); ls=$(stat -c%s "$f")
  rs=$(ssh -n h200 "stat -c%s '$RDIR/videos/$b' 2>/dev/null || echo 0")
  if [ "$ls" != "$rs" ]; then
    id=$(echo "$b" | cut -d' ' -f1)
    pending+=("$id")
  fi
done
if [ ${#pending[@]} -gt 0 ]; then
  log "PENDING TRANSFERS: ${pending[*]}"
else
  log "No pending transfers."
fi

total=$(wc -l < "$URLS")
log "========================================================"
log "Local residential-IP downloader"
log "Total URLs: $total"
log "Already done: $(wc -l < "$ARCHIVE")"
log "PID: $$"
log "========================================================"

cooldown=900

n=0
while IFS= read -r url <&3; do
  [ -z "$url" ] && continue
  n=$((n+1))
  id=$(basename "$url" | sed 's/.*v=//; s/&.*//')

  # archive.txt lines are "youtube <id>"
  # A video needing a pending re-ship must NOT be skipped even if in archive.
  is_pending=0
  for p in ${pending+"${pending[@]}"}; do
    [ "$p" = "$id" ] && { is_pending=1; break; }
  done

  if [ $is_pending -eq 0 ]; then
    if awk -v i="$id" '$2==i{f=1} END{exit !f}' "$ARCHIVE"; then
      continue
    fi
    if awk -v i="$id" '$1==i{f=1} END{exit !f}' "$FAILED"; then
      continue
    fi
  fi

  # Disk guards
  stage_gb=$(du -sm "$VIDEOS" 2>/dev/null | awk '{print $1}')
  stage_gb=${stage_gb:-0}
  [ "$stage_gb" -gt "$((MAX_STAGE_GB * 1024))" ] && { log "FATAL: staging dir over ${MAX_STAGE_GB}GB"; exit 1; }

  remote_free=$(ssh -n h200 "df -BG /mnt/volume_d2wey28 | tail -1 | awk '{print \$4}'" | tr -dc '0-9')
  if [ -n "$remote_free" ] && [ "$remote_free" -lt "$MIN_REMOTE_FREE_GB" ]; then
    log "FATAL: H200 free space ${remote_free}GB < ${MIN_REMOTE_FREE_GB}GB - stopping"
    exit 1
  fi

  log "[$n/$total] Downloading ID: $id"
  success=0

  # If the file is already downloaded locally and just needs re-shipping,
  # skip the download step entirely.
  if [ $is_pending -eq 1 ]; then
    log "[$n/$total] File already local; skipping download, going straight to transfer"
    success=1
  fi

  for attempt in 1 2 3 4; do
    [ $is_pending -eq 1 ] && break
    err=$(mktemp)
    yt-dlp \
      --download-archive "$ARCHIVE" \
      --paths "home:$VIDEOS" \
      --paths "temp:$PART" \
      -o '%(id)s - %(title).150B.%(ext)s' \
      --restrict-filenames --trim-filenames 200 \
      -f "bv*" \
      --remux-video mp4 \
      --sleep-requests 2 \
      --sleep-interval 20 --max-sleep-interval 45 \
      --limit-rate 8M \
      --concurrent-fragments 2 \
      --retries 8 --fragment-retries 12 \
      --retry-sleep 'http:exp=5:180' \
      --socket-timeout 30 \
      --no-playlist --no-warnings \
      --write-info-json \
      --no-overwrites --continue \
      "$url" 2>"$err"
    rc=$?
    msg=$(cat "$err"); rm -f "$err"

    if [ $rc -eq 0 ]; then
      success=1; cooldown=900
      log "[$n/$total] DOWNLOAD OK: $id"
      break
    fi

    if echo "$msg" | grep -qiE 'private video|video unavailable|has been removed|removed by the uploader|account associated|terminated|does not exist|members-only|Join this channel|age-restricted|copyright'; then
      echo "$id" >> "$FAILED"
      log "[$n/$total] PERMANENT FAILURE: $id"
      rsync -a "$ARCHIVE" "$FAILED" "$REMOTE/" 2>/dev/null
      success=1
      break
    fi

    if echo "$msg" | grep -qiE '429|Too Many Requests|Sign in to confirm|not a bot|failed to extract any player response|The following content is not available'; then
      log "[$n/$total] THROTTLED on $id - cooling down ${cooldown}s"
      sleep "$cooldown"
      cooldown=$(( cooldown < 3600 ? cooldown * 2 : 3600 ))
      continue
    fi

    log "[$n/$total] RETRY $attempt/4 for $id (rc=$rc)"
    sleep $(( 30 + RANDOM % 40 ))
  done

  if [ $success -eq 0 ]; then
    log "[$n/$total] GIVING UP for now: $id - will retry next run"
    break
  fi

  # Ship to H200 via .incoming staging, then atomic move into videos/.
  # Use --partial --append-verify so an interrupted ~1GB transfer resumes
  # instead of restarting from zero on the slow uplink.
  files=()
  while IFS= read -r -d '' f; do files+=("$f"); done < <(find "$VIDEOS" -maxdepth 1 -name "$id*" -print0 2>/dev/null)
  if [ ${#files[@]} -eq 0 ]; then
    log "[$n/$total] No local file for $id - skipping transfer"
    continue
  fi

  # Skip if already present & complete on H200.
  h200_has=1
  for f in "${files[@]}"; do
    b=$(basename "$f"); ls=$(stat -c%s "$f")
    rs=$(ssh -n h200 "stat -c%s '$RDIR/videos/$b' 2>/dev/null || echo 0")
    if [ "$ls" != "$rs" ]; then h200_has=0; fi
  done
  if [ $h200_has -eq 1 ]; then
    rm -f $files
    rsync -a "$ARCHIVE" "$FAILED" "$REMOTE/" 2>/dev/null
    log "[$n/$total] DONE $id (already on H200; local copy removed)"
    continue
  fi

  log "[$n/$total] Transferring $id to H200 via chunked upload..."
  # The local uplink drops long single SSH sessions, so shard each file into
  # ~150 MB pieces, upload each with retries, then reassemble on H200.
  SSH_OPTS="-o ServerAliveInterval=20 -o ServerAliveCountMax=10 -o TCPKeepAlive=yes -o ConnectTimeout=30"
  remote_chunks="$RDIR/.incoming/.chunks_$id"
  ssh -n h200 "mkdir -p '$remote_chunks'"

  chunk_dir=$(mktemp -d /tmp/ayekoo_chunks.XXXXXX)
  chunk_mb=150
  for f in "${files[@]}"; do
    b=$(basename "$f")
    split -b ${chunk_mb}M -d "$f" "$chunk_dir/${b}.chunk."
  done

  xfer_ok=1
  for c in "$chunk_dir"/*; do
    [ -e "$c" ] || continue
    cb=$(basename "$c")
    ok=0
    for t in 1 2 3 4 5 6 7 8; do
      rsync -a -e "ssh $SSH_OPTS" --partial "$c" "h200:$remote_chunks/" </dev/null
      if [ $? -eq 0 ]; then ok=1; break; fi
      log "[$n/$total] Chunk $cb try $t/8 failed; retrying in 10s"
      sleep 10
    done
    if [ $ok -ne 1 ]; then xfer_ok=0; break; fi
  done

  if [ $xfer_ok -eq 1 ]; then
    # Reassemble chunks using the exact (space-containing) filenames.
    assemble=""
    for f in "${files[@]}"; do
      b=$(basename "$f")
      assemble="$assemble
      if ls \"$b.chunk.\"* >/dev/null 2>&1; then cat \"$b.chunk.\"* > \"$b\"; rm -f \"$b.chunk.\"*; fi"
    done
    ssh -n h200 "set -e
      R='$remote_chunks'
      V='$RDIR/videos'
      cd \"\$R\"
      $assemble
      mkdir -p \"\$V\"
      mv -f * \"\$V/\" 2>/dev/null || true
      cd /; rmdir \"\$R\" 2>/dev/null || true"
  fi
  rm -rf "$chunk_dir"

  if [ $xfer_ok -ne 1 ]; then
    log "[$n/$total] TRANSFER FAILED for $id - local copy kept for resume"
    continue
  fi

  ssh -n h200 "rm -rf '$remote_chunks' 2>/dev/null || true"

  # Verify sizes match before deleting the local copy
  bad=0
  for f in "${files[@]}"; do
    b=$(basename "$f"); ls=$(stat -c%s "$f")
    rs=$(ssh -n h200 "stat -c%s '$RDIR/videos/$b' 2>/dev/null || echo 0")
    [ "$ls" != "$rs" ] && { log "SIZE MISMATCH $b ($ls vs $rs)"; bad=1; }
  done
  if [ $bad -eq 0 ]; then
    rm -f $files
    rsync -a "$ARCHIVE" "$FAILED" "$REMOTE/" 2>/dev/null
    log "[$n/$total] DONE $id (transferred, local copy removed)"
  else
    log "[$n/$total] Verification failed for $id - local copy kept"
  fi

  sleep $(( 15 + RANDOM % 30 ))
done 3< "$URLS"

log "========================================================"
log "LOCAL DOWNLOADER FINISHED"
log "Archive: $(wc -l < "$ARCHIVE") | Failed: $(wc -l < "$FAILED")"
log "========================================================"