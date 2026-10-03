#!/usr/bin/env bash
# Wait for every video to finish extracting, then flatten all frames
# (images/<id>/*.jpg + clean/*.jpg + removed/*.jpg) into a single
# frames/ directory and write metadata that points at the flat layout.
#
# Usage: consolidate_frames.sh [PROJECT_DIR]
set -uo pipefail

P="${1:-${AYEKOO_PROJECT_DIR:-/mnt/volume_d2wey28/projects/ayekoo-videos}}"
URLS="$P/urls.txt"
FRAMES="$P/frames"
LOG="$P/logs/consolidate.log"

mkdir -p "$P/logs"
exec >>"$LOG" 2>&1

log() { echo "[$(date '+%F %T')] $*"; }

N=$(wc -l < "$URLS")
log "waiting for extraction of $N videos"

while true; do
    done=$(find "$P/images" -name .done 2>/dev/null | wc -l)
    failed=$(find "$P/images" -name .failed 2>/dev/null | wc -l)
    log "status: done=$done failed=$failed total=$((done + failed))/$N"
    if [ "$((done + failed))" -ge "$N" ]; then
        break
    fi
    sleep 60
done

log "extraction complete; stopping extractor sessions"
for s in ayekoo_extract_0 ayekoo_extract_1 ayekoo_extract_2; do
    tmux kill-session -t "$s" 2>/dev/null || true
done
sleep 3

mkdir -p "$FRAMES"
log "flattening frames into $FRAMES"
find "$P/images" -type f -name '*.jpg' -exec mv -n -t "$FRAMES" {} + 2>/dev/null || true
find "$P/clean" "$P/removed" -type f -name '*.jpg' -exec mv -n -t "$FRAMES" {} + 2>/dev/null || true
n=$(find "$FRAMES" -maxdepth 1 -type f -name '*.jpg' | wc -l)
log "frames in $FRAMES: $n"

log "writing metadata (relative_path -> frames/)"
PY="$P/.venv/bin/python"
"$PY" - "$P" <<'PYEOF'
import csv
import os
import sys

P = sys.argv[1]
src = os.path.join(P, "metadata.csv")
dst_csv = os.path.join(P, "frames", "metadata.csv")
dst_pq = os.path.join(P, "frames", "metadata.parquet")

rows = []
with open(src, newline="") as fh:
    reader = csv.DictReader(fh)
    fields = list(reader.fieldnames)
    for row in reader:
        row["relative_path"] = "frames/" + row["image_filename"]
        rows.append(row)

with open(dst_csv, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print("metadata rows:", len(rows))

try:
    import pandas as pd

    pd.DataFrame(rows).to_parquet(dst_pq, index=False)
    print("parquet written:", dst_pq)
except Exception as exc:  # pragma: no cover
    print("parquet skipped:", exc)
PYEOF

log "done: flat frames=$n"
