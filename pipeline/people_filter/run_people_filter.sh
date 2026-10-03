#!/usr/bin/env bash
# Wait for consolidation (frames/metadata.csv), then run YOLO person detection
# in parallel shards over every frame. Frames with no person are hardlinked
# into no_people/ and a matching no_people/metadata.csv (+parquet) is written.
#
# Usage: run_people_filter.sh [PROJECT_DIR] [SHARDS] [THREADS]
set -uo pipefail

P="${1:-${AYEKOO_PROJECT_DIR:-/mnt/volume_d2wey28/projects/ayekoo-videos}}"
SHARDS="${2:-8}"
THREADS="${3:-2}"
FRAMES="$P/frames"
LOG="$P/logs/people_filter.log"

mkdir -p "$P/logs" "$P/no_people"
exec >>"$LOG" 2>&1

log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for consolidation ($FRAMES/metadata.csv)"
while [ ! -f "$FRAMES/metadata.csv" ]; do
    sleep 60
done
n=$(find "$FRAMES" -maxdepth 1 -type f -name '*.jpg' | wc -l)
log "consolidation detected: $n frames"

export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
cd "$P"
source "$P/.venv/bin/activate"

pids=()
for i in $(seq 0 $((SHARDS - 1))); do
    python filter_people.py \
        --project-dir "$P" --src frames --out no_people \
        --shard-index "$i" --shard-count "$SHARDS" \
        --imgsz 960 --conf 0.20 --batch 32 --device cpu \
        --report "$P/logs/people_scan_shard$i.csv" \
        >>"$P/logs/people_shard$i.log" 2>&1 &
    pids+=($!)
done
log "launched $SHARDS shards (threads/proc=$THREADS): ${pids[*]}"

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
log "all shards finished (status=$status)"

python - "$P" "$SHARDS" <<'PYEOF'
import csv
import os
import sys

P, shards = sys.argv[1], int(sys.argv[2])
logn = os.path.join(P, "logs")
rows, fields = [], None
for i in range(shards):
    path = os.path.join(logn, f"people_scan_shard{i}.csv")
    if not os.path.exists(path):
        print("missing shard report:", path)
        continue
    with open(path) as fh:
        reader = csv.DictReader(fh)
        if fields is None:
            fields = reader.fieldnames
        rows.extend(reader)

scan = os.path.join(P, "people_scan.csv")
with open(scan, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
kept = sum(1 for r in rows if r["kept"] == "1")
print(f"scan rows={len(rows)} kept={kept} dropped={len(rows) - kept}")

src = os.path.join(P, "frames", "metadata.csv")
dst = os.path.join(P, "no_people", "metadata.csv")
kept_names = {r["image_filename"] for r in rows if r["kept"] == "1"}
out_rows, mfields = [], None
with open(src, newline="") as fh:
    reader = csv.DictReader(fh)
    mfields = list(reader.fieldnames)
    for row in reader:
        if row["image_filename"] in kept_names:
            row["relative_path"] = "no_people/" + row["image_filename"]
            out_rows.append(row)
with open(dst, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=mfields)
    writer.writeheader()
    writer.writerows(out_rows)
print(f"no_people metadata rows={len(out_rows)}")

try:
    import pandas as pd

    pd.DataFrame(out_rows).to_parquet(os.path.join(P, "no_people", "metadata.parquet"), index=False)
    print("parquet written")
except Exception as exc:
    print("parquet skipped:", exc)
PYEOF

files=$(find "$P/no_people" -maxdepth 1 -type f -name '*.jpg' | wc -l)
log "done: no_people jpgs=$files"
