#!/usr/bin/env bash
# Detect when extraction + consolidation + Qwen people filtering are all done,
# then verify the result, write a final report, and build an audit sample.
set -uo pipefail

P="${1:-${AYEKOO_PROJECT_DIR:-/mnt/volume_d2wey28/projects/ayekoo-videos}}"
LOG="$P/logs/finalize.log"
mkdir -p "$P/logs"
exec >>"$LOG" 2>&1

log() { echo "[$(date '+%F %T')] $*"; }

log "finalize monitor started"

# 1. wait for consolidated frames
while [ ! -f "$P/frames/metadata.csv" ]; do
    sleep 60
done
total=$(( $(wc -l < "$P/frames/metadata.csv") - 1 ))
log "frames consolidated: $total"

# 2. wait until the people filter has finished. Drop images are purged during
#    the run, so completion is tracked via the CSV record count, not file counts.
while :; do
    np=$(find "$P/no_people" -maxdepth 1 -type f -name '*.jpg' 2>/dev/null | wc -l)
    pp=$(find "$P/people" -maxdepth 1 -type f -name '*.jpg' 2>/dev/null | wc -l)
    proc=$(pgrep -fc "filter_people_qwen.py" 2>/dev/null || true)
    proc=${proc:-0}
    rows=0
    if [ -f "$P/people_filtered.csv" ]; then
        rows=$(( $(wc -l < "$P/people_filtered.csv") - 1 ))
    fi
    log "progress: csv_rows=$rows no_people=$np people=$pp / $total (filter procs=$proc)"
    if [ "$proc" -eq 0 ] && [ "$rows" -ge "$total" ]; then
        break
    fi
    sleep 120
done
log "people filter complete"

# 3. report (counts from the CSV, since DROP jpgs are purged during the run)
python3 - "$P" <<'PYEOF'
import csv
import os
import sys

P = sys.argv[1]
filt = os.path.join(P, "people_filtered.csv")
keep = drop = unclear = total = 0
if os.path.exists(filt):
    with open(filt, newline="") as fh:
        for row in csv.DictReader(fh):
            total += 1
            s = (row.get("filter_status") or "").strip()
            if s == "KEEP":
                keep += 1
            elif s == "DROP":
                drop += 1
            else:
                unclear += 1

mc = os.path.join(P, "no_people", "metadata.csv")
meta_rows = 0
if os.path.exists(mc):
    with open(mc) as fh:
        meta_rows = sum(1 for _ in fh) - 1

np_jpg = len([f for f in os.listdir(os.path.join(P, "no_people")) if f.endswith(".jpg")]) \
    if os.path.isdir(os.path.join(P, "no_people")) else 0
pp_jpg = len([f for f in os.listdir(os.path.join(P, "people")) if f.endswith(".jpg")]) \
    if os.path.isdir(os.path.join(P, "people")) else 0

lines = [
    "AYEKOO people-filter final report",
    f"frames total     : {total}",
    f"no_people (KEEP) : {keep} ({keep / max(1, total) * 100:.1f}%)",
    f"people   (DROP)  : {drop} ({drop / max(1, total) * 100:.1f}%)",
    f"unclear          : {unclear}",
    f"no_people files  : {np_jpg}",
    f"people files kept: {pp_jpg} (rest purged)",
    f"no_people metadata rows: {meta_rows}",
]
lines.append(f"status: {'OK' if meta_rows == keep else 'MISMATCH (metadata rows != KEEP)'}")
lines.append(f"metadata: {mc}")
open(os.path.join(P, "FINAL_REPORT.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PYEOF

# 4. audit sample (hardlinks) + contact sheets
python3 - "$P" <<'PYEOF'
import glob
import os
import random
import sys

P = sys.argv[1]
for name, src, n in (("no_person", "no_people", 200), ("person", "people", 100)):
    dst = os.path.join(P, "final_audit", name)
    os.makedirs(dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(P, src, "*.jpg")))
    random.Random(1).shuffle(files)
    for f in files[:n]:
        d = os.path.join(dst, os.path.basename(f))
        if not os.path.exists(d):
            os.link(f, d)
print("audit sample built")
PYEOF

if command -v ffmpeg >/dev/null 2>&1; then
    (
        cd "$P/final_audit"
        ffmpeg -y -loglevel error -pattern_type glob -i 'no_person/*.jpg' \
            -vf "scale=320:240:force_original_aspect_ratio=decrease,pad=320:240:(ow-iw)/2:(oh-ih)/2,tile=10x20" \
            -frames:v 1 no_person_sheet.jpg 2>/dev/null || true
        ffmpeg -y -loglevel error -pattern_type glob -i 'person/*.jpg' \
            -vf "scale=320:240:force_original_aspect_ratio=decrease,pad=320:240:(ow-iw)/2:(oh-ih)/2,tile=10x10" \
            -frames:v 1 person_sheet.jpg 2>/dev/null || true
    )
fi

date "+%F %T" >"$P/DONE"
log "FINALIZED"
