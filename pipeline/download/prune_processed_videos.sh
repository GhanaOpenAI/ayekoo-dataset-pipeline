#!/usr/bin/env bash
# Delete source videos whose extraction is already complete (DRY=1 to preview).
# Completion = video_id present in metadata.csv (written after a video finishes)
# or a .done marker in images/<video_id>/.
#
# Usage: prune_processed_videos.sh [PROJECT_DIR] [DRY(1|0)]
set -uo pipefail

P="${1:-${AYEKOO_PROJECT_DIR:-/mnt/volume_d2wey28/projects/ayekoo-videos}}"
DRY="${2:-1}"

python3 - "$P" "$DRY" <<'PYEOF'
import csv
import glob
import os
import sys

P, dry = sys.argv[1], sys.argv[2] == "1"

completed = set()
meta = os.path.join(P, "metadata.csv")
if os.path.exists(meta):
    try:
        with open(meta, newline="") as fh:
            for row in csv.DictReader(fh):
                vid = (row.get("video_id") or "").strip()
                if vid:
                    completed.add(vid)
    except Exception as exc:  # metadata.csv may be mid-write; .done markers still cover us
        print("metadata read skipped:", exc)
for marker in glob.glob(os.path.join(P, "images", "*", ".done")):
    completed.add(os.path.basename(os.path.dirname(marker)))

print(f"completed video ids: {len(completed)}")

targets = []
for path in glob.glob(os.path.join(P, "videos", "*.mp4")):
    name = os.path.basename(path)
    vid = name.split(" ")[0].split(".")[0]
    if vid in completed:
        targets.append((path, os.path.getsize(path), vid))

total = sum(size for _, size, _ in targets)
print(f"mp4 completed & deletable: {len(targets)}  ({total / 1e9:.1f} GB)")

if dry:
    print("DRY RUN — nothing deleted")
else:
    freed = 0
    for path, size, vid in targets:
        try:
            os.remove(path)
            freed += size
        except OSError as exc:
            print("skip", path, exc)
        for suffix in (".info.json", ".json"):
            extra = os.path.join(P, "videos", vid + suffix)
            if os.path.exists(extra):
                os.remove(extra)
    print(f"deleted {len(targets)} mp4, freed {freed / 1e9:.1f} GB")
PYEOF
