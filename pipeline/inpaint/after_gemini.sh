#!/usr/bin/env bash
# Wait for the Gemini detection run to finish, then builds masks, LaMa-inpaints
# the overlays, and assembles the final dataset. Safe to re-run (idempotent).
set -euo pipefail

P="${AYEKOO_PROJECT_DIR:-/mnt/volume_d2wey28/projects/ayekoo-videos}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$P/logs"
cd "$P"
source "$P/.venv/bin/activate"
LOG="$P/logs/finalize.log"
exec >>"$LOG" 2>&1

echo "=== waiting for Gemini to finish: $(date) ==="
while ! grep -q "DONE processed=" "$P/logs/gemini.log" 2>/dev/null; do
  sleep 60
done

echo "=== Gemini done: $(date); building masks ==="
rm -rf "$P/masks" "$P/inpaint_input"
python "$HERE/make_masks.py" --detections gemini_detections.jsonl \
  --masks-dir masks --input-dir inpaint_input --dropped-list gemini_dropped.txt

echo "=== inpainting (LaMa): $(date) ==="
python "$HERE/run_inpaint.py" --input-dir inpaint_input --masks-dir masks --output-dir inpainted

echo "=== assembling final dataset: $(date) ==="
python "$HERE/../finalize/build_dataset.py" --detections gemini_detections.jsonl \
  --inpainted-dir inpainted --out-dir final_dataset

echo "=== DONE $(date) ==="
touch "$P/.finalize.done"
