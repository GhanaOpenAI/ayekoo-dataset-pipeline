#!/usr/bin/env bash
# Wait for the recaption run to finish, then:
#   clean text-referencing descriptions -> merge descriptive_text ->
#   rebuild HF parquet shards -> upload.
set -euo pipefail

P="${AYEKOO_PROJECT_DIR:-/mnt/volume_d2wey28/projects/ayekoo-videos}"
cd "$P"

echo "[finalize] waiting for recaption DONE..."
while ! grep -q "DONE processed=" logs/recaption.log 2>/dev/null; do
  sleep 60
done
echo "[finalize] recaption finished"

source .venv/bin/activate
echo "[finalize] cleaning descriptions that mention text/graphics"
python clean_descriptive.py --project-dir "$P" --key-file "$P/.secrets/gemini_key" \
  --workers 32

echo "[finalize] merging descriptions"
python merge_descriptive.py --project-dir "$P" --prompts detailed_prompts.clean.jsonl

echo "[finalize] rebuilding parquet shards"
source .venv-hf/bin/activate
python make_parquet_shards.py --project-dir "$P" --out-dir hf_dataset --shards 50

echo "[finalize] uploading to HuggingFace"
python upload_to_hf.py --project-dir "$P" --data-dir hf_dataset --card "$P/dataset_card.md"

echo "[finalize] DONE"