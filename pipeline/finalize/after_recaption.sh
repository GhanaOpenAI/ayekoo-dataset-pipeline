#!/usr/bin/env bash
# Wait for the recaption run to finish, then:
#   merge descriptive_text -> metadata.csv -> rebuild HF parquet shards -> upload.
set -euo pipefail

P="${AYEKOO_PROJECT_DIR:-/mnt/volume_d2wey28/projects/ayekoo-videos}"
cd "$P"

echo "[finalize] waiting for recaption DONE..."
while ! grep -q "DONE processed=" logs/recaption.log 2>/dev/null; do
  sleep 60
done
echo "[finalize] recaption finished; merging descriptions"

source .venv/bin/activate
python merge_descriptive.py --project-dir "$P"

echo "[finalize] rebuilding parquet shards"
source .venv-hf/bin/activate
python make_parquet_shards.py --project-dir "$P" --out-dir hf_dataset --shards 50

echo "[finalize] uploading to HuggingFace"
python upload_to_hf.py --project-dir "$P" --data-dir hf_dataset --card "$P/dataset_card.md"

echo "[finalize] DONE"
