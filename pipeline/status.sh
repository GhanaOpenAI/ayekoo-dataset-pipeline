#!/usr/bin/env bash
P="${AYEKOO_PROJECT_DIR:-/mnt/volume_d2wey28/projects/ayekoo-videos}"

echo "=========================================================="
echo "               AYEKOO PIPELINE STATUS                     "
echo "=========================================================="

# Services check
if tmux has-session -t "ayekoo_dl" 2>/dev/null; then
  dl_pid=$(cat "$P/download.pid" 2>/dev/null || echo "active")
  echo "Downloader   : RUNNING (tmux: ayekoo_dl, PID: $dl_pid)"
else
  echo "Downloader   : STOPPED"
fi

if tmux has-session -t "ayekoo_extract" 2>/dev/null; then
  echo "Extractor    : RUNNING (tmux: ayekoo_extract - watch mode)"
else
  echo "Extractor    : STOPPED"
fi

if tmux has-session -t "ayekoo_filter" 2>/dev/null; then
  echo "Qwen2.5 Filter: RUNNING (tmux: ayekoo_filter)"
else
  echo "Qwen2.5 Filter: IDLE / READY"
fi

# Video counts
total=$(wc -l < "$P/urls.txt" 2>/dev/null || echo 0)
archived=$(wc -l < "$P/archive.txt" 2>/dev/null || echo 0)
failed=$(wc -l < "$P/failed.txt" 2>/dev/null || echo 0)
total=${total// /}
archived=${archived// /}
failed=${failed// /}
remaining=$(( total - archived - failed ))
(( remaining < 0 )) && remaining=0

vids_count=$(find "$P/videos" -maxdepth 1 -name "*.mp4" 2>/dev/null | wc -l)
vids_count=${vids_count// /}
vids_size=$(du -sh "$P/videos" 2>/dev/null | awk '{print $1}')
free_gb=$(df -h /mnt/volume_d2wey28 | tail -1 | awk '{print $4}')

# Image dataset counts
imgs_extracted_vids=0
imgs_total=0
if [[ -d "$P/images" ]]; then
  imgs_extracted_vids=$(find "$P/images" -mindepth 1 -maxdepth 1 -type d ! -name "_raw_temp" 2>/dev/null | wc -l)
  imgs_total=$(find "$P/images" -name "*.jpg" 2>/dev/null | wc -l)
fi
imgs_extracted_vids=${imgs_extracted_vids// /}
imgs_total=${imgs_total// /}
imgs_size=$(du -sh "$P/images" 2>/dev/null | awk '{print $1}' || echo "0B")

echo "----------------------------------------------------------"
echo "VIDEOS (Download):"
echo "  Total URLs : $total"
echo "  Downloaded : $archived (Files: $vids_count, Size: $vids_size)"
echo "  Unavailable: $failed"
echo "  Remaining  : $remaining"
echo "  Disk Avail : $free_gb on /mnt/volume_d2wey28"
echo "----------------------------------------------------------"
echo "RAW IMAGE DATASET (1 frame / 2s):"
echo "  Converted  : $imgs_extracted_vids / $vids_count videos"
echo "  Total JPGs : $imgs_total images ($imgs_size)"
if [[ -f "$P/metadata.csv" ]]; then
  meta_rows=$(wc -l < "$P/metadata.csv")
  meta_rows=${meta_rows// /}
  echo "  Metadata   : metadata.csv ($(( meta_rows - 1 )) frames indexed)"
fi
echo "----------------------------------------------------------"

# Filtered dataset counts
if [[ -f "$P/metadata_filtered.csv" ]]; then
  filt_rows=$(wc -l < "$P/metadata_filtered.csv")
  filt_rows=${filt_rows// /}
  filt_count=$(( filt_rows - 1 ))
  kept_count=$(grep -c ",KEEP," "$P/metadata_filtered.csv" 2>/dev/null)
  drop_count=$(grep -c ",DROP," "$P/metadata_filtered.csv" 2>/dev/null)
  kept_count=$(grep -c ",KEEP," "$P/metadata_filtered.csv" 2>/dev/null)
  drop_count=$(grep -c ",DROP," "$P/metadata_filtered.csv" 2>/dev/null)
  echo "QWEN2.5-VL FILTERED DATASET:"
  echo "  Evaluated  : $filt_count frames"
  echo "  KEPT (Farm): $kept_count frames"
  echo "  DROPPED    : $drop_count frames (talking-heads / interviews)"
  echo "  Metadata   : metadata_filtered.csv & metadata_filtered.parquet"
  echo "----------------------------------------------------------"
fi

latest_log=$(ls -t "$P/logs"/run-*.log 2>/dev/null | head -1)
if [[ -n "$latest_log" && -f "$latest_log" ]]; then
  echo "Recent Download Log:"
  tr '\r' '\n' < "$latest_log" | grep -v "^\s*$" | tail -n 5
fi
echo "=========================================================="
