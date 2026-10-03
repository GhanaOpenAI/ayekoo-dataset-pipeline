#!/usr/bin/env python3
"""
Ayekoo Video-to-Image Dataset Converter (sharpest-frame selection).

For every keep-interval (default 2s) a batch of candidate frames is decoded
(default 5 per second, i.e. 10 candidates per 2s window) and the sharpest one
(highest variance of Laplacian) is kept. This avoids motion-blurred / mid-pan
frames that fixed-interval sampling can pick.

Output images: <video_id>_fNNNNNN.jpg, plus metadata.csv / metadata.parquet.
"""

import os
import re
import sys
import json
import time
import zlib
import fcntl
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy import ndimage

DEFAULT_PROJECT_DIR = os.environ.get("AYEKOO_PROJECT_DIR", "/mnt/volume_d2wey28/projects/ayekoo-videos")


def probe_display_size(video_path: Path):
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_streams", "-of", "json", str(video_path)], text=True))
    s = info["streams"][0]
    w, h = int(s["width"]), int(s["height"])
    rot = 0
    for sd in s.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(sd["rotation"])
    if "tags" in s and "rotate" in s["tags"]:
        rot = int(s["tags"]["rotate"])
    if abs(rot) % 180 == 90:
        w, h = h, w
    return w, h


def load_playlist_metadata(project_dir: Path):
    tsv_path = project_dir / "playlist.tsv"
    meta = {}
    if tsv_path.exists():
        df = pd.read_csv(tsv_path, sep="\t")
        for _, row in df.iterrows():
            vid = str(row["id"]).strip()
            meta[vid] = {
                "title": str(row.get("title", "")).strip(),
                "duration_seconds": row.get("duration_seconds", None),
                "url": str(row.get("url", f"https://www.youtube.com/watch?v={vid}")).strip()
            }
    return meta


def extract_video_id(filename: str):
    m = re.match(r"^([a-zA-Z0-9_-]{11})", filename)
    return m.group(1) if m else None


def sharpness(rgb_bytes: bytes, h: int, w: int, scale: int = 4) -> float:
    arr = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(h, w, 3)
    g = arr[::scale, ::scale, 1].astype(np.int16)
    lap = ndimage.laplace(g)
    return float(lap.var())


def process_video(video_path: Path, images_dir: Path, meta_map: dict,
                  fps: float, cand_fps: float, quality: int):
    vid = extract_video_id(video_path.name)
    if not vid:
        return []

    vid_img_dir = images_dir / vid
    done_marker = vid_img_dir / ".done"
    if done_marker.exists() or (vid_img_dir / ".failed").exists():
        return []

    vid_img_dir.mkdir(parents=True, exist_ok=True)

    keep_interval = 1.0 / fps
    w, h = probe_display_size(video_path)
    frame_bytes = w * h * 3

    print(f"--- Extracting frames for {vid}: {video_path.name} "
          f"({w}x{h}, keep 1/{fps:g}s, {cand_fps:g} cand/s) ---")
    start_t = time.time()

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-map", "0:v:0",
        "-vf", f"fps={cand_fps}",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo", "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    title = meta_map.get(vid, {}).get("title", "")
    url = meta_map.get(vid, {}).get("url", f"https://www.youtube.com/watch?v={vid}")

    keep_idx = 0
    cand_idx = 0
    cur_window = -1
    best_score = -1.0
    best_arr = None
    best_t = 0.0
    records = []

    def flush_best():
        nonlocal keep_idx, records
        if best_arr is None:
            return
        keep_idx += 1
        name = f"{vid}_f{keep_idx:06d}.jpg"
        Image.fromarray(best_arr.reshape(h, w, 3)).save(vid_img_dir / name, quality=quality)
        records.append({
            "image_id": f"{vid}_f{keep_idx:06d}",
            "image_filename": name,
            "relative_path": f"images/{vid}/{name}",
            "video_id": vid,
            "video_title": title,
            "timestamp_seconds": round(best_t, 2),
            "frame_index": keep_idx,
            "fps_sampled": fps,
            "video_url": url,
        })

    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            t = cand_idx / cand_fps
            window = int(t / keep_interval)
            score = sharpness(buf, h, w)
            if window != cur_window:
                flush_best()
                cur_window = window
                best_score = -1.0
                best_arr = None
            if score > best_score:
                best_score = score
                best_arr = np.frombuffer(buf, dtype=np.uint8).copy()
                best_t = t
            cand_idx += 1
        flush_best()
    finally:
        proc.stdout.close()
        proc.wait()

    if keep_idx == 0:
        print(f"No frames extracted for {vid}")
        return []

    elapsed = time.time() - start_t
    print(f"Kept {keep_idx} sharpest frames for {vid} from {cand_idx} candidates "
          f"in {elapsed:.1f}s")

    with open(done_marker, "w") as f:
        f.write(f"extracted_frames={keep_idx}\nfps={fps}\ncand_fps={cand_fps}\n")

    return records


def update_metadata_files(project_dir: Path, new_records: list):
    if not new_records:
        return
    csv_path = project_dir / "metadata.csv"
    parquet_path = project_dir / "metadata.parquet"
    new_df = pd.DataFrame(new_records)
    lock_path = project_dir / ".metadata.lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if csv_path.exists():
            existing_df = pd.read_csv(csv_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["image_id"]).reset_index(drop=True)
        else:
            combined_df = new_df
        tmp_csv = csv_path.parent / (csv_path.name + ".tmp")
        combined_df.to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, csv_path)
        tmp_pq = parquet_path.parent / (parquet_path.name + ".tmp")
        combined_df.to_parquet(tmp_pq, index=False)
        os.replace(tmp_pq, parquet_path)
        fcntl.flock(lock, fcntl.LOCK_UN)
    print(f"Metadata updated: Total {len(combined_df):,} images recorded")


def run_extraction(project_dir: Path, fps: float, cand_fps: float, quality: int,
                   watch: bool, interval: int, shard_index: int = 0,
                   shard_count: int = 1):
    videos_dir = project_dir / "videos"
    images_dir = project_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_map = load_playlist_metadata(project_dir)

    print("==========================================================")
    print("Ayekoo Frame Extractor (sharpest-frame selection)")
    print(f"Project Dir      : {project_dir}")
    print(f"Keep Rate        : 1 frame every {1/fps:.1f}s (fps={fps})")
    print(f"Candidate Rate   : {cand_fps:g} frames/s")
    print(f"JPEG Quality     : {quality}")
    print(f"Watch Mode       : {watch}")
    print(f"Shard            : {shard_index + 1}/{shard_count}")
    print("==========================================================")

    while True:
        video_files = sorted([
            f for f in videos_dir.glob("*.mp4")
            if not f.name.endswith(".part") and not f.name.startswith(".")
        ])
        if shard_count > 1:
            # Stable hash partition so file-list changes never reassign a video
            # that another shard may already be extracting.
            video_files = [
                f for f in video_files
                if zlib.crc32((extract_video_id(f.name) or f.name).encode())
                % shard_count == shard_index
            ]
        for vid_file in video_files:
            vid = extract_video_id(vid_file.name)
            try:
                recs = process_video(vid_file, images_dir, meta_map, fps, cand_fps, quality)
            except Exception as e:
                print(f"!!! Extraction failed for {vid_file.name}: {e}", flush=True)
                if vid:
                    fail_dir = images_dir / vid
                    fail_dir.mkdir(parents=True, exist_ok=True)
                    (fail_dir / ".failed").write_text(f"{e}\n")
                continue
            if recs:
                update_metadata_files(project_dir, recs)
        if not watch:
            print(f"Extraction complete for {len(video_files)} video files.")
            break
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract sharpest video frames to image dataset")
    parser.add_argument("--project-dir", type=str, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--fps", type=float, default=0.5, help="Kept frames per second (0.5 = 1 frame / 2s)")
    parser.add_argument("--cand-fps", type=float, default=5.0, help="Candidate frames decoded per second")
    parser.add_argument("--quality", type=int, default=95, help="PIL JPEG quality (1-100)")
    parser.add_argument("--watch", action="store_true", help="Continuously watch videos/ folder")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval for --watch")
    parser.add_argument("--shard-index", type=int, default=0, help="This worker's index (0-based) for running N parallel extractors")
    parser.add_argument("--shard-count", type=int, default=1, help="Total number of parallel extractors sharing the videos/ folder")
    args = parser.parse_args()
    if not (0 <= args.shard_index < args.shard_count):
        parser.error("--shard-index must satisfy 0 <= index < --shard-count")
    run_extraction(Path(args.project_dir), args.fps, args.cand_fps, args.quality,
                   args.watch, args.interval, args.shard_index, args.shard_count)
