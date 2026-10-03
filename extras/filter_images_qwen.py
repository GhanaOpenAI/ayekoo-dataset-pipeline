#!/usr/bin/env python3
"""
Ayekoo Image Dataset Filter using Qwen2.5-VL.
Drops talking heads, microphone/interview setups (even a single person), and all
non-agricultural interiors, keeping genuine farm, field, crop, machinery,
livestock, and labor footage plus agricultural interiors (barns, greenhouses,
sheds, etc.).

With --action move the frames are reorganised into two flat folders:
  clean/<video_id>_fNNNNNN.jpg    (KEEP)
  removed/<video_id>_fNNNNNN.jpg  (DROP)
"""

import os
import re
import sys
import time
import shutil
import argparse
import pandas as pd
from pathlib import Path
from PIL import Image

os.environ["HF_HOME"] = "/mnt/volume_d2wey28/hf_cache"
DEFAULT_PROJECT_DIR = os.environ.get("AYEKOO_PROJECT_DIR", "/mnt/volume_d2wey28/projects/ayekoo-videos")
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT_TEXT = (
    "Task: classify this single video frame as KEEP or DROP for an agricultural dataset.\n\n"
    "DROP the frame if ANY of the following is true:\n"
    "- The scene is indoors and is NOT an agricultural interior. Offices, studios, "
    "meeting/conference rooms, halls, homes, classrooms, restaurants and any other "
    "non-agricultural interior are DROP.\n"
    "- A person is facing or speaking toward the camera (talking head: presenter, "
    "reporter, host, official, farmer, or interviewee), even if it is only one person.\n"
    "- A person is holding or speaking into a microphone, or is part of a mic/interview "
    "setup, even if they are alone.\n"
    "- Two or more people are posed in front of the camera in an interview.\n\n"
    "KEEP the frame only if it is agricultural and nobody is addressing the camera:\n"
    "- outdoor scenes: crops, plants, fields, farms, landscapes, livestock, machinery, "
    "equipment, hands working, or workers doing labor who are NOT posing/speaking to the "
    "camera;\n"
    "- agricultural interiors: barns, greenhouses, poultry houses, milking parlours, "
    "silos, farm storage sheds, or farm machinery indoors.\n\n"
    "Respond with ONLY one word: KEEP or DROP."
)

def load_vlm(model_name: str, device: str = "cuda:0", max_pixels: int = 401408,
             attn: str = "sdpa"):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"Loading Qwen2.5-VL model: {model_name} on {device} "
          f"(bfloat16, {attn}, max_pixels={max_pixels})...")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn,
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=256 * 28 * 28,
        max_pixels=max_pixels
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model, processor

def classify_batch(model, processor, image_paths: list, device: str = "cuda:0"):
    import torch
    from qwen_vl_utils import process_vision_info

    messages_batch = []
    valid_paths = []
    
    for img_p in image_paths:
        if not img_p.exists():
            continue
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(img_p)},
                {"type": "text", "text": PROMPT_TEXT}
            ]
        }]
        messages_batch.append(messages)
        valid_paths.append(img_p)

    if not messages_batch:
        return []

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages_batch
    ]
    image_inputs, video_inputs = process_vision_info(messages_batch)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=6,
            do_sample=False,
            temperature=None,
            top_p=None
        )

    # Trim input tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    results = []
    for img_p, out_txt in zip(valid_paths, output_texts):
        cleaned = out_txt.strip().upper()
        if "DROP" in cleaned:
            status = "DROP"
        elif "KEEP" in cleaned:
            status = "KEEP"
        else:
            status = "KEEP" if "FARM" in cleaned or "CROP" in cleaned else "DROP"
        results.append((img_p, status, cleaned))

    return results

def run_filtering(project_dir: Path, model_name: str, batch_size: int, 
                  action: str, watch: bool, interval: int, limit: int = 0,
                  max_pixels: int = 401408, attn: str = "sdpa",
                  save_every: int = 500):
    images_dir = project_dir / "images"
    clean_dir = project_dir / "clean"
    removed_dir = project_dir / "removed"
    if action in ("move", "delete"):
        clean_dir.mkdir(parents=True, exist_ok=True)
    if action == "move":
        removed_dir.mkdir(parents=True, exist_ok=True)

    meta_csv = project_dir / "metadata.csv"
    meta_parquet = project_dir / "metadata.parquet"
    filt_csv = project_dir / "metadata_filtered.csv"
    filt_parquet = project_dir / "metadata_filtered.parquet"

    model, processor = load_vlm(model_name, max_pixels=max_pixels, attn=attn)

    print(f"==========================================================")
    print(f"Ayekoo Qwen2.5-VL Image Filter")
    print(f"Project Dir : {project_dir}")
    print(f"Model       : {model_name}")
    print(f"Batch Size  : {batch_size}")
    print(f"Max Pixels  : {max_pixels}")
    print(f"Attention   : {attn}")
    print(f"Save Every  : {save_every}")
    print(f"Action      : {action.upper()}")
    print(f"Watch Mode  : {watch}")
    print(f"==========================================================")

    # Load existing results as full rows and keep them in memory so incremental
    # saves never have to re-read the ever-growing CSV (avoids O(n^2) I/O).
    classified_records = {}
    if filt_csv.exists():
        try:
            prev_df = pd.read_csv(filt_csv)
        except Exception as e:
            print(f"Could not read {filt_csv.name} ({e}); starting fresh")
            prev_df = None
        if prev_df is not None:
            for _, row in prev_df.iterrows():
                rec = row.to_dict()
                img_id = str(rec.get("image_id", "")).strip()
                if img_id:
                    classified_records[img_id] = rec
        print(f"Loaded {len(classified_records):,} previously classified records from {filt_csv.name}")

    since_save = 0
    total_processed = 0
    total_kept = sum(1 for v in classified_records.values() if v["filter_status"] == "KEEP")
    total_dropped = sum(1 for v in classified_records.values() if v["filter_status"] == "DROP")

    while True:
        if not meta_csv.exists():
            if not watch:
                print("No metadata.csv found.")
                break
            time.sleep(interval)
            continue

        try:
            df_meta = pd.read_csv(meta_csv)
        except Exception as e:
            if not watch:
                raise
            print(f"metadata.csv not ready ({e}); retrying in {interval}s")
            time.sleep(interval)
            continue
        # Find unclassified rows
        unclassified = []
        for idx, row in df_meta.iterrows():
            img_id = str(row["image_id"]).strip()
            if img_id not in classified_records:
                rel_path = str(row["relative_path"]).strip()
                full_path = project_dir / rel_path
                if full_path.exists():
                    unclassified.append((row, full_path))

        if not unclassified:
            if not watch:
                print(f"All {len(classified_map):,} images already classified.")
                break
            time.sleep(interval)
            continue

        print(f"Found {len(unclassified):,} unclassified images to process.")
        if limit > 0:
            unclassified = unclassified[:limit]
            print(f"Limited to first {limit} images.")

        # Process in batches
        for b_start in range(0, len(unclassified), batch_size):
            batch = unclassified[b_start:b_start + batch_size]
            batch_paths = [p for _, p in batch]
            batch_rows = [r for r, _ in batch]

            results = classify_batch(model, processor, batch_paths)
            
            for (r, (img_p, status, raw_resp)) in zip(batch_rows, results):
                img_id = str(r["image_id"]).strip()
                rec = dict(r)
                rec["filter_status"] = status
                rec["raw_response"] = raw_resp

                total_processed += 1
                if status == "KEEP":
                    total_kept += 1
                else:
                    total_dropped += 1

                # Organise into flat pools: clean/ (KEEP) and removed/ (DROP)
                if action in ("move", "delete"):
                    if status == "KEEP":
                        dest = clean_dir / img_p.name
                        shutil.move(str(img_p), str(dest))
                        rec["relative_path"] = str(dest.relative_to(project_dir))
                    elif action == "move":
                        dest = removed_dir / img_p.name
                        shutil.move(str(img_p), str(dest))
                        rec["relative_path"] = str(dest.relative_to(project_dir))
                    elif action == "delete":
                        img_p.unlink(missing_ok=True)

                    # Remove the source video dir once it is empty
                    try:
                        parent = img_p.parent
                        if parent != images_dir and not any(parent.iterdir()):
                            parent.rmdir()
                    except OSError:
                        pass

                classified_records[img_id] = rec
                since_save += 1

            if since_save >= save_every or (b_start + batch_size >= len(unclassified)):
                # Save from the in-memory record store (no disk re-read).
                all_filt_df = pd.DataFrame(list(classified_records.values()))
                all_filt_df.to_csv(filt_csv.parent / (filt_csv.name + ".tmp"), index=False)
                os.replace(filt_csv.parent / (filt_csv.name + ".tmp"), filt_csv)
                all_filt_df.to_parquet(filt_parquet.parent / (filt_parquet.name + ".tmp"), index=False)
                os.replace(filt_parquet.parent / (filt_parquet.name + ".tmp"), filt_parquet)
                since_save = 0

                pct_kept = (total_kept / max(1, total_processed)) * 100
                print(f"Progress: {total_processed:,} processed | Kept: {total_kept:,} ({pct_kept:.1f}%) | Dropped: {total_dropped:,}")

        if not watch or (limit > 0):
            break

        time.sleep(interval)

    print(f"==========================================================")
    print(f"Filter Complete!")
    print(f"Total Processed: {total_processed:,}")
    print(f"Kept           : {total_kept:,}")
    print(f"Dropped        : {total_dropped:,}")
    print(f"Saved Filtered Metadata: {filt_csv.name} & {filt_parquet.name}")
    print(f"==========================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter images using Qwen2.5-VL")
    parser.add_argument("--project-dir", type=str, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--action", type=str, choices=["flag", "move", "delete"], default="flag",
                        help="flag: only record status in metadata; move: put KEEP in clean/ and DROP in removed/; delete: put KEEP in clean/ and delete DROP")
    parser.add_argument("--watch", action="store_true", help="Continuously watch metadata.csv for new images")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0, help="Test limit for number of images")
    parser.add_argument("--max-pixels", type=int, default=401408,
                        help="Max pixels per image fed to the model (lower = faster, less detail)")
    parser.add_argument("--attn", type=str, default="sdpa",
                        choices=["sdpa", "flash_attention_2", "eager"],
                        help="Attention implementation (flash_attention_2 needs flash-attn installed)")
    parser.add_argument("--save-every", type=int, default=500,
                        help="Write metadata every N newly classified images")

    args = parser.parse_args()
    run_filtering(
        Path(args.project_dir),
        args.model,
        args.batch_size,
        args.action,
        args.watch,
        args.interval,
        args.limit,
        args.max_pixels,
        args.attn,
        args.save_every,
    )
