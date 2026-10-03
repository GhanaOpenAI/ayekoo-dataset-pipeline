#!/usr/bin/env python3
"""Ayekoo people filter using Qwen2.5-VL.

Keeps ONLY frames that contain no person and no part of a human body. Frames
with any human presence (person, face, head, hands, arms, legs, silhouette,
background person, ...) are dropped.

With --action move the frames are reorganised into two flat folders:
  no_people/<video_id>_fNNNNNN.jpg   (KEEP: nobody present)
  people/<video_id>_fNNNNNN.jpg      (DROP: a person or body part is visible)
"""

import argparse
import os
import re
import shutil
import time
from pathlib import Path

import pandas as pd

os.environ.setdefault("HF_HOME", "/mnt/volume_d2wey28/hf_cache")
DEFAULT_PROJECT_DIR = os.environ.get("AYEKOO_PROJECT_DIR", "/mnt/volume_d2wey28/projects/ayekoo-videos")
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT_TEXT = (
    "Answer YES if this image should be excluded from a photo dataset. Exclude it if "
    "EITHER is true:\n"
    "1) it shows any person or any part of a human body (face, head, hair, hand, "
    "finger, arm, leg, foot, torso, shoulder, skin, a silhouette, or a person in "
    "the background); OR\n"
    "2) it is not a real photograph, for example a title or credits screen, a "
    "presentation slide, a poster or graphic made of text, a logo, a chart or "
    "diagram, a caption/subtitle card, or an image dominated by on-screen text.\n\n"
    "Answer NO only if it is a real camera photograph showing no person and no human "
    "body part.\n\n"
    "Answer with only one word: YES or NO."
)


def load_vlm(model_name: str, device: str = "cuda:0", max_pixels: int = 401408,
             attn: str = "sdpa"):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"Loading Qwen2.5-VL model: {model_name} on {device} "
          f"(bfloat16, {attn}, max_pixels={max_pixels})...")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=256 * 28 * 28,
        max_pixels=max_pixels,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model, processor


def classify_batch(model, processor, image_paths: list, device: str = "cuda:0"):
    import torch
    from qwen_vl_utils import process_vision_info

    messages_batch, valid_paths = [], []
    for img_p in image_paths:
        if not img_p.exists():
            continue
        messages_batch.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": str(img_p)},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }])
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
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, max_new_tokens=6, do_sample=False, temperature=None, top_p=None
        )

    trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    results = []
    for img_p, out_txt in zip(valid_paths, output_texts):
        cleaned = out_txt.strip()
        up = cleaned.upper()
        tokens = re.sub(r"[^A-Z]", " ", up).split()
        first = tokens[0] if tokens else ""
        if first == "YES" or "PEOPLE" in up:
            status = "DROP"
        elif first == "NO" or "NONE" in up:
            status = "KEEP"
        else:
            status = "UNCLEAR"
        results.append((img_p, status, cleaned))
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Keep frames with no people using Qwen2.5-VL")
    p.add_argument("--project-dir", type=str, default=DEFAULT_PROJECT_DIR)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--src", type=str, default="frames",
                   help="subdir holding the flat frames + metadata.csv")
    p.add_argument("--keep-dir", type=str, default="no_people")
    p.add_argument("--drop-dir", type=str, default="people")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--action", choices=["flag", "move", "delete"], default="move")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-pixels", type=int, default=401408)
    p.add_argument("--attn", type=str, default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=int, default=30)
    return p.parse_args()


def run(args):
    project_dir = Path(args.project_dir)
    src_dir = project_dir / args.src
    keep_dir = project_dir / args.keep_dir
    drop_dir = project_dir / args.drop_dir
    if args.action in ("move", "delete"):
        keep_dir.mkdir(parents=True, exist_ok=True)
    if args.action == "move":
        drop_dir.mkdir(parents=True, exist_ok=True)

    meta_csv = src_dir / "metadata.csv"
    filt_csv = project_dir / "people_filtered.csv"
    filt_parquet = project_dir / "people_filtered.parquet"

    model, processor = load_vlm(args.model, device=args.device,
                                max_pixels=args.max_pixels, attn=args.attn)

    print("=" * 58)
    print("Ayekoo Qwen2.5-VL People Filter")
    print(f"Project  : {project_dir}")
    print(f"Source   : {src_dir}")
    print(f"Model    : {args.model}")
    print(f"Batch    : {args.batch_size}  MaxPixels: {args.max_pixels}  Attn: {args.attn}")
    print(f"Action   : {args.action.upper()}  Device: {args.device}")
    print("=" * 58)

    records = {}
    if filt_csv.exists():
        try:
            prev = pd.read_csv(filt_csv)
            for _, row in prev.iterrows():
                rec = row.to_dict()
                img_id = str(rec.get("image_id", "")).strip()
                if img_id:
                    records[img_id] = rec
        except Exception as exc:
            print(f"Could not read {filt_csv.name} ({exc}); starting fresh")
        print(f"Loaded {len(records):,} previously classified records")

    since_save = processed = 0
    kept = sum(1 for v in records.values() if v["filter_status"] == "KEEP")
    dropped = sum(1 for v in records.values() if v["filter_status"] == "DROP")

    while True:
        if not meta_csv.exists():
            if not args.watch:
                print(f"{meta_csv} not found.")
                break
            time.sleep(args.interval)
            continue

        df_meta = pd.read_csv(meta_csv)
        unclassified = []
        for _, row in df_meta.iterrows():
            img_id = str(row["image_id"]).strip()
            if img_id in records:
                continue
            full_path = project_dir / str(row["relative_path"]).strip()
            if full_path.exists():
                unclassified.append((row, full_path))

        if not unclassified:
            if not args.watch:
                print("All images already classified.")
                break
            time.sleep(args.interval)
            continue

        print(f"Found {len(unclassified):,} unclassified images")
        if args.sample > 0:
            import random

            random.Random(args.seed).shuffle(unclassified)
            unclassified = unclassified[: args.sample]
        if args.limit > 0:
            unclassified = unclassified[: args.limit]

        for b_start in range(0, len(unclassified), args.batch_size):
            batch = unclassified[b_start:b_start + args.batch_size]
            paths = [p for _, p in batch]
            rows = [r for r, _ in batch]

            results = classify_batch(model, processor, paths, device=args.device)
            for (r, (img_p, status, raw)) in zip(rows, results):
                img_id = str(r["image_id"]).strip()
                rec = dict(r)
                rec["filter_status"] = status
                rec["raw_response"] = raw
                processed += 1
                if status == "KEEP":
                    kept += 1
                else:
                    dropped += 1

                if args.action in ("move", "delete"):
                    if status == "KEEP":
                        dest = keep_dir / img_p.name
                        shutil.move(str(img_p), str(dest))
                        rec["relative_path"] = str(dest.relative_to(project_dir))
                    elif args.action == "move":
                        dest = drop_dir / img_p.name
                        shutil.move(str(img_p), str(dest))
                        rec["relative_path"] = str(dest.relative_to(project_dir))
                    else:
                        img_p.unlink(missing_ok=True)
                    try:
                        parent = img_p.parent
                        if parent not in (src_dir, project_dir) and not any(parent.iterdir()):
                            parent.rmdir()
                    except OSError:
                        pass

                records[img_id] = rec
                since_save += 1

            if since_save >= args.save_every or b_start + args.batch_size >= len(unclassified):
                df = pd.DataFrame(list(records.values()))
                df.to_csv(filt_csv.parent / (filt_csv.name + ".tmp"), index=False)
                os.replace(filt_csv.parent / (filt_csv.name + ".tmp"), filt_csv)
                df.to_parquet(filt_parquet.parent / (filt_parquet.name + ".tmp"), index=False)
                os.replace(filt_parquet.parent / (filt_parquet.name + ".tmp"), filt_parquet)
                since_save = 0
                pct = kept / max(1, processed) * 100
                print(f"Progress: {processed:,} done | no_people: {kept:,} ({pct:.1f}%) | people: {dropped:,}", flush=True)

        if args.limit > 0 or args.sample > 0:
            break
        if not args.watch:
            break
        time.sleep(args.interval)

    # Final: materialise the no_people metadata (KEEP rows) pointing at keep_dir
    if records:
        df = pd.DataFrame(list(records.values()))
        keep_df = df[df["filter_status"] == "KEEP"].copy()
        keep_df["relative_path"] = keep_df["image_filename"].map(
            lambda n: f"{args.keep_dir}/{n}"
        )
        out_csv = keep_dir / "metadata.csv"
        keep_df.to_csv(out_csv, index=False)
        keep_df.to_parquet(keep_dir / "metadata.parquet", index=False)
        print(f"no_people metadata rows: {len(keep_df):,} -> {out_csv}")

    print("=" * 58)
    print(f"Filter complete: processed={processed:,} no_people={kept:,} people={dropped:,}")


if __name__ == "__main__":
    run(parse_args())
