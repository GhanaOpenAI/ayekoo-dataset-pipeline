#!/usr/bin/env python3
"""Package the final dataset as HF-ready sharded parquet with embedded images.

Reads final_dataset/metadata.csv and writes:
    <out-dir>/data/train-00000-of-000NN.parquet   images + captions
    <out-dir>/metadata.csv                        sidecar (no image bytes)

Each row embeds the JPEG bytes in a datasets `Image` feature so the Hub
Dataset Viewer renders the image and `load_dataset(repo, split="train")`
returns an `image` column with no external files needed.  Shards are built
one at a time to bound memory.
"""
import argparse
import math
import shutil
from pathlib import Path

import pandas as pd
from datasets import Dataset, Features, Image, Value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--metadata", default="final_dataset/metadata.csv")
    ap.add_argument("--image-dir", default="final_dataset")
    ap.add_argument("--out-dir", default="hf_dataset")
    ap.add_argument("--shards", type=int, default=6)
    args = ap.parse_args()

    P = Path(args.project_dir)
    meta = P / args.metadata
    img_dir = P / args.image_dir
    out = P / args.out_dir
    data = out / "data"
    if out.exists():
        shutil.rmtree(out)
    data.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(meta)
    cols = ["image_id", "image_filename", "video_id", "video_title",
            "video_url", "timestamp_seconds", "caption", "had_overlay",
            "overlay_inpainted", "n_boxes"]
    df = df[cols]
    features = Features({
        "image": Image(),
        "image_id": Value("string"),
        "image_filename": Value("string"),
        "video_id": Value("string"),
        "video_title": Value("string"),
        "video_url": Value("string"),
        "timestamp_seconds": Value("float64"),
        "caption": Value("string"),
        "had_overlay": Value("bool"),
        "overlay_inpainted": Value("bool"),
        "n_boxes": Value("int64"),
    })
    n = len(df)
    per = math.ceil(n / args.shards)
    print(f"{n} rows -> {args.shards} shards of ~{per}", flush=True)

    for i in range(args.shards):
        lo, hi = i * per, min((i + 1) * per, n)
        if lo >= hi:
            continue
        recs = []
        for row in df.iloc[lo:hi].itertuples(index=False):
            b = (img_dir / row.image_filename).read_bytes()
            recs.append({
                "image": {"bytes": b, "path": row.image_filename},
                "image_id": row.image_id,
                "image_filename": row.image_filename,
                "video_id": row.video_id,
                "video_title": row.video_title,
                "video_url": row.video_url,
                "timestamp_seconds": row.timestamp_seconds,
                "caption": row.caption,
                "had_overlay": bool(row.had_overlay),
                "overlay_inpainted": bool(row.overlay_inpainted),
                "n_boxes": int(row.n_boxes),
            })
        part = f"train-{i:05d}-of-{args.shards:05d}.parquet"
        Dataset.from_list(recs, features=features).to_parquet(str(data / part))
        print(f"  wrote {part} ({hi - lo} rows)", flush=True)
        del recs

    shutil.copy2(meta, out / "metadata.csv")
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
