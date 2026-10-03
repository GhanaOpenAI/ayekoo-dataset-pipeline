#!/usr/bin/env python3
"""Assemble the final clean training dataset.

Reads gemini_detections.jsonl and keeps every frame with drop=false (i.e. no
person / graphic / low-resolution frame). For kept frames whose overlays were
inpainted (inpainted/<stem>.jpg exists) the cleaned copy is used, otherwise the
original no_people frame is copied.

Outputs:
    final_dataset/<image_filename>      cleaned frame
    final_dataset/metadata.csv          caption + source-video metadata
    final_dataset/metadata.parquet      same, parquet if pyarrow is available
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Assemble the final Ayekoo dataset")
    ap.add_argument("--project-dir",
                    default=os.environ.get("AYEKOO_PROJECT_DIR",
                                           "/mnt/volume_d2wey28/projects/ayekoo-videos"))
    ap.add_argument("--detections", default="gemini_detections.jsonl")
    ap.add_argument("--inpainted-dir", default="inpainted")
    ap.add_argument("--out-dir", default="final_dataset")
    ap.add_argument("--link", action="store_true",
                    help="hardlink instead of copying (same filesystem only)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    P = Path(args.project_dir)
    out = P / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    inp = P / args.inpainted_dir

    counts = dict(keep=0, dropped=0, inpainted=0, original=0,
                  missing=0, error=0)
    rows = []
    with (P / args.detections).open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("error") or r.get("drop") is None:
                counts["error"] += 1
                continue
            if r["drop"]:
                counts["dropped"] += 1
                continue
            name = r["image_filename"]
            stem = Path(name).stem
            src_inp = inp / f"{stem}.jpg"
            src_orig = P / str(r["relative_path"]).strip()
            had_boxes = bool(r.get("boxes"))
            if src_inp.exists():
                src, used = src_inp, "inpainted"
            else:
                src, used = src_orig, "original"
            if not src.exists():
                counts["missing"] += 1
                continue
            dst = out / name
            if not dst.exists():
                try:
                    if args.link:
                        os.link(src, dst)
                    else:
                        shutil.copy2(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
            counts["keep"] += 1
            counts[used] += 1
            rows.append({
                "image_id": r.get("image_id"),
                "image_filename": name,
                "relative_path": f"{args.out_dir}/{name}",
                "video_id": r.get("video_id"),
                "video_title": r.get("video_title"),
                "video_url": r.get("video_url"),
                "timestamp_seconds": r.get("timestamp_seconds"),
                "caption": r.get("caption"),
                "had_overlay": had_boxes,
                "overlay_inpainted": used == "inpainted",
                "n_boxes": len(r.get("boxes") or []),
            })
            if args.limit and counts["keep"] >= args.limit:
                break

    df = pd.DataFrame(rows)
    df.to_csv(out / "metadata.csv", index=False)
    try:
        df.to_parquet(out / "metadata.parquet", index=False)
    except Exception:  # noqa: BLE001
        pass
    print(json.dumps(counts, indent=2))
    print(f"final rows: {len(df)} -> {out}")


if __name__ == "__main__":
    main()
