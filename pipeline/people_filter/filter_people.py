#!/usr/bin/env python3
"""Keep frames that contain no people (or body parts) using YOLO detection.

Scans a directory of JPEGs, runs COCO person detection (class 0), and writes a
per-image report. Images with no detection at/above --conf are "kept" and, when
--action is hardlink/copy, placed into <project>/<out>/.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

PERSON_CLASS = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-dir", required=True, type=Path)
    p.add_argument("--src", default="frames", help="dir of jpgs (relative to project-dir, or absolute)")
    p.add_argument("--out", default="no_people", help="output dir relative to project-dir")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--conf", type=float, default=0.20)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--action", choices=["hardlink", "copy", "none"], default="hardlink")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sample", type=int, default=0, help="randomly sample N files before sharding")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--report", default="")
    return p.parse_args()


def video_id_of(name: str) -> str:
    return name.rsplit("_f", 1)[0] if "_f" in name else Path(name).stem


def main() -> int:
    args = parse_args()
    src = Path(args.src)
    if not src.is_absolute():
        src = args.project_dir / src
    out = args.project_dir / args.out

    files = sorted(src.rglob("*.jpg") if args.recursive else src.glob("*.jpg"))
    if not files:
        print(f"no jpgs in {src}", file=sys.stderr)
        return 1
    if args.sample:
        import random

        random.Random(args.seed).shuffle(files)
        files = files[: args.sample]
    if args.shard_count > 1:
        files = [f for i, f in enumerate(files) if i % args.shard_count == args.shard_index]
    if args.limit:
        files = files[: args.limit]

    report = Path(args.report) if args.report else (
        args.project_dir / "logs" / f"people_scan_shard{args.shard_index}.csv"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    if args.action != "none":
        out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(args.model)
    print(
        f"shard {args.shard_index}/{args.shard_count}: {len(files)} images "
        f"model={args.model} conf={args.conf} imgsz={args.imgsz} dev={args.device}",
        flush=True,
    )

    kept = dropped = 0
    t0 = time.time()
    with open(report, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_filename", "video_id", "person_count", "max_person_conf", "kept"])
        results = model.predict(
            source=[str(f) for f in files],
            conf=args.conf,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            stream=True,
            verbose=False,
        )
        for idx, (f, r) in enumerate(zip(files, results), 1):
            confs = []
            boxes = r.boxes
            if boxes is not None and len(boxes):
                for cls, cf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                    if int(cls) == PERSON_CLASS and cf >= args.conf:
                        confs.append(cf)
            pc = len(confs)
            is_kept = pc == 0
            w.writerow(
                [f.name, video_id_of(f.name), pc, round(max(confs), 4) if confs else 0.0, int(is_kept)]
            )
            if is_kept:
                kept += 1
                if args.action != "none":
                    dst = out / f.name
                    if not dst.exists():
                        try:
                            os.link(f, dst)
                        except OSError:
                            shutil.copy2(f, dst)
            else:
                dropped += 1
            if idx % 200 == 0:
                el = time.time() - t0
                print(
                    f"  {idx}/{len(files)} kept={kept} dropped={dropped} {idx / el:.1f} img/s",
                    flush=True,
                )

    el = time.time() - t0
    print(
        f"shard {args.shard_index} done: kept={kept} dropped={dropped} "
        f"in {el:.1f}s ({len(files) / el:.1f} img/s) -> {report}",
        flush=True,
    )
    report.with_suffix(".done").write_text("ok\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
