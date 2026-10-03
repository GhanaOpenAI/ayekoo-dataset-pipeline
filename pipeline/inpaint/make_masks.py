#!/usr/bin/env python3
"""Build LaMa/IOPaint masks from Gemini detections.

Reads gemini_detections.jsonl and, for every KEEP image that has detected
watermark/logo/lower-third boxes, writes:
  <masks>/<name>.png          binary mask (white = inpaint) at image resolution
  <inpaint_input>/<name>.jpg  symlink to the source frame

Images flagged drop=true are listed in gemini_dropped.txt (to be removed from
the final dataset).
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default=os.environ.get("AYEKOO_PROJECT_DIR", "/mnt/volume_d2wey28/projects/ayekoo-videos"))
    ap.add_argument("--detections", default="gemini_detections.jsonl")
    ap.add_argument("--keep-dir", default="no_people")
    ap.add_argument("--masks-dir", default="masks")
    ap.add_argument("--input-dir", default="inpaint_input")
    ap.add_argument("--dropped-list", default="gemini_dropped.txt")
    ap.add_argument("--pad", type=int, default=8, help="pixels to dilate each box")
    args = ap.parse_args()

    P = Path(args.project_dir)
    masks = P / args.masks_dir
    inputs = P / args.input_dir
    masks.mkdir(exist_ok=True)
    inputs.mkdir(exist_ok=True)

    n_keep = n_drop = n_boxes = n_masks = n_err = 0
    dropped = []
    with (P / args.detections).open() as fh:
        for line in fh:
            r = json.loads(line)
            name = r["image_filename"]
            src = P / str(r["relative_path"]).strip()
            if r.get("error") or r.get("drop") is None:
                n_err += 1
                continue
            if r["drop"]:
                n_drop += 1
                dropped.append(name)
                continue
            n_keep += 1
            boxes = r.get("boxes") or []
            if not boxes or not src.exists():
                continue
            try:
                im = Image.open(src).convert("RGB")
            except Exception:  # noqa: BLE001
                n_err += 1
                continue
            w, h = im.size
            mask = Image.new("L", (w, h), 0)
            dr = ImageDraw.Draw(mask)
            for b in boxes:
                box = b.get("box")
                if not box or len(box) != 4:
                    continue
                ymin, xmin, ymax, xmax = box
                x0 = max(0, int(xmin / 1000 * w) - args.pad)
                y0 = max(0, int(ymin / 1000 * h) - args.pad)
                x1 = min(w, int(xmax / 1000 * w) + args.pad)
                y1 = min(h, int(ymax / 1000 * h) + args.pad)
                if x1 <= x0 or y1 <= y0:
                    continue
                dr.rectangle([x0, y0, x1, y1], fill=255)
                n_boxes += 1
            mask.save(masks / f"{Path(name).stem}.png")
            link = inputs / name
            if not link.exists():
                try:
                    os.symlink(src, link)
                except FileExistsError:
                    pass
            n_masks += 1

    (P / args.dropped_list).write_text("\n".join(sorted(dropped)) + "\n")
    print(json.dumps({
        "keep": n_keep, "drop": n_drop, "errors_or_unclear": n_err,
        "masks": n_masks, "boxes_burned": n_boxes,
        "dropped_list": str(P / args.dropped_list),
    }, indent=2))


if __name__ == "__main__":
    main()
