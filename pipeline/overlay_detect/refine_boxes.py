#!/usr/bin/env python3
"""Refine Gemini boxes so wide overlays cover the whole graphic band.

For detections that are wide (a lower-third/header/footer bar), grow the box to
the outer rectangle of the overlay: full image width when the band spans it, and
the band's true top/bottom found by colour-similarity (connected component of the
bar's background colour). Narrow boxes (logos on photos) are left untouched.

Writes a refined JSONL and optional side-by-side visualisations.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def refine_box(bgr, box, w, h, tol=38, min_width_frac=0.20, max_area_frac=0.6):
    ymin, xmin, ymax, xmax = box
    x0 = max(0, int(xmin / 1000 * w))
    y0 = max(0, int(ymin / 1000 * h))
    x1 = min(w, int(xmax / 1000 * w))
    y1 = min(h, int(ymax / 1000 * h))
    if x1 <= x0 or y1 <= y0 or (x1 - x0) < min_width_frac * w:
        return box

    pad = 6
    rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
    rx1, ry1 = min(w, x1 + pad), min(h, y1 + pad)
    patch = bgr[ry0:ry1, rx0:rx1]
    inside = np.zeros(patch.shape[:2], bool)
    inside[(y0 - ry0):(y1 - ry0), (x0 - rx0):(x1 - rx0)] = True
    ring = patch[~inside]
    if ring.size == 0:
        return box
    q = (ring // 16).astype(np.int32)
    vals, counts = np.unique(q, axis=0, return_counts=True)
    bg = (vals[counts.argmax()] * 16 + 8).astype(np.float32)

    diff = np.linalg.norm(bgr.astype(np.float32) - bg, axis=2)
    m = (diff < tol).astype(np.uint8)
    num, labels = cv2.connectedComponents(m, connectivity=8)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    lab = labels[cy, cx]
    if lab == 0:
        ys, xs = np.where(labels[y0:y1, x0:x1] > 0)
        if len(ys) == 0:
            return box
        lab = labels[y0 + ys[0], x0 + xs[0]]
    comp = labels == lab
    ys, xs = np.where(comp)
    nx0, nx1, ny0, ny1 = xs.min(), xs.max(), ys.min(), ys.max()

    if (nx1 - nx0) >= 0.5 * w:
        nx0, nx1 = 0, w - 1
    if (nx1 - nx0) * (ny1 - ny0) > max_area_frac * w * h:
        return box
    return [ny0 / h * 1000, nx0 / w * 1000, ny1 / h * 1000, nx1 / w * 1000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default=os.environ.get("AYEKOO_PROJECT_DIR", "/mnt/volume_d2wey28/projects/ayekoo-videos"))
    ap.add_argument("--detections", default="gemini_pilot.jsonl")
    ap.add_argument("--out", default="gemini_pilot_refined.jsonl")
    ap.add_argument("--viz-dir", default="refine_viz")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tol", type=int, default=38)
    args = ap.parse_args()

    P = Path(args.project_dir)
    viz = P / args.viz_dir
    viz.mkdir(exist_ok=True)
    rows = [json.loads(l) for l in (P / args.detections).open()]
    if args.limit:
        rows = rows[: args.limit]

    n_ref = 0
    with (P / args.out).open("w") as fo:
        for r in rows:
            if r.get("drop") or not r.get("boxes"):
                fo.write(json.dumps(r) + "\n")
                continue
            src = P / str(r["relative_path"]).strip()
            if not src.exists():
                fo.write(json.dumps(r) + "\n")
                continue
            img = cv2.imread(str(src))
            h, w = img.shape[:2]
            new_boxes = []
            for b in r["boxes"]:
                nb = dict(b)
                nb["box"] = refine_box(img, b["box"], w, h, tol=args.tol)
                if nb["box"] != b["box"]:
                    n_ref += 1
                new_boxes.append(nb)
            r2 = dict(r)
            r2["boxes"] = new_boxes
            r2["boxes_original"] = r["boxes"]
            fo.write(json.dumps(r2) + "\n")

            vis = img.copy()
            for b in r["boxes"]:
                ymin, xmin, ymax, xmax = b["box"]
                cv2.rectangle(vis, (int(xmin / 1000 * w), int(ymin / 1000 * h)),
                              (int(xmax / 1000 * w), int(ymax / 1000 * h)),
                              (0, 255, 0), 3)
            for b in new_boxes:
                ymin, xmin, ymax, xmax = b["box"]
                cv2.rectangle(vis, (int(xmin / 1000 * w), int(ymin / 1000 * h)),
                              (int(xmax / 1000 * w), int(ymax / 1000 * h)),
                              (0, 0, 255), 3)
            cv2.imwrite(str(viz / (Path(r["image_filename"]).stem + ".jpg")), vis)

    print(json.dumps({"records": len(rows), "boxes_refined": n_ref,
                      "out": str(P / args.out), "viz": str(viz)}))


if __name__ == "__main__":
    main()
