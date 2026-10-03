#!/usr/bin/env python3
"""Inpaint masked regions with LaMa (simple-lama-inpainting).

Reads inpaint_input/*.jpg and the matching masks/*.png (white = fill),
writes cleaned frames to inpainted/<name>. Resumable: existing outputs skipped.
"""

import argparse
import os
import time
from pathlib import Path

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default=os.environ.get("AYEKOO_PROJECT_DIR", "/mnt/volume_d2wey28/projects/ayekoo-videos"))
    ap.add_argument("--input-dir", default="inpaint_input")
    ap.add_argument("--masks-dir", default="masks")
    ap.add_argument("--output-dir", default="inpainted")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    P = Path(args.project_dir)
    inp, masks, out = P / args.input_dir, P / args.masks_dir, P / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    from simple_lama_inpainting import SimpleLama
    lama = SimpleLama()

    names = sorted(p.name for p in inp.iterdir() if p.is_file())
    if args.limit > 0:
        names = names[: args.limit]
    t0 = time.time()
    done = skipped = errs = 0
    for name in names:
        stem = Path(name).stem
        dst = out / f"{stem}.jpg"
        if dst.exists():
            skipped += 1
            continue
        mask_p = masks / f"{stem}.png"
        if not mask_p.exists():
            continue
        try:
            img = Image.open(inp / name).convert("RGB")
            mask = Image.open(mask_p).convert("L")
            res = lama(img, mask)
            res.save(dst, quality=95)
            done += 1
        except Exception as e:  # noqa: BLE001
            errs += 1
            print(f"ERR {name}: {type(e).__name__}: {e}", flush=True)
        if done and done % 25 == 0:
            el = time.time() - t0
            print(f"{done} inpainted | {done / el:.1f} img/s | errs={errs}", flush=True)
    el = time.time() - t0
    print(f"DONE inpainted={done} skipped={skipped} errors={errs} "
          f"in {el:.0f}s ({done / el if el else 0:.1f} img/s)")


if __name__ == "__main__":
    main()
