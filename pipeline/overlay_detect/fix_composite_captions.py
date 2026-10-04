#!/usr/bin/env python3
"""Targeted cleanup of residual composite/image-referencing captions.

Rewrites only captions that refer to the image/composite itself or to
on-screen text; genuine real-world objects (physical signs, printed
labels) are returned unchanged.
"""
import argparse
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd

STRONG = re.compile(
    r"\blower.?third\b|\bticker\b|\bon.?screen\b|\btext overlay\b|\boverlay\b|"
    r"\bscreenshot\b|\bwatermark\b|\bsubtitle\b|\bcredits?\b|\bUI\b|"
    r"\bimage shows\b|\bsplit[- ]image\b|\bsplit[- ]screen\b|\bchannel bug\b",
    re.I,
)

PROMPT = (
    "You fix captions for a clean image-generation dataset. Rewrite ONLY those "
    "captions that refer to the image/composite/graphic itself or to text on "
    "screen (e.g. 'a split image shows X on the left and Y on the right', "
    "'a landscape image shows ...', 'the image shows', an on-screen overlay, "
    "lower-third or ticker). Turn those into plain real-world scene "
    "descriptions that do not mention an image, panel, side, split, frame or "
    "on-screen text. If a caption instead mentions a genuine real-world object "
    "- a physical sign beside a road, a printed label on a product, a logo "
    "printed on a sack - return it EXACTLY unchanged. Return JSON only. "
    "Captions:\n"
)

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "i": {"type": "INTEGER"},
                    "caption": {"type": "STRING"},
                },
                "required": ["i", "caption"],
            },
        }
    },
    "required": ["items"],
}


def load_cc(path):
    spec = importlib.util.spec_from_file_location("cc", path)
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)
    return cc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--cleaner", required=True)
    ap.add_argument("--metadata", default="final_dataset/metadata.csv")
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    args = ap.parse_args()

    P = Path(args.project_dir)
    cc = load_cc(args.cleaner)
    key = Path(args.key_file).read_text().strip()
    meta = P / args.metadata
    df = pd.read_csv(meta)
    caps = df["caption"].fillna("").astype(str)
    targets = caps[caps.str.contains(STRONG, na=False)].tolist()
    unique = list(dict.fromkeys(targets))
    print(f"rows={len(df)} matched={len(targets)} unique={len(unique)}")

    newmap = {}
    B = 40
    for k in range(0, len(unique), B):
        sub = unique[k:k + B]
        payload = json.dumps([{"i": i, "caption": c} for i, c in enumerate(sub)])
        text, err = cc.call_text(key, args.model, PROMPT + payload, SCHEMA, 180, 5)
        if text is None:
            print(f"  batch {k}: error {err}")
            continue
        try:
            items = cc.parse_json(text)["items"]
        except Exception as e:  # noqa: BLE001
            print(f"  batch {k}: parse error {e}")
            continue
        for it in items:
            if isinstance(it.get("i"), int) and it.get("caption"):
                newmap[sub[it["i"]]] = it["caption"].strip()
        print(f"  processed {min(k + B, len(unique))}/{len(unique)}")
    changed = 0
    for old, new in newmap.items():
        if new and new != old:
            df.loc[df["caption"] == old, "caption"] = new
            changed += 1
    print(f"rewritten unique captions: {changed}")
    df.to_csv(meta, index=False)
    try:
        df.to_parquet(meta.with_suffix(".parquet"), index=False)
    except Exception as e:  # noqa: BLE001
        print("parquet skipped:", e)
    rem = df["caption"].fillna("").astype(str).str.contains(STRONG, na=False)
    print(f"remaining matches: {int(rem.sum())}")
    for c in df.loc[rem, "caption"].head(20):
        print("  -", c)


if __name__ == "__main__":
    main()
