#!/usr/bin/env python3
"""Merge detailed descriptions into final_dataset/metadata.csv.

Reads the JSONL produced by recaption_detailed.py and adds a `descriptive_text`
column to metadata.csv (and metadata.parquet).  Resumable runs may contain the
older key name `prompt_text`; both are accepted.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--prompts", default="detailed_prompts.jsonl")
    ap.add_argument("--metadata", default="final_dataset/metadata.csv")
    args = ap.parse_args()

    P = Path(args.project_dir)
    meta_path = P / args.metadata

    prompts = {}
    src = P / args.prompts
    with src.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            txt = r.get("descriptive_text") or r.get("prompt_text")
            if txt:
                prompts[str(r["image_id"]).strip()] = txt

    df = pd.read_csv(meta_path)
    df["descriptive_text"] = df["image_id"].astype(str).str.strip().map(prompts)

    order = list(df.columns)
    order.remove("descriptive_text")
    order.insert(order.index("caption") + 1, "descriptive_text")
    df = df[order]

    df.to_csv(meta_path, index=False)
    df.to_parquet(meta_path.with_suffix(".parquet"), index=False)

    got = df["descriptive_text"].notna().sum()
    print(f"descriptions: {len(prompts):,} read, {got:,}/{len(df):,} rows filled")
    print(f"wrote {meta_path} and {meta_path.with_suffix('.parquet')}")


if __name__ == "__main__":
    main()
