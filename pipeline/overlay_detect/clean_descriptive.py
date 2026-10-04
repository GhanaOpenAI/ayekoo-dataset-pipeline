#!/usr/bin/env python3
"""Clean descriptive_text values that mention text or graphic elements.

Same two-pass design as clean_captions.py, but a cheap regex prefilter keeps
the Gemini calls down to the few candidates that could mention text:

  Stage A (classify): candidate description -> 0 or 1
      0 = describes only the real-world scene
      1 = mentions visible text/writing/printed characters, or an added
          on-screen/graphic element
  Stage B (rewrite): only the ones flagged 1 are rewritten to drop those
      references and describe the objects themselves.

Output `detailed_prompts.clean.jsonl` is the input with each description
replaced by its cleaned version (unchanged when flag == 0).

Usage:
    python clean_descriptive.py --project-dir "$AYEKOO_PROJECT_DIR" \
        --key-file /path/to/gemini_key --workers 32
"""
import argparse
import json
import os
import re
from pathlib import Path

from clean_captions import (Appender, DEFAULT_MODEL, call_text, load_jsonl_map,
                            parse_json, run_batches)

DEFAULT_PROJECT_DIR = os.environ.get("AYEKOO_PROJECT_DIR", ".")

# Cheap prefilter: anything that could plausibly reference visible text or an
# added graphic element.  Gemini stage A makes the actual call.
TEXTY = re.compile(
    r"\b(watermarks?|logos?|logotypes?|labels?|labell?ed|printed|prints?|printing"
    r"|signage|signboards?|signs?|banners?|subtitles?|captions?|overlays?|texts?"
    r"|wording|lettering|inscriptions?|writing|written|placards?|posters?"
    r"|billboards?|tickers?|stickers?|seals?|stamps?|screen|screenshot)\b"
    r"|\bon screen\b|\bthe image\b|\bthis image\b|\bin the corner\b"
    r"|\bat the bottom\b|\bforeground text\b",
    re.I,
)

CLASSIFY_PROMPT = (
    "You audit image descriptions for an image-generation dataset. A good "
    "description covers ONLY the real-world scene. Return 1 if it mentions any "
    "visible text or writing (printed, painted, engraved or handwritten "
    "characters), including text on packaging, containers, labels, product "
    "names, price tags, road signs, signboards, banners, posters, placards and "
    "vehicle branding; or any added on-screen/graphic element: watermark, logo, "
    "channel bug, subtitle, caption, ticker, lower-third, border, UI; or "
    "phrases like 'the image shows', 'on screen', 'screenshot', 'in the "
    "corner', 'at the bottom'. Otherwise return 0.\n\n"
    "Label each description below. Return JSON only.\nDescriptions:\n"
)

REWRITE_PROMPT = (
    "Rewrite each image description so it covers ONLY the real-world scene and "
    "contains NO mention of any text, writing, printed characters, labels, "
    "packaging text, signs, signboards, banners, posters, placards, subtitles, "
    "watermarks, logos, captions or on-screen graphics. Describe the objects "
    "themselves instead of what is written on them. Keep the same scene and as "
    "much concrete detail as possible: a leading sentence (subject + count + "
    "attributes + action + setting) followed by 3-5 semicolon-separated facets "
    "(other objects, environment/background, lighting and time of day, camera "
    "framing/angle, and Ghana/West-African agricultural context), 30-60 words "
    "total. Do not add anything not already implied. Return JSON only. "
    "Descriptions:\n"
)

CLASSIFY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "labels": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "i": {"type": "INTEGER"},
                    "v": {"type": "STRING", "enum": ["0", "1"]},
                },
                "required": ["i", "v"],
            },
        }
    },
    "required": ["labels"],
}

REWRITE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "i": {"type": "INTEGER"},
                    "description": {"type": "STRING"},
                },
                "required": ["i", "description"],
            },
        }
    },
    "required": ["items"],
}


def log(msg):
    import time
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def classify_batch(api_key, model, descs, timeout, retries):
    payload = json.dumps([{"i": i, "description": d} for i, d in enumerate(descs)])
    text, err = call_text(api_key, model, CLASSIFY_PROMPT + payload,
                          CLASSIFY_SCHEMA, timeout, retries)
    if text is None:
        return None, err
    try:
        labels = parse_json(text)["labels"]
    except Exception as e:  # noqa: BLE001
        return None, f"parse: {e}"
    out = {}
    for it in labels:
        v = it.get("v")
        if isinstance(it.get("i"), int) and v in (0, 1, "0", "1"):
            out[it["i"]] = int(v)
    if len(out) != len(descs):
        return None, f"misaligned: {len(out)}/{len(descs)}"
    return [out[i] for i in range(len(descs))], None


def rewrite_batch(api_key, model, descs, timeout, retries):
    payload = json.dumps([{"i": i, "description": d} for i, d in enumerate(descs)])
    text, err = call_text(api_key, model, REWRITE_PROMPT + payload,
                          REWRITE_SCHEMA, timeout, retries)
    if text is None:
        return None, err
    try:
        items = parse_json(text)["items"]
    except Exception as e:  # noqa: BLE001
        return None, f"parse: {e}"
    out = {}
    for it in items:
        if isinstance(it.get("i"), int) and it.get("description"):
            out[it["i"]] = it["description"].strip()
    if len(out) != len(descs):
        return None, f"misaligned: {len(out)}/{len(descs)}"
    return [out[i] for i in range(len(descs))], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    ap.add_argument("--prompts", default="detailed_prompts.jsonl")
    ap.add_argument("--out", default="detailed_prompts.clean.jsonl")
    ap.add_argument("--classify-file", default="descriptive_classify.jsonl")
    ap.add_argument("--rewrite-file", default="descriptive_rewrite.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--key-file", default="")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--classify-batch", type=int, default=20)
    ap.add_argument("--rewrite-batch", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=5)
    args = ap.parse_args()

    if not args.key_file:
        raise SystemExit("need --key-file")
    api_key = Path(args.key_file).read_text().strip()
    proj = Path(args.project_dir)
    src = proj / args.prompts

    records = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    unique, seen = [], set()
    for r in records:
        d = (r.get("descriptive_text") or "").strip()
        if d and d not in seen:
            seen.add(d)
            unique.append(d)
    candidates = [d for d in unique if TEXTY.search(d)]
    log(f"{len(records):,} records, {len(unique):,} unique descriptions, "
        f"{len(candidates):,} regex candidates")

    classify_path = proj / args.classify_file
    rewrite_path = proj / args.rewrite_file

    flags = load_jsonl_map(classify_path, "description", "flag")
    to_classify = [d for d in candidates if d not in flags]
    log(f"stage A (classify): {len(to_classify)} remaining / {len(candidates)}")
    if to_classify:
        app = Appender(classify_path)

        def on_flag(gi, v):
            d = to_classify[gi]
            flags[d] = int(v)
            app.add({"description": d, "flag": int(v)})

        run_batches(classify_batch, api_key, args.model, to_classify,
                    args.classify_batch, args.workers, args.timeout,
                    args.retries, "classify", on_flag)
    nflag = sum(1 for d in candidates if flags.get(d) == 1)
    log(f"stage A complete: {nflag:,} flagged of {len(candidates):,}")

    cleans = load_jsonl_map(rewrite_path, "description", "clean")
    to_rewrite = [d for d in candidates if flags.get(d) == 1 and not cleans.get(d)]
    log(f"stage B (rewrite): {len(to_rewrite)} remaining / {nflag:,} flagged")
    if to_rewrite:
        app = Appender(rewrite_path)

        def on_clean(gi, v):
            d = to_rewrite[gi]
            v = (v or "").strip()
            if v:
                cleans[d] = v
                app.add({"description": d, "clean": v})

        run_batches(rewrite_batch, api_key, args.model, to_rewrite,
                    args.rewrite_batch, args.workers, args.timeout,
                    args.retries, "rewrite", on_clean)

    out_path = proj / args.out
    cleaned = 0
    with out_path.open("w") as f:
        for r in records:
            d = (r.get("descriptive_text") or "").strip()
            if flags.get(d) == 1 and cleans.get(d):
                r["descriptive_text"] = cleans[d]
                cleaned += 1
            f.write(json.dumps(r) + "\n")
    log(f"wrote {out_path.name}: {len(cleans):,} rewritten, "
        f"{cleaned:,} records updated")


if __name__ == "__main__":
    main()