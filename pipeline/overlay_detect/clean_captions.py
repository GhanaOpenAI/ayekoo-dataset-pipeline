#!/usr/bin/env python3
"""Clean captions that mention on-screen / graphic elements.

Two Gemini text passes (no image needed):

  Stage A (classify): every unique caption -> 0 or 1
      0 = caption describes only the real-world scene
      1 = caption mentions an on-screen/graphic/interface element
  Stage B (rewrite): only the captions flagged 1 are rewritten to drop
      those references, describing the real-world scene instead.

All requests are batched and fully resumable (captions_map.jsonl).  The
output `gemini_detections.clean.jsonl` is a copy of the input with each
caption replaced by its cleaned version (unchanged when flag == 0).

Usage:
    python clean_captions.py --project-dir "$AYEKOO_PROJECT_DIR" \
        --detections gemini_detections.jsonl \
        --out gemini_detections.clean.jsonl
"""
import argparse
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_PROJECT_DIR = os.environ.get("AYEKOO_PROJECT_DIR", ".")
DEFAULT_MODEL = os.environ.get("AYEKOO_GEMINI_MODEL", "gemini-3.5-flash-lite")
DEFAULT_KEY_FILE = os.environ.get("AYEKOO_GEMINI_KEY_FILE", "")

CLASSIFY_PROMPT = (
    "You audit captions for an image-generation dataset. A good caption "
    "describes ONLY the real-world scene. Return 1 if the caption mentions "
    "any on-screen, graphic, or interface element added to the video: "
    "watermark, logo, banner, ticker, lower-third, subtitle, caption text, "
    "on-screen text, text overlay, sign/billboard text, channel bug, border, "
    "frame, UI, or phrases like 'the image shows', 'on screen', 'screenshot', "
    "'in the corner', 'at the bottom'. Otherwise return 0.\n\n"
    "Rank each caption below. Return JSON only.\nCaptions:\n"
)

REWRITE_PROMPT = (
    "Rewrite each caption so it describes ONLY the real-world scene and "
    "contains no mention of any on-screen, graphic, or interface element "
    "(watermark, logo, banner, ticker, lower-third, subtitle, text, sign "
    "text, overlay, screenshot, 'the image shows', 'on screen', etc.). Keep "
    "the same scene and as much concrete detail as possible; 10-30 words, "
    "one natural English sentence. If removing those references leaves "
    "nothing meaningful, describe the background scene that would remain. "
    "Do not add anything not already implied. Return JSON only. Captions:\n"
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
                    "v": {"type": "INTEGER", "enum": [0, 1]},
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
                    "caption": {"type": "STRING"},
                },
                "required": ["i", "caption"],
            },
        }
    },
    "required": ["items"],
}


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def call_text(api_key, model, prompt, schema, timeout, max_retries):
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "maxOutputTokens": 8192,
        },
    }
    data = json.dumps(body).encode()
    last = ""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                d = json.load(resp)
            return d["candidates"][0]["content"]["parts"][0]["text"], None
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode()[:200]}"
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(min(60, 2 ** attempt) + random.random())
                continue
            return None, last
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            if attempt < max_retries - 1:
                time.sleep(min(30, 2 ** attempt) + random.random())
                continue
            return None, last
    return None, last or "retries exhausted"


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _chunks(seq, n):
    for k in range(0, len(seq), n):
        yield seq[k:k + n]


def classify_batch(api_key, model, caps, timeout, retries):
    payload = json.dumps([{"i": i, "caption": c} for i, c in enumerate(caps)])
    text, err = call_text(api_key, model, CLASSIFY_PROMPT + payload,
                          CLASSIFY_SCHEMA, timeout, retries)
    if text is None:
        return None, err
    try:
        labels = parse_json(text)["labels"]
    except Exception as e:
        return None, f"parse: {e}"
    out = {}
    for it in labels:
        if isinstance(it.get("i"), int) and it.get("v") in (0, 1):
            out[it["i"]] = int(it["v"])
    if len(out) != len(caps):
        return None, f"misaligned: {len(out)}/{len(caps)}"
    return [out[i] for i in range(len(caps))], None


def rewrite_batch(api_key, model, caps, timeout, retries):
    payload = json.dumps([{"i": i, "caption": c} for i, c in enumerate(caps)])
    text, err = call_text(api_key, model, REWRITE_PROMPT + payload,
                          REWRITE_SCHEMA, timeout, retries)
    if text is None:
        return None, err
    try:
        items = parse_json(text)["items"]
    except Exception as e:
        return None, f"parse: {e}"
    out = {}
    for it in items:
        if isinstance(it.get("i"), int) and it.get("caption"):
            out[it["i"]] = it["caption"].strip()
    if len(out) != len(caps):
        return None, f"misaligned: {len(out)}/{len(caps)}"
    return [out[i] for i in range(len(caps))], None


def run_batches(fn, api_key, model, caps, batch_size, workers,
                timeout, retries, label):
    """fn returns (list_aligned_or_None, err). Falls back to per-item on failure."""
    results = [None] * len(caps)
    batches = list(_chunks(list(range(len(caps))), batch_size))
    done = 0
    errs = 0
    lock = threading.Lock()

    def work(idx_range):
        sub = [caps[i] for i in idx_range]
        vals, err = fn(api_key, model, sub, timeout, retries)
        if vals is not None:
            return idx_range, vals, None
        if len(idx_range) > 1:
            vals2, err2 = fn(api_key, model, sub, timeout, retries)
            if vals2 is not None:
                return idx_range, vals2, None
        out = []
        bad = 0
        for c in sub:
            v, e = fn(api_key, model, [c], timeout, retries)
            if v is not None:
                out.append(v[0])
            else:
                out.append(None)
                bad += 1
        return idx_range, out, (bad or err)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, br) for br in batches]
        for fut in as_completed(futs):
            idx_range, vals, err = fut.result()
            with lock:
                for off, i in enumerate(idx_range):
                    results[i] = vals[off]
                done += len(idx_range)
                if err:
                    errs += 1
                if done % 500 < batch_size or done == len(caps):
                    log(f"  {label}: {done}/{len(caps)} batches processed "
                        f"({errs} with per-item fallback)")
    return results, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    ap.add_argument("--detections", default="gemini_detections.jsonl")
    ap.add_argument("--out", default="gemini_detections.clean.jsonl")
    ap.add_argument("--map", default="captions_map.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--key-file", default=DEFAULT_KEY_FILE)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--classify-batch", type=int, default=50)
    ap.add_argument("--rewrite-batch", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    if not args.key_file:
        raise SystemExit("need --key-file or AYEKOO_GEMINI_KEY_FILE")
    api_key = Path(args.key_file).read_text().strip()
    proj = Path(args.project_dir)
    det_path = proj / args.detections
    out_path = proj / args.out
    map_path = proj / args.map

    records = [json.loads(l) for l in det_path.read_text().splitlines() if l.strip()]
    log(f"loaded {len(records)} detections from {det_path}")

    unique = []
    seen = set()
    for r in records:
        if r.get("error"):
            continue
        c = (r.get("caption") or "").strip()
        if c and c not in seen:
            seen.add(c)
            unique.append(c)
    log(f"{len(unique)} unique captions")

    mapping = {}
    if map_path.exists():
        for l in map_path.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                mapping[d["caption"]] = d

    to_classify = [c for c in unique if c not in mapping]
    log(f"stage A (classify): {len(to_classify)} to process")
    if to_classify:
        flags, errs = run_batches(
            classify_batch, api_key, args.model, to_classify,
            args.classify_batch, args.workers, args.timeout, args.retries,
            "classify")
        with map_path.open("a") as f:
            for c, v in zip(to_classify, flags):
                flag = int(v) if v in (0, 1) else 0
                rec = {"caption": c, "flag": flag}
                mapping[c] = rec
                f.write(json.dumps(rec) + "\n")
        log(f"stage A done ({errs} fallbacks)")

    to_rewrite = [c for c in unique
                  if mapping.get(c, {}).get("flag") == 1
                  and not mapping.get(c, {}).get("clean")]
    log(f"stage B (rewrite): {len(to_rewrite)} flagged captions")
    if to_rewrite:
        rewrites, errs = run_batches(
            rewrite_batch, api_key, args.model, to_rewrite,
            args.rewrite_batch, args.workers, args.timeout, args.retries,
            "rewrite")
        updates = {}
        for c, v in zip(to_rewrite, rewrites):
            if v:
                updates[c] = v
        log(f"stage B done: {len(updates)} rewritten ({errs} fallbacks)")
        tmp = map_path.with_suffix(".tmp")
        with tmp.open("w") as f:
            for c in unique:
                rec = mapping[c]
                if c in updates:
                    rec = {"caption": c, "flag": 1, "clean": updates[c]}
                f.write(json.dumps(rec) + "\n")
        tmp.replace(map_path)
        for c, v in updates.items():
            mapping[c]["clean"] = v

    cleaned = 0
    with out_path.open("w") as f:
        for r in records:
            c = (r.get("caption") or "").strip()
            m = mapping.get(c)
            if m and m.get("clean"):
                r["caption"] = m["clean"]
                cleaned += 1
            f.write(json.dumps(r) + "\n")
    flagged = sum(1 for c in unique if mapping.get(c, {}).get("flag") == 1)
    log(f"wrote {out_path.name}: {flagged} flagged captions, {cleaned} records rewritten")


if __name__ == "__main__":
    main()
