#!/usr/bin/env python3
"""Clean captions that mention on-screen / graphic elements.

Two Gemini text passes (no image needed):

  Stage A (classify): every unique kept caption -> 0 or 1
      0 = caption describes only the real-world scene
      1 = caption mentions an on-screen/graphic/interface element
  Stage B (rewrite): only the captions flagged 1 are rewritten to drop
      those references, describing the real-world scene instead.

Requests are batched and results are persisted INCREMENTALLY, so the job
is fully resumable (kill/restart anytime):
    captions_classify.jsonl : {"caption":..,"flag":0|1}
    captions_rewrite.jsonl  : {"caption":..,"clean":".."}

Final output `gemini_detections.clean.jsonl` is the input with each kept
caption replaced by its cleaned version (unchanged when flag == 0).

Usage:
    python clean_captions.py --project-dir "$AYEKOO_PROJECT_DIR" \
        --key-file /path/to/gemini_key --workers 32 --classify-batch 100
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
    "Label each caption below. Return JSON only.\nCaptions:\n"
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
        v = it.get("v")
        if isinstance(it.get("i"), int) and v in (0, 1, "0", "1"):
            out[it["i"]] = int(v)
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


class Appender:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.count = 0

    def add(self, obj):
        line = json.dumps(obj) + "\n"
        with self.lock:
            with self.path.open("a") as f:
                f.write(line)
            self.count += 1


def run_batches(fn, api_key, model, items, batch_size, workers,
                timeout, retries, label, on_item):
    total = len(items)
    done = 0
    errs = 0
    t0 = time.time()
    lock = threading.Lock()

    def work(idx_range):
        sub = [items[i] for i in idx_range]
        vals, err = fn(api_key, model, sub, timeout, retries)
        if vals is None:
            vals, err = fn(api_key, model, sub, timeout, retries)
        if vals is None:
            out = []
            for c in sub:
                v, e = fn(api_key, model, [c], timeout, retries)
                out.append(v[0] if v is not None else None)
            return idx_range, out, (err or "fallback")
        return idx_range, vals, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, br) for br in _chunks(list(range(total)), batch_size)]
        for fut in as_completed(futs):
            idx_range, vals, err = fut.result()
            with lock:
                for off, gi in enumerate(idx_range):
                    if vals[off] is not None:
                        on_item(gi, vals[off])
                done += len(idx_range)
                if err:
                    errs += 1
                if done % 500 == 0 or done >= total:
                    rate = done / max(1e-9, time.time() - t0)
                    log(f"  {label}: {done}/{total} | {rate:.0f}/s | "
                        f"{errs} fallbacks")
    return errs


def load_jsonl_map(path, key, val):
    d = {}
    if path.exists():
        for l in path.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                d[r[key]] = r[val]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    ap.add_argument("--detections", default="gemini_detections.jsonl")
    ap.add_argument("--out", default="gemini_detections.clean.jsonl")
    ap.add_argument("--classify-file", default="captions_classify.jsonl")
    ap.add_argument("--rewrite-file", default="captions_rewrite.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--key-file", default=DEFAULT_KEY_FILE)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--classify-batch", type=int, default=100)
    ap.add_argument("--rewrite-batch", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=5)
    args = ap.parse_args()

    if not args.key_file:
        raise SystemExit("need --key-file or AYEKOO_GEMINI_KEY_FILE")
    api_key = Path(args.key_file).read_text().strip()
    proj = Path(args.project_dir)
    det_path = proj / args.detections
    classify_path = proj / args.classify_file
    rewrite_path = proj / args.rewrite_file

    records = [json.loads(l) for l in det_path.read_text().splitlines() if l.strip()]
    unique, seen = [], set()
    for r in records:
        if r.get("error") or r.get("drop"):
            continue
        c = (r.get("caption") or "").strip()
        if c and c not in seen:
            seen.add(c)
            unique.append(c)
    log(f"{len(unique)} unique kept captions")

    flags = load_jsonl_map(classify_path, "caption", "flag")
    to_classify = [c for c in unique if c not in flags]
    log(f"stage A (classify): {len(to_classify)} remaining / {len(unique)}")
    if to_classify:
        app = Appender(classify_path)

        def on_flag(gi, v):
            c = to_classify[gi]
            flags[c] = int(v)
            app.add({"caption": c, "flag": int(v)})

        run_batches(classify_batch, api_key, args.model, to_classify,
                    args.classify_batch, args.workers, args.timeout,
                    args.retries, "classify", on_flag)
    nflag = sum(1 for c in unique if flags.get(c) == 1)
    log(f"stage A complete: {nflag} flagged of {len(unique)}")

    cleans = load_jsonl_map(rewrite_path, "caption", "clean")
    to_rewrite = [c for c in unique if flags.get(c) == 1 and not cleans.get(c)]
    log(f"stage B (rewrite): {len(to_rewrite)} remaining / {nflag} flagged")
    if to_rewrite:
        app = Appender(rewrite_path)

        def on_clean(gi, v):
            c = to_rewrite[gi]
            v = (v or "").strip()
            if v:
                cleans[c] = v
                app.add({"caption": c, "clean": v})

        run_batches(rewrite_batch, api_key, args.model, to_rewrite,
                    args.rewrite_batch, args.workers, args.timeout,
                    args.retries, "rewrite", on_clean)

    out_path = proj / args.out
    cleaned = 0
    with out_path.open("w") as f:
        for r in records:
            c = (r.get("caption") or "").strip()
            if flags.get(c) == 1 and cleans.get(c):
                r["caption"] = cleans[c]
                cleaned += 1
            f.write(json.dumps(r) + "\n")
    log(f"wrote {out_path.name}: {len(cleans)} rewritten captions, "
        f"{cleaned} records updated")


if __name__ == "__main__":
    main()
