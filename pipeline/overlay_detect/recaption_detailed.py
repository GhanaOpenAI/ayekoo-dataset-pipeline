#!/usr/bin/env python3
"""Generate detailed, grounded descriptive text for the final frames.

For each image in the finalized dataset this asks Gemini for a rich description:

    leading sentence (subject + count + attributes + action + setting),
    then semicolon-separated facets: props, environment, lighting/time,
    camera framing, and Ghana/West-African location context.

The model is told the images come from Ghana and show agriculture/farming so it
can use plausible Ghanaian context that is consistent with what is visible,
without inventing details.

Output is appended to a JSONL file (one record per image) so the run is
fully resumable.
"""

import argparse
import base64
import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

DEFAULT_PROJECT_DIR = "/mnt/volume_d2wey28/projects/ayekoo-videos"
DEFAULT_MODEL = "gemini-3.5-flash-lite"

PROMPT = (
    "You write detailed, grounded captions that will be used as text-to-image "
    "training prompts. The images come from Ghana and show agriculture and "
    "farming; use that context only to inform plausible Ghanaian / West African "
    "details that are consistent with what is visible, and never assert what "
    "you cannot see.\n"
    "Describe ONLY what is visible. Produce:\n"
    "- one leading sentence naming the main subject(s) with count and visual "
    "attributes (colours, breeds, crop types, materials), the action/pose, and "
    "the immediate setting;\n"
    "- then 3-5 semicolon-separated facets covering: other key objects/props, "
    "the surrounding environment/background, lighting and time of day, camera "
    "framing/angle, and location/context if clearly implied.\n"
    "Rules: 30-60 words total; describe the real-world scene only; never mention "
    "text, logos, watermarks, captions, overlays or the image itself; do not use "
    "quality/style tags such as \"masterpiece\", \"8k\", \"photorealistic\", "
    "\"high resolution\"; do not invent details; do not name people.\n"
    "Respond with JSON only.\n"
    "Example: {\"descriptive_text\": \"About a dozen white-and-brown sheep and goats "
    "feed on dry hay inside an open-sided enclosure with rough timber posts and a "
    "corrugated roof; bare earth ground; soft overcast daylight; medium-wide "
    "eye-level shot; rural farm.\"}"
)

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {"descriptive_text": {"type": "STRING"}},
    "required": ["descriptive_text"],
}


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def call_gemini(api_key, model, image_path, prompt, timeout, max_retries):
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={api_key}")
    body = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
        ]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
            "maxOutputTokens": 512,
        },
    }
    data = json.dumps(body).encode()
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                d = json.load(resp)
            txt = d["candidates"][0]["content"]["parts"][0]["text"]
            return txt, d.get("usageMetadata"), None
        except urllib.error.HTTPError as e:
            code = e.code
            msg = e.read().decode()[:300]
            if code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(min(60, 2 ** attempt) + random.random())
                continue
            return None, None, f"HTTP {code}: {msg}"
        except Exception as e:  # noqa: BLE001
            if attempt < max_retries - 1:
                time.sleep(min(30, 2 ** attempt) + random.random())
                continue
            return None, None, f"{type(e).__name__}: {e}"
    return None, None, "retries exhausted"


def parse_json_object(text):
    if not text:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def parse_args():
    p = argparse.ArgumentParser(description="Detailed T2I prompt generation")
    p.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    p.add_argument("--metadata", default="final_dataset/metadata.csv")
    p.add_argument("--image-dir", default="final_dataset")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default="detailed_prompts.jsonl")
    p.add_argument("--api-key-file", default=".secrets/gemini_key")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max-retries", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    P = Path(args.project_dir)
    key = Path(P / args.api_key_file).read_text().strip()
    df = pd.read_csv(P / args.metadata)
    img_dir = P / args.image_dir
    rows = list(df.to_dict("records"))
    if args.sample > 0:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.sample]
    if args.limit > 0:
        rows = rows[: args.limit]

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = P / out_path
    done = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                done.add(str(json.loads(line)["image_id"]).strip())
            except Exception:  # noqa: BLE001
                pass
    todo = [r for r in rows if str(r["image_id"]).strip() not in done]
    log(f"rows={len(rows):,}  done={len(done):,}  todo={len(todo):,}")

    write_lock = threading.Lock()
    counts = {"ok": 0, "error": 0}
    t0 = time.time()
    processed = 0

    def handle(rec):
        img = img_dir / str(rec["image_filename"])
        img_id = str(rec["image_id"]).strip()
        txt, usage, err = call_gemini(
            key, args.model, img, PROMPT,
            args.timeout, args.max_retries)
        parsed = parse_json_object(txt) if not err else None
        prompt = (parsed or {}).get("descriptive_text")
        if isinstance(prompt, str):
            prompt = prompt.strip()[:800] or None
        out = {
            "image_id": img_id,
            "image_filename": rec.get("image_filename"),
            "descriptive_text": prompt,
            "model": args.model,
            "usage": usage,
            "error": err,
            "raw_response": txt,
            "ts": time.strftime("%FT%T"),
        }
        with write_lock:
            nonlocal processed
            processed += 1
            if err or not prompt:
                counts["error"] += 1
            else:
                counts["ok"] += 1
            with out_path.open("a") as fh:
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            if processed % 25 == 0 or processed == len(todo):
                el = time.time() - t0
                rate = processed / el if el else 0
                eta = (len(todo) - processed) / rate / 3600 if rate else 0
                log(f"{processed:,}/{len(todo):,} | ok={counts['ok']} "
                    f"err={counts['error']} | {rate:.2f} img/s ETA {eta:.1f}h")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(handle, r) for r in todo]
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc is not None:
                log(f"worker error: {type(exc).__name__}: {exc}")

    log(f"DONE processed={processed:,} -> {out_path}")
    log(f"summary {json.dumps(counts)}")


if __name__ == "__main__":
    main()
