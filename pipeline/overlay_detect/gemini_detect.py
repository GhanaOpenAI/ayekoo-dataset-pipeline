#!/usr/bin/env python3
"""Detect watermarks/logos/lower-third overlays and flag unsuitable frames via Gemini.

For every no_people frame we ask Gemini to:
  * return bounding boxes for watermarks / logos / lower-third text overlays
    as [ymin, xmin, ymax, xmax] on a 0-1000 normalized scale, and
  * decide whether the frame should be dropped (person / graphic / low_resolution).

Results are appended to a JSONL file (one record per image) so the run is fully
resumable. Optionally draws the detected boxes onto copies for visual review.
"""

import argparse
import base64
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

DEFAULT_PROJECT_DIR = os.environ.get("AYEKOO_PROJECT_DIR", "/mnt/volume_d2wey28/projects/ayekoo-videos")
DEFAULT_MODEL = "gemini-3.5-flash-lite"

PROMPT = (
    "You are preparing a photo dataset for training an image generation model. "
    "Analyze the image and respond with ONLY a JSON object, no markdown.\n"
    "Tasks:\n"
    "1. Detection: locate every watermark, logo, text overlay and graphic "
    "overlay burned into the image (watermarks, corner logos, captions, name "
    "straps, tickers, subtitles, headers, footers, banners, lower-third boxes). "
    "For each give a bounding box [ymin, xmin, ymax, xmax] on a 0-1000 "
    "normalized scale (0=top/left, 1000=bottom/right) and a label from: "
    "watermark, logo, lower_third_text, graphic_overlay.\n"
    "IMPORTANT graphic_overlay: whenever text or a logo sits inside a coloured or "
    "rectangular graphic (a lower-third box, banner, bar, or header/footer "
    "strip), return a SEPARATE box labelled graphic_overlay that outlines the "
    "ENTIRE rectangle of that graphic, including the parts with no text. If the "
    "band spans the full image width, its box MUST have xmin=0 and xmax=1000. "
    "This box may overlap the text/logo boxes.\n"
    "Rules: a box must never be smaller than the visible overlay; do not return "
    "individual words or letters as separate boxes; each distinct region must be "
    "listed only once; each box must be exactly four numbers.\n"
    "2. Drop decision: set \"drop\" true if the image is NOT suitable as a clean "
    "training photo, i.e. ANY of: it contains a person or people or any body part "
    "(face, head, hand, arm, leg, torso, silhouette); it is a graphic or designed "
    "image rather than a real photograph (poster, title card, slide, chart, "
    "diagram, drawing, logo-only, text-heavy graphic); or it is low-resolution, "
    "blurry, or too small to be usable for training.\n"
    "3. Caption: write ONE concise English sentence (10-30 words) describing the "
    "main content of the photo: scene, setting, notable objects, activity and "
    "mood. Describe only what is actually visible in the image. If a video "
    "title/URL is provided below, use it only as background context; never "
    "invent details that are not in the image and do not name people.\n"
    "Return a JSON object with exactly these keys: \"drop\" (boolean), "
    "\"drop_reason\" (one of \"none\", \"person\", \"graphic\", "
    "\"low_resolution\"), \"caption\" (string), and \"boxes\" (array; empty if "
    "nothing detected). Respond with JSON only.\n"
    "Example response:\n"
    "{\"drop\": false, \"drop_reason\": \"none\", "
    "\"caption\": \"An outdoor market scene with stalls of colourful produce.\", "
    "\"boxes\": "
    "[{\"label\": \"graphic_overlay\", \"box\": [820, 0, 1000, 1000]}, "
    "{\"label\": \"lower_third_text\", \"box\": [860, 40, 960, 620]}, "
    "{\"label\": \"watermark\", \"box\": [902, 848, 985, 998]}]}"
)


def build_prompt(title, url):
    """Append the source-video metadata as context for the caption."""
    if not (title or url):
        return PROMPT
    return (PROMPT + "Video context (background only, do not copy verbatim): "
            f"title=\"{title or ''}\", url={url or ''}.\n")


RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "drop": {"type": "BOOLEAN"},
        "drop_reason": {"type": "STRING",
                        "enum": ["person", "graphic", "low_resolution", "none"]},
        "caption": {"type": "STRING"},
        "boxes": {
            "type": "ARRAY",
            "maxItems": 20,
            "items": {
                "type": "OBJECT",
                "properties": {
                    "label": {"type": "STRING",
                              "enum": ["watermark", "logo", "lower_third_text",
                                       "graphic_overlay"]},
                    "box": {"type": "ARRAY", "maxItems": 4, "items": {"type": "NUMBER"}},
                },
                "required": ["label", "box"],
            },
        },
    },
    "required": ["drop", "drop_reason", "caption", "boxes"],
}

_write_lock = threading.Lock()


def log(msg: str):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def call_gemini(api_key, model, image_path, prompt, timeout, thinking_budget, max_retries):
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={api_key}")
    gen = {"temperature": 0, "responseMimeType": "application/json",
           "responseSchema": RESPONSE_SCHEMA, "maxOutputTokens": 2048}
    if thinking_budget is not None and thinking_budget >= 0:
        gen["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    body = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
        ]}],
        "generationConfig": gen,
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
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
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


def draw_boxes(image_path, boxes, out_path):
    try:
        from PIL import Image, ImageDraw
    except Exception:  # noqa: BLE001
        return
    try:
        im = Image.open(image_path).convert("RGB")
    except Exception:  # noqa: BLE001
        return
    w, h = im.size
    dr = ImageDraw.Draw(im)
    for b in boxes or []:
        box = b.get("box")
        if not box or len(box) != 4:
            continue
        ymin, xmin, ymax, xmax = [max(0, min(1000, float(v))) for v in box]
        col = (0, 0, 255) if b.get("label") == "graphic_overlay" else (255, 0, 0)
        dr.rectangle([xmin / 1000 * w, ymin / 1000 * h,
                      xmax / 1000 * w, ymax / 1000 * h], outline=col, width=3)
        dr.text((xmin / 1000 * w + 4, ymin / 1000 * h + 4),
                str(b.get("label", "")), fill=col)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, quality=90)


def parse_args():
    p = argparse.ArgumentParser(description="Gemini watermark/logo detection + drop filter")
    p.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    p.add_argument("--metadata", default="", help="defaults to <keep-dir>/metadata.csv")
    p.add_argument("--keep-dir", default="no_people")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default="gemini_detections.jsonl")
    p.add_argument("--api-key-file", default=".secrets/gemini_key")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--thinking-budget", type=int, default=-1,
                   help="-1 omits thinkingConfig (default; required for 3.5-lite); "
                        "0 disables thinking where supported")
    p.add_argument("--overlay-dir", default="")
    return p.parse_args()


def main():
    args = parse_args()
    P = Path(args.project_dir)
    key = Path(P / args.api_key_file).read_text().strip()
    meta = Path(args.metadata) if args.metadata else P / args.keep_dir / "metadata.csv"
    df = pd.read_csv(meta)
    df = df[df["relative_path"].map(
        lambda r: (P / str(r).strip()).exists() if pd.notna(r) else False)]
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
                done.add(json.loads(line)["image_id"])
            except Exception:  # noqa: BLE001
                pass
    todo = [r for r in rows if str(r["image_id"]).strip() not in done]
    log(f"metadata rows={len(rows):,}  already done={len(done):,}  to process={len(todo):,}")

    write_lock = threading.Lock()
    counts = {"keep": 0, "drop": 0, "error": 0, "boxes": 0}
    t0 = time.time()
    processed = 0

    def handle(rec):
        img = P / str(rec["relative_path"]).strip()
        img_id = str(rec["image_id"]).strip()
        txt, usage, err = call_gemini(
            key, args.model, img,
            build_prompt(rec.get("video_title"), rec.get("video_url")),
            args.timeout, args.thinking_budget, args.max_retries)
        parsed = parse_json_object(txt) if not err else None
        out = {
            "image_id": img_id,
            "image_filename": rec.get("image_filename"),
            "relative_path": rec.get("relative_path"),
            "video_id": rec.get("video_id"),
            "video_title": rec.get("video_title"),
            "video_url": rec.get("video_url"),
            "timestamp_seconds": rec.get("timestamp_seconds"),
            "model": args.model,
            "drop": None,
            "drop_reason": None,
            "caption": None,
            "boxes": [],
            "usage": usage,
            "error": err,
            "raw_response": txt,
            "ts": time.strftime("%FT%T"),
        }
        if parsed is not None:
            out["drop"] = bool(parsed.get("drop"))
            dr = parsed.get("drop_reason")
            out["drop_reason"] = None if dr in (None, "none", "null", "") else dr
            cap = parsed.get("caption")
            out["caption"] = (str(cap).strip()[:400] if cap else None)
            valid = []
            seen = set()
            for b in parsed.get("boxes") or []:
                if not (isinstance(b, dict) and isinstance(b.get("box"), list)
                        and len(b["box"]) == 4):
                    continue
                try:
                    vals = [max(0.0, min(1000.0, float(x))) for x in b["box"]]
                except (TypeError, ValueError):
                    continue
                ymin, xmin, ymax, xmax = vals
                if ymax <= ymin or xmax <= xmin:
                    continue
                sig = (b.get("label"), tuple(round(v) for v in vals))
                if sig in seen:
                    continue
                seen.add(sig)
                valid.append({"label": b.get("label"), "box": vals})
            out["boxes"] = valid
            if args.overlay_dir:
                draw_boxes(img, out["boxes"], Path(args.overlay_dir) / Path(img).name)
        with write_lock:
            nonlocal processed
            processed += 1
            if err:
                counts["error"] += 1
            elif out["drop"]:
                counts["drop"] += 1
            else:
                counts["keep"] += 1
            if out["boxes"]:
                counts["boxes"] += 1
            with out_path.open("a") as fh:
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            if processed % 25 == 0 or processed == len(todo):
                el = time.time() - t0
                rate = processed / el if el else 0
                eta = (len(todo) - processed) / rate / 3600 if rate else 0
                log(f"{processed:,}/{len(todo):,} | keep={counts['keep']} "
                    f"drop={counts['drop']} err={counts['error']} "
                    f"with_boxes={counts['boxes']} | {rate:.2f} img/s ETA {eta:.1f}h")

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
