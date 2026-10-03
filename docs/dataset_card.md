---
license: mit
pretty_name: Ayekoo
task_categories:
  - text-to-image
language:
  - en
tags:
  - image-captioning
  - ghana
  - west-africa
  - agriculture
  - image-inpainting
configs:
  - config_name: default
    data_files:
      - split: train
        path: metadata.csv
---

# Ayekoo

**Author:** [Ghana Open AI](https://huggingface.co/ghanaopenai)

Ayekoo is a captioned still-image dataset built from Ghanaian agriculture
television programming. Each frame is a real photograph with on-screen overlays
removed and no people, paired with a short grounded English caption.

## How it was built

1. **Frames** — one sharp frame every 2 seconds (sharpest in each window) from the
   source videos.
2. **People filter** — frames containing a person or any body part were dropped
   (Qwen2.5-VL).
3. **Overlay detection + captioning** — Gemini detected watermarks, logos and
   lower-third graphics, flagged non-photo/low-resolution frames, and wrote a
   one-sentence caption from the frame plus its source-video title/URL.
4. **Inpainting** — detected overlays were removed with LaMa, filling the whole
   graphic rectangle.

Code: [GhanaOpenAI/ayekoo-dataset-pipeline](https://github.com/GhanaOpenAI/ayekoo-dataset-pipeline)

## Files

- `*.jpg` — cleaned frames (1920×1080).
- `metadata.csv` — one row per frame.

### Metadata fields

| field | description |
|-------|-------------|
| `image_id` | stable frame id |
| `image_filename` | file name in this repository |
| `relative_path` | path to the frame |
| `video_id` | source YouTube video id |
| `video_title` | source video title |
| `video_url` | source video URL |
| `timestamp_seconds` | frame position in the source video |
| `caption` | grounded English caption (Gemini) |
| `had_overlay` | whether any overlay was detected |
| `overlay_inpainted` | whether this frame was LaMa-inpainted |
| `n_boxes` | number of detected overlay boxes |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("ghanaopenai/ayekoo", split="train")
print(ds[0]["caption"], ds[0]["image"].size)
```

## License and attribution

Released under the MIT License. Built by [Ghana Open AI](https://huggingface.co/ghanaopenai).
Source videos remain the property of their original creators; this dataset
contains derived still frames with on-screen graphics removed.
