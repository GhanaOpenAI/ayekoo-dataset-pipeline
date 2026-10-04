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
        path: data/train-*.parquet
---

# Ayekoo

**Author:** [Ghana Open AI](https://huggingface.co/ghanaopenai)

**Supported by** [Ghana NLP](https://ghananlp.org)

Ayekoo is a captioned still-image dataset built from **Ayekoo**, the Ghanaian
agriculture television programme broadcast on **UTV** (Ghana). Each frame is a
real photograph with on-screen overlays removed and no people, paired with a
short grounded English caption.

> **Acknowledgement.** This dataset is derived from the *Ayekoo* programme
> hosted on **UTV**. We gratefully acknowledge the *Ayekoo* team and UTV Ghana —
> their work creating and broadcasting the programme is what made this dataset
> possible. All underlying footage remains the property of its original
> creators.

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
5. **Detailed descriptions** — Gemini re-read every cleaned frame, with Ghana /
   agriculture context, and wrote a detailed, grounded `descriptive_text` for
   each image.

Code: [GhanaOpenAI/ayekoo-dataset-pipeline](https://github.com/GhanaOpenAI/ayekoo-dataset-pipeline)

## Dataset at a glance

| | |
|--|--|
| Frames | **88,158** |
| Source videos | 294 (*Ayekoo*, UTV) |
| Resolution | 1920×1080 |
| Captions | 1 grounded English sentence per frame (Gemini) |
| Overlays removed | 87,248 frames LaMa-inpainted; 910 frames had no overlay |
| People | none |

## Files

- `data/train-*-of-*.parquet` — frames + captions (an `image` column with embedded JPEGs and a `caption` column).
- `metadata.csv` — the same rows without image bytes, for quick inspection.

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
| `caption` | short grounded English caption (Gemini) |
| `descriptive_text` | detailed visual description of the frame (subject + attributes + setting + lighting + camera) |
| `had_overlay` | whether any overlay was detected |
| `overlay_inpainted` | whether this frame was LaMa-inpainted |
| `n_boxes` | number of detected overlay boxes |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("ghanaopenai/ayekoo", split="train")
print(ds[0]["caption"], ds[0]["image"].size)
```

## Acknowledgements

The images were extracted from ***Ayekoo***, the Ghanaian agricultural
television programme **hosted on UTV**. We thank the *Ayekoo* team and UTV
Ghana for producing and broadcasting the programme; it is the work that made
this dataset possible. Please credit *Ayekoo* (UTV) when using this dataset.

## License and attribution

Released under the MIT License. Built by [Ghana Open AI](https://huggingface.co/ghanaopenai).
The source footage is from the ***Ayekoo*** programme hosted on **UTV** (Ghana);
the programme and its creators retain all rights to the underlying video. This
dataset contains derived still frames with on-screen graphics removed and is
shared for research and non-commercial use with acknowledgement to *Ayekoo*
(UTV).
