#!/usr/bin/env python3
"""Publish the finalized Ayekoo dataset to the HuggingFace Hub.

Pushes final_dataset/ (frames + metadata) to a HuggingFace *dataset* repo,
default: ghanaopenai/ayekoo.

Requires a token with write access, e.g.:
    export HF_TOKEN=hf_xxx
    python upload_to_hf.py --repo-id ghanaopenai/ayekoo
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main():
    ap = argparse.ArgumentParser(description="Upload the Ayekoo dataset to HuggingFace")
    ap.add_argument("--project-dir",
                    default=os.environ.get("AYEKOO_PROJECT_DIR",
                                           "/mnt/volume_d2wey28/projects/ayekoo-videos"))
    ap.add_argument("--data-dir", default="final_dataset")
    ap.add_argument("--repo-id", default=os.environ.get("AYEKOO_HF_REPO",
                                                        "ghanaopenai/ayekoo"))
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()

    folder = Path(args.project_dir) / args.data_dir
    if not folder.exists():
        raise SystemExit(f"data dir not found: {folder}")

    api = HfApi(token=args.token)
    create_repo(args.repo_id, repo_type="dataset", exist_ok=True,
                private=args.private, token=args.token)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        commit_message="Upload Ayekoo frame dataset",
    )
    print(f"uploaded -> https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
