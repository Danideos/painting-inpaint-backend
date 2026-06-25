"""Publish the verified rank-64 FLUX-Canny LoRA to a private Hugging Face repo."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_REPO_ID = "danideos/durer-flux-canny-lora"
DEFAULT_WEIGHTS = Path(
    "local_assets/flux_canny_rank64_final/pytorch_lora_weights.safetensors"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required and will not be printed.")
    manifest = args.weights.parent / "artifact_manifest.json"
    if not args.weights.is_file() or not manifest.is_file():
        raise FileNotFoundError("Run prepare_flux_canny_lora.py before publishing.")

    from huggingface_hub import CommitOperationAdd, HfApi

    readme = """---
license: other
base_model: black-forest-labs/FLUX.1-Canny-dev
---

# Durer FLUX-Canny LoRA

Private deployment artifact for painting restoration. This is the final rank-64 LoRA
from training run `20260606_212825_flux1_canny_dev_curated_durer_lora_rank64`.
It was trained for FLUX.1-Canny-dev; LanPaint is used only during inference.
"""
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)
    commit = api.create_commit(
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Publish verified rank-64 final FLUX-Canny LoRA",
        operations=[
            CommitOperationAdd(
                path_in_repo="pytorch_lora_weights.safetensors",
                path_or_fileobj=str(args.weights),
            ),
            CommitOperationAdd(
                path_in_repo="artifact_manifest.json",
                path_or_fileobj=str(manifest),
            ),
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode("utf-8")),
        ],
        token=token,
    )
    result = {"repo_id": args.repo_id, "private": True, "commit": commit.oid}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
