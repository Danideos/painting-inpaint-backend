"""Extract and verify only the selected rank-64 final FLUX-Canny LoRA."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

ARCHIVE_MEMBER = (
    "20260606_212825_flux1_canny_dev_curated_durer_lora_rank64/"
    "pytorch_lora_weights.safetensors"
)
EXPECTED_SIZE = 359_115_856
EXPECTED_RANK = 64
DEFAULT_OUTPUT = Path(
    "local_assets/flux_canny_rank64_final/pytorch_lora_weights.safetensors"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_safetensors(path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    ranks: set[int] = set()
    tensor_count = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        tensor_count = len(keys)
        for key in keys:
            shape = tuple(handle.get_slice(key).get_shape())
            lowered = key.lower()
            if "lora_a" in lowered and len(shape) >= 1:
                ranks.add(int(shape[0]))
            elif "lora_b" in lowered and len(shape) >= 2:
                ranks.add(int(shape[1]))
        metadata = dict(handle.metadata() or {})
    if ranks != {EXPECTED_RANK}:
        raise RuntimeError(f"Expected only LoRA rank {EXPECTED_RANK}, found {sorted(ranks)}.")
    return {
        "tensor_count": tensor_count,
        "detected_ranks": sorted(ranks),
        "safetensors_metadata": metadata,
    }


def prepare(archive: Path, output: Path, *, force: bool = False) -> dict[str, Any]:
    if not archive.is_file():
        raise FileNotFoundError(f"Missing FLUX-Canny training archive: {archive}")
    if output.exists() and not force:
        raise FileExistsError(f"Output already exists; pass --force to replace it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            member = bundle.getmember(ARCHIVE_MEMBER)
            if not member.isfile():
                raise RuntimeError(f"Archive member is not a file: {ARCHIVE_MEMBER}")
            if member.size != EXPECTED_SIZE:
                raise RuntimeError(
                    f"Expected archive member size {EXPECTED_SIZE}, found {member.size}."
                )
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read archive member: {ARCHIVE_MEMBER}")
            with source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        if temporary.stat().st_size != EXPECTED_SIZE:
            raise RuntimeError("Extracted LoRA size does not match the archive manifest.")
        inspection = _inspect_safetensors(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "source_archive": str(archive.resolve()),
        "archive_member": ARCHIVE_MEMBER,
        "output": str(output.resolve()),
        "size_bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "rank": EXPECTED_RANK,
        "checkpoint": "final",
        "training_run": "20260606_212825_flux1_canny_dev_curated_durer_lora_rank64",
        **inspection,
    }
    manifest_path = output.parent / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = prepare(args.archive, args.output, force=args.force)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
