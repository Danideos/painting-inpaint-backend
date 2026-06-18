"""Dataset preparation helpers for LoRA training."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Formatter
from typing import Any

from PIL import Image, UnidentifiedImageError

from painting_inpaint.experiments.manifest import write_json
from painting_inpaint.masks import allow_large_images
from painting_inpaint.paths import resolve_data_path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
HTML_MARKERS = (b"<!doctype html", b"<html", b"<head", b"<body")
ALLOWED_CAPTION_FIELDS = {"trigger_word"}


@dataclass(frozen=True)
class ImageRecord:
    """Inspection result for one image-like file."""

    path: str
    relative_path: str
    valid: bool
    format: str | None
    width: int | None
    height: int | None
    bytes: int | None
    problem: str | None = None


@dataclass(frozen=True)
class PreparedDataset:
    """Summary of a prepared Diffusers imagefolder dataset."""

    source_dir: Path
    prepared_dir: Path
    train_dir: Path
    metadata_path: Path
    manifest_path: Path
    image_count: int
    invalid_count: int
    trigger_word: str


def looks_like_html(path: Path) -> bool:
    """Return whether a file starts with common HTML markers."""

    try:
        head = path.read_bytes()[:512].lstrip().lower()
    except OSError:
        return False
    return any(head.startswith(marker) for marker in HTML_MARKERS)


def inspect_image(path: Path, root: Path) -> ImageRecord:
    """Inspect one image-like path without keeping the image open."""

    record = {
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "valid": False,
        "format": None,
        "width": None,
        "height": None,
        "bytes": path.stat().st_size if path.exists() else None,
        "problem": None,
    }

    if looks_like_html(path):
        record["problem"] = "looks_like_html"
        return ImageRecord(**record)

    try:
        with allow_large_images(), Image.open(path) as img:
            img.verify()
        with allow_large_images(), Image.open(path) as img:
            record["format"] = img.format
            record["width"], record["height"] = img.size
            record["valid"] = True
    except UnidentifiedImageError:
        record["problem"] = "unidentified_image"
    except Exception as exc:
        record["problem"] = f"{type(exc).__name__}: {exc}"

    return ImageRecord(**record)


def scan_image_folder(
    root: str | Path,
    *,
    suffixes: set[str] | None = None,
) -> list[ImageRecord]:
    """Return deterministic inspection records for image-like files under ``root``."""

    folder = Path(root)
    if not folder.exists():
        raise FileNotFoundError(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    allowed_suffixes = {suffix.lower() for suffix in (suffixes or IMAGE_SUFFIXES)}
    files = sorted(
        (
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in allowed_suffixes
        ),
        key=lambda p: str(p.relative_to(folder)).lower(),
    )
    return [inspect_image(path, folder) for path in files]


def validate_caption_template(template: str) -> None:
    """Reject filename-derived caption templates for style LoRA training."""

    fields = {
        field_name.split(".", 1)[0].split("[", 1)[0]
        for _literal, field_name, _format_spec, _conversion in Formatter().parse(template)
        if field_name
    }
    unsupported = sorted(fields - ALLOWED_CAPTION_FIELDS)
    if unsupported:
        joined = ", ".join(f"{{{field}}}" for field in unsupported)
        raise ValueError(
            "caption_template may only use {trigger_word}. "
            f"Unsupported field(s): {joined}. "
            "Filename-derived captions are disabled because they can be inaccurate."
        )


def caption_for_image(_record: ImageRecord, trigger_word: str, template: str) -> str:
    """Build the per-image training caption for one inspected record."""

    validate_caption_template(template)
    return template.format(trigger_word=trigger_word)


def _prepared_image_name(index: int, relative_path: str) -> str:
    src = Path(relative_path)
    safe_stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", src.stem).strip("._-") or "image"
    return f"{index:05d}_{safe_stem}{src.suffix.lower()}"


def _write_metadata_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def prepare_lora_dataset(
    dataset_config: dict[str, Any],
    *,
    overwrite: bool | None = None,
) -> PreparedDataset:
    """Create a captioned Diffusers imagefolder dataset from an image directory."""

    source_dir = resolve_data_path(dataset_config.get("source_dir"))
    prepared_dir = resolve_data_path(dataset_config.get("prepared_dir"))
    if source_dir is None or prepared_dir is None:
        raise ValueError("Dataset config must define source_dir and prepared_dir.")

    source_dir = source_dir.resolve()
    prepared_dir = prepared_dir.resolve()
    train_dir = prepared_dir / "train"
    metadata_filename = dataset_config.get("metadata_filename", "metadata.jsonl")
    metadata_path = train_dir / metadata_filename
    manifest_path = prepared_dir / "dataset_manifest.json"
    trigger_word = str(dataset_config.get("trigger_word", "")).strip()
    if not trigger_word:
        raise ValueError("Dataset config must define a non-empty trigger_word.")

    caption_template = dataset_config.get(
        "caption_template",
        "{trigger_word}",
    )
    validate_caption_template(caption_template)
    do_overwrite = dataset_config.get("overwrite", False) if overwrite is None else overwrite

    records = scan_image_folder(source_dir)
    valid_records = [record for record in records if record.valid]
    min_images = int(dataset_config.get("min_images", 1))
    if len(valid_records) < min_images:
        raise ValueError(
            f"Expected at least {min_images} valid image(s), found "
            f"{len(valid_records)} in {source_dir}"
        )

    if prepared_dir.exists():
        if not do_overwrite:
            raise FileExistsError(f"Prepared dataset already exists: {prepared_dir}")
        shutil.rmtree(prepared_dir)
    train_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, str]] = []
    copied_images: list[dict[str, Any]] = []
    for index, record in enumerate(valid_records, start=1):
        dest_name = _prepared_image_name(index, record.relative_path)
        src_path = source_dir / record.relative_path
        dest_path = train_dir / dest_name
        shutil.copy2(src_path, dest_path)
        caption = caption_for_image(record, trigger_word, caption_template)
        metadata_rows.append({"file_name": dest_name, "text": caption})
        copied_images.append(
            {
                "source": record.relative_path,
                "prepared": f"train/{dest_name}",
                "caption": caption,
                "width": record.width,
                "height": record.height,
                "format": record.format,
            }
        )

    _write_metadata_jsonl(metadata_path, metadata_rows)

    manifest = {
        "name": dataset_config.get("name"),
        "kind": dataset_config.get("kind"),
        "source_dir": str(source_dir),
        "prepared_dir": str(prepared_dir),
        "train_dir": str(train_dir),
        "metadata_path": str(metadata_path),
        "trigger_word": trigger_word,
        "caption_template": caption_template,
        "image_count": len(valid_records),
        "invalid_count": len(records) - len(valid_records),
        "invalid_images": [asdict(record) for record in records if not record.valid],
        "images": copied_images,
    }
    write_json(manifest_path, manifest)

    return PreparedDataset(
        source_dir=source_dir,
        prepared_dir=prepared_dir,
        train_dir=train_dir,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        image_count=len(valid_records),
        invalid_count=len(records) - len(valid_records),
        trigger_word=trigger_word,
    )
