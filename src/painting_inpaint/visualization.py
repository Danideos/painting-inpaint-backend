"""Preview and contact-sheet helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def resize_to_long_side(
    image: Image.Image,
    long_side: int,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    """Resize an image so its longest side equals ``long_side``."""

    if long_side <= 0:
        raise ValueError("long_side must be positive")
    width, height = image.size
    scale = long_side / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if new_size == image.size:
        return image.copy()
    return image.resize(new_size, resample=resample)


def fit_to_max_side(image: Image.Image, max_side: int) -> Image.Image:
    """Return a copy constrained to ``max_side`` without upscaling."""

    if max_side <= 0:
        raise ValueError("max_side must be positive")
    img = image.convert("RGB")
    scale = min(max_side / img.width, max_side / img.height, 1.0)
    if scale >= 1.0:
        return img.copy()
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    return img.resize(size, Image.Resampling.LANCZOS)


def save_preview(
    image: Image.Image,
    output_path: str | Path,
    max_side: int = 1800,
) -> Image.Image:
    """Save and return an RGB preview constrained to ``max_side``."""

    preview = fit_to_max_side(image.convert("RGB"), max_side=max_side)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    preview.save(out)
    return preview


def make_comparison_sheet(
    images: list[Image.Image],
    labels: list[str],
    output_path: str | Path | None = None,
    max_panel_side: int = 512,
    padding: int = 16,
    label_height: int = 24,
) -> Image.Image:
    """Create a horizontal side-by-side comparison sheet."""

    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")
    if not images:
        raise ValueError("at least one image is required")

    panels = [fit_to_max_side(img, max_panel_side) for img in images]
    widths = [p.width for p in panels]
    heights = [p.height for p in panels]
    sheet_w = sum(widths) + padding * (len(panels) + 1)
    sheet_h = max(heights) + padding * 2 + label_height
    sheet = Image.new("RGB", (sheet_w, sheet_h), (248, 248, 248))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    x = padding
    for panel, label in zip(panels, labels, strict=True):
        draw.text((x, padding), label, fill=(20, 20, 20), font=font)
        y = padding + label_height
        sheet.paste(panel, (x, y))
        x += panel.width + padding

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out)
    return sheet


def draw_windows_overview(
    image: Image.Image,
    windows: Sequence[Mapping[str, object]],
    output_path: str | Path | None = None,
    max_side: int = 1800,
    colors: Sequence[tuple[int, int, int]] | None = None,
) -> Image.Image:
    """Draw labeled rectangular experiment windows on a downscaled preview."""

    preview = fit_to_max_side(image.convert("RGB"), max_side=max_side)
    scale_x = preview.width / image.width
    scale_y = preview.height / image.height
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    palette = colors or [
        (230, 57, 70),
        (29, 53, 87),
        (42, 157, 143),
        (244, 162, 97),
        (131, 56, 236),
    ]

    for idx, window in enumerate(windows):
        x = int(window["x"])
        y = int(window["y"])
        width = int(window["width"])
        height = int(window["height"])
        x0 = round(x * scale_x)
        y0 = round(y * scale_y)
        x1 = round((x + width) * scale_x)
        y1 = round((y + height) * scale_y)
        color = palette[idx % len(palette)]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=4)

        label = str(window.get("window_id", f"window_{idx + 1:02d}"))
        if "mask_fraction" in window:
            label = f"{label} mask={float(window['mask_fraction']):.3%}"
        text_box = draw.textbbox((x0 + 6, y0 + 6), label, font=font)
        draw.rectangle(text_box, fill=(255, 255, 255))
        draw.text((x0 + 6, y0 + 6), label, fill=color, font=font)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        preview.save(out)
    return preview
