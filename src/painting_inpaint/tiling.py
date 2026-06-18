"""Tile/window helpers for large painting experiments."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Tile:
    """A rectangular image window."""

    x: int
    y: int
    width: int
    height: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


def _pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value, value
    if len(value) != 2 or value[0] <= 0 or value[1] <= 0:
        raise ValueError(f"{name} must be a positive int or pair")
    return int(value[0]), int(value[1])


def tile_starts(length: int, window: int, stride: int) -> list[int]:
    """Return start coordinates that cover an axis including the final edge."""

    if length <= 0:
        raise ValueError("length must be positive")
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    if length <= window:
        return [0]

    starts = list(range(0, length - window + 1, stride))
    final = length - window
    if starts[-1] != final:
        starts.append(final)
    return starts


def generate_tiles(
    image_size: tuple[int, int],
    tile_size: int | tuple[int, int],
    stride: int | tuple[int, int],
) -> list[Tile]:
    """Generate covering tiles for ``image_size``."""

    image_w, image_h = image_size
    tile_w, tile_h = _pair(tile_size, "tile_size")
    stride_x, stride_y = _pair(stride, "stride")
    xs = tile_starts(image_w, min(tile_w, image_w), stride_x)
    ys = tile_starts(image_h, min(tile_h, image_h), stride_y)
    return [
        Tile(x=x, y=y, width=min(tile_w, image_w - x), height=min(tile_h, image_h - y))
        for y in ys
        for x in xs
    ]


def crop_tile(image: Image.Image, tile: Tile) -> Image.Image:
    """Crop one tile from an image."""

    return image.crop(tile.box)


def tile_from_top_left_fraction(
    image_size: tuple[int, int],
    tile_size: int | tuple[int, int],
    x_fraction: float,
    y_fraction: float,
) -> Tile:
    """Return a tile positioned by fractions of the available top-left range."""

    image_w, image_h = image_size
    tile_w, tile_h = _pair(tile_size, "tile_size")
    if image_w < tile_w or image_h < tile_h:
        raise ValueError(
            f"Image must be at least {tile_w}x{tile_h}; got {image_w}x{image_h}"
        )

    x_frac = max(0.0, min(1.0, float(x_fraction)))
    y_frac = max(0.0, min(1.0, float(y_fraction)))
    x = round(x_frac * (image_w - tile_w))
    y = round(y_frac * (image_h - tile_h))
    return Tile(x=int(x), y=int(y), width=tile_w, height=tile_h)


def crop_tile_set(
    tile: Tile,
    image: Image.Image,
    mask: Image.Image,
    reference: Image.Image | None = None,
    control: Image.Image | None = None,
) -> dict[str, Image.Image | None]:
    """Crop image, mask, and optional conditioning inputs with the same tile."""

    images = [image, mask, reference, control]
    sizes = [img.size for img in images if img is not None]
    if any(size != image.size for size in sizes):
        raise ValueError(f"All tiled inputs must share image size; got {sizes}")
    return {
        "image": crop_tile(image, tile),
        "mask": crop_tile(mask, tile),
        "reference": crop_tile(reference, tile) if reference is not None else None,
        "control": crop_tile(control, tile) if control is not None else None,
    }


def tile_intersects_mask(mask: Image.Image, tile: Tile, threshold: int = 127) -> bool:
    """Return whether a tile contains at least one editable mask pixel."""

    crop = crop_tile(mask.convert("L"), tile)
    arr = np.asarray(crop, dtype=np.uint8)
    return bool(np.any(arr > threshold))


def filter_tiles_intersecting_mask(
    tiles: Iterable[Tile],
    mask: Image.Image,
    threshold: int = 127,
) -> list[Tile]:
    """Keep only tiles intersecting the editable mask region."""

    return [tile for tile in tiles if tile_intersects_mask(mask, tile, threshold=threshold)]
