"""Common backend interface for inpainting methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class InpaintingRequest:
    """Input for an inpainting backend."""

    image: Image.Image
    mask: Image.Image
    config: Mapping[str, Any]
    output_dir: Path
    seed: int | None = None
    conditioning_image: Image.Image | None = None
    control_image: Image.Image | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InpaintingResult:
    """Output from an inpainting backend."""

    raw_image: Image.Image
    composite_image: Image.Image
    metadata: dict[str, Any] = field(default_factory=dict)


class InpaintingBackend(ABC):
    """Base class for all inpainting backends."""

    name: str = "base"

    @abstractmethod
    def run(self, request: InpaintingRequest) -> InpaintingResult:
        """Run inpainting for a request."""
