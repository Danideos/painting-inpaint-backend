"""Reference-image and mask alignment utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class AlignmentResult:
    """Aligned reference image plus metadata about how it was produced."""

    image: Image.Image
    metadata: dict[str, Any] = field(default_factory=dict)


def resize_reference_to_target(reference: Image.Image, target: Image.Image) -> AlignmentResult:
    """Resize a reference image to the target size as a conservative baseline alignment."""

    aligned = reference.convert("RGB").resize(target.size, Image.Resampling.LANCZOS)
    return AlignmentResult(
        image=aligned,
        metadata={"method": "resize_to_target", "target_size": list(target.size)},
    )


def estimate_homography(
    source_points_xy: np.ndarray | list[list[float]],
    target_points_xy: np.ndarray | list[list[float]],
) -> np.ndarray:
    """Estimate a projective transform mapping source coordinates to target coordinates.

    Points must be provided as ``[[x, y], ...]`` pairs. Four or more correspondences
    are accepted; with more than four points the least-squares SVD solution is used.
    """

    src = np.asarray(source_points_xy, dtype=np.float64)
    dst = np.asarray(target_points_xy, dtype=np.float64)
    if src.shape != dst.shape:
        raise ValueError(f"source and target point shapes differ: {src.shape} != {dst.shape}")
    if src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if src.shape[0] < 4:
        raise ValueError("at least four point pairs are required")

    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])

    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    homography = vh[-1].reshape(3, 3)
    if np.isclose(homography[2, 2], 0.0):
        raise ValueError("degenerate homography; check point ordering and separation")
    return homography / homography[2, 2]


def apply_homography_to_points(
    points_xy: np.ndarray | list[list[float]],
    source_to_target_homography: np.ndarray,
) -> np.ndarray:
    """Map ``[[x, y], ...]`` points through a homography."""

    points = np.asarray(points_xy, dtype=np.float64)
    homography = np.asarray(source_to_target_homography, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if homography.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3)")

    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    mapped = homogeneous @ homography.T
    return mapped[:, :2] / mapped[:, 2:3]


def homography_residuals(
    source_points_xy: np.ndarray | list[list[float]],
    target_points_xy: np.ndarray | list[list[float]],
    source_to_target_homography: np.ndarray,
) -> dict[str, Any]:
    """Return point residual diagnostics for a source-to-target homography."""

    src = np.asarray(source_points_xy, dtype=np.float64)
    dst = np.asarray(target_points_xy, dtype=np.float64)
    if src.shape != dst.shape:
        raise ValueError(f"source and target point shapes differ: {src.shape} != {dst.shape}")

    mapped = apply_homography_to_points(src, source_to_target_homography)
    residual_vectors = mapped - dst
    distances = np.linalg.norm(residual_vectors, axis=1)
    return {
        "mapped_points_xy": mapped.tolist(),
        "residual_vectors_xy": residual_vectors.tolist(),
        "residual_distances_px": distances.tolist(),
        "mean_residual_px": float(np.mean(distances)) if len(distances) else 0.0,
        "max_residual_px": float(np.max(distances)) if len(distances) else 0.0,
        "rms_residual_px": float(np.sqrt(np.mean(distances**2))) if len(distances) else 0.0,
    }


def scale_points_between_sizes(
    points_xy: np.ndarray | list[list[float]],
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> np.ndarray:
    """Scale ``[[x, y], ...]`` points from one image size to another."""

    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")

    from_w, from_h = from_size
    to_w, to_h = to_size
    if from_w <= 0 or from_h <= 0:
        raise ValueError(f"invalid source size: {from_size}")

    scaled = points.copy()
    scaled[:, 0] *= to_w / from_w
    scaled[:, 1] *= to_h / from_h
    return scaled


def pil_perspective_coefficients(source_to_target_homography: np.ndarray) -> tuple[float, ...]:
    """Return PIL perspective coefficients for a source-to-target homography.

    ``Image.transform(..., PERSPECTIVE, coeffs)`` expects output-to-input coefficients,
    so this function inverts the provided source-to-target matrix.
    """

    homography = np.asarray(source_to_target_homography, dtype=np.float64)
    if homography.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3)")

    inverse = np.linalg.inv(homography)
    inverse = inverse / inverse[2, 2]
    return tuple(float(v) for v in inverse.flatten()[:8])


def warp_perspective_pil(
    image: Image.Image,
    source_to_target_homography: np.ndarray,
    target_size: tuple[int, int],
    resample: Image.Resampling = Image.Resampling.BICUBIC,
    fillcolor: int | tuple[int, int, int] = 0,
) -> Image.Image:
    """Warp an image from source coordinates into the target coordinate system."""

    return image.transform(
        target_size,
        Image.Transform.PERSPECTIVE,
        pil_perspective_coefficients(source_to_target_homography),
        resample=resample,
        fillcolor=fillcolor,
    )


def warp_binary_mask(
    mask: Image.Image,
    source_to_target_homography: np.ndarray,
    target_size: tuple[int, int],
    threshold: int = 127,
) -> Image.Image:
    """Warp a mask with nearest-neighbor sampling and rebinarize it."""

    warped = warp_perspective_pil(
        mask.convert("L"),
        source_to_target_homography,
        target_size,
        resample=Image.Resampling.NEAREST,
        fillcolor=0,
    )
    arr = np.asarray(warped, dtype=np.uint8)
    return Image.fromarray((arr > threshold).astype(np.uint8) * 255)


def make_checkerboard_overlay(
    first: Image.Image,
    second: Image.Image,
    tile: int = 64,
) -> Image.Image:
    """Return a checkerboard preview alternating between two same-sized RGB images."""

    a = first.convert("RGB")
    b = second.convert("RGB")
    if a.size != b.size:
        raise ValueError(f"image sizes differ: {a.size} != {b.size}")
    if tile <= 0:
        raise ValueError("tile must be positive")

    arr_a = np.asarray(a, dtype=np.uint8)
    arr_b = np.asarray(b, dtype=np.uint8)
    yy, xx = np.indices((a.height, a.width))
    keep_first = ((xx // tile) + (yy // tile)) % 2 == 0
    out = arr_a.copy()
    out[~keep_first] = arr_b[~keep_first]
    return Image.fromarray(out)


def blend_images(first: Image.Image, second: Image.Image, alpha: float = 0.5) -> Image.Image:
    """Return a simple RGB blend of two same-sized images."""

    a = first.convert("RGB")
    b = second.convert("RGB")
    if a.size != b.size:
        raise ValueError(f"image sizes differ: {a.size} != {b.size}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    out = (1.0 - alpha) * arr_a + alpha * arr_b
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
