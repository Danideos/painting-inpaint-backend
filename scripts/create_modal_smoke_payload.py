"""Create a provider-neutral local smoke-test payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from painting_inpaint_backend.core.image_io import image_to_base64


def make_images(size: int) -> tuple[Image.Image, Image.Image]:
    image = Image.new("RGB", (size, size), (166, 126, 84))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, size - 24, size - 24), outline=(70, 42, 24), width=5)
    draw.line((0, size // 2, size, size // 2), fill=(220, 190, 140), width=3)
    draw.ellipse((size // 3, size // 3, 2 * size // 3, 2 * size // 3), fill=(112, 62, 36))

    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    pad = size // 3
    mask_draw.rectangle((pad, pad, size - pad, size - pad), fill=255)
    return image, mask


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("generated/modal_smoke_payload.json"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument(
        "--method",
        choices=("flux_fill", "flux_canny_lanpaint"),
        default="flux_fill",
    )
    parser.add_argument("--include-control-image", action="store_true")
    args = parser.parse_args()

    image, mask = make_images(args.size)
    payload = {
        "method": args.method,
        "image_base64": image_to_base64(image, output_format="png"),
        "mask_base64": image_to_base64(mask, output_format="png"),
        "prompt": "DURER_RESTO",
        "partial_noise": 1.0,
        "num_inference_steps": 8,
        "seed": 123,
        "lora_scale": 1.0,
        "output_format": "png",
    }
    if args.method == "flux_fill":
        payload["guidance_scale"] = 30.0
    if args.include_control_image:
        payload["control_image_base64"] = image_to_base64(image, output_format="png")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
