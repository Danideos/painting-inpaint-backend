"""Populate the Modal Volume with SDXL BrushNet weights."""

from __future__ import annotations

import argparse
import json

from painting_inpaint_backend.modal.sd_volume_setup import app, populate_sdxl_brushnet_volume


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with app.run():
        result = populate_sdxl_brushnet_volume.remote(force=args.force)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

