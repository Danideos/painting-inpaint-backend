"""Inspect expected paths in the Modal model Volume."""

from __future__ import annotations

import json

from painting_inpaint_backend.modal.volume_setup import app, inspect_volume


def main() -> int:
    with app.run():
        result = inspect_volume.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
