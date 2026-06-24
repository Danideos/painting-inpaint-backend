"""Run a tiny Modal CPU function to verify local auth and deployment."""

from __future__ import annotations

import time

from painting_inpaint_backend.modal.smoke import app, ping


def main() -> int:
    started = time.perf_counter()
    with app.run():
        result = ping.remote()
    print({"result": result, "observed_seconds": time.perf_counter() - started})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
