"""Download pinned upstream datasets and build local snapshots (needs network once).

Usage: uv run --group prep python scripts/prepare_data.py gsm8k
"""

from __future__ import annotations

import sys

from bioalloc.evaluation import gsm8k_data

PREPARERS = {"gsm8k": gsm8k_data.prepare}


def main(names: list[str]) -> int:
    for name in names or list(PREPARERS):
        path = PREPARERS[name]()
        print(f"{name}: manifest written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
