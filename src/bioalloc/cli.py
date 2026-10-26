"""Command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from bioalloc import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bioalloc", description=__doc__)
    parser.add_argument("--version", action="version", version=f"bioalloc {__version__}")
    parser.add_subparsers(dest="command")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
