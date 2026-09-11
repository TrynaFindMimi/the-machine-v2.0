from __future__ import annotations

import argparse
import sys

from app.registry import TEST_NAMES
from app.runner import run
from config.strings import USAGE


def _parse_args(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="the-machine — detección de manos/gestos",
        epilog=f"modos: {', '.join(TEST_NAMES)}",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=TEST_NAMES,
        help="modo inicial",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if args.mode is None:
        return 0
    return TEST_NAMES.index(args.mode)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    try:
        start = _parse_args(argv)
    except SystemExit as exc:
        if exc.code == 2 and argv and argv[0] not in TEST_NAMES and not argv[0].startswith("-"):
            names = ", ".join(TEST_NAMES)
            print(USAGE.format(names))
            raise SystemExit(1) from None
        raise
    run(start)


if __name__ == "__main__":
    main()
