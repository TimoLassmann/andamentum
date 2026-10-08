"""``python -m andamentum.compose.harness.examples DEST``: write the SaltLog example project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import make_example


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m andamentum.compose.harness.examples",
        description="Write the invented SaltLog example project into a new folder.",
    )
    parser.add_argument("dest", type=Path, help="folder to create (must not exist)")
    args = parser.parse_args(argv)
    project = make_example(args.dest)
    print(f"example project: {project.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
