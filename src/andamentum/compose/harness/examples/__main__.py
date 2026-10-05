"""``python -m andamentum.compose.harness.examples DEST``: write the SaltLog example project."""

from __future__ import annotations

import sys
from pathlib import Path

from . import make_example


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(
            "usage: python -m andamentum.compose.harness.examples DEST",
            file=sys.stderr,
        )
        return 2
    project = make_example(Path(argv[0]))
    print(f"example project: {project.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
