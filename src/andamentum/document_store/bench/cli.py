"""``python -m andamentum.document_store.bench`` — the retrieval benchmark CLI.

    fetch    --workdir DIR [--suite S]          download raw dev sets once
    freeze   --workdir DIR --suite S [--dev]    draw and record the sample (once, ever)
    run      --workdir DIR --suite S --arm A --tier T [--embedding-model M]
    show     RUN_DIR                            print a run's summary
    compare  RUN_A RUN_B                        paired-bootstrap B - A

``--workdir`` has no default: datasets, the embedding cache, built stores and
run records live wherever the caller says, never in a hidden home directory.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .arms import ARMS
from .data import RUN_TIERS, SOURCES, fetch, freeze, freeze_dev
from .runner import compare, format_summary, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m andamentum.document_store.bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("--workdir", type=Path, required=True)
    p_fetch.add_argument("--suite", choices=sorted(SOURCES), action="append")

    p_freeze = sub.add_parser("freeze")
    p_freeze.add_argument("--workdir", type=Path, required=True)
    p_freeze.add_argument("--suite", choices=sorted(SOURCES), required=True)
    p_freeze.add_argument(
        "--dev", action="store_true", help="draw the dev sample (disjoint from test)"
    )

    p_run = sub.add_parser("run")
    p_run.add_argument("--workdir", type=Path, required=True)
    p_run.add_argument("--suite", choices=sorted(SOURCES), required=True)
    p_run.add_argument("--arm", choices=sorted(ARMS), required=True)
    p_run.add_argument("--tier", choices=RUN_TIERS, required=True)
    p_run.add_argument(
        "--embedding-model", help="Ollama embedding model, e.g. embeddinggemma:latest"
    )

    p_show = sub.add_parser("show")
    p_show.add_argument("run_dir", type=Path)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("run_a", type=Path)
    p_cmp.add_argument("run_b", type=Path)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    if args.cmd == "fetch":
        for suite in args.suite or sorted(SOURCES):
            print(f"{suite}: {fetch(suite, workdir=args.workdir)}")
    elif args.cmd == "freeze":
        fn = freeze_dev if args.dev else freeze
        print(fn(args.suite, workdir=args.workdir))
    elif args.cmd == "run":
        run_dir = asyncio.run(
            run(
                suite=args.suite,
                tier=args.tier,
                arm_name=args.arm,
                workdir=args.workdir,
                embedding_model=args.embedding_model,
            )
        )
        print(format_summary(run_dir))
        print(f"\nrecorded in {run_dir}")
    elif args.cmd == "show":
        print(format_summary(args.run_dir))
    elif args.cmd == "compare":
        print(compare(args.run_a, args.run_b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
