"""``python -m andamentum.document_store.bench`` — the retrieval benchmark CLI.

    fetch     --workdir DIR [--suite S]          download raw dev sets once
    freeze    --workdir DIR --suite S            draw and record dev + test (once, ever)
    run       --workdir DIR --out RUN_DIR --suite S --arm A --tier T
              [--embedding-model M] [--driver-model M] [--concurrency N] [--repeat N]
    show      RUN_DIR                            print a run's summary
    compare   RUN_A RUN_B                        linked-group bootstrap B - A, every metric
    contrast  --a RUN_A --b RUN_B --metric M --out FILE.json
    row       RUN_DIR --out FILE.json            one outcome-table row for a run

``--workdir`` and ``--out`` have no default: datasets, the embedding cache,
built stores and run records live wherever the caller says, never in a hidden
home directory. A failed run exits nonzero and leaves its manifest marked
``failed``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .arms import ARMS
from .data import RUN_TIERS, SOURCES, fetch, freeze
from .runner import compare, contrast, format_summary, run, table_row


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="python -m andamentum.document_store.bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("--workdir", type=Path, required=True)
    p_fetch.add_argument("--suite", choices=sorted(SOURCES), action="append")

    p_freeze = sub.add_parser("freeze")
    p_freeze.add_argument("--workdir", type=Path, required=True)
    p_freeze.add_argument("--suite", choices=sorted(SOURCES), required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--workdir", type=Path, required=True)
    p_run.add_argument("--out", type=Path, required=True, help="the run's directory")
    p_run.add_argument("--suite", choices=sorted(SOURCES), required=True)
    p_run.add_argument("--arm", choices=sorted(ARMS), required=True)
    p_run.add_argument("--tier", choices=RUN_TIERS, required=True)
    p_run.add_argument(
        "--embedding-model", help="Ollama embedding model, e.g. embeddinggemma-2:latest"
    )
    p_run.add_argument(
        "--driver-model",
        help="LLM driving the answering arms, e.g. ollama:gemma4:31b-cloud",
    )
    p_run.add_argument(
        "--concurrency",
        type=int,
        default=1,
        choices=range(1, 5),
        help="questions in flight at once (1-4; cloud models only above 1)",
    )
    p_run.add_argument(
        "--repeat", type=int, default=1, help="repeat number, recorded in the manifest"
    )

    p_show = sub.add_parser("show")
    p_show.add_argument("run_dir", type=Path)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("run_a", type=Path)
    p_cmp.add_argument("run_b", type=Path)

    p_con = sub.add_parser("contrast")
    p_con.add_argument("--a", type=Path, required=True)
    p_con.add_argument("--b", type=Path, required=True)
    p_con.add_argument("--metric", required=True)
    p_con.add_argument("--out", type=Path, required=True)

    p_row = sub.add_parser("row")
    p_row.add_argument("run_dir", type=Path)
    p_row.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    if args.cmd == "fetch":
        for suite in args.suite or sorted(SOURCES):
            print(f"{suite}: {fetch(suite, workdir=args.workdir)}")
    elif args.cmd == "freeze":
        print(freeze(args.suite, workdir=args.workdir))
    elif args.cmd == "run":
        run_dir = asyncio.run(
            run(
                suite=args.suite,
                tier=args.tier,
                arm_name=args.arm,
                workdir=args.workdir,
                out_dir=args.out,
                embedding_model=args.embedding_model,
                driver_model=args.driver_model,
                concurrency=args.concurrency,
                repeat=args.repeat,
                argv=raw_argv,
            )
        )
        print(format_summary(run_dir))
        print(f"\nrecorded in {run_dir}")
    elif args.cmd == "show":
        print(format_summary(args.run_dir))
    elif args.cmd == "compare":
        print(compare(args.run_a, args.run_b))
    elif args.cmd == "contrast":
        result = contrast(args.a, args.b, args.metric)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    elif args.cmd == "row":
        args.out.write_text(json.dumps(table_row(args.run_dir), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
