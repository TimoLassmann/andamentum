"""Standalone CLI: run the benchmark, write a markdown report.

Usage::

    uv run python -m benchmarks.chunker.cli                       # default model + cases
    uv run python -m benchmarks.chunker.cli --model X             # one model override
    uv run python -m benchmarks.chunker.cli --output report.md    # write report to file
    uv run python -m benchmarks.chunker.cli --case academic_short # one case only

Loads .env so OPENAI_API_KEY etc. are available.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from andamentum.chunker.extractor import make_runner_executor
from andamentum.core.agents import AgentRunner

from .conftest import DEFAULT_MODEL, discover_cases
from .loader import load_case
from .report import to_markdown_table
from .runner import run_case


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chunker-bench")
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to test (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only the named case(s); may be given multiple times",
    )
    p.add_argument(
        "--output", "-o", default="-", help="Output markdown file (default: stdout)"
    )
    return p


async def _main_async(args: argparse.Namespace) -> int:
    runner = AgentRunner(model=args.model)
    executor = make_runner_executor(runner)
    executor.label = args.model  # type: ignore[attr-defined]

    all_paths = discover_cases()
    if args.case:
        wanted = set(args.case)
        all_paths = [
            p for p in all_paths if p.name.removesuffix(".truth.json") in wanted
        ]
        if not all_paths:
            print(f"No cases matched: {args.case}", file=sys.stderr)
            return 1

    runs = []
    for path in all_paths:
        case = load_case(path)
        print(f"running {case.name}…", file=sys.stderr, flush=True)
        run = await run_case(case, primary_executor=executor, model_label=args.model)
        runs.append(run)

    md = to_markdown_table(runs)
    if args.output == "-":
        print(md)
    else:
        Path(args.output).write_text(md)
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _build_parser().parse_args(argv)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
