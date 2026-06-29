"""Standalone CLI: run the forge benchmark, print + optionally write a report.

Usage::

    uv run python -m benchmarks.forge.cli --model <id>
    uv run python -m benchmarks.forge.cli --model <id> --runs 5
    uv run python -m benchmarks.forge.cli --model <id> --output report.md
    uv run python -m benchmarks.forge.cli --model <id> --case loop --case branch
    uv run python -m benchmarks.forge.cli --model <id> --full   # Tier-2 hook (TODO)

``--model`` is required — forge's explicit-model rule: no default, no env var. Loads
.env so the resolved provider's API key (e.g. OPENAI_API_KEY) is available.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from .cases import CASES
from .report import render
from .runner import run_all
from .types import Case


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forge-bench")
    p.add_argument(
        "--model",
        required=True,
        help="Model id to drive forge (required; no default, no env var)",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Repetitions per case (default: 3)",
    )
    p.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only case(s) whose brief contains this substring; repeatable",
    )
    p.add_argument(
        "--output",
        "-o",
        default=None,
        help="Write the markdown report to this file (default: stdout only)",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Tier-2 end-to-end build (hook only; currently behaves like Tier 1)",
    )
    return p


def _select_cases(substrings: list[str]) -> list[Case]:
    """Filter ``CASES`` to those whose brief contains any of ``substrings``."""
    if not substrings:
        return list(CASES)
    needles = [s.lower() for s in substrings]
    return [c for c in CASES if any(n in c.brief.lower() for n in needles)]


async def _main_async(args: argparse.Namespace) -> int:
    cases = _select_cases(args.case)
    if not cases:
        print(f"No cases matched: {args.case}", file=sys.stderr)
        return 1

    scores = await run_all(cases, model=args.model, runs=args.runs, full=args.full)
    report = render(scores, model=args.model)
    print(report)

    if args.output is not None:
        Path(args.output).write_text(report)
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _build_parser().parse_args(argv)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
