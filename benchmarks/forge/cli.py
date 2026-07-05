"""Standalone CLI: run the forge benchmark, print + optionally write a report.

Usage::

    uv run python -m benchmarks.forge.cli --model <id>
    uv run python -m benchmarks.forge.cli --model <id> --runs 5
    uv run python -m benchmarks.forge.cli --model <id> --output report.md
    uv run python -m benchmarks.forge.cli --model <id> --case loop --case branch
    uv run python -m benchmarks.forge.cli --model <id> --full   # Tier-2: end-to-end
    uv run python -m benchmarks.forge.cli --model <id> --full --sandbox podman
    uv run python -m benchmarks.forge.cli --model <id> --golden # Tier-3: run + score output

``--model`` is required — forge's explicit-model rule: no default, no env var. Loads
.env so the resolved provider's API key (e.g. OPENAI_API_KEY) is available.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from .cases import CASES
from .golden import GOLDEN_CASES, GoldenCase, GoldenOutcome, render_golden, run_golden
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
        default=None,
        help="Repetitions per case (default: 3; 1 with --golden — builds are expensive)",
    )
    p.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only case(s) whose brief (or golden key) contains this substring; repeatable",
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
        help="Tier-2: render + agent-author + sandbox-audit, score on whether it works",
    )
    p.add_argument(
        "--golden",
        action="store_true",
        help="Tier-3: build, EXECUTE on a real input, score whether the output covers the task",
    )
    p.add_argument(
        "--sandbox",
        choices=("subprocess", "podman"),
        default="subprocess",
        help="Tier-2/3 execution seam (default: subprocess; podman for network briefs)",
    )
    return p


def _select_cases(substrings: list[str]) -> list[Case]:
    """Filter ``CASES`` to those whose brief contains any of ``substrings``."""
    if not substrings:
        return list(CASES)
    needles = [s.lower() for s in substrings]
    return [c for c in CASES if any(n in c.brief.lower() for n in needles)]


def _select_golden_cases(substrings: list[str]) -> list[GoldenCase]:
    """Filter ``GOLDEN_CASES`` to those whose key or brief contains any of ``substrings``."""
    if not substrings:
        return list(GOLDEN_CASES)
    needles = [s.lower() for s in substrings]
    return [
        c
        for c in GOLDEN_CASES
        if any(n in c.key.lower() or n in c.brief.lower() for n in needles)
    ]


async def _main_golden(args: argparse.Namespace, runs: int) -> str | None:
    """Run the Tier-3 golden corpus; return the report (None when nothing matched)."""
    cases = _select_golden_cases(args.case)
    if not cases:
        return None

    results: list[tuple[GoldenCase, list[GoldenOutcome]]] = []
    for case in cases:
        outcomes: list[GoldenOutcome] = []
        for _ in range(runs):
            # A fresh destination per run keeps each build isolated.
            with tempfile.TemporaryDirectory(prefix="forge-golden-") as tmp:
                outcomes.append(
                    await run_golden(
                        case,
                        model=args.model,
                        dest_root=Path(tmp),
                        sandbox_backend=args.sandbox,
                    )
                )
        results.append((case, outcomes))
    return render_golden(results, model=args.model)


async def _main_async(args: argparse.Namespace) -> int:
    if args.golden:
        runs = args.runs if args.runs is not None else 1
        report = await _main_golden(args, runs)
        if report is None:
            print(f"No golden cases matched: {args.case}", file=sys.stderr)
            return 1
    else:
        cases = _select_cases(args.case)
        if not cases:
            print(f"No cases matched: {args.case}", file=sys.stderr)
            return 1
        scores = await run_all(
            cases,
            model=args.model,
            runs=args.runs if args.runs is not None else 3,
            full=args.full,
            sandbox_backend=args.sandbox,
        )
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
