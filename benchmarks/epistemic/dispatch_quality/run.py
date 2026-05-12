"""CLI entry point for the dispatch-quality benchmark.

Usage:

    # Run Tier 1 with the model from $ANDAMENTUM_MAIN_LLM_MODEL, write
    # reports to ./results/dispatch_quality/<timestamp>/
    uv run python -m benchmarks.epistemic.dispatch_quality.run

    # Choose a specific dispatch model
    uv run python -m benchmarks.epistemic.dispatch_quality.run \\
        --model openai:gpt-5.4-nano

    # Held-out evaluation (stricter: hide each tested example from
    # the dispatch agent's in-context teaching during its own test)
    uv run python -m benchmarks.epistemic.dispatch_quality.run --held-out

    # Subset of providers
    uv run python -m benchmarks.epistemic.dispatch_quality.run \\
        --providers pubmed,europepmc,arxiv

The CLI returns nonzero if any provider's triage accuracy falls below
the PRD's acceptance threshold (0.80 by default; configurable).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from andamentum.core.agents import AgentRunner

from andamentum.epistemic.providers import get_all_providers

from .harness import run_and_report

logger = logging.getLogger(__name__)


DEFAULT_OUTPUT_ROOT = Path(__file__).parent.parent / "results" / "dispatch_quality"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the dispatch-quality benchmark (Tier 1).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ANDAMENTUM_MAIN_LLM_MODEL"),
        help=(
            "Dispatch-agent model (e.g. openai:gpt-5.4-nano, "
            "anthropic:claude-haiku-4-5). Falls back to "
            "$ANDAMENTUM_MAIN_LLM_MODEL."
        ),
    )
    parser.add_argument(
        "--providers",
        default="",
        help="Comma-separated subset of providers to evaluate. Empty = all.",
    )
    parser.add_argument(
        "--held-out",
        action="store_true",
        help=(
            "Mask each tested example from the dispatch agent's "
            "in-context teaching block during its own evaluation. "
            "Stricter generalisation test."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Per-provider triage-accuracy threshold for acceptance.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the report. Default: ./benchmarks/.../results/<timestamp>/",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="DEBUG logging.",
    )
    return parser.parse_args(argv)


async def _async_main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    if not args.model:
        print(
            "ERROR: --model is required (or set $ANDAMENTUM_MAIN_LLM_MODEL).",
            file=sys.stderr,
        )
        return 2

    all_providers = get_all_providers()
    if args.providers:
        wanted = {p.strip() for p in args.providers.split(",") if p.strip()}
        unknown = wanted - set(all_providers)
        if unknown:
            print(
                f"ERROR: unknown providers: {sorted(unknown)}. "
                f"Available: {sorted(all_providers)}",
                file=sys.stderr,
            )
            return 2
        providers = {k: v for k, v in all_providers.items() if k in wanted}
    else:
        providers = all_providers

    if args.output_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        held_tag = "_held_out" if args.held_out else ""
        output_dir = DEFAULT_OUTPUT_ROOT / f"{ts}{held_tag}"
    else:
        output_dir = args.output_dir

    runner = AgentRunner(model=args.model)

    print(f"Running Tier 1 against {len(providers)} provider(s) using {args.model}")
    print(f"Held-out: {args.held_out}")
    print(f"Output: {output_dir}")
    print()

    results = await run_and_report(
        providers=providers,
        agent_runner=runner,
        held_out=args.held_out,
        output_dir=output_dir,
    )

    # Acceptance: every provider's triage accuracy must clear the threshold.
    failures = [r for r in results if r.triage_accuracy < args.threshold]
    if failures:
        print()
        print(
            f"FAIL: {len(failures)} provider(s) below "
            f"triage_accuracy={args.threshold}:"
        )
        for r in failures:
            print(f"  - {r.provider}: {r.triage_accuracy:.2f}")
        return 1

    print()
    print(f"PASS: all providers ≥ {args.threshold} triage accuracy")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    sys.exit(main())
