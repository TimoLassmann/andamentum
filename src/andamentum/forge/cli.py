"""Command-line adapter for forge — design or build an agentic system from a brief.

    andamentum-forge build "Manage my reading list."  --model anthropic:claude-haiku-4-5 --out ./out
    andamentum-forge design "Triage support tickets."  --model anthropic:claude-haiku-4-5

``build`` renders + verifies a package; ``design`` stops at the validated spec. The
model resolves via ``--model`` or ``$ANDAMENTUM_MAIN_LLM_MODEL`` (no hidden default).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from andamentum.core import resolve_model_from_args

from .graph import run_forge
from .schemas import ForgeResult


def _print_summary(result: ForgeResult) -> None:
    spec = result.spec
    heads = sum(1 for n in spec.nodes if n.kind.value == "head")
    print(f"\nsystem:  {spec.name}")
    print(f"purpose: {spec.description}")
    print(f"nodes:   {len(spec.nodes)} ({heads} head, {len(spec.nodes) - heads} spine)")
    print(
        f"agents:  {len(spec.agents)}   entities: {len(spec.entities)}   loop-caps: {len(spec.loop_caps)}"
    )
    if result.notes:
        print("notes:")
        for n in result.notes:
            print(f"  - {n}")
    if result.report is not None:
        verdict = "WORKS" if result.report.works else "INCOMPLETE"
        print(f"\nverification: {verdict}  (score {result.report.score:.2f})")
        for c in result.report.checks:
            print(f"  [{'pass' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
    if result.rendered_files:
        print(
            f"\nrendered {len(result.rendered_files)} files under {Path(result.rendered_files[0]).parent}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="andamentum-forge", description="Build agentic systems from a brief."
    )
    parser.add_argument(
        "command",
        choices=["build", "design"],
        help="build = render + verify; design = spec only",
    )
    parser.add_argument("brief", help="the natural-language brief")
    parser.add_argument(
        "--model",
        default=None,
        help="pydantic-ai model id (or set ANDAMENTUM_MAIN_LLM_MODEL)",
    )
    parser.add_argument(
        "--out", default="out", help="output directory for `build` (default: ./out)"
    )
    args = parser.parse_args(argv)

    model = resolve_model_from_args(args.model)
    dest = Path(args.out) if args.command == "build" else None

    try:
        result = asyncio.run(run_forge(args.brief, model=model, dest=dest))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    _print_summary(result)
    if result.report is not None and not result.report.works:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
