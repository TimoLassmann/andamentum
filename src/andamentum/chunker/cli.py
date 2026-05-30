"""Command-line entry point: andamentum-chunker.

Usage::

    andamentum-chunker INPUT --model openai:gpt-4o-mini --domain academic -o units.json

Outputs JSON with {units, gaps, coverage, gap_fraction, model_calls}.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from andamentum.core.agents import AgentRunner
from andamentum.core.models import resolve_model_from_args

from .extractor import extract_units, make_runner_executor


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="andamentum-chunker",
        description="Verifiable semantic chunking of long text.",
    )
    from andamentum import __version__ as _ver

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s (andamentum {_ver})",
    )
    parser.add_argument("input", help="Path to text/markdown file to chunk")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "LLM model (e.g. anthropic:claude-haiku-4-5, openai:gpt-4o-mini, "
            "ollama:gemma2:31b) — or set $ANDAMENTUM_MAIN_LLM_MODEL"
        ),
    )
    parser.add_argument(
        "--escalate-to",
        action="append",
        default=[],
        help=(
            "Backup model(s) to try if the primary fails. May be given "
            "multiple times. Tried in order."
        ),
    )
    parser.add_argument(
        "--domain",
        default="general",
        choices=["academic", "web", "code", "transcript", "general"],
        help="Domain hint that adjusts the prompt (default: general)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        help="Output JSON file path (default: stdout)",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Emit only the unit text bodies as a markdown-style listing",
    )
    return parser


async def _main_async(args: argparse.Namespace) -> int:
    src = Path(args.input).read_text()
    primary_model = resolve_model_from_args(args.model)
    primary_runner = AgentRunner(model=primary_model)
    backup_runners = [AgentRunner(model=m) for m in args.escalate_to]

    primary_executor = make_runner_executor(primary_runner)
    primary_executor.label = primary_model  # type: ignore[attr-defined]

    backup_executors = []
    for runner, model_str in zip(backup_runners, args.escalate_to):
        ex = make_runner_executor(runner)
        ex.label = model_str  # type: ignore[attr-defined]
        backup_executors.append(ex)

    result = await extract_units(
        src,
        primary_executor=primary_executor,
        backup_executors=backup_executors,
        domain=args.domain,
    )

    if args.text_only:
        body = "\n\n---\n\n".join(f"## {u.title}\n\n{u.text}" for u in result.units)
    else:
        body = json.dumps(
            {
                "units": [u.model_dump() for u in result.units],
                "gaps": [g.model_dump() for g in result.gaps],
                "coverage": result.coverage,
                "gap_fraction": result.gap_fraction,
                "total_chars": result.total_chars,
                "model_calls": result.model_calls,
                "windows_processed": result.windows_processed,
            },
            indent=2,
        )

    if args.output == "-":
        print(body)
    else:
        Path(args.output).write_text(body)
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _build_parser().parse_args(argv)
    if not Path(args.input).exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1
    return asyncio.run(_main_async(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
