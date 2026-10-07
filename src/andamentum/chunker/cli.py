"""Command-line entry point: andamentum-chunker.

Usage::

    andamentum-chunker INPUT -o units.json
    andamentum-chunker INPUT --judge-model openai:gpt-5.4-nano -o units.json

Without ``--judge-model`` the chunker makes no LLM call (stages 1 and 2 only);
stage 3, the grey-zone merge judge, runs only when that flag names a model.
``--decision-model NAME`` (a local Ollama decision model) answers the grey-zone
questions instead of ``--judge-model``.

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

from andamentum.core.decisions import DecisionError, DecisionModel

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
        "--judge-model",
        default=None,
        help=(
            "Run stage 3, the LLM judge for grey-zone boundaries, with this "
            "model (e.g. anthropic:claude-haiku-4-5, ollama:gemma4:26b). "
            "Omitted: stage 3 is skipped and no LLM is called."
        ),
    )
    parser.add_argument(
        "--decision-model",
        default=None,
        help=(
            "A local Ollama decision model (e.g. nimble:9b) that answers the "
            "grey-zone merge question instead of --judge-model."
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
    judge_executor = (
        make_runner_executor(AgentRunner(model=args.judge_model))
        if args.judge_model
        else None
    )

    if args.decision_model is None:
        result = await extract_units(
            src, judge_executor=judge_executor, domain=args.domain
        )
    else:
        try:
            async with await DecisionModel.open(args.decision_model) as decision_model:
                result = await extract_units(
                    src,
                    judge_executor=judge_executor,
                    decision_model=decision_model,
                    domain=args.domain,
                )
        except DecisionError as exc:
            print(f"Error: decision model: {exc}", file=sys.stderr)
            return 2

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
                "decision_calls": result.decision_calls,
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
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not Path(args.input).exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1
    return asyncio.run(_main_async(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
