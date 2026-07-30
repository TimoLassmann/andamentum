"""Command-line entry point: ``andamentum-llm-judge``.

Two subcommands: ``score <output>`` and ``compare <output_a> <output_b>``.
``--model`` is required (no hidden default). Pass it once for the FAST
(single-judge) path, or repeat it / pass a comma-separated list for the
PANEL path — a 3-model panel over the 6 default criteria is 18+ strictly
sequential model calls (plus any PromptedOutput retries), so panel runs are
noticeably slower than fast runs, especially against local Ollama. This is
accepted: correctness over speed, and Ollama serialises requests anyway so
there is nothing to parallelise.

Output is JSON to stdout by default; use ``-o`` to write a file.

Exit codes:
    0 — success
    1 — argument error
    2 — input error (e.g. a criteria file that doesn't parse)
    3 — judge call failed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import NoReturn, Sequence

from dotenv import load_dotenv

from . import judge_compare, judge_score
from .criteria import DEFAULT_CRITERIA
from .schemas import Criterion


def _die(code: int, message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(code)


def _parse_models(raw: list[str]) -> str | list[str]:
    """Collapse repeated ``--model`` flags and comma-separated values into
    either a single model string (fast path) or a list (panel path)."""
    models: list[str] = []
    for entry in raw:
        models.extend(part.strip() for part in entry.split(",") if part.strip())
    if not models:
        raise ValueError("--model given but resolved to no model ids")
    return models[0] if len(models) == 1 else models


def _load_criteria(path: str | None) -> list[Criterion]:
    if path is None:
        return DEFAULT_CRITERIA
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Criterion(**entry) for entry in data]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="andamentum-llm-judge",
        description=(
            "LLM-as-judge: score one output against criteria, or compare "
            "two outputs. Pass --model once for a fast single judge, or "
            "multiple times (or comma-separated) for a sequential panel "
            "with an agreement gate."
        ),
    )
    from andamentum import __version__ as _ver

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s (andamentum {_ver})"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # `score` takes a positional named `output` (the text to judge), so the
    # destination-file flag MUST carry an explicit, different dest — sharing
    # `output` makes argparse silently overwrite one with the other.
    def _add_shared(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--model",
            required=True,
            action="append",
            metavar="MODEL",
            help=(
                "Required. Repeatable, or comma-separated, for a panel. "
                "One value = fast single judge (temperature 0). "
                "Examples: anthropic:claude-haiku-4-5, ollama:gemma4:31b-nvfp4."
            ),
        )
        sp.add_argument(
            "--context",
            default=None,
            metavar="TEXT",
            help="Optional task/prompt the output(s) were answering.",
        )
        sp.add_argument(
            "--criteria",
            default=None,
            metavar="PATH",
            help="Optional JSON file: a list of {name, description} objects. "
            "Defaults to the built-in six-criterion set.",
        )
        sp.add_argument(
            "-o",
            "--output",
            dest="out_path",
            default="-",
            metavar="FILE",
            help="Output JSON file path. Default: '-' (stdout).",
        )

    score_parser = subparsers.add_parser(
        "score", help="Score one output against criteria."
    )
    score_parser.add_argument("output", help="The text to judge.")
    _add_shared(score_parser)

    compare_parser = subparsers.add_parser("compare", help="Compare two outputs.")
    compare_parser.add_argument("output_a", help="The first candidate output.")
    compare_parser.add_argument("output_b", help="The second candidate output.")
    _add_shared(compare_parser)

    return parser


async def _run(args: argparse.Namespace) -> int:
    try:
        model = _parse_models(args.model)
    except ValueError as exc:
        _die(1, str(exc))

    try:
        criteria = _load_criteria(args.criteria)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _die(2, f"could not load criteria from {args.criteria!r}: {exc}")

    try:
        if args.command == "score":
            result = await judge_score(
                args.output, criteria=criteria, context=args.context, model=model
            )
        else:
            result = await judge_compare(
                args.output_a,
                args.output_b,
                criteria=criteria,
                context=args.context,
                model=model,
            )
    except Exception as exc:
        # Broad catch here is deliberate CLI-boundary behaviour (matches
        # andamentum-vision-critique): report a clear message on stderr for
        # what is a routine user-facing tool, rather than an unfriendly
        # traceback. The library layer itself never swallows errors.
        _die(3, f"judge call failed: {type(exc).__name__}: {exc}")

    payload = result.model_dump_json(indent=2)
    if args.out_path == "-":
        sys.stdout.write(payload + "\n")
    else:
        out = Path(args.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        _die(1, "interrupted")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
