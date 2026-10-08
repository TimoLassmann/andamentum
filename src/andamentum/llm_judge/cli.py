"""Command-line entry point: ``andamentum-llm-judge``.

Two subcommands: ``score TEXT`` and ``compare TEXT_A TEXT_B``. Each text is a
literal argument, or ``-`` to read it from stdin (for ``compare`` at most one
of the two may be ``-``). ``--model`` is required (no default). Pass it once
for the FAST path (a single judge), or repeat it / pass a comma-separated
list for the PANEL path - a 3-model panel over the 6 default criteria is
18+ strictly sequential model calls (plus any PromptedOutput retries), so
panel runs are noticeably slower than fast runs, especially against local
Ollama. This is accepted: correctness over speed, and Ollama serialises
requests anyway so there is nothing to parallelise.

The result is JSON on stdout, or in the ``-o/--out`` file.

Exit codes:
    0 - success (the verdict is in the JSON; a "fails" or "b" verdict is not an error)
    2 - usage error (a wrong or missing argument)
    3 - input unreadable (a criteria file that is missing or does not parse)
    4 - the run failed (model or network error, or no valid structured answer)

A judge verdict is data rather than a pass/fail check, so 1 is not used.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import judge_compare, judge_score
from .criteria import DEFAULT_CRITERIA
from .errors import InvalidJudgeInputError
from .schemas import Criterion

_EXIT_CODES = """\
exit codes:
  0  success (the verdict is in the JSON; a "fails" verdict is not an error)
  2  usage error (a wrong or missing argument)
  3  input unreadable (a criteria file that is missing or does not parse)
  4  the run failed (model or network error, or no valid structured answer)
"""


class _Exit(Exception):
    """Stops the run with an exit code and a one-line message."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _parse_models(raw: list[str]) -> str | list[str]:
    """Collapse repeated ``--model`` flags and comma-separated values into
    either a single model string (fast path) or a list (panel path)."""
    models: list[str] = []
    for entry in raw:
        models.extend(part.strip() for part in entry.split(",") if part.strip())
    if not models:
        raise InvalidJudgeInputError("--model given but resolved to no model ids")
    return models[0] if len(models) == 1 else models


def _load_criteria(path: str | None) -> list[Criterion]:
    if path is None:
        return DEFAULT_CRITERIA
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Criterion(**entry) for entry in data]


def _read_text(value: str, *, stdin_text: dict[str, str]) -> str:
    """A literal argument, or all of stdin for ``-`` (read once)."""
    if value != "-":
        return value
    if "text" not in stdin_text:
        stdin_text["text"] = sys.stdin.read()
    return stdin_text["text"]


def build_parser() -> argparse.ArgumentParser:
    from andamentum import __version__

    parser = argparse.ArgumentParser(
        prog="andamentum-llm-judge",
        description=(
            "LLM-as-judge: score one text against criteria, or compare two "
            "texts. Pass --model once for a fast single judge, or several "
            "times (or comma-separated) for a sequential panel with an "
            "agreement gate. Prints JSON."
        ),
        epilog=_EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"andamentum-llm-judge {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_shared(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--model",
            required=True,
            action="append",
            metavar="MODEL",
            help=(
                "model that judges. One value = fast mode, a single judge at "
                "temperature 0. Repeat the flag, or comma-separate, for panel "
                "mode: several judges, run one after another, with an agreement "
                "gate. Examples: anthropic:claude-haiku-4-5, ollama:gemma4:31b"
            ),
        )
        sp.add_argument(
            "--context",
            default=None,
            metavar="TEXT",
            help="the task or question the text(s) were answering (optional)",
        )
        sp.add_argument(
            "--criteria",
            default=None,
            metavar="FILE",
            help=(
                "JSON file holding a list of {name, description} objects "
                "(default: the built-in six criteria)"
            ),
        )
        sp.add_argument(
            "-o",
            "--out",
            dest="out_path",
            default="-",
            metavar="FILE",
            help="write the JSON result to FILE (default: print to stdout)",
        )

    score_parser = subparsers.add_parser(
        "score",
        help="score one text against criteria",
        description="Score one text against criteria. Prints JSON.",
        epilog=_EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    score_parser.add_argument(
        "text", metavar="TEXT", help="the text to judge; use - to read it from stdin"
    )
    _add_shared(score_parser)

    compare_parser = subparsers.add_parser(
        "compare",
        help="say which of two texts is better",
        description=(
            "Compare two texts. Both presentation orders are always run. Prints JSON."
        ),
        epilog=_EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    compare_parser.add_argument(
        "text_a", metavar="TEXT_A", help="the first text; use - to read it from stdin"
    )
    compare_parser.add_argument(
        "text_b",
        metavar="TEXT_B",
        help="the second text; use - to read it from stdin (only one of the two may be -)",
    )
    _add_shared(compare_parser)

    return parser


async def _run(args: argparse.Namespace) -> int:
    try:
        model = _parse_models(args.model)
    except InvalidJudgeInputError as exc:
        raise _Exit(2, str(exc)) from exc

    if args.command == "compare" and args.text_a == "-" and args.text_b == "-":
        raise _Exit(2, "only one of TEXT_A and TEXT_B may be - (stdin)")

    try:
        criteria = _load_criteria(args.criteria)
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise _Exit(
            3, f"could not load criteria from {args.criteria!r}: {exc}"
        ) from exc

    stdin_text: dict[str, str] = {}
    try:
        if args.command == "score":
            text = _read_text(args.text, stdin_text=stdin_text)
            result = await judge_score(
                text, criteria=criteria, context=args.context, model=model
            )
        else:
            text_a = _read_text(args.text_a, stdin_text=stdin_text)
            text_b = _read_text(args.text_b, stdin_text=stdin_text)
            result = await judge_compare(
                text_a,
                text_b,
                criteria=criteria,
                context=args.context,
                model=model,
            )
    except InvalidJudgeInputError as exc:
        raise _Exit(2, str(exc)) from exc
    except Exception as exc:
        # Broad catch here is deliberate CLI-boundary behaviour (matches
        # andamentum-vision-critique): report a clear message on stderr for
        # what is a routine user-facing tool, rather than an unfriendly
        # traceback. The library layer itself never swallows errors.
        raise _Exit(4, f"judge call failed: {type(exc).__name__}: {exc}") from exc

    payload = result.model_dump_json(indent=2)
    if args.out_path == "-":
        sys.stdout.write(payload + "\n")
    else:
        out = Path(args.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    from dotenv import load_dotenv

    args = build_parser().parse_args(argv)
    load_dotenv()  # provider API keys only
    try:
        return asyncio.run(_run(args))
    except _Exit as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.code
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 4


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
