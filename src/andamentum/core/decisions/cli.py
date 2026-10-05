"""``python -m andamentum.core.decisions check --model NAME [--base-url URL]``.

Checks the server (Ollama 0.35 or later), that the model is installed and
declares the ``decision`` capability, prints its digest, and asks one question.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

import httpx

from ..embeddings import DEFAULT_OLLAMA_URL
from .api import Question
from .errors import DecisionError
from .model import DecisionModel

SMOKE_STATE = "The kettle boiled and the water was poured over the tea."
SMOKE_QUESTION = Question(
    type="noul", instructions="Is this sentence about making a drink?"
)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


async def _check(
    args: argparse.Namespace, transport: httpx.AsyncBaseTransport | None
) -> int:
    async with await DecisionModel.open(
        args.model, base_url=args.base_url, transport=transport
    ) as model:
        print(f"server      {model.base_url}")
        print(f"model       {model.name}")
        print(f"capability  {', '.join(model.capabilities or ['(none reported)'])}")
        print(f"digest      {model.digest}")
        result = await model.decide_with_usage(SMOKE_STATE, {"drink": SMOKE_QUESTION})
        answer = result.answers["drink"]
        print(
            f"smoke       p(true) = {_fmt(answer.p_true)} in {result.ms:.0f} ms, "
            f"{result.usage.input_tokens} tokens in"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m andamentum.core.decisions",
        description=(__doc__ or "").split("\n")[0],
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser(
        "check", help="check the server and the model, ask one question"
    )
    check.add_argument("--model", required=True, help="Ollama model, e.g. nimble:9b")
    check.add_argument("--base-url", default=DEFAULT_OLLAMA_URL)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> int:
    """Run the CLI; ``transport`` exists for tests."""
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_check(args, transport))
    except (DecisionError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
