"""Command-line entry point: ``andamentum-vision-critique``.

Single positional argument (an image path or URL), ``--model`` is
required (no hidden default — same convention as the other LLM-using
CLIs). Output is JSON to stdout by default; use ``-o`` to write a file.

Exit codes:
    0 — success
    1 — argument error
    2 — image could not be loaded (file missing, URL fetch failed)
    3 — vision call failed / structured output didn't hold
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import NoReturn, Sequence

from dotenv import load_dotenv

from .api import critique_figure


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="andamentum-vision-critique",
        description=(
            "Vision-critique a rendered figure for layout and "
            "readability problems. Produces a bounded JSON critique "
            "(label overlap, legibility, legend placement, aspect "
            "ratio, suggested fixes from a fixed set, confidence)."
        ),
    )
    from andamentum import __version__ as _ver

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s (andamentum {_ver})",
    )
    parser.add_argument(
        "image",
        help="Local image path or http(s)://... URL (PNG/JPEG/WebP/GIF).",
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="MODEL",
        help=(
            "Required. pydantic-ai multimodal model id. "
            "Validated default for local use: ollama:gemma4:e4b-it-q4_K_M. "
            "Cloud examples: anthropic:claude-haiku-4-5, openai:gpt-5.4-nano."
        ),
    )
    parser.add_argument(
        "--context",
        default=None,
        metavar="TEXT",
        help=(
            "Optional extra context for the model — e.g. "
            "'this is a panel from a Cell-format manuscript figure'."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        metavar="FILE",
        help="Output JSON file path. Default: '-' (stdout).",
    )
    return parser


def _die(code: int, message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(code)


async def _run(args: argparse.Namespace) -> int:
    try:
        critique = await critique_figure(
            args.image,
            model=args.model,
            extra_context=args.context,
        )
    except FileNotFoundError as exc:
        _die(2, f"image not found: {exc}")
    except OSError as exc:
        _die(2, f"could not read image {args.image!r}: {exc}")
    except Exception as exc:
        # Catch the broad case for clear stderr reporting; pydantic-ai's
        # UnexpectedModelBehavior, httpx errors, ollama-connect errors,
        # all land here. Better to print the message than re-raise into
        # an unfriendly traceback for what is a routine user-facing tool.
        _die(3, f"vision critique failed: {type(exc).__name__}: {exc}")

    payload = critique.model_dump_json(indent=2)
    if args.output == "-":
        sys.stdout.write(payload + "\n")
    else:
        out = Path(args.output)
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
