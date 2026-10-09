"""Command-line entry point: ``andamentum-vision-critique``.

One positional argument (an image path or URL); ``--model`` is required (no
default). The critique is printed as JSON on stdout, or written with
``-o/--out``.

Exit codes:
    0 - success, the critique found no issue
    1 - success, the critique found at least one issue (the JSON is still written)
    2 - usage error (a wrong or missing argument)
    3 - the image could not be read (file missing, URL unreachable, not a
        PNG, JPEG, WEBP or GIF)
    4 - the run failed (model or network error, or no valid structured answer)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

_EXIT_CODES = """\
exit codes:
  0  success, no issue found
  1  success, the critique found at least one issue (the JSON is still written)
  2  usage error (a wrong or missing argument)
  3  the image could not be read (missing file, unreachable URL, not PNG/JPEG/WEBP/GIF)
  4  the run failed (model or network error, or no valid structured answer)
"""


def build_parser() -> argparse.ArgumentParser:
    from andamentum import __version__

    parser = argparse.ArgumentParser(
        prog="andamentum-vision-critique",
        description=(
            "Check a rendered figure for layout and readability problems with a "
            "vision model. Prints a bounded JSON critique: label overlap, "
            "legibility, legend placement, aspect ratio, suggested fixes from a "
            "fixed set, and a confidence."
        ),
        epilog=(
            "examples:\n"
            "  andamentum-vision-critique figure.png --model ollama:qwen3.8\n"
            "  andamentum-vision-critique figure.png --model ollama:qwen3.8 \\\n"
            '      --context "a two-column journal figure" -o critique.json\n'
            f"\n{_EXIT_CODES}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"andamentum-vision-critique {__version__}",
    )
    parser.add_argument(
        "image",
        help="image to check: a file path or an http(s) URL (PNG, JPEG, WEBP or GIF; "
        "render a PDF or SVG to PNG first)",
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="MODEL",
        help=(
            "multimodal model that looks at the image, e.g. ollama:qwen3.8 "
            "(validated for local use), anthropic:claude-haiku-4-5 or "
            "openai:gpt-5.4-nano"
        ),
    )
    parser.add_argument(
        "--context",
        default=None,
        metavar="TEXT",
        help=(
            "extra context for the model, e.g. "
            "'this is a panel from a Cell-format manuscript figure'"
        ),
    )
    parser.add_argument(
        "-o",
        "--out",
        default="-",
        metavar="FILE",
        help="write the JSON critique to FILE (default: print to stdout)",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    from .api import critique_figure
    from .errors import UnsupportedImageError

    try:
        critique = await critique_figure(
            args.image,
            model=args.model,
            context=args.context,
        )
    except UnsupportedImageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except FileNotFoundError as exc:
        print(f"error: image not found: {exc}", file=sys.stderr)
        return 3
    except OSError as exc:
        print(f"error: could not read image {args.image!r}: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        # pydantic-ai's UnexpectedModelBehavior, httpx errors and model
        # connection errors all land here: a routine user-facing failure, so
        # one stderr line rather than a traceback.
        print(
            f"error: vision critique failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 4

    payload = critique.model_dump_json(indent=2)
    if args.out == "-":
        sys.stdout.write(payload + "\n")
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
    return 1 if getattr(critique, "has_issues", False) else 0


def main(argv: Sequence[str] | None = None) -> int:
    from andamentum.core.secrets import load_secrets

    args = build_parser().parse_args(argv)
    load_secrets()  # provider API keys only
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 4


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
