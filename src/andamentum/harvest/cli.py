"""Command-line entry point: ``andamentum-harvest``.

    andamentum-harvest SOURCE [-o FILE] [--mode fast|accurate]
        [--figure-model MODEL] [--no-ocr] [--preserve-superscripts]
        [--tdm-host HOST ...] [-v]

``SOURCE`` is a URL or a local file path; the markdown goes to stdout, or to
the file named by ``-o/--out``. No language model is called in ``fast`` mode.
``--mode accurate`` (PDF only, needs ``uv sync --extra vlm``) needs
``--figure-model``: there is no default.

Exit codes:
    0 - success
    2 - usage error: a wrong or missing argument, ``--mode accurate`` without
        ``--figure-model``, or without the optional ``vlm`` extra installed
    3 - the source could not be read: fetch failed, file missing, unsupported format
    4 - the run failed: extraction failed, or the figure model is unavailable
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from .api import extract
from .errors import (
    ExtractionError,
    FetchError,
    FigureModelRequired,
    HarvestError,
    UnsupportedFormatError,
)

EXIT_USAGE = 2
EXIT_INPUT = 3
EXIT_FAILED = 4

_LOGGER_NAME = "andamentum.harvest"

_EPILOG = """\
exit status:
  0  success
  2  usage error (a wrong or missing argument; --mode accurate without
     --figure-model or without the vlm extra installed)
  3  the source could not be read (fetch failed, file missing, unsupported format)
  4  the run failed (extraction failed, or the figure model is unavailable)
"""


def build_parser() -> argparse.ArgumentParser:
    from andamentum import __version__

    parser = argparse.ArgumentParser(
        prog="andamentum-harvest",
        description=(
            "Convert a URL or document file to clean markdown. "
            "Auto-detects PDF / HTML / DOCX / PPTX / Markdown / plain text; "
            "for HTML, races trafilatura and Docling when the page metadata "
            "is ambiguous and picks the better-structured output."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"andamentum-harvest {__version__}"
    )
    parser.add_argument(
        "source",
        help="URL (http://... / https://... / file://...) or local file path.",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        metavar="FILE",
        help="write the markdown to this file instead of stdout.",
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "accurate"),
        default="fast",
        help=(
            "fast (default) runs docling's standard pipeline on its stock "
            "settings. accurate runs the VLM pipeline: one model reads the page "
            "layout, and figures get a written description. PDF only; roughly "
            "an order of magnitude slower per page; needs 'uv sync --extra vlm' "
            "and --figure-model."
        ),
    )
    parser.add_argument(
        "--figure-model",
        default=None,
        metavar="MODEL",
        help=(
            "the model that writes the figure descriptions; required with "
            "--mode accurate, ignored by --mode fast. A docling preset id "
            "(smolvlm / granite_vision / qwen / pixtral) downloads weights and "
            "runs them in-process. 'ollama:<model>' uses a vision model your "
            "local Ollama already serves, downloading nothing, "
            "e.g. --figure-model ollama:minicpm-v4.6:1b."
        ),
    )
    parser.add_argument(
        "--no-ocr",
        dest="ocr",
        action="store_false",
        help=(
            "skip OCR in --mode fast. For a born-digital PDF the text layer "
            "is already there and OCR is the expensive stage, so this is the "
            "fastest correct path. A scan read with --no-ocr yields no text "
            "and fails. No effect on --mode accurate."
        ),
    )
    parser.add_argument(
        "--preserve-superscripts",
        action="store_true",
        help=(
            "PDF only: keep raised text (citation numbers, units) and page "
            "footnotes marked in the markdown instead of merged into the body."
        ),
    )
    parser.add_argument(
        "--tdm-host",
        action="append",
        default=[],
        metavar="HOST",
        help=(
            "a hostname for which you attest to holding a text-and-data-mining "
            "licence; turns off the paywalled-publisher check for it. May be "
            "repeated, e.g. --tdm-host nature.com --tdm-host sciencedirect.com"
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print progress to stderr."
    )
    return parser


def _error(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


async def _run(args: argparse.Namespace) -> int:
    tdm_hosts = frozenset(h.strip().lower() for h in args.tdm_host if h.strip())
    try:
        markdown = await extract(
            args.source,
            mode=args.mode,
            ocr=args.ocr,
            figure_model=args.figure_model,
            tdm_allowed_hosts=tdm_hosts,
            preserve_superscripts=args.preserve_superscripts,
        )
    except ImportError as exc:
        # mode="accurate" without the optional VLM extra installed.
        _error(str(exc))
        return EXIT_USAGE
    except FigureModelRequired as exc:
        _error(str(exc))
        return EXIT_USAGE
    except FetchError as exc:
        _error(f"could not fetch {args.source!r}: {exc}")
        return EXIT_INPUT
    except UnsupportedFormatError as exc:
        _error(f"unsupported format for {args.source!r}: {exc}")
        return EXIT_INPUT
    except ExtractionError as exc:
        _error(f"extraction failed for {args.source!r}: {exc}")
        return EXIT_FAILED
    except HarvestError as exc:
        _error(f"harvest failed for {args.source!r}: {exc}")
        return EXIT_FAILED

    if args.out is None:
        sys.stdout.write(markdown)
        if not markdown.endswith("\n"):
            sys.stdout.write("\n")
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
    return 0


def _configure_logging(verbose: bool) -> None:
    """Route harvest logging to stderr when --verbose; otherwise warnings only."""
    logger = logging.getLogger(_LOGGER_NAME)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.propagate = False


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "accurate" and args.figure_model is None:
        parser.error("--mode accurate needs --figure-model")
    _configure_logging(args.verbose)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        _error("interrupted")
        return EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
