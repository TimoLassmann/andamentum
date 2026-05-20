"""Command-line entry point: ``andamentum-harvest``.

Single positional argument (a URL or a local path) and one output flag.
No LLM, no model resolution — harvest is a pure source → markdown pipe.

Exit codes:
    0 — success
    1 — argument error
    2 — fetch / unsupported-format error (FetchError, UnsupportedFormatError)
    3 — extraction failed (ExtractionError, other HarvestError)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import NoReturn, Sequence

from .api import extract
from .errors import ExtractionError, FetchError, HarvestError, UnsupportedFormatError


_LOGGER_NAME = "andamentum.harvest"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="andamentum-harvest",
        description=(
            "Convert any URL or document file to clean markdown. "
            "Auto-detects PDF / HTML / DOCX / PPTX / Markdown / plain text; "
            "for HTML, races trafilatura and Docling when the page metadata "
            "is ambiguous and picks the better-structured output."
        ),
    )
    parser.add_argument(
        "source",
        help="URL (http://... / https://... / file://...) or local file path.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        metavar="FILE",
        help="Output markdown file path. Default: '-' (stdout).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress to stderr.",
    )
    parser.add_argument(
        "--tdm-host",
        action="append",
        default=[],
        metavar="HOST",
        help=(
            "Hostname for which you attest to holding a text-and-data-mining "
            "licence. Disarms the paywalled-publisher tripwire for this host. "
            "Repeatable. Example: --tdm-host nature.com --tdm-host sciencedirect.com"
        ),
    )
    return parser


def _die(code: int, message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(code)


async def _run(args: argparse.Namespace) -> int:
    tdm_hosts = frozenset(h.strip().lower() for h in args.tdm_host if h.strip())
    try:
        markdown = await extract(args.source, tdm_allowed_hosts=tdm_hosts)
    except FetchError as exc:
        _die(2, f"could not fetch {args.source!r}: {exc}")
    except UnsupportedFormatError as exc:
        _die(2, f"unsupported format for {args.source!r}: {exc}")
    except ExtractionError as exc:
        _die(3, f"extraction failed for {args.source!r}: {exc}")
    except HarvestError as exc:
        _die(3, f"harvest failed for {args.source!r}: {exc}")

    if args.output == "-":
        sys.stdout.write(markdown)
        if not markdown.endswith("\n"):
            sys.stdout.write("\n")
    else:
        out = Path(args.output)
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
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        _die(1, "interrupted")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
