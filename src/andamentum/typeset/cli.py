"""Command-line entry point: ``andamentum-typeset``.

Reads a Markdown file (or stdin) and renders it to HTML or PDF using one
of the typeset styles. The output format is inferred from the output
filename's extension (``.html``, ``.htm``, or ``.pdf``).

Exit codes:
    0 — success
    1 — argument error
    2 — input file not found
    3 — render failed (e.g. PDF backend missing)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn, Sequence

from .renderer import render, render_to_file
from .styles import STYLES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="andamentum-typeset",
        description=(
            "Render a Markdown file to a typeset HTML or PDF document. "
            "Format is inferred from the output extension (.html / .pdf)."
        ),
    )
    from andamentum import __version__ as _ver

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s (andamentum {_ver})",
    )
    parser.add_argument(
        "source",
        help="Markdown file path. Use '-' to read from stdin.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        metavar="FILE",
        help=(
            "Output file path. Extension determines format: .html/.htm "
            "for HTML, .pdf for PDF. Default: '-' (HTML to stdout)."
        ),
    )
    parser.add_argument(
        "--style",
        default="article",
        choices=sorted(STYLES.keys()),
        help="Document style preset. Default: article.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="HTML <title> (defaults to first heading in the document).",
    )
    parser.add_argument(
        "--footer",
        default="",
        help="Footer text (appears at the bottom of every PDF page).",
    )
    return parser


def _die(code: int, message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(code)


def _read_source(source: str) -> str:
    if source == "-":
        return sys.stdin.read()
    path = Path(source)
    if not path.exists():
        _die(2, f"input file not found: {source}")
    return path.read_text(encoding="utf-8")


def _is_pdf(output: str) -> bool:
    return output != "-" and Path(output).suffix.lower() == ".pdf"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    markdown = _read_source(args.source)

    if args.output == "-":
        try:
            html = render(
                markdown,
                style=args.style,
                title=args.title,
                footer=args.footer,
            )
        except Exception as exc:  # noqa: BLE001 — surface backend errors
            _die(3, f"render failed: {exc}")
        sys.stdout.write(html)
        if not html.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if _is_pdf(args.output):
        try:
            from .renderer import render_pdf
        except ImportError as exc:
            _die(
                3,
                f"PDF rendering requires WeasyPrint: {exc}. "
                "Install with: uv pip install weasyprint",
            )
        try:
            render_pdf(
                markdown,
                output_path,
                style=args.style,
                title=args.title,
                footer=args.footer,
            )
        except Exception as exc:  # noqa: BLE001
            _die(3, f"PDF render failed: {exc}")
    else:
        try:
            render_to_file(
                markdown,
                output_path,
                style=args.style,
                title=args.title,
                footer=args.footer,
            )
        except Exception as exc:  # noqa: BLE001
            _die(3, f"render failed: {exc}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
