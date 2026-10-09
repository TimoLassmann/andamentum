"""Command-line entry point: ``andamentum-typeset``.

Reads a Markdown file (or stdin) and renders it to HTML or PDF using one
of the typeset styles. The output format is inferred from the extension of
the ``-o/--out`` file (``.html``, ``.htm`` or ``.pdf``); without ``-o`` the
HTML is printed to stdout.

Exit codes:
    0 - success
    2 - usage error (a wrong or missing argument)
    3 - the input could not be read (missing file, not UTF-8 text)
    4 - the run failed (rendering failed, or PDF output needs the ``pdf`` extra)

A render has no negative result, so 1 is not used.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from .errors import PdfUnavailableError
from .renderer import render, render_to_file
from .styles import STYLES

_EXIT_CODES = """\
examples:
  andamentum-typeset notes.md -o notes.html
  andamentum-typeset notes.md --style report -o notes.pdf

exit codes:
  0  success
  2  usage error (a wrong or missing argument)
  3  the input could not be read (missing file, not UTF-8 text)
  4  the run failed (rendering failed, or PDF output needs the pdf extra; see the README, Installation)
"""


def build_parser() -> argparse.ArgumentParser:
    from andamentum import __version__

    parser = argparse.ArgumentParser(
        prog="andamentum-typeset",
        description=(
            "Render a Markdown file to a typeset HTML or PDF document. "
            "The format follows the extension of the -o file (.html or .pdf). "
            "PDF output needs the pdf extra (see the README, Installation)."
        ),
        epilog=_EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"andamentum-typeset {__version__}"
    )
    parser.add_argument(
        "source",
        help="Markdown file to render; use - to read from stdin",
    )
    parser.add_argument(
        "-o",
        "--out",
        default="-",
        metavar="FILE",
        help=(
            "file to write; the extension sets the format: .html or .htm "
            "for HTML, .pdf for PDF (default: print HTML to stdout)"
        ),
    )
    parser.add_argument(
        "--style",
        default="article",
        choices=sorted(STYLES.keys()),
        help="document style (default: article)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help='page title (default: the first # heading in the document, else "Document")',
    )
    parser.add_argument(
        "--footer",
        default="",
        help="footer text, shown at the bottom of every PDF page",
    )
    return parser


class _Exit(Exception):
    """Stops the run with an exit code and a one-line message."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _die(code: int, message: str) -> NoReturn:
    raise _Exit(code, message)


def _read_source(source: str) -> str:
    if source == "-":
        return sys.stdin.read()
    path = Path(source)
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        _die(3, f"input file not found: {source}")
    except (OSError, UnicodeDecodeError) as exc:
        _die(3, f"cannot read {source} as UTF-8 text: {exc}")


def _is_pdf(output: str) -> bool:
    return output != "-" and Path(output).suffix.lower() == ".pdf"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except _Exit as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.code


def _run(args: argparse.Namespace) -> int:
    markdown = _read_source(args.source)

    if args.out == "-":
        try:
            html = render(
                markdown,
                style=args.style,
                title=args.title,
                footer=args.footer,
            )
        except Exception as exc:  # noqa: BLE001 — surface backend errors
            _die(4, f"render failed: {exc}")
        sys.stdout.write(html)
        if not html.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if _is_pdf(args.out):
        from .renderer import render_pdf

        try:
            render_pdf(
                markdown,
                output_path,
                style=args.style,
                title=args.title,
                footer=args.footer,
            )
        except PdfUnavailableError as exc:
            _die(4, str(exc))
        except Exception as exc:  # noqa: BLE001
            _die(4, f"PDF render failed: {exc}")
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
            _die(4, f"render failed: {exc}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
