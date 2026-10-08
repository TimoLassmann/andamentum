"""Command-line entry point: andamentum-chunker.

Usage::

    andamentum-chunker INPUT --max-chars 4000 -o chunks.json
    andamentum-chunker - --max-chars 4000 --text-only < doc.md

Prints JSON with {chunks: [{text, start, end, heading_path}], total_chars}.

Exit codes: 0 success; 2 usage error; 3 the input could not be read
(missing file, not UTF-8 text). The chunker has no negative result, so 1 and
4 are not used.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .splitter import ChunkerError, chunk

_EXIT_CODES = """\
exit codes:
  0  success
  2  usage error (a wrong or missing argument)
  3  the input could not be read (missing file, not UTF-8 text)
"""


def build_parser() -> argparse.ArgumentParser:
    from andamentum import __version__

    parser = argparse.ArgumentParser(
        prog="andamentum-chunker",
        description="Split markdown into chunks at headings, paragraphs and sentences. "
        "Code blocks, tables and front matter are never cut. No model is used.",
        epilog=_EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"andamentum-chunker {__version__}"
    )
    parser.add_argument("input", help="path to a text or markdown file, or - for stdin")
    parser.add_argument(
        "--max-chars",
        type=int,
        required=True,
        help="maximum characters per chunk (code blocks and tables may exceed it)",
    )
    parser.add_argument(
        "-o",
        "--out",
        default="-",
        metavar="FILE",
        help="write the result to FILE (default: print to stdout)",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="print only the chunk texts, separated by a blank line and ---",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_chars < 1:
        print("error: --max-chars must be at least 1", file=sys.stderr)
        return 2
    try:
        if args.input == "-":
            source = sys.stdin.read()
        else:
            source = Path(args.input).read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 3
    except (OSError, UnicodeDecodeError) as exc:
        print(f"error: cannot read {args.input} as UTF-8 text: {exc}", file=sys.stderr)
        return 3

    try:
        chunks = chunk(source, max_chars=args.max_chars)
    except ChunkerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.text_only:
        body = "\n\n---\n\n".join(c.text for c in chunks)
    else:
        body = json.dumps(
            {
                "chunks": [
                    {
                        "text": c.text,
                        "start": c.start,
                        "end": c.end,
                        "heading_path": list(c.heading_path),
                    }
                    for c in chunks
                ],
                "total_chars": len(source),
            },
            indent=2,
        )
    if args.out == "-":
        print(body)
    else:
        try:
            Path(args.out).write_text(body, encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot write {args.out}: {exc}", file=sys.stderr)
            return 4
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
