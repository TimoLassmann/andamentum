"""Command-line entry point: andamentum-chunker.

Usage::

    andamentum-chunker INPUT --max-chars 4000 -o chunks.json
    andamentum-chunker - --max-chars 4000 --text-only < doc.md

Outputs JSON with {chunks: [{text, start, end, heading_path}], total_chars}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .splitter import chunk


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="andamentum-chunker",
        description="Deterministic chunking of markdown by headings, paragraphs and sentences.",
    )
    from andamentum import __version__ as _ver

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s (andamentum {_ver})"
    )
    parser.add_argument("input", help="Path to a text/markdown file, or - for stdin")
    parser.add_argument(
        "--max-chars",
        type=int,
        required=True,
        help="Maximum characters per chunk (code blocks and tables may exceed it)",
    )
    parser.add_argument(
        "-o", "--output", default="-", help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Print only the chunk texts, separated by a blank line and ---",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.max_chars < 1:
        print("Error: --max-chars must be >= 1", file=sys.stderr)
        return 2
    if args.input == "-":
        source = sys.stdin.read()
    else:
        path = Path(args.input)
        if not path.exists():
            print(f"Error: input file not found: {args.input}", file=sys.stderr)
            return 1
        source = path.read_text()

    chunks = chunk(source, max_chars=args.max_chars)
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
    if args.output == "-":
        print(body)
    else:
        Path(args.output).write_text(body)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
