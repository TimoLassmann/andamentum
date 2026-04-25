"""Command-line entry point: andamentum-scribe.

Subcommands mirror document-tools:doc-draft so users can swap mental
models 1:1:
  init           Create an empty document (optionally from a scaffold).
  list-sections  Print sections with block and word counts.
  read-section   Print the content of a named section.
  write-section  Replace a named section's body with content from a file.
  insert-figure  Append a figure block (or insert into a named section).
  insert-table   Append a table block (or insert into a named section)
                 from a CSV file.
  render         Render the document to .docx.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .api import Document, Figure, Table


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="andamentum-scribe")
    sub = parser.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init", help="Create a new document")
    init.add_argument("--database", required=True)
    init.add_argument("--title", required=True)
    init.add_argument("--template", default=None)
    init.add_argument(
        "--scaffold",
        default=None,
        choices=["article", "grant"],
        help="Pre-populate canonical sections",
    )

    ls = sub.add_parser("list-sections", help="List sections with counts")
    ls.add_argument("--database", required=True)
    ls.add_argument("--id", required=True)

    rs = sub.add_parser("read-section", help="Print a section's content")
    rs.add_argument("--database", required=True)
    rs.add_argument("--id", required=True)
    rs.add_argument("--section", required=True)

    ws = sub.add_parser("write-section", help="Replace a section's body")
    ws.add_argument("--database", required=True)
    ws.add_argument("--id", required=True)
    ws.add_argument("--section", required=True)
    ws.add_argument("--content-file", required=True)
    ws.add_argument("--reason", default=None)

    ifig = sub.add_parser("insert-figure", help="Append a figure block")
    ifig.add_argument("--database", required=True)
    ifig.add_argument("--id", required=True)
    ifig.add_argument("--image", required=True)
    ifig.add_argument("--caption", required=True)
    ifig.add_argument("--label", required=True)
    ifig.add_argument("--width-in", type=float, default=None)
    ifig.add_argument(
        "--section",
        default=None,
        help="If set, append the figure as the last block of this section",
    )

    itab = sub.add_parser("insert-table", help="Append a table block from CSV")
    itab.add_argument("--database", required=True)
    itab.add_argument("--id", required=True)
    itab.add_argument("--csv", required=True, help="Path to CSV file")
    itab.add_argument("--caption", default="")
    itab.add_argument("--label", default="")
    itab.add_argument("--no-header", action="store_true")
    itab.add_argument(
        "--section",
        default=None,
        help="If set, append the table as the last block of this section",
    )

    render = sub.add_parser("render", help="Render document to .docx")
    render.add_argument("--database", required=True)
    render.add_argument("--id", required=True)
    render.add_argument("--output", required=True)

    return parser


def _append_to_section(doc: Document, section_name: str, block_spec: dict) -> str:
    """Insert a block as the last child of a named section."""
    import json as _json

    from .api import _new_id, _now_iso
    from .database import open_db

    section_blocks = doc.section(section_name)
    last = section_blocks[-1]
    insert_pos = last.position + 1
    bid = _new_id()
    now = _now_iso()

    with open_db(doc.database) as conn:
        conn.execute(
            "UPDATE scribe_blocks "
            "SET position = position + 1 "
            "WHERE doc_id = ? AND position >= ?",
            (doc.id, insert_pos),
        )
        conn.execute(
            "INSERT INTO scribe_blocks "
            "(id, doc_id, type, content, position, parent_id, metadata, "
            " revision, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?, 1, ?, ?)",
            (
                bid,
                doc.id,
                block_spec["type"],
                block_spec.get("content", ""),
                insert_pos,
                _json.dumps(block_spec.get("metadata", {})),
                now,
                now,
            ),
        )
        conn.commit()
    return bid


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "init":
        doc = Document.create(
            title=args.title,
            database=args.database,
            template=args.template,
            scaffold=args.scaffold,
        )
        print(doc.id)
        return 0

    if args.cmd == "list-sections":
        doc = Document.open(args.id, database=args.database)
        for s in doc.list_sections():
            print(
                f"{s['name']:30s}  blocks={s['block_count']:3d}  words={s['word_count']:5d}"
            )
        return 0

    if args.cmd == "read-section":
        doc = Document.open(args.id, database=args.database)
        for blk in doc.section(args.section):
            if blk.type == "heading":
                level = int(blk.metadata.get("level", 1))
                print(f"{'#' * level} {blk.content}\n")
            elif blk.type == "paragraph":
                print(f"{blk.content}\n")
            elif blk.type == "figure":
                m = blk.metadata
                print(
                    f"![{m.get('caption', '')}]({m.get('path', '')}) "
                    f"{{#{m.get('label', '')}}}\n"
                )
            elif blk.type == "table":
                rows = blk.metadata.get("rows", [])
                for row in rows:
                    print(" | ".join(row))
                print()
        return 0

    if args.cmd == "write-section":
        doc = Document.open(args.id, database=args.database)
        content = Path(args.content_file).read_text()
        doc.replace_section(args.section, content, reason=args.reason)
        return 0

    if args.cmd == "insert-figure":
        doc = Document.open(args.id, database=args.database)
        spec = Figure(
            path=args.image,
            caption=args.caption,
            label=args.label,
            width_in=args.width_in,
        )
        if args.section:
            bid = _append_to_section(doc, args.section, spec)
        else:
            bid = doc.append(spec)
        print(bid)
        return 0

    if args.cmd == "insert-table":
        doc = Document.open(args.id, database=args.database)
        with open(args.csv, newline="") as f:
            rows = [row for row in csv.reader(f)]
        spec = Table(
            rows=rows,
            header_row=not args.no_header,
            caption=args.caption,
            label=args.label,
        )
        if args.section:
            bid = _append_to_section(doc, args.section, spec)
        else:
            bid = doc.append(spec)
        print(bid)
        return 0

    if args.cmd == "render":
        doc = Document.open(args.id, database=args.database)
        doc.render(args.output, format="docx")
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
