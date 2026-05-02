"""Command-line entry point: andamentum-scribe.

Subcommands mirror document-tools:doc-draft so users can swap mental
models 1:1:
  init             Create an empty document (optionally from a scaffold).
  list-sections    Print sections with block and word counts.
  read-section     Print the content of a named section.
  write-section    Replace a named section's body with content from a file.
  insert-figure    Append a figure block (or insert into a named section).
  insert-table     Append a table block (or insert into a named section)
                   from a CSV file.
  add-reference    Attach a bibliographic reference (cite key + optional
                   BibTeX entry).
  list-references  Print all references attached to the document.
  list-citations   Print all citation keys used in paragraph blocks.
  validate         Run structural validators (missing citations, missing
                   figure files, unresolved markers).
  render           Render the document to .docx.
"""

from __future__ import annotations

import argparse
import csv
import json
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

    addref = sub.add_parser(
        "add-reference",
        help="Attach a bibliographic reference (cite key + optional BibTeX)",
    )
    addref.add_argument("--database", required=True)
    addref.add_argument("--id", required=True)
    addref.add_argument("--cite-key", required=True)
    addref.add_argument(
        "--bibtex",
        default=None,
        help="BibTeX entry text (literal). Mutually exclusive with --bibtex-file.",
    )
    addref.add_argument(
        "--bibtex-file",
        default=None,
        help="Path to a file containing the BibTeX entry. Mutually exclusive with --bibtex.",
    )
    addref.add_argument(
        "--metadata-json",
        default=None,
        help="Optional JSON object with extra reference metadata",
    )

    lrefs = sub.add_parser("list-references", help="List bibliographic references")
    lrefs.add_argument("--database", required=True)
    lrefs.add_argument("--id", required=True)

    lcites = sub.add_parser(
        "list-citations",
        help="List unique citation keys used in paragraph blocks",
    )
    lcites.add_argument("--database", required=True)
    lcites.add_argument("--id", required=True)

    val = sub.add_parser(
        "validate",
        help="Run structural validators (missing cites, missing figures, …)",
    )
    val.add_argument("--database", required=True)
    val.add_argument("--id", required=True)

    render = sub.add_parser("render", help="Render document to .docx")
    render.add_argument("--database", required=True)
    render.add_argument("--id", required=True)
    render.add_argument("--output", required=True)

    return parser


def _resolve_bibtex(literal: str | None, path: str | None) -> str | None:
    """Pick whichever of --bibtex or --bibtex-file was supplied."""
    if literal is not None and path is not None:
        print(
            "error: pass either --bibtex or --bibtex-file, not both",
            file=sys.stderr,
        )
        sys.exit(2)
    if path is not None:
        return Path(path).read_text(encoding="utf-8")
    return literal


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
            bid = doc.insert_into_section(args.section, spec)
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
            bid = doc.insert_into_section(args.section, spec)
        else:
            bid = doc.append(spec)
        print(bid)
        return 0

    if args.cmd == "add-reference":
        doc = Document.open(args.id, database=args.database)
        bibtex = _resolve_bibtex(args.bibtex, args.bibtex_file)
        metadata = json.loads(args.metadata_json) if args.metadata_json else None
        rid = doc.add_reference(
            cite_key=args.cite_key, bibtex=bibtex, metadata=metadata
        )
        print(rid)
        return 0

    if args.cmd == "list-references":
        doc = Document.open(args.id, database=args.database)
        for ref in doc.references():
            preview = (ref.bibtex_entry or "").split("\n", 1)[0][:80]
            print(f"{ref.cite_key:30s}  {preview}")
        return 0

    if args.cmd == "list-citations":
        doc = Document.open(args.id, database=args.database)
        for key in doc.citations():
            print(key)
        return 0

    if args.cmd == "validate":
        doc = Document.open(args.id, database=args.database)
        issues = doc.validate()
        if not issues:
            print("ok: no validation issues")
            return 0
        for issue in issues:
            print(f"[{issue.severity:7s}] {issue.location}: {issue.message}")
        # Non-zero exit only if any error-severity issues present
        return 1 if any(i.severity == "error" for i in issues) else 0

    if args.cmd == "render":
        doc = Document.open(args.id, database=args.database)
        doc.render(args.output, format="docx")
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
