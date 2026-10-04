"""Command-line entry point: andamentum-scribe.

Subcommands work on a bundle directory; there is no default location.
  init      Create a bundle (optionally from the article scaffold).
  sections  List sections with revision, status and word count.
  show      Print one section (frontmatter fields, then the body).
  write     Replace a section's body from a file or stdin (needs --revision).
  add       Add a section, at the end or after a given id.
  validate  Run the structural checks; exit 1 on any error.
  render    Render to docx, md, html or txt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .bundle import Bundle
from .models import ScribeError
from .scaffolds import SCAFFOLDS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="andamentum-scribe")
    from andamentum import __version__ as _ver

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s (andamentum {_ver})"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init", help="Create a new bundle")
    init.add_argument("path")
    init.add_argument("--title", required=True)
    init.add_argument("--scaffold", default=None, choices=sorted(SCAFFOLDS))

    ls = sub.add_parser("sections", help="List sections")
    ls.add_argument("path")

    show = sub.add_parser("show", help="Print one section")
    show.add_argument("path")
    show.add_argument("section_id")

    write = sub.add_parser("write", help="Replace a section's body")
    write.add_argument("path")
    write.add_argument("section_id")
    write.add_argument(
        "--from", dest="source", required=True, help="FILE, or - for stdin"
    )
    write.add_argument(
        "--revision", type=int, required=True, help="the revision you read"
    )

    add = sub.add_parser("add", help="Add a section")
    add.add_argument("path")
    add.add_argument("--title", required=True)
    add.add_argument("--after", default=None, help="section id to insert after")

    val = sub.add_parser("validate", help="Run structural checks")
    val.add_argument("path")

    render = sub.add_parser("render", help="Render the bundle")
    render.add_argument("path")
    render.add_argument(
        "--format", required=True, choices=["docx", "md", "html", "txt"]
    )
    render.add_argument("--out", required=True)

    return parser


def _run(args: argparse.Namespace) -> int:
    if args.cmd == "init":
        bundle = Bundle.create(args.path, title=args.title, scaffold=args.scaffold)
        print(bundle.path)
        return 0

    bundle = Bundle.open(args.path)

    if args.cmd == "sections":
        for s in bundle.sections():
            words = len(s.body.split())
            print(
                f"{s.id}  r{s.revision:<3d} {s.status:10s} {words:6d} words  {s.title}"
            )
        return 0

    if args.cmd == "show":
        s = bundle.section(args.section_id)
        print(
            f"id: {s.id}\ntitle: {s.title}\norder: {s.order}\n"
            f"revision: {s.revision}\nstatus: {s.status}\n"
        )
        print(s.body)
        return 0

    if args.cmd == "write":
        body = (
            sys.stdin.read()
            if args.source == "-"
            else Path(args.source).read_text(encoding="utf-8")
        )
        s = bundle.write_section(
            args.section_id, body=body, expected_revision=args.revision
        )
        print(f"{s.id} now at revision {s.revision}")
        return 0

    if args.cmd == "add":
        s = bundle.add_section(title=args.title, after=args.after)
        print(s.id)
        return 0

    if args.cmd == "validate":
        issues = bundle.validate()
        if not issues:
            print("ok: no validation issues")
            return 0
        for issue in issues:
            print(f"[{issue.severity:7s}] {issue.location}: {issue.message}")
        return 1 if any(i.severity == "error" for i in issues) else 0

    if args.cmd == "render":
        print(bundle.render(args.out, format=args.format))
        return 0

    return 1  # pragma: no cover — argparse requires a known subcommand


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return _run(args)
    except ScribeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
