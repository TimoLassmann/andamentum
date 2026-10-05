"""``python -m andamentum.compose.harness.spike``: make, probe and survival commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .make import make_spike_docx
from .probe import evaluate, format_findings, probe
from .survival import render_report, run_survival


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m andamentum.compose.harness.spike",
        description="M0 spike: do anchors, fields and comments survive Word and other tools?",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    make = commands.add_parser("make", help="write the spike document")
    make.add_argument("path", type=Path)

    probe_command = commands.add_parser("probe", help="print what survived in a file")
    probe_command.add_argument("file", type=Path)
    probe_command.add_argument(
        "--json", action="store_true", help="print the raw probe report"
    )

    survival = commands.add_parser(
        "survival", help="run the python-docx, LibreOffice and whetstone tests"
    )
    survival.add_argument("--workdir", type=Path, required=True)
    survival.add_argument("--report", type=Path, help="write the markdown report here")

    args = parser.parse_args(argv)
    if args.command == "make":
        print(make_spike_docx(args.path))
        return 0
    if args.command == "probe":
        report = probe(args.file)
        if args.json:
            print(report.model_dump_json(indent=2))
        else:
            print(format_findings(report, evaluate(report)))
        return 0
    results = run_survival(args.workdir)
    text = render_report(results)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
