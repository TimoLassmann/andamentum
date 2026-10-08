"""Command-line entry point: ``andamentum-references``.

Check the citations and the reference list of a draft, in any citation style:
does every citation point to an entry, is every entry cited, and (unless
``--no-lookup``) does each entry exist in Crossref with the details written.

    andamentum-references DRAFT [--model M] [--decision-model M]
        [--no-lookup] [--mailto EMAIL] [--json] [-o FILE]

DRAFT is a markdown or plain-text file, or ``-`` for text on stdin. A Word,
PDF or HTML file is converted to markdown first with ``andamentum.harvest``.
Only the text of the reference entries is sent to Crossref, never the draft.

This is the ONE place inside ``andamentum.references`` that imports
``andamentum.harvest`` (lazily, only for a file that is not already text);
the library proper depends on ``core`` alone.

Exit codes:
    0 — success: no problem found
    1 — the check ran and found problems (or found no reference list)
    2 — usage error
    3 — the input could not be read (missing file, unsupported format)
    4 — the run failed (model or network error, crash)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, Sequence

if TYPE_CHECKING:
    from . import ReferenceReport

_EXIT_NEGATIVE = 1
_EXIT_USAGE = 2
_EXIT_INPUT = 3
_EXIT_FAILED = 4

#: Files read as they are; anything else goes through harvest.
_TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".text"})

_EPILOG = """\
exit codes:
  0  no problem found
  1  the check ran and found problems (or found no reference list)
  2  usage error
  3  the input could not be read
  4  the run failed (model or network error, crash)
"""


def build_parser() -> argparse.ArgumentParser:
    from andamentum import __version__

    parser = argparse.ArgumentParser(
        prog="andamentum-references",
        description=(
            "Check a draft's citations against its reference list, in any "
            "citation style, then look each entry up in Crossref: does the "
            "work exist, and do its title, year and first author match? Only "
            "the text of the reference entries is sent to Crossref, never "
            "the draft; --no-lookup keeps everything offline."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "draft",
        metavar="DRAFT",
        help=(
            "The draft: a markdown or text file, '-' for text on stdin, or a "
            "Word, PDF or HTML file (converted to markdown first)."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help=(
            "A model to settle what the text leaves open: how the document "
            "cites, and whether an ambiguous number is a citation. Used only "
            "then, with short excerpts; a local model (ollama:...) keeps them "
            "local. Without it, such cases are reported as undecided."
        ),
    )
    parser.add_argument(
        "--decision-model",
        default=None,
        metavar="MODEL",
        help=(
            "A local Ollama decision model (for example nimble:9b) that "
            "answers each citation-candidate question instead of --model. "
            "The server and the model are checked before the draft is read."
        ),
    )
    parser.add_argument(
        "--no-lookup",
        action="store_true",
        help="Check citations against the list only; send nothing to Crossref.",
    )
    parser.add_argument(
        "--mailto",
        default=None,
        metavar="EMAIL",
        help=(
            "An email address sent to Crossref with each lookup, which puts "
            "the requests in its faster 'polite pool'. Optional."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write the result to this file instead of stdout.",
    )
    return parser


def _die(code: int, message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(code)


async def _read_draft(draft: str) -> str:
    """The draft as text: stdin, a text file as it is, anything else through
    harvest."""
    if draft == "-":
        return sys.stdin.read()
    path = Path(draft)
    if not path.is_file():
        _die(_EXIT_INPUT, f"not a file: {draft!r}")
    if path.suffix.lower() in _TEXT_SUFFIXES:
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            _die(_EXIT_INPUT, f"could not read {draft!r}: {exc}")
    # Deferred: harvest is only for files that are not already text, and
    # --help must not import it.
    from andamentum.harvest import HarvestError, extract

    try:
        return await extract(path)
    except HarvestError as exc:
        _die(_EXIT_INPUT, f"could not read {draft!r}: {exc}")


_STYLE_NAMES = {
    "numeric_brackets": "numbers in square brackets, like [3]",
    "numeric_superscript": "superscript numbers",
    "numeric_parentheses": "numbers in parentheses, like (3)",
    "author_year": "author and year, like (Smith et al., 2020)",
    "notes": "footnotes or endnotes",
    "pandoc_keys": "Pandoc citation keys, like [@smith2020]",
}


def _human(report: "ReferenceReport", text: str, *, lookup: bool) -> str:
    """The result in plain words: how the document cites, then each problem."""
    if not report.found_list and report.style is None:
        return (
            "No reference list was found, so nothing was checked. The checker "
            "looks for a section headed References, Bibliography, Works cited "
            "or Literature cited."
        )
    lines: list[str] = []
    style = report.style
    if report.entries:
        lines.append(f"Checked {len(report.entries)} references.")
    if style is None or style.style == "undecided":
        lines.append(
            "How the document cites: undecided. "
            + (style.reason if style else "")
            + " Citations were not checked."
        )
    else:
        lines.append(
            f"How the document cites: {_STYLE_NAMES.get(style.style, style.style)} "
            f"({style.reason})."
        )
        unjudged = sum(1 for j in report.judged if j.reason.startswith("not judged"))
        if unjudged:
            lines.append(
                f"{unjudged} candidate citation(s) were left unjudged; pass "
                "--model or --decision-model to judge them."
            )
    if report.looked_up:
        counts = {"found": 0, "not_found": 0, "details_differ": 0, "service_error": 0}
        for result in report.lookups:
            counts[result.status] += 1
        lines.append(
            f"Crossref lookup: {counts['found']} found as written, "
            f"{counts['details_differ']} with details that differ, "
            f"{counts['not_found']} not found"
            + (
                f", {counts['service_error']} not checked (Crossref did not answer)"
                if counts["service_error"]
                else ""
            )
            + "."
        )
    elif not lookup:
        lines.append("Crossref lookup: off, so the works themselves were not checked.")
    if not report.problems:
        lines.append("")
        lines.append("No problems found.")
        return "\n".join(lines)
    lines.append("")
    lines.append(f"{len(report.problems)} problem(s):")
    for problem in report.problems:
        lines.append("")
        lines.append(f"[{problem.severity}] {problem.headline}")
        quoted = " ".join(text[problem.start : problem.end].split())
        if quoted:
            lines.append(f"  at: {quoted[:160]}")
        lines.append(f"  {problem.issue}")
        lines.append(f"  Fix: {problem.fix}")
    return "\n".join(lines)


def _as_json(report: "ReferenceReport", text: str) -> str:
    data = report.model_dump(mode="json")
    for problem, entry in zip(report.problems, data["problems"]):
        entry["text"] = text[problem.start : problem.end]
    return json.dumps(data, indent=2, ensure_ascii=False)


async def _run(args: argparse.Namespace) -> int:
    from . import check_references

    # The decision model is opened (and so checked) before the draft is read.
    handle = None
    if args.decision_model:
        from andamentum.core.decisions import DecisionError, DecisionModel

        try:
            handle = await DecisionModel.open(args.decision_model)
        except DecisionError as exc:
            _die(_EXIT_FAILED, f"--decision-model {args.decision_model}: {exc}")
    try:
        text = await _read_draft(args.draft)
        try:
            report = await check_references(
                text,
                model=args.model,
                lookup=not args.no_lookup,
                mailto=args.mailto,
                decision_model=handle,
            )
        except Exception as exc:  # a model or network failure, not a verdict
            _die(_EXIT_FAILED, f"the check failed: {exc}")
    finally:
        if handle is not None:
            await handle.aclose()

    output = (
        _as_json(report, text)
        if args.json
        else _human(report, text, lookup=not args.no_lookup)
    )
    if args.out is None:
        sys.stdout.write(output + "\n")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    nothing_checked = not report.found_list and report.style is None
    return _EXIT_NEGATIVE if report.problems or nothing_checked else 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return _EXIT_FAILED
    except SystemExit as exc:
        if exc.code is None:
            return 0
        return exc.code if isinstance(exc.code, int) else _EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
