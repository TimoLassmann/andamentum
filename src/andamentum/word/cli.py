"""``andamentum-word``: work on an existing Word file.

Subcommands: ``check``, ``respond``, ``refresh``, ``bind-numbers``,
``draft``, ``find-citation``, ``paragraphs``, ``insert-citation`` and ``mcp``. The input file
is never edited. A subcommand that writes puts an edited copy, a report and
``state/`` (what a later run into the same folder picks up) into the folder
named by ``-o/--out``, and nothing anywhere else; ``find-citation`` and
``paragraphs`` only print.

Exit codes: 0 success, 1 the run finished with a negative result (errors found
in the document, something left for the authors, a work that did not verify;
the output is still written), 2 usage error, 3 an input could not be read,
4 the run failed (model or network error, crash).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

EXIT_OK, EXIT_NEGATIVE, EXIT_USAGE, EXIT_INPUT, EXIT_FAILED = 0, 1, 2, 3, 4

EXIT_EPILOG = (
    "exit codes: 0 success; 1 finished with a negative result (output still "
    "written); 2 usage error; 3 an input could not be read; 4 the run failed"
)


def _version() -> str:
    from andamentum import __version__

    return f"andamentum-word {__version__}"


def _out(parser: argparse.ArgumentParser, *, what: str) -> None:
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        required=True,
        metavar="DIR",
        help=f"folder to write into: {what}, the report and a state/ subfolder "
        "(the folder keeps state/ so a later run, for example refresh after "
        "bind-numbers, finds the earlier results); nothing is written anywhere else",
    )


def _json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json", action="store_true", help="print the result as JSON instead of text"
    )


def _file(parser: argparse.ArgumentParser, *, what: str) -> None:
    parser.add_argument("file", type=Path, help=f"the .docx to {what}")


def _library(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--library",
        type=Path,
        help="your references: a CSL JSON (.json) or BibTeX (.bib) file",
    )
    parser.add_argument(
        "--zotero",
        action="store_true",
        help="also read your Zotero library (read only)",
    )


def _mailto(parser: argparse.ArgumentParser, *, what: str = "Crossref") -> None:
    parser.add_argument(
        "--mailto", metavar="EMAIL", help=f"your email, for the polite pool of {what}"
    )


def _offline(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Make no network call besides the model: no Crossref, retraction or "
        "web checks, and a web address as --source is refused (a local Zotero "
        "library is still read)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="andamentum-word",
        description="Work on an existing Word file: check it, answer reviewer "
        "comments, bind and refresh numbers, insert citations, or draft into it. "
        "The input file is never edited; results go into --out DIR.",
        epilog="examples:\n"
        "  andamentum-word check draft.docx -o checked\n"
        "  andamentum-word respond draft.docx --reviews reviews.txt -o answered "
        "--model anthropic:claude-haiku-4-5\n"
        '  andamentum-word draft "A methods paragraph on the SaltLog assay" '
        "--into draft.docx -o drafted --model anthropic:claude-haiku-4-5\n"
        f"{EXIT_EPILOG}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=_version())
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add(name: str, summary: str, example: str) -> argparse.ArgumentParser:
        return sub.add_parser(
            name,
            help=summary,
            description=summary,
            epilog=f"example: {example}\n{EXIT_EPILOG}",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

    check = add(
        "check",
        "Check a .docx: citations, cross-references, statistics, numbers; "
        "findings become comments in a copy.",
        "andamentum-word check collaborator.docx --out checked --results results",
    )
    _file(check, what="check")
    _out(check, what="the commented copy")
    check.add_argument(
        "--results",
        type=Path,
        metavar="DIR",
        help="folder of result files (csv, tsv, json) to check the numbers against",
    )
    check.add_argument(
        "--checks",
        action="append",
        default=[],
        metavar="ID",
        help="run only this check (repeatable or comma-separated); default: all "
        "whose needs are met",
    )
    check.add_argument(
        "--model", help="model for the checks that need one (the numbers check)"
    )
    check.add_argument(
        "--decision-model",
        metavar="MODEL",
        help="local decision model for the citation pre-gate",
    )
    _mailto(check)
    _offline(check)
    _json(check)

    respond = add(
        "respond",
        "Answer the comments in a .docx and a reviewers' report: tracked edits, "
        "replies and a response letter.",
        "andamentum-word respond paper.docx --reviews reviews.txt --out answered --model M",
    )
    _file(respond, what="answer in")
    _out(respond, what="the answered copy and the response letter")
    respond.add_argument(
        "--reviews",
        type=Path,
        help="reviewers' report to answer too (.txt, .md, .docx or .pdf)",
    )
    respond.add_argument(
        "--model", help="model that triages, edits and answers (needed with --reviews)"
    )
    respond.add_argument(
        "--decision-model",
        metavar="MODEL",
        help="local decision model that triages instead of --model when it is sure",
    )
    respond.add_argument(
        "--results",
        type=Path,
        metavar="DIR",
        help="folder of result files, for the numbers check",
    )
    _mailto(respond)
    _offline(respond)
    _json(respond)

    refresh = add(
        "refresh",
        "Recompute the numbers bound to result files as tracked changes.",
        "andamentum-word refresh answered/paper.bound.docx --results results --out answered",
    )
    _file(refresh, what="refresh")
    _out(refresh, what="the refreshed copy (use the folder that bind-numbers wrote to)")
    refresh.add_argument(
        "--results",
        type=Path,
        required=True,
        metavar="DIR",
        help="folder of result files (csv, tsv, json)",
    )
    _json(refresh)

    bind = add(
        "bind-numbers",
        "Tie the numbers written in a .docx to the result files they come from.",
        "andamentum-word bind-numbers collaborator.docx --results results --out bound --model M",
    )
    _file(bind, what="bind")
    _out(bind, what="the bound copy")
    bind.add_argument(
        "--results",
        type=Path,
        required=True,
        metavar="DIR",
        help="folder of result files (csv, tsv, json)",
    )
    bind.add_argument("--model", required=True, help="model that proposes the matches")
    _json(bind)

    draft = add(
        "draft",
        "Draft a document from a task into Word: tracked changes, comments, "
        "verified citations and bound numbers.",
        'andamentum-word draft "Results section" --out drafted --into paper.docx '
        "--results results --library refs.json --model M",
    )
    draft.add_argument("task", help="what to write, in your own words")
    _out(draft, what="draft.docx")
    draft.add_argument(
        "--model",
        required=True,
        help="model id; it drafts, labels the claims and checks the numbers",
    )
    draft.add_argument(
        "--into",
        type=Path,
        help="extend a copy of this .docx as tracked changes (it is never edited)",
    )
    draft.add_argument("--genre", help="genre pack name or genre .yaml file")
    draft.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="FILE_OR_URL",
        help="a file or URL to draw on (repeatable)",
    )
    draft.add_argument(
        "--results",
        type=Path,
        metavar="DIR",
        help="folder of result files (csv, tsv, json); numbers are bound to them",
    )
    _library(draft)
    draft.add_argument(
        "--template", type=Path, help="a .docx or .dotx whose styles the document uses"
    )
    _offline(draft)
    draft.add_argument(
        "--confirm-own-sources",
        action="store_true",
        help="proceed although a source looks confidential",
    )
    _mailto(draft, what="Crossref and OpenAlex")
    draft.add_argument(
        "--research",
        choices=["auto", "web", "scholarly", "both"],
        metavar="WHERE",
        help="research the task once and cite what it finds (needs the network): "
        "web, scholarly or both, or auto to let the genre choose (scholarly for "
        "article, grant, report; web for memo, email, howto; both for proposal)",
    )
    draft.add_argument(
        "--searxng-url",
        help="with --research: your SearxNG address (web search needs this or --browser)",
    )
    draft.add_argument(
        "--browser",
        action="store_true",
        help="with --research: search the web through an isolated Chrome in a "
        "podman container instead of SearxNG",
    )
    _json(draft)

    find = add(
        "find-citation",
        "Find works for a title, topic or claim and verify them; prints the "
        "result and writes nothing.",
        "andamentum-word find-citation 'daily salt logging' --library refs.json",
    )
    find.add_argument("query", help="a title, a topic or a claim to find works for")
    _library(find)
    find.add_argument(
        "--limit", type=int, default=5, help="most works to look up (default 5)"
    )
    _mailto(find)
    _json(find)

    paragraphs = add(
        "paragraphs",
        "List the paragraphs of a .docx with their ids (what insert-citation "
        "needs); prints the list and writes nothing.",
        "andamentum-word paragraphs paper.docx",
    )
    _file(paragraphs, what="list")
    _json(paragraphs)

    insert = add(
        "insert-citation",
        "Cite verified works at a place in a .docx, as tracked changes in a copy.",
        "andamentum-word insert-citation paper.docx --paragraph-id p3-ab12 --at 118 "
        "--doi 10.5555/saltlog.2020.1 --out cited",
    )
    _file(insert, what="cite in")
    _out(insert, what="the cited copy")
    insert.add_argument(
        "--paragraph-id",
        required=True,
        metavar="ID",
        help="id of the paragraph to cite in; list the ids with "
        "'andamentum-word paragraphs FILE.docx'",
    )
    insert.add_argument(
        "--at",
        type=int,
        required=True,
        metavar="N",
        help="character offset in the paragraph text where the citation goes",
    )
    insert.add_argument(
        "--doi",
        action="append",
        default=[],
        help="DOI of a work to cite (repeatable)",
    )
    insert.add_argument(
        "--key",
        action="append",
        default=[],
        help="id of an item of --library or --zotero to cite (repeatable)",
    )
    _library(insert)
    insert.add_argument("--locator", help="for example 'p. 12'")
    insert.add_argument("--prefix", help="text before the citation")
    insert.add_argument("--suffix", help="text after the citation")
    _mailto(insert)
    _json(insert)

    add(
        "mcp",
        "Serve these jobs to Claude Code or Claude for Word over MCP (stdio). "
        "Each tool call names the file it writes.",
        "andamentum-word mcp\n"
        "to register it with Claude Code: claude mcp add word -- andamentum-word mcp",
    )
    return parser


def _must_exist(path: Path | None, *, flag: str, folder: bool = False) -> None:
    from andamentum.compose.harness.errors import WordInputError

    if path is None:
        return
    if (folder and not path.is_dir()) or (not folder and not path.is_file()):
        kind = "folder" if folder else "file"
        raise WordInputError(f"{flag} {path} is not an existing {kind}")


def _checks(values: Sequence[str]) -> list[str]:
    return [part for value in values for part in value.split(",") if part]


def _emit(run: object, *, as_json: bool, text: list[str]) -> None:
    if as_json:
        print(run.model_dump_json(indent=2))  # type: ignore[attr-defined]
    else:
        for line in text:
            print(line)


async def _dispatch(args: argparse.Namespace) -> int:
    from andamentum import word

    command = args.command
    if command == "check":
        _must_exist(args.results, flag="--results", folder=True)
        run = await word.check_document(
            args.file,
            out=args.out,
            results=args.results,
            checks=_checks(args.checks),
            model=args.model,
            decision_model=args.decision_model,
            mailto=args.mailto,
            offline=args.offline,
        )
        report = run.report
        text = [
            f"commented copy: {run.output}",
            f"report: {run.report_md} and {run.report_json}",
            f"{report.counts['error']} error(s), {report.counts['warning']} warning(s), "
            f"{report.counts['info']} info; checks run: {', '.join(report.checks_run)}",
            *(f"skipped {s.id}: {s.reason}" for s in report.skipped),
        ]
        _emit(run, as_json=args.json, text=text)
        return report.exit_code

    if command == "respond":
        _must_exist(args.reviews, flag="--reviews")
        _must_exist(args.results, flag="--results", folder=True)
        run = await word.respond_to_reviews(
            args.file,
            out=args.out,
            reviews=args.reviews,
            model=args.model,
            decision_model=args.decision_model,
            results=args.results,
            mailto=args.mailto,
            offline=args.offline,
        )
        report = run.report
        text = [
            f"copy: {run.output}",
            f"report: {run.report_md} and {run.report_json}",
            *(f"response letter: {path}" for path in report.letter),
            ", ".join(f"{n} {c}" for c, n in report.counts.items()),
        ]
        _emit(run, as_json=args.json, text=text)
        return report.exit_code

    if command == "refresh":
        _must_exist(args.results, flag="--results", folder=True)
        run = await word.refresh_numbers(args.file, out=args.out, results=args.results)
        report = run.report
        text = [
            f"refreshed copy: {run.output}",
            f"report: {run.report_md} and {run.report_json}",
            f"{len(report.changes)} number(s) changed, {report.unchanged} unchanged, "
            f"{report.errors} error(s)",
        ]
        _emit(run, as_json=args.json, text=text)
        return report.exit_code

    if command == "bind-numbers":
        _must_exist(args.results, flag="--results", folder=True)
        run = await word.bind_numbers(
            args.file, out=args.out, results=args.results, model=args.model
        )
        report = run.report
        text = [
            f"bound copy: {run.output}",
            f"report: {run.report_md} and {run.report_json}",
            f"{len(report.bound)} number(s) bound, {len(report.unmatched)} not bound",
        ]
        _emit(run, as_json=args.json, text=text)
        return EXIT_OK

    if command == "draft":
        if args.offline:
            from andamentum.compose.harness.errors import WordUsageError

            for source in args.source or ():
                if source.startswith(("http://", "https://")):
                    raise WordUsageError(
                        f"--offline rules out fetching {source}; save it as a file first"
                    )
        _must_exist(args.into, flag="--into")
        _must_exist(args.results, flag="--results", folder=True)
        _must_exist(args.library, flag="--library")
        _must_exist(args.template, flag="--template")
        result = await word.draft_document(
            args.task,
            out=args.out,
            model=args.model,
            into=args.into,
            genre=args.genre,
            sources=args.source,
            results=args.results,
            library=args.library,
            zotero=args.zotero,
            template=args.template,
            offline=args.offline,
            confirm_own_sources=args.confirm_own_sources,
            mailto=args.mailto,
            research=args.research is not None,
            research_mode=None if args.research == "auto" else args.research,
            searxng_url=args.searxng_url,
            browser=args.browser,
            progress=lambda line: print(line, file=sys.stderr, flush=True),
        )
        text = [
            f"document: {result.output}",
            f"report: {result.report}",
            f"state: {result.project}",
            f"claims ledger: {result.ledger}",
            f"{len(result.sections)} section(s), {result.citations} citation(s), "
            f"{result.numbers_bound} number(s) bound",
            *(f"open: {line}" for line in result.open_errors),
            *(f"refused: {line}" for line in result.refused),
            *(f"question: {line}" for line in result.author_questions),
        ]
        _emit(result, as_json=args.json, text=text)
        return result.exit_code

    if command == "find-citation":
        _must_exist(args.library, flag="--library")
        found = await word.find_citation(
            args.query,
            library=args.library,
            zotero=args.zotero,
            limit=args.limit,
            mailto=args.mailto,
        )
        if args.json:
            print(found.model_dump_json(indent=2))
        else:
            for work in found.verified:
                csl = work.csl
                title = csl.get("title", "")
                print(f"verified: {title} [{work.uri}] ({work.source})")
            for work in found.unverified:
                print(f"not verified: {work.title or work.doi}: {work.reason}")
            if not found.verified and not found.unverified:
                print("no works found")
        return EXIT_OK

    if command == "paragraphs":
        items = await word.list_paragraphs(args.file)
        if args.json:
            print(json.dumps([item.model_dump() for item in items], indent=2))
        else:
            for item in items:
                if item.heading_level is not None:
                    kind = f"heading {item.heading_level}"
                else:
                    kind = item.style or "body"
                start = " ".join(item.text.split())
                if len(start) > 60:
                    start = start[:60] + "..."
                print(f"{item.id}  [{kind}]  {start}")
        return EXIT_OK

    if command == "insert-citation":
        _must_exist(args.library, flag="--library")
        done = await word.insert_citation(
            args.file,
            out=args.out,
            paragraph_id=args.paragraph_id,
            at=args.at,
            doi=args.doi,
            key=args.key,
            library=args.library,
            zotero=args.zotero,
            locator=args.locator,
            prefix=args.prefix,
            suffix=args.suffix,
            mailto=args.mailto,
        )
        if args.json:
            print(done.model_dump_json(indent=2))
        else:
            print(done.message, file=sys.stdout if done.cited else sys.stderr)
        return EXIT_OK if done.cited else EXIT_NEGATIVE

    raise AssertionError(command)  # pragma: no cover


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if args.command == "mcp":
        import anyio

        from andamentum.compose.harness.mcp_server import serve_stdio

        anyio.run(serve_stdio)
        return EXIT_OK

    from andamentum.core.secrets import load_secrets

    from andamentum.compose.harness.errors import WordInputError, WordUsageError

    load_secrets()  # credentials only (API keys); configuration comes from flags
    try:
        return asyncio.run(_dispatch(args))
    except WordUsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except WordInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except Exception as exc:  # a crash is reported with its traceback, exit 4
        traceback.print_exc()
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
