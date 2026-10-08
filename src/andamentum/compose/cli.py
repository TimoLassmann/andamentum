"""``andamentum-compose``: a task in, a finished document out. Never sends anything.

``write TASK -o DIR`` drafts the document into the folder DIR: markdown
sections, the evidence, a report and the rendered files in ``DIR/out/``.
``continue DIR`` picks a folder up again, ``check DIR`` runs its checks once,
``status DIR`` says where it stands, ``render DIR`` renders it again and
``genres`` lists the kinds of document. ``new``, ``sections``, ``show``,
``edit``, ``add`` and ``validate`` read and edit a document folder and call no
model. ``eval`` compares compose with a single model call on synthetic tasks.
Working on an existing Word file is the separate tool ``andamentum-word``.

Exit codes: 0 success (the goal was met), 1 the run finished but the goal is
not met or the folder has problems (``validate`` found an error; the output is
still written), 2 usage error, 3 an input could not be read (the folder is
missing or is not a document folder, a source file does not exist), 4 the run
failed (model or network error, crash).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from .schemas import Roles

EXIT_OK, EXIT_NOT_MET, EXIT_USAGE, EXIT_INPUT, EXIT_FAILED = 0, 1, 2, 3, 4

EXIT_EPILOG = (
    "exit codes: 0 success; 1 finished but the goal is not met or problems were "
    "found; 2 usage error; 3 an input could not be read; 4 the run failed"
)
FORMATS = ["docx", "md", "html", "txt"]


def _progress(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _version() -> str:
    from andamentum import __version__

    return f"andamentum-compose {__version__}"


def _model(
    parser: argparse.ArgumentParser,
    *,
    jobs: str = "it drafts, reviews, checks the evidence and reads the result",
) -> None:
    parser.add_argument("--model", required=True, help=f"model id; {jobs}")


def _output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        action="append",
        dest="formats",
        choices=FORMATS,
        help="output format (repeatable); default: the genre's formats",
    )
    parser.add_argument(
        "--no-provenance",
        action="store_true",
        help="leave out provenance markers (refused for genres that require them)",
    )


RESEARCH_MODES = ["web", "scholarly", "both"]


def _research_options(parser: argparse.ArgumentParser, *, enable: bool) -> None:
    group = parser.add_argument_group(
        "research",
        "research the task once and cite what it finds"
        if enable
        else "only for a folder started with --research whose research did "
        "not finish; a finished research result is reused",
    )
    if enable:
        group.add_argument(
            "--research",
            choices=["auto", *RESEARCH_MODES],
            metavar="WHERE",
            help="search for the task and cite what is found (needs the network): "
            "web, scholarly or both, or auto to let the genre choose (scholarly "
            "for article, grant, report; web for memo, email, howto; both for "
            "proposal)",
        )
    group.add_argument(
        "--searxng-url",
        help="your SearxNG address (web search needs this or --browser)",
    )
    group.add_argument(
        "--browser",
        action="store_true",
        help="search the web through an isolated Chrome in a podman container "
        "instead of SearxNG",
    )
    group.add_argument(
        "--mailto",
        metavar="EMAIL",
        help="your email, for the polite pools of the sources",
    )


def _folder(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("folder", type=Path, help="the document folder")


def _json(parser: argparse.ArgumentParser, *, what: str = "the result") -> None:
    parser.add_argument(
        "--json", action="store_true", help=f"print {what} as JSON instead of text"
    )


def build_parser() -> argparse.ArgumentParser:
    from .bundle.scaffolds import SCAFFOLDS

    parser = argparse.ArgumentParser(
        prog="andamentum-compose",
        description="A task in, a finished document out, in a folder. Never "
        "sends anything. For an existing Word file use andamentum-word.",
        epilog=EXIT_EPILOG,
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

    write = add(
        "write",
        "Write a document from a task into a new folder.",
        'andamentum-compose write "A memo on the SaltLog pilot" --out memo '
        "--model anthropic:claude-haiku-4-5",
    )
    write.add_argument("task", help="what to write, in your own words")
    write.add_argument(
        "-o",
        "--out",
        type=Path,
        required=True,
        metavar="DIR",
        help="the new folder to write the document into (must not exist or be empty)",
    )
    write.add_argument(
        "--genre", help="kind of document: a name from 'genres' or a .yaml file"
    )
    write.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="FILE_OR_URL",
        help="a file or URL to draw on (repeatable)",
    )
    write.add_argument(
        "--confirm-own-sources",
        action="store_true",
        help="proceed although a source looks confidential",
    )
    _research_options(write, enable=True)
    _model(write)
    _output_options(write)
    _json(write)

    cont = add(
        "continue",
        "Continue a document folder with the settings 'write' recorded in it: "
        "finished steps are skipped, checks resume.",
        "andamentum-compose continue memo --model anthropic:claude-haiku-4-5",
    )
    _folder(cont)
    cont.add_argument(
        "--confirm-own-sources",
        action="store_true",
        help="proceed although a source looks confidential",
    )
    _research_options(cont, enable=False)
    _model(cont)
    _json(cont)

    status = add(
        "status",
        "Show where a document folder stands.",
        "andamentum-compose status memo",
    )
    _folder(status)

    check = add(
        "check",
        "Run a document folder's checks once, without revising.",
        "andamentum-compose check memo --model anthropic:claude-haiku-4-5",
    )
    _folder(check)
    _model(check, jobs="it reviews, checks the evidence and reads the text")
    _json(check)

    render = add(
        "render",
        "Render a document folder's text again (into its out/ folder).",
        "andamentum-compose render memo --format docx",
    )
    _folder(render)
    render.add_argument(
        "--model", help="model named in the provenance (default: the one recorded)"
    )
    render.add_argument(
        "--plain",
        action="store_true",
        help="plain render without citations, sources list or provenance; "
        "needs exactly one --format and -o FILE",
    )
    render.add_argument(
        "-o", "--out", type=Path, metavar="FILE", help="file to write (with --plain)"
    )
    _output_options(render)

    genres = add("genres", "List the kinds of document.", "andamentum-compose genres")
    _json(genres, what="the list")

    ev = add(
        "eval",
        "Compare compose with a single model call on the synthetic tasks.",
        "andamentum-compose eval --genre memo --out eval-out --model anthropic:claude-haiku-4-5",
    )
    ev.add_argument(
        "--genre",
        action="append",
        dest="genres",
        help="genre to evaluate (repeatable) or 'all' (default)",
    )
    ev.add_argument(
        "-o",
        "--out",
        type=Path,
        required=True,
        metavar="DIR",
        help="folder to write the documents and eval.json / eval.md into",
    )
    _model(ev)

    new = add(
        "new",
        "Create an empty document folder (no model).",
        'andamentum-compose new --out paper --title "SaltLog pilot" --scaffold article',
    )
    new.add_argument(
        "-o",
        "--out",
        type=Path,
        required=True,
        metavar="DIR",
        help="the new folder (must not exist or be empty)",
    )
    new.add_argument("--title", required=True, help="title of the document")
    new.add_argument(
        "--scaffold",
        choices=sorted(SCAFFOLDS),
        help="start with this set of empty sections",
    )

    sections = add(
        "sections",
        "List a document folder's sections.",
        "andamentum-compose sections paper",
    )
    _folder(sections)
    _json(sections, what="the list")

    show = add("show", "Print one section.", "andamentum-compose show paper s01")
    _folder(show)
    show.add_argument("section_id", help="id of the section, as 'sections' lists it")

    edit = add(
        "edit",
        "Replace a section's body.",
        "andamentum-compose edit paper s01 --from new-body.md --revision 1",
    )
    _folder(edit)
    edit.add_argument("section_id", help="id of the section, as 'sections' lists it")
    edit.add_argument(
        "--from",
        dest="source",
        required=True,
        metavar="FILE",
        help="file with the new body, or - for stdin",
    )
    edit.add_argument(
        "--revision",
        type=int,
        required=True,
        help="the revision you read; the edit is refused if the section moved on",
    )

    addp = add(
        "add",
        "Add a section.",
        'andamentum-compose add paper --title "Limitations" --after s03',
    )
    _folder(addp)
    addp.add_argument("--title", required=True, help="title of the new section")
    addp.add_argument(
        "--after", help="id of the section to insert after (default: at the end)"
    )

    validate = add(
        "validate", "Run the structural checks.", "andamentum-compose validate paper"
    )
    _folder(validate)
    _json(validate, what="the issues")
    return parser


def _open_folder(path: Path):  # noqa: ANN202
    """The document folder at *path*, or ``ComposeInputError`` (exit 3)."""
    from .bundle import Bundle, BundleError
    from .errors import ComposeInputError

    if not path.is_dir():
        raise ComposeInputError(f"{path} is not a folder")
    try:
        return Bundle.open(path)
    except BundleError as exc:
        raise ComposeInputError(str(exc)) from exc


def _research(  # noqa: ANN202
    args: argparse.Namespace, *, wanted: bool, mode: str | None, recorded: bool = False
):
    """The research settings, or None when research is not wanted.

    *mode* is where to search (``None``: the genre's choice). *recorded* is
    True for ``continue``, whose research setting comes from the folder.
    """
    from .errors import ComposeUsageError
    from .research import research_setup

    idle = [
        flag
        for flag, on in (
            ("--searxng-url", args.searxng_url is not None),
            ("--browser", args.browser),
            ("--mailto", args.mailto is not None),
        )
        if on
    ]
    if not wanted:
        if idle:
            if recorded:
                raise ComposeUsageError(
                    f"this folder was written without research, so {', '.join(idle)} "
                    "has nothing to do; leave it out"
                )
            raise ComposeUsageError(f"{', '.join(idle)} only works with --research")
        return None
    return research_setup(
        mailto=args.mailto,
        searxng_url=args.searxng_url,
        browser=args.browser,
        mode=mode,  # type: ignore[arg-type]  # argparse choices / the record
    )


def _check_sources(sources: Sequence[str]) -> None:
    from .errors import ComposeInputError

    for source in sources:
        if "://" not in source and not Path(source).exists():
            raise ComposeInputError(f"source {source} does not exist")


def _print_result(result, *, as_json: bool) -> int:  # noqa: ANN001
    if as_json:
        print(result.model_dump_json(indent=2))
    else:
        for line in result.summary_lines():
            print(line)
    return EXIT_OK if result.goal_met else EXIT_NOT_MET


async def _write(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from .api import write_document

    roles = Roles.all(args.model)
    _check_sources(args.source)
    if args.genre and args.genre.lower().endswith((".yaml", ".yml")):
        from .errors import ComposeInputError

        if not Path(args.genre).is_file():
            raise ComposeInputError(f"genre file {args.genre} does not exist")
    result = await write_document(
        args.task,
        out=args.out,
        roles=roles,
        genre=args.genre,
        sources=args.source,
        research=_research(
            args,
            wanted=args.research is not None,
            mode=None if args.research == "auto" else args.research,
        ),
        confirm_own_sources=args.confirm_own_sources,
        provenance=not args.no_provenance,
        formats=args.formats,
        progress=_progress,
    )
    return _print_result(result, as_json=args.json)


async def _continue(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from .api import continue_document, recorded_settings

    _open_folder(args.folder)
    settings = recorded_settings(args.folder)
    result = await continue_document(
        args.folder,
        roles=Roles.all(args.model),
        research=_research(args, wanted=True, mode=settings.research_mode)
        if settings.research
        else _research(args, wanted=False, mode=None, recorded=True),
        confirm_own_sources=settings.confirm_own_sources or args.confirm_own_sources,
        provenance=settings.provenance,
        formats=settings.formats,
        progress=_progress,
    )
    return _print_result(result, as_json=args.json)


async def _check(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from .api import check_folder

    _open_folder(args.folder)
    result = await check_folder(
        args.folder, roles=Roles.all(args.model), progress=_progress
    )
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        for item in result.checks:
            where = f" [{item.section_id}]" if item.section_id else ""
            print(f"{'pass' if item.passed else 'FAIL'}  {item.name}{where}")
            if not item.passed:
                for detail in item.details:
                    print(f"      {detail}")
        print(f"goal met: {'yes' if result.goal_met else 'no'}")
    return EXIT_OK if result.goal_met else EXIT_NOT_MET


async def _eval(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from .evals.runner import run_eval

    genres = None if not args.genres or "all" in args.genres else args.genres
    report = await run_eval(
        genres=genres,
        roles=Roles.all(args.model),
        out=args.out,
        progress=_progress,
    )
    print(report.markdown())
    print(f"written: {args.out / 'eval.json'} and {args.out / 'eval.md'}")
    return EXIT_OK


def _folder_command(args: argparse.Namespace) -> int:
    """The commands that read or edit a folder and call no model."""
    from .bundle import Bundle

    if args.command == "new":
        folder = Bundle.create(args.out, title=args.title, scaffold=args.scaffold)
        print(folder.path)
        return EXIT_OK

    folder = _open_folder(args.folder)

    if args.command == "sections":
        rows = folder.sections()
        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "id": s.id,
                            "revision": s.revision,
                            "status": s.status,
                            "words": len(s.body.split()),
                            "title": s.title,
                        }
                        for s in rows
                    ],
                    indent=2,
                )
            )
        else:
            for s in rows:
                words = len(s.body.split())
                print(
                    f"{s.id}  r{s.revision:<3d} {s.status:10s} {words:6d} words  {s.title}"
                )
        return EXIT_OK

    if args.command == "show":
        s = folder.section(args.section_id)
        print(
            f"id: {s.id}\ntitle: {s.title}\norder: {s.order}\n"
            f"revision: {s.revision}\nstatus: {s.status}\n"
        )
        print(s.body)
        return EXIT_OK

    if args.command == "edit":
        if args.source == "-":
            body = sys.stdin.read()
        else:
            from .errors import ComposeInputError

            source = Path(args.source)
            if not source.is_file():
                raise ComposeInputError(f"{source} is not a file")
            body = source.read_text(encoding="utf-8")
        s = folder.write_section(
            args.section_id, body=body, expected_revision=args.revision
        )
        print(f"{s.id} now at revision {s.revision}")
        return EXIT_OK

    if args.command == "add":
        s = folder.add_section(title=args.title, after=args.after)
        print(s.id)
        return EXIT_OK

    if args.command == "validate":
        issues = folder.validate()
        if args.json:
            print(json.dumps([i.model_dump() for i in issues], indent=2))
        elif not issues:
            print("ok: no validation issues")
        else:
            for issue in issues:
                print(f"[{issue.severity:7s}] {issue.location}: {issue.message}")
        return EXIT_NOT_MET if any(i.severity == "error" for i in issues) else EXIT_OK

    raise AssertionError(args.command)  # pragma: no cover


def _render(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    folder = _open_folder(args.folder)
    if args.plain:
        if not args.formats or len(args.formats) != 1 or args.out is None:
            parser.error("render --plain needs exactly one --format and -o FILE")
        print(folder.render(args.out, format=args.formats[0]))
        return EXIT_OK
    if args.out is not None:
        parser.error(
            "-o/--out is for --plain; without it the files go to the folder's out/"
        )
    from .api import recorded_roles, render_folder

    model = args.model
    if not model and not args.no_provenance:
        recorded = recorded_roles(args.folder)
        model = recorded.drafter if recorded else None
    if not model and not args.no_provenance:
        parser.error("render needs --model (no model is recorded in the folder)")
    for path in render_folder(
        args.folder,
        model=model or "",
        formats=args.formats,
        provenance=not args.no_provenance,
    ):
        print(path)
    return EXIT_OK


def _genres(args: argparse.Namespace) -> int:
    from .genres import GENRES

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "name": name,
                        "min_words": genre.word_range[0],
                        "max_words": genre.word_range[1],
                        "description": genre.description,
                    }
                    for name, genre in GENRES.items()
                ],
                indent=2,
            )
        )
        return EXIT_OK
    for name, genre in GENRES.items():
        lo, hi = genre.word_range
        print(f"{name:<10} {lo}-{hi} words  {genre.description}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])

    from dotenv import load_dotenv

    from .errors import ComposeError, ComposeInputError

    load_dotenv()  # credentials only (API keys); configuration comes from flags
    try:
        if args.command == "write":
            return asyncio.run(_write(args, parser))
        if args.command == "continue":
            return asyncio.run(_continue(args, parser))
        if args.command == "check":
            return asyncio.run(_check(args, parser))
        if args.command == "eval":
            return asyncio.run(_eval(args, parser))
        if args.command == "status":
            from .api import folder_status

            _open_folder(args.folder)
            print(folder_status(args.folder))
            return EXIT_OK
        if args.command == "render":
            return _render(args, parser)
        if args.command == "genres":
            return _genres(args)
        return _folder_command(args)
    except ComposeInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except ComposeError as exc:  # a bad argument or a refused folder operation
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:  # a crash is reported with its traceback, exit 4
        traceback.print_exc()
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
