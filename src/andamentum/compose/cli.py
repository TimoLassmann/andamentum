"""``andamentum-compose``: a task in, a finished document out. Never sends anything.

Exit codes for ``write`` and ``run``: 0 goal met, 4 goal not met (outputs are
still written), 3 crash. A missing role model or a bad argument exits 2.

``check FILE.docx`` is the harness verb (see ``harness/check.py``: commented copy
and report, exit 0/1/2/3); ``check-bundle DIR`` is the bundle's own one-pass check.

The bundle verbs (``new``, ``sections``, ``show``, ``edit``, ``add``,
``validate``, ``render --raw``) call no model. A bundle error prints
``error: ...`` and exits 2; ``validate`` exits 1 when it finds an error.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

from .bundle import Bundle, BundleError
from .bundle.scaffolds import SCAFFOLDS
from .harness.kernel.verbs import build_subparsers, dispatch_verb, load_builtin_verbs
from .schemas import ROLE_NAMES, Roles

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_CRASH = 3
EXIT_NOT_MET = 4
BUNDLE_COMMANDS = ("new", "sections", "show", "edit", "add", "validate")


def _progress(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _add_models(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "models", "--model fills every role not named; every role needs a model"
    )
    group.add_argument("--model", help="model id for every role not set below")
    for role in ROLE_NAMES:
        group.add_argument(f"--{role}", metavar="MODEL", help=f"model for the {role}")


def _add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        action="append",
        dest="formats",
        choices=["docx", "md", "html", "txt"],
        help="output format (repeatable); default: the genre's formats",
    )
    parser.add_argument(
        "--no-provenance",
        action="store_true",
        help="leave out provenance markers (refused for genres that require them)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="andamentum-compose",
        description="A task in, a finished document out. Never sends anything.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    write = sub.add_parser("write", help="write a document from a task")
    write.add_argument("task", help="what to write, in your own words")
    write.add_argument("--out", required=True, type=Path, help="new bundle folder")
    write.add_argument("--genre", help="pack name or path to a genre .yaml")
    write.add_argument(
        "--source",
        action="append",
        default=[],
        dest="sources",
        help="a file or URL to draw on (repeatable)",
    )
    write.add_argument("--max-rounds", type=int, default=4)
    write.add_argument(
        "--research",
        action="store_true",
        help="run web research once and add its report as a source (needs SearxNG)",
    )
    write.add_argument(
        "--confirm-own-sources",
        action="store_true",
        help="proceed although a source looks confidential",
    )
    _add_models(write)
    _add_output(write)

    run = sub.add_parser("run", help="continue an existing bundle")
    run.add_argument("bundle", type=Path)
    run.add_argument("--max-rounds", type=int, default=4)
    run.add_argument("--confirm-own-sources", action="store_true")
    _add_models(run)
    _add_output(run)

    status = sub.add_parser("status", help="show where a bundle stands")
    status.add_argument("bundle", type=Path)

    check = sub.add_parser(
        "check-bundle", help="run a bundle's checks once, without revising"
    )
    check.add_argument("bundle", type=Path)
    _add_models(check)

    render = sub.add_parser("render", help="render a bundle's text to out/ again")
    render.add_argument("bundle", type=Path)
    render.add_argument(
        "--model", help="producing model for provenance (default: recorded)"
    )
    render.add_argument(
        "--raw",
        action="store_true",
        help="plain bundle render (no citations, sources list or provenance); "
        "needs exactly one --format and --out FILE",
    )
    render.add_argument("--out", type=Path, help="output file (with --raw)")
    _add_output(render)

    new = sub.add_parser("new", help="create an empty bundle (no model)")
    new.add_argument("bundle", type=Path)
    new.add_argument("--title", required=True)
    new.add_argument("--scaffold", default=None, choices=sorted(SCAFFOLDS))

    sections = sub.add_parser("sections", help="list a bundle's sections")
    sections.add_argument("bundle", type=Path)

    show = sub.add_parser("show", help="print one section")
    show.add_argument("bundle", type=Path)
    show.add_argument("section_id")

    edit = sub.add_parser("edit", help="replace a section's body")
    edit.add_argument("bundle", type=Path)
    edit.add_argument("section_id")
    edit.add_argument(
        "--from", dest="source", required=True, help="FILE, or - for stdin"
    )
    edit.add_argument(
        "--revision", type=int, required=True, help="the revision you read"
    )

    add = sub.add_parser("add", help="add a section")
    add.add_argument("bundle", type=Path)
    add.add_argument("--title", required=True)
    add.add_argument("--after", default=None, help="section id to insert after")

    validate = sub.add_parser("validate", help="run the structural checks")
    validate.add_argument("bundle", type=Path)

    sub.add_parser("genres", help="list the genre packs")

    ev = sub.add_parser(
        "eval", help="compose against a single-call baseline on the synthetic tasks"
    )
    ev.add_argument(
        "--genre",
        action="append",
        dest="genres",
        help="genre to evaluate (repeatable) or 'all' (default)",
    )
    ev.add_argument("--out", required=True, type=Path)
    ev.add_argument("--max-rounds", type=int, default=4)
    _add_models(ev)
    load_builtin_verbs()
    build_subparsers(sub)  # harness verbs (registered in the kernel) become subcommands
    return parser


def resolve_roles(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    fallback: Roles | None = None,
) -> Roles:
    """Roles from the flags; a role with no model is a usage error (exit 2)."""
    chosen: dict[str, str] = {}
    missing: list[str] = []
    for role in ROLE_NAMES:
        value = getattr(args, role, None) or args.model
        if not value and fallback is not None:
            value = getattr(fallback, role)
        if value:
            chosen[role] = value
        else:
            missing.append(role)
    if missing:
        parser.error(
            f"no model for: {', '.join(missing)}. Pass --model MODEL to use one "
            "model for every role, or --" + ", --".join(missing) + " individually."
        )
    return Roles(**chosen)


def _print_result(
    result_bundle: Path,
    *,
    goal_met: bool,
    stop: str,
    outputs: list[Path],
    open_questions: list[str],
) -> None:
    print(f"bundle: {result_bundle}")
    print(f"goal met: {'yes' if goal_met else 'no'} ({stop})")
    for path in outputs:
        print(f"output: {path}")
    print(f"report: {result_bundle / 'report.md'}")
    for question in open_questions:
        print(f"open: {question}")


async def _write_or_run(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> int:
    from .api import compose_bundle, recorded_roles, write_document

    if args.command == "write":
        roles = resolve_roles(args, parser)
        result = await write_document(
            args.task,
            out=args.out,
            roles=roles,
            genre=args.genre,
            sources=args.sources,
            max_rounds=args.max_rounds,
            research=args.research,
            confirm_own_sources=args.confirm_own_sources,
            provenance=not args.no_provenance,
            formats=args.formats,
            progress=_progress,
        )
    else:
        roles = resolve_roles(args, parser, fallback=recorded_roles(args.bundle))
        result = await compose_bundle(
            args.bundle,
            roles=roles,
            max_rounds=args.max_rounds,
            confirm_own_sources=args.confirm_own_sources,
            provenance=not args.no_provenance,
            formats=args.formats,
            progress=_progress,
        )
    _print_result(
        result.bundle,
        goal_met=result.goal_met,
        stop=result.stop_reason,
        outputs=result.outputs,
        open_questions=result.open_questions,
    )
    return EXIT_OK if result.goal_met else EXIT_NOT_MET


async def _check(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from .api import check_bundle, recorded_roles

    roles = resolve_roles(args, parser, fallback=recorded_roles(args.bundle))
    result = await check_bundle(args.bundle, roles=roles, progress=_progress)
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

    roles = resolve_roles(args, parser)
    genres = None if not args.genres or "all" in args.genres else args.genres
    report = await run_eval(
        genres=genres,
        roles=roles,
        out=args.out,
        max_rounds=args.max_rounds,
        progress=_progress,
    )
    print(report.markdown())
    print(f"written: {args.out / 'eval.json'} and {args.out / 'eval.md'}")
    return EXIT_OK


def _bundle_command(args: argparse.Namespace) -> int:
    """The no-model bundle verbs."""
    if args.command == "new":
        bundle = Bundle.create(args.bundle, title=args.title, scaffold=args.scaffold)
        print(bundle.path)
        return EXIT_OK

    bundle = Bundle.open(args.bundle)

    if args.command == "sections":
        for s in bundle.sections():
            words = len(s.body.split())
            print(
                f"{s.id}  r{s.revision:<3d} {s.status:10s} {words:6d} words  {s.title}"
            )
        return EXIT_OK

    if args.command == "show":
        s = bundle.section(args.section_id)
        print(
            f"id: {s.id}\ntitle: {s.title}\norder: {s.order}\n"
            f"revision: {s.revision}\nstatus: {s.status}\n"
        )
        print(s.body)
        return EXIT_OK

    if args.command == "edit":
        body = (
            sys.stdin.read()
            if args.source == "-"
            else Path(args.source).read_text(encoding="utf-8")
        )
        s = bundle.write_section(
            args.section_id, body=body, expected_revision=args.revision
        )
        print(f"{s.id} now at revision {s.revision}")
        return EXIT_OK

    if args.command == "add":
        s = bundle.add_section(title=args.title, after=args.after)
        print(s.id)
        return EXIT_OK

    if args.command == "validate":
        issues = bundle.validate()
        if not issues:
            print("ok: no validation issues")
            return EXIT_OK
        for issue in issues:
            print(f"[{issue.severity:7s}] {issue.location}: {issue.message}")
        return 1 if any(i.severity == "error" for i in issues) else EXIT_OK

    raise AssertionError(args.command)  # pragma: no cover


def _render_raw(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if not args.formats or len(args.formats) != 1 or args.out is None:
        parser.error("render --raw needs exactly one --format and --out FILE")
    print(Bundle.open(args.bundle).render(args.out, format=args.formats[0]))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # secrets only (API keys); configuration comes from flags
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    try:
        if getattr(args, "verb", None) is not None:
            return dispatch_verb(args)
        if args.command in ("write", "run"):
            return asyncio.run(_write_or_run(args, parser))
        if args.command == "check-bundle":
            return asyncio.run(_check(args, parser))
        if args.command == "eval":
            return asyncio.run(_eval(args, parser))
        if args.command == "status":
            from .api import bundle_status

            print(bundle_status(args.bundle))
            return EXIT_OK
        if args.command in BUNDLE_COMMANDS or (args.command == "render" and args.raw):
            try:
                if args.command == "render":
                    return _render_raw(args, parser)
                return _bundle_command(args)
            except BundleError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return EXIT_USAGE
        if args.command == "render":
            from .api import recorded_roles, render_bundle

            model = args.model
            if not model and not args.no_provenance:
                recorded = recorded_roles(args.bundle)
                model = recorded.drafter if recorded else None
            if not model and not args.no_provenance:
                parser.error("render needs --model (no recorded model in the bundle)")
            for path in render_bundle(
                args.bundle,
                model=model or "",
                formats=args.formats,
                provenance=not args.no_provenance,
            ):
                print(path)
            return EXIT_OK
        if args.command == "genres":
            from .genres import GENRES

            for name, genre in GENRES.items():
                lo, hi = genre.word_range
                print(f"{name:<10} {lo}-{hi} words  {genre.description}")
            return EXIT_OK
    except Exception as exc:  # a crash is reported loudly with its traceback, exit 3
        traceback.print_exc()
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CRASH
    parser.error(f"unknown command {args.command!r}")
    return EXIT_USAGE  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
