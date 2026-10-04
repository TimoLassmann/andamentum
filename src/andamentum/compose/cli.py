"""``andamentum-compose``: a task in, a finished document out. Never sends anything.

Exit codes for ``write`` and ``run``: 0 goal met, 4 goal not met (outputs are
still written), 3 crash. A missing role model or a bad argument exits 2.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

from .schemas import ROLE_NAMES, Roles

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_CRASH = 3
EXIT_NOT_MET = 4


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

    check = sub.add_parser("check", help="run the checks once, without revising")
    check.add_argument("bundle", type=Path)
    _add_models(check)

    render = sub.add_parser("render", help="render a bundle's text to out/ again")
    render.add_argument("bundle", type=Path)
    render.add_argument(
        "--model", help="producing model for provenance (default: recorded)"
    )
    _add_output(render)

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


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # secrets only (API keys); configuration comes from flags
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    try:
        if args.command in ("write", "run"):
            return asyncio.run(_write_or_run(args, parser))
        if args.command == "check":
            return asyncio.run(_check(args, parser))
        if args.command == "eval":
            return asyncio.run(_eval(args, parser))
        if args.command == "status":
            from .api import bundle_status

            print(bundle_status(args.bundle))
            return EXIT_OK
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
