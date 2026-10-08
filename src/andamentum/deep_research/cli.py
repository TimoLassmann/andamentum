"""``andamentum-research``: research a question over the web and scholarly sources.

    andamentum-research [ask] "QUESTION" --mode {web,scholarly,both} --model M
        [-o DIR] [--json] [--store PATH] [--cache PATH]
        [--depth {quick,standard,deep}] [--searxng-url URL | --browser]
        [--mailto EMAIL] [--local-corpus PATH] [--tdm-host HOST ...] [--quiet]
    andamentum-research mcp [--store PATH] [--model M] [--searxng-url URL | --browser]
        [--mailto EMAIL]
    andamentum-research cache info --cache PATH
    andamentum-research cache prune --cache PATH [--max-bytes N]

``ask`` is the default: when the first argument is not a command or a flag it
is taken to be the question. ``--mode web`` and ``--mode both`` need a search
address, ``--searxng-url URL`` or ``--browser``. ``--depth`` picks one of three
named budgets (``Budget.for_depth``); ``standard``, the default, is ``Budget()``.
The scholarly sources are OpenAlex, Europe PMC and arXiv; ``--local-corpus PATH``
adds your own documents to them. Every other setting is a keyword argument of
the Python function ``research()``.

Every model and address is a flag; nothing has a hidden default. Without
``--store`` nothing persists. The contact address for the OpenAlex and Crossref
polite pools is ``--mailto``. Secrets are read here only, from the environment
after ``load_dotenv()``: ``OPENALEX_API_KEY``, ``SEMANTIC_SCHOLAR_API_KEY`` and,
as a fallback when ``--mailto`` is not given, ``OPENALEX_EMAIL``.

Output of ``ask``: without ``-o`` the report (Markdown, or the whole result as
JSON with ``--json``) goes to stdout and no file is written. With ``-o DIR`` the
folder gets ``report.md`` (``report.json`` with ``--json``) and the run log as
``log.md`` and ``log.jsonl`` (what happened at each step); a failed run writes
the log so far. Nothing is written anywhere else.

Exit status: 0 the run finished clean; 1 it finished degraded (a source or model
failed; the result is still written and the reasons are on stderr); 2 usage
error; 3 an input could not be read (``--local-corpus`` is missing); 4 the run
failed (model, network or crash).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, get_args

from dotenv import load_dotenv

from .engine import research
from .runlog import write_log
from .sources import SourceConfig
from .store import ResearchStore
from .types import Budget, Depth, Mode, ResearchResult, RunLog

EXIT_OK = 0
EXIT_DEGRADED = 1
EXIT_USAGE = 2
EXIT_INPUT = 3
EXIT_FAILED = 4

ScholarlyName = Literal["openalex", "europepmc", "arxiv", "local_corpus"]
_DEFAULT_SOURCES: tuple[ScholarlyName, ...] = ("openalex", "europepmc", "arxiv")

_COMMANDS = ("ask", "mcp", "cache")

_EPILOG = """\
web search needs an address:
  --mode web and --mode both need --searxng-url URL or --browser.

exit status:
  0  the run finished clean
  1  the run finished degraded (a source or model failed); the result is still written
  2  usage error: a wrong or missing argument
  3  an input could not be read (for example --local-corpus does not exist)
  4  the run failed (model or network error, or a crash)
"""


def _version() -> str:
    from andamentum import __version__

    return f"andamentum-research {__version__}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="andamentum-research",
        description="Research a question over the web and scholarly sources; "
        "the report cites verified findings by id. `ask` is the default command: "
        "andamentum-research QUESTION --mode M --model M. Modes web and both "
        "need --searxng-url URL or --browser.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=_version())
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    _add_ask(sub)
    _add_mcp(sub)
    _add_cache(sub)
    return parser


def _add_ask(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    ask = sub.add_parser(
        "ask",
        help="research a question and write the report",
        description="Research a question and write a report that cites verified "
        "findings by id. Without -o the report goes to stdout and no file is "
        "written; with -o DIR the folder gets report.md (or report.json), log.md "
        "and log.jsonl.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ask.add_argument("question", help="the research question; - reads it from stdin")
    ask.add_argument(
        "--mode",
        required=True,
        choices=get_args(Mode),
        help="where to search: web (needs --searxng-url or --browser), scholarly "
        "(OpenAlex, Europe PMC, arXiv) or both (also needs --searxng-url or "
        "--browser)",
    )
    ask.add_argument(
        "--model",
        required=True,
        help="the language model, a pydantic-ai id (e.g. ollama:gemma4:31b-cloud)",
    )
    ask.add_argument(
        "-o",
        "--out",
        default=None,
        metavar="DIR",
        help="write the report and the run log into this folder (report.md or "
        "report.json, log.md, log.jsonl); without it the report goes to stdout "
        "and nothing is written",
    )
    ask.add_argument(
        "--json",
        action="store_true",
        help="give the whole result as JSON instead of the Markdown report",
    )
    ask.add_argument(
        "--depth",
        choices=get_args(Depth),
        default="standard",
        help="how hard to search (default: %(default)s). standard: 3 rounds per "
        "sub-question, 3 queries and 5 pages read a round, at most 200 model "
        "calls and 1 hour. quick: 1 round, 2 queries and 3 pages, at most 60 "
        "calls and 15 minutes. deep: 5 rounds, 4 queries and 8 pages, a wider "
        "page window, at most 500 calls and 2 hours. A spent limit ends the "
        "search and the report is still written. The quick and deep values are "
        "starting points, not measured settings",
    )
    ask.add_argument(
        "--store",
        default=None,
        metavar="PATH",
        help="keep the run in this research store (a SQLite file; page copies go "
        "beside it); without it nothing persists",
    )
    ask.add_argument(
        "--cache",
        default=None,
        metavar="PATH",
        help="a shared HTTP cache (a SQLite file; page copies go beside it) that "
        "every source reads first and writes into, kept apart from --store",
    )
    ask.add_argument(
        "--searxng-url",
        default=None,
        help="address of the SearxNG instance used for web search; --mode web "
        "and --mode both need this or --browser",
    )
    ask.add_argument(
        "--browser",
        action="store_true",
        help="search the web and read pages through an isolated Chrome in a podman "
        "container the run starts and removes, instead of SearxNG (needs podman; "
        "the image is built on first use); --mode web and --mode both need "
        "this or --searxng-url",
    )
    ask.add_argument(
        "--mailto",
        default=None,
        metavar="EMAIL",
        help="your contact address, sent to OpenAlex and Crossref with each "
        "request (their faster 'polite pool'); optional",
    )
    ask.add_argument(
        "--local-corpus",
        type=Path,
        default=None,
        metavar="PATH",
        help="a folder of your own documents (corpus.jsonl and its index), "
        "searched together with the scholarly sources",
    )
    ask.add_argument(
        "--tdm-host",
        action="append",
        default=[],
        metavar="HOST",
        help="a publisher host whose full text you are licensed to mine; may be repeated",
    )
    ask.add_argument("--quiet", action="store_true", help="no progress lines on stderr")


def _add_mcp(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    mcp = sub.add_parser(
        "mcp",
        help="serve the engine and its store to an agent over MCP (stdio)",
        description="Serve the research engine and its store over MCP (stdio); "
        "example: claude mcp add research -- andamentum-research mcp --store research.db",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mcp.add_argument(
        "--store",
        default=None,
        metavar="PATH",
        help="the research store (a SQLite file; page copies go beside it); "
        "without it the store lives in memory and nothing persists",
    )
    mcp.add_argument(
        "--model",
        default=None,
        help="the language model for reading and for the research tool",
    )
    mcp.add_argument(
        "--searxng-url",
        default=None,
        help="address of the SearxNG instance used by the web search tool",
    )
    mcp.add_argument(
        "--browser",
        action="store_true",
        help="search the web (and read pages a plain fetch cannot) through an "
        "isolated Chrome in a podman container the server starts and removes, "
        "instead of SearxNG",
    )
    mcp.add_argument(
        "--mailto",
        default=None,
        metavar="EMAIL",
        help="your contact address, sent to OpenAlex and Crossref with each "
        "request (their faster 'polite pool'); optional",
    )


def _add_cache(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    cache = sub.add_parser(
        "cache",
        help="look at, or trim, an HTTP cache file",
        description="Look at, or trim, an HTTP cache (the --cache file of ask).",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    jobs = cache.add_subparsers(dest="cache_command", metavar="JOB", required=True)
    info = jobs.add_parser(
        "info", help="size, rows per source, oldest and newest entry"
    )
    info.add_argument(
        "--cache", required=True, metavar="PATH", help="the HTTP cache file"
    )
    prune = jobs.add_parser(
        "prune", help="evict expired entries, then the least recently used"
    )
    prune.add_argument(
        "--cache", required=True, metavar="PATH", help="the HTTP cache file"
    )
    prune.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        metavar="N",
        help="the size to bring the cache under (default: the store's cap, 200 MB)",
    )


def _source_config(args: argparse.Namespace) -> SourceConfig:
    scholarly = _DEFAULT_SOURCES
    if args.local_corpus is not None:
        scholarly = (*scholarly, "local_corpus")
    return SourceConfig(
        mailto=args.mailto or os.environ.get("OPENALEX_EMAIL") or None,
        openalex_api_key=os.environ.get("OPENALEX_API_KEY") or None,
        semantic_scholar_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None,
        searxng_url=args.searxng_url,
        browser=args.browser,
        tdm_allowed_hosts=frozenset(args.tdm_host),
        scholarly=scholarly,
        local_corpus=args.local_corpus,
    )


def _render(result: ResearchResult, *, as_json: bool) -> str:
    if as_json:
        return result.model_dump_json(indent=2) + "\n"
    return result.report_markdown


async def _run(args: argparse.Namespace) -> ResearchResult:
    budget = Budget.for_depth(args.depth)

    def progress(line: str) -> None:
        if not args.quiet:
            print(line, file=sys.stderr, flush=True)

    cache = ResearchStore(Path(args.cache)) if args.cache else None
    try:
        return await research(
            args.question,
            mode=args.mode,
            model=args.model,
            store=Path(args.store) if args.store else None,
            budget=budget,
            sources=_source_config(args),
            http_cache=cache,
            progress=progress,
        )
    finally:
        if cache is not None:
            cache.close()


def _write_log(out: str | None, log: RunLog) -> None:
    """The run log as ``log.md`` and ``log.jsonl`` in *out*; nothing without it."""
    if not out:
        return
    md, jsonl = write_log(log, md_path=Path(out) / "log.md")
    print(f"log: {md} (and {jsonl.name})", file=sys.stderr)


def _error(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


def _ask(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.question == "-":
        args.question = sys.stdin.read().strip()
    if not args.question:
        parser.error("the question is empty")
    if args.mode in ("web", "both") and not (args.searxng_url or args.browser):
        parser.error(f"--mode {args.mode} needs --searxng-url or --browser")
    if args.local_corpus is not None and args.mode == "web":
        parser.error(
            "--local-corpus is searched with the scholarly sources; use --mode scholarly or both"
        )
    if args.out and Path(args.out).is_file():
        parser.error(f"--out must be a folder, but {args.out} is a file")
    if args.local_corpus is not None and not args.local_corpus.exists():
        _error(f"--local-corpus {args.local_corpus} does not exist")
        return EXIT_INPUT
    load_dotenv()  # secrets only (API keys, contact mail); configuration is flags
    try:
        result = asyncio.run(_run(args))
    except KeyboardInterrupt as stop:
        partial = getattr(stop, "run_log", None)
        if isinstance(partial, RunLog):
            _write_log(args.out, partial)
        _error("research cancelled")
        return EXIT_FAILED
    except Exception as failure:  # reported, then a failing exit status
        partial = getattr(failure, "run_log", None)
        if isinstance(partial, RunLog):
            _write_log(args.out, partial)
        _error(f"{type(failure).__name__}: {failure}")
        return EXIT_FAILED
    text = _render(result, as_json=args.json)
    if args.out:
        folder = Path(args.out)
        folder.mkdir(parents=True, exist_ok=True)
        report = folder / ("report.json" if args.json else "report.md")
        report.write_text(text, encoding="utf-8")
        print(f"report: {report}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    print(
        f"stop: {result.stop_reason}; {len(result.findings)} findings from "
        f"{len(result.works)} works; {len(result.verification)} verification flags",
        file=sys.stderr,
    )
    _write_log(args.out, result.log)
    if result.notes:
        print(f"notes ({len(result.notes)}):", file=sys.stderr)
        for note in result.notes:
            print(f"  - {note}", file=sys.stderr)
    if result.degraded:
        print(f"degraded ({len(result.degraded)}):", file=sys.stderr)
        for reason in result.degraded:
            print(f"  - {reason}", file=sys.stderr)
        return EXIT_DEGRADED
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command named by *argv* (default: the process arguments); return
    the exit status."""
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] not in _COMMANDS and not raw[0].startswith("-"):
        raw = ["ask", *raw]
    args = parser.parse_args(raw)
    if args.command is None:
        parser.print_help(sys.stderr)
        return EXIT_USAGE
    if args.command == "ask":
        return _ask(parser, args)
    if args.command == "mcp":
        from .mcp_server import run as run_mcp

        return run_mcp(args)
    from .cache_cli import run as run_cache

    return run_cache(args)
