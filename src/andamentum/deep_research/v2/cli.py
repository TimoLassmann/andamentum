"""``andamentum-research v2``: the evidence-first engine from the command line.

    andamentum-research v2 "QUESTION" --mode both --model M
        [--preset baseline|fast|recall|hybrid|criteria]
        [--decision-model NAME] [--embedding-model NAME] [--store PATH] [--cache PATH] [--searxng-url URL]
        [--sources openalex,europepmc,arxiv] [--local-corpus PATH]
        [--openalex-search keyword]
        [--tdm-host HOST ...] [--results-per-query N]
        [--max-rounds N ... every Budget field] [--json | --md] [--out FILE]
        [--log PATH]

Every model and address is a flag; nothing has a hidden default. ``--preset``
names the run's ``ResearchConfig`` (``config.PRESETS``; ``baseline`` unless
another is named). Without
``--store`` nothing persists. Secrets are read here only, from the environment
after ``load_dotenv()``: ``OPENALEX_API_KEY``, ``OPENALEX_EMAIL`` (the polite
contact address for OpenAlex and Crossref) and ``SEMANTIC_SCHOLAR_API_KEY``.

Every run writes its log (``RunLog``: what happened at each step) as two
files, ``<stem>.log.md`` and ``<stem>.log.jsonl``: beside ``--out FILE`` (the
out file's stem), else in the current directory as
``research-<run id prefix>.log.md``; ``--log PATH`` (a ``.md`` path, the
``.jsonl`` beside it) puts it elsewhere. A failed run writes the log so far.

Exit status: 0 for a clean run, 4 for a run that finished degraded (a source
or model failed; the result is still written and the reasons are on stderr),
1 for an error, 2 for a usage error.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Literal, get_args

from dotenv import load_dotenv

from .config import DEFAULT_PRESET, PRESETS
from .graph import research
from .runlog import write_log
from .sources import SourceConfig
from .store import ResearchStore
from .types import Budget, Mode, ResearchResult, RunLog

EXIT_DEGRADED = 4
ScholarlyName = Literal[
    "openalex", "europepmc", "arxiv", "semantic_scholar", "local_corpus"
]
_SCHOLARLY: tuple[ScholarlyName, ...] = get_args(ScholarlyName)


def _flag(field: str) -> str:
    return "--" + field.replace("_", "-")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="andamentum-research v2",
        description="Research a question over the web and scholarly sources; "
        "the report cites verified findings by id.",
    )
    parser.add_argument("question", help="the research question")
    parser.add_argument(
        "--mode",
        required=True,
        choices=get_args(Mode),
        help="web (SearxNG), scholarly (OpenAlex, Europe PMC, arXiv) or both",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="the language model, a pydantic-ai id (e.g. ollama:gemma4:31b-cloud)",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=DEFAULT_PRESET,
        help="what the run does at every swappable step: baseline (one generic "
        "relevance question per candidate), fast (the baseline with a lexical "
        "pre-filter and code-first claim-support checks), recall (screening "
        "queue by source rank), hybrid (the raw question searched too, dense "
        "similarity in the rank fusion, the screen ranks and drops nothing; "
        "needs --embedding-model) or criteria (LLM-compiled criteria); "
        "default: %(default)s",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        metavar="NAME",
        help="a local Ollama embedding model (e.g. embeddinggemma:latest) for a "
        "preset that ranks by dense similarity (hybrid); other presets ignore it",
    )
    parser.add_argument(
        "--decision-model",
        default=None,
        metavar="NAME",
        help="a local Ollama decision model for screening and checks; without "
        "it the language model answers those questions",
    )
    parser.add_argument(
        "--store",
        default=None,
        metavar="PATH",
        help="the research store (a SQLite file; blobs go beside it). Without "
        "it nothing persists",
    )
    parser.add_argument(
        "--cache",
        default=None,
        metavar="PATH",
        help="a shared HTTP cache (a SQLite file; blobs go beside it) every "
        "source reads first and writes into, kept apart from --store; the "
        "run's own store holds a copy of each entry the run used",
    )
    parser.add_argument(
        "--searxng-url", default=None, help="the SearxNG instance (web and both modes)"
    )
    parser.add_argument(
        "--browser-cdp-url",
        default=None,
        metavar="URL",
        help="search the web through this Chrome instead of SearxNG (its debugging "
        "address, e.g. http://127.0.0.1:9222; start Chrome with "
        "--remote-debugging-port=9222 and its own --user-data-dir)",
    )
    parser.add_argument(
        "--sources",
        type=scholarly_names,
        default=",".join(SourceConfig().scholarly),
        help=f"scholarly sources, comma-separated, from {', '.join(_SCHOLARLY)} "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--local-corpus",
        type=Path,
        default=None,
        metavar="PATH",
        help="the folder of the local_corpus source (corpus.jsonl and its index, "
        "see sources/local_corpus.py); needed when --sources names local_corpus",
    )
    parser.add_argument(
        "--openalex-search",
        choices=("keyword", "semantic", "both"),
        default="keyword",
        help="OpenAlex endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--tdm-host",
        action="append",
        default=[],
        metavar="HOST",
        help="a publisher host whose text you are licensed to mine (repeatable)",
    )
    parser.add_argument(
        "--results-per-query",
        type=int,
        default=SourceConfig().results_per_query,
        help="results asked of each source per query (default: %(default)s)",
    )
    limits = parser.add_argument_group("budget (every bound of the run)")
    for name, info in Budget.model_fields.items():
        kind = float if info.annotation is float else int
        limits.add_argument(
            _flag(name), type=kind, default=info.default, help="default: %(default)s"
        )
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--json", action="store_true", help="print the whole result as JSON"
    )
    output.add_argument(
        "--md", action="store_true", help="print the report as Markdown (default)"
    )
    parser.add_argument(
        "--out", default=None, metavar="FILE", help="write the output here, not stdout"
    )
    parser.add_argument(
        "--log",
        default=None,
        metavar="PATH",
        help="write the run log here (a .md path; the .jsonl goes beside it). "
        "Default: <stem>.log.md beside --out, else research-<run id>.log.md "
        "in the current directory",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="no progress lines on stderr"
    )
    return parser


def scholarly_names(text: str) -> tuple[ScholarlyName, ...]:
    """The scholarly sources named in *text* (comma-separated), in its order."""
    names: list[ScholarlyName] = []
    for raw in text.split(","):
        name = raw.strip()
        known: list[ScholarlyName] = [s for s in _SCHOLARLY if s == name]
        if not known:
            raise argparse.ArgumentTypeError(
                f"unknown source {name!r}; known: {', '.join(_SCHOLARLY)}"
            )
        names += known
    return tuple(dict.fromkeys(names))


def _source_config(args: argparse.Namespace) -> SourceConfig:
    return SourceConfig(
        mailto=os.environ.get("OPENALEX_EMAIL") or None,
        openalex_api_key=os.environ.get("OPENALEX_API_KEY") or None,
        semantic_scholar_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None,
        searxng_url=args.searxng_url,
        browser_cdp_url=args.browser_cdp_url,
        tdm_allowed_hosts=frozenset(args.tdm_host),
        scholarly=args.sources,
        local_corpus=args.local_corpus,
        openalex_search=args.openalex_search,
        results_per_query=args.results_per_query,
    )


def _render(result: ResearchResult, *, as_json: bool) -> str:
    if as_json:
        return result.model_dump_json(indent=2) + "\n"
    return result.report_markdown


async def _run(args: argparse.Namespace) -> ResearchResult:
    budget = Budget(**{name: getattr(args, name) for name in Budget.model_fields})

    def progress(line: str) -> None:
        if not args.quiet:
            print(line, file=sys.stderr, flush=True)

    cache = ResearchStore(Path(args.cache)) if args.cache else None
    try:
        return await research(
            args.question,
            mode=args.mode,
            model=args.model,
            decision_model=args.decision_model,
            embedding_model=args.embedding_model,
            store=Path(args.store) if args.store else None,
            budget=budget,
            sources=_source_config(args),
            config=PRESETS[args.preset],
            http_cache=cache,
            progress=progress,
        )
    finally:
        if cache is not None:
            cache.close()


def log_path(*, log: str | None, out: str | None, run_id: str) -> Path:
    """Where the run log's Markdown file goes: *log* when given, else
    ``<stem>.log.md`` beside *out*, else ``research-<run id[:8]>.log.md`` in
    the current directory."""
    if log:
        return Path(log)
    if out:
        target = Path(out)
        return target.with_name(f"{target.stem}.log.md")
    return Path.cwd() / f"research-{run_id[:8] or 'run'}.log.md"


def _write_log(args: argparse.Namespace, log: RunLog) -> None:
    md, jsonl = write_log(
        log, md_path=log_path(log=args.log, out=args.out, run_id=log.run_id)
    )
    print(f"log: {md} (and {jsonl.name})", file=sys.stderr)


def main(argv: list[str]) -> None:
    """Run one research question; *argv* is what follows ``v2``."""
    load_dotenv()  # secrets only (API keys, contact mail); configuration is flags
    args = build_parser().parse_args(argv)
    try:
        result = asyncio.run(_run(args))
    except KeyboardInterrupt as stop:
        partial = getattr(stop, "run_log", None)
        if isinstance(partial, RunLog):
            _write_log(args, partial)
        print("\nResearch cancelled.", file=sys.stderr)
        sys.exit(1)
    except Exception as error:  # reported, then a failing exit status
        partial = getattr(error, "run_log", None)
        if isinstance(partial, RunLog):
            _write_log(args, partial)
        print(f"Error: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
    text = _render(result, as_json=args.json)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    print(
        f"preset: {result.config_name} ({result.config_hash[:12]}); "
        f"stop: {result.stop_reason}; {len(result.findings)} findings from "
        f"{len(result.works)} works; {len(result.verification)} verification flags",
        file=sys.stderr,
    )
    _write_log(args, result.log)
    if result.notes:
        print(f"notes ({len(result.notes)}):", file=sys.stderr)
        for note in result.notes:
            print(f"  - {note}", file=sys.stderr)
    if result.degraded:
        print(f"degraded ({len(result.degraded)}):", file=sys.stderr)
        for reason in result.degraded:
            print(f"  - {reason}", file=sys.stderr)
        sys.exit(EXIT_DEGRADED)
