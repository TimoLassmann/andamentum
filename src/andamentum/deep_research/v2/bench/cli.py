"""The benchmark runner: ``python -m andamentum.deep_research.v2.bench``.

    download litsearch-local --dir DIR [--force]
    experiment run FILE --out-dir DIR [--arm NAME ...] [--only-sets NAME ...] ...
    experiment report FILE --out-dir DIR [--out FILE.md]
    experiment losses FILE --out-dir DIR --simpleqa-dir DIR [--arm NAME]
                           [--only-failed] [--format md|json]

Every directory is named on the command line; nothing is read from or written to
a default location. API keys and the OpenAlex contact address come from the
environment (read here, after ``load_dotenv``, never in the library):
``SEMANTIC_SCHOLAR_API_KEY``, ``OPENALEX_API_KEY``, ``OPENALEX_EMAIL``. A keyless
experiment file (``keyless: true``) reads neither key: the source configuration is
built without them whatever the environment holds. The exit status of
``experiment run`` is 2 when any cell failed (the failure is in the registry).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from ..sources.base import SourceConfig
from ..sources.extraction import PDF_EXTRACTIONS, PdfExtraction
from ..store import ResearchStore
from .attribution import collect_losses
from .concurrency import MAX_CLOUD_CONCURRENCY
from .download import DatasetMissing, write_atomic
from .experiment import Experiment, load_experiment
from .experiment_report import report_from_registry
from .experiment_run import (
    DataDirs,
    InfrastructureDown,
    Services,
    run_experiment,
    set_mode,
)
from .health import SearxDown, SearxHealth, require_browser, require_searxng
from .registry import RegistryMismatch
from .simpleqa import AnswerGrader

DEFAULT_SEARXNG_URL = "http://127.0.0.1:4070"
EXIT_ITEM_FAILED = 2
# The bench reads PDFs from their text layer (fast); the library default is docling.
BENCH_PDF_EXTRACTION: PdfExtraction = "fast"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m andamentum.deep_research.v2.bench",
        description="Benchmark harness for deep_research v2.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="fetch a dataset into a directory")
    sets = download.add_subparsers(dest="dataset", required=True)
    local = sets.add_parser(
        "litsearch-local",
        help="LitSearch queries and its own corpus (titles, abstracts, citation "
        "ids) with a BM25 index; needs pyarrow (the bench-data extra)",
    )
    local.add_argument("--dir", type=Path, required=True)
    local.add_argument(
        "--force",
        action="store_true",
        help="fetch the files again even when the folder holds them",
    )

    experiment = commands.add_parser("experiment", help="pre-registered comparisons")
    actions = experiment.add_subparsers(dest="experiment_command", required=True)

    run = actions.add_parser("run", help="run every arm of an experiment file")
    run.add_argument("file", type=Path)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument(
        "--litsearch-local-dir",
        type=Path,
        default=None,
        help="the litsearch-local download (queries and gold papers)",
    )
    run.add_argument(
        "--simpleqa-dir",
        type=Path,
        default=None,
        help="the folder that holds simple_qa_test_set.csv (the simpleqa set)",
    )
    run.add_argument("--searxng-url", default=DEFAULT_SEARXNG_URL)
    run.add_argument(
        "--browser-cdp-url",
        default=None,
        metavar="URL",
        help="search the web through this Chrome (its debugging address) instead "
        "of SearxNG, for every arm",
    )
    run.add_argument(
        "--only-sets",
        nargs="+",
        default=None,
        help="run only these sets of the file (the rest wait for a later run of "
        "the same command; the file and its registry are unchanged)",
    )
    run.add_argument(
        "--arm",
        action="append",
        default=None,
        metavar="NAME",
        help="run only this arm (repeatable); the order stays the file's, solo "
        "arms last",
    )
    run.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=f"cells processed at once, 1 to {MAX_CLOUD_CONCURRENCY}; more than 1 "
        "only when every language model is an Ollama '-cloud' model (a local "
        "model answers one request at a time)",
    )
    run.add_argument(
        "--cache",
        type=Path,
        default=None,
        metavar="PATH",
        help="a shared HTTP cache (one research store file; blobs go beside "
        "it) that every source reads first and writes into, so no cell "
        "downloads an answer another already holds",
    )
    run.add_argument(
        "--pdf-extraction",
        choices=PDF_EXTRACTIONS,
        default=BENCH_PDF_EXTRACTION,
        help="how a fetched PDF becomes text: 'fast' (the PDF's text layer, "
        "no OCR; the bench default) or 'docling' (layout-aware, slow)",
    )

    summary = actions.add_parser("report", help="the experiment's markdown report")
    summary.add_argument("file", type=Path)
    summary.add_argument("--out-dir", type=Path, required=True)
    summary.add_argument("--out", type=Path, default=None)

    losses = actions.add_parser(
        "losses",
        help="where each lost answer was lost, from the run logs (no model, no network)",
    )
    losses.add_argument("file", type=Path)
    losses.add_argument("--out-dir", type=Path, required=True)
    losses.add_argument(
        "--simpleqa-dir",
        type=Path,
        default=None,
        help="the folder that holds simple_qa_test_set.csv (needed when the "
        "registry holds simpleqa cells)",
    )
    losses.add_argument(
        "--litsearch-local-dir",
        type=Path,
        default=None,
        help="the litsearch-local download (needed when the registry holds "
        "litsearch-web cells)",
    )
    losses.add_argument(
        "--arm",
        action="append",
        default=None,
        metavar="NAME",
        help="only this arm (repeatable); default: the experiment's v2 arms",
    )
    losses.add_argument(
        "--only-failed",
        action="store_true",
        help="LitSearch: leave out the gold works that were found",
    )
    losses.add_argument("--format", choices=("md", "json"), default="md")
    losses.add_argument("--out", type=Path, default=None)
    return parser


def _source_config(
    searxng_url: str | None,
    pdf_extraction: PdfExtraction = BENCH_PDF_EXTRACTION,
    *,
    keyless: bool = False,
    browser_cdp_url: str | None = None,
) -> SourceConfig:
    """The sources' configuration. Keyless: neither API key is looked up at all
    (not read, not checked for), whatever the environment holds."""
    return SourceConfig(
        searxng_url=searxng_url,
        browser_cdp_url=browser_cdp_url,
        pdf_extraction=pdf_extraction,
        mailto=os.environ.get("OPENALEX_EMAIL"),
        openalex_api_key=None if keyless else os.environ.get("OPENALEX_API_KEY"),
        semantic_scholar_api_key=(
            None if keyless else os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        ),
    )


def _download_litsearch_local(args: argparse.Namespace) -> int:
    from .litsearch_local import download_litsearch_local

    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        done = download_litsearch_local(
            args.dir,
            client=client,
            today=date.today(),
            progress=lambda line: print(line, file=sys.stderr, flush=True),
            force=args.force,
        )
    print(
        f"LitSearch (local): {done.n_queries} queries, {done.index.n_records} corpus "
        f"records ({done.index.n_with_abstract} with an abstract), "
        f"{done.index.n_citation_edges} citation edges, {done.n_gold_in_corpus} of "
        f"{done.n_gold_ids} gold papers in the corpus; {done.bytes_fetched / 1e6:.0f} MB "
        f"fetched into {args.dir}"
    )
    print(f"provenance: {args.dir / 'PROVENANCE.md'}")
    return 0


async def _cmd_download(args: argparse.Namespace) -> int:
    return await asyncio.to_thread(_download_litsearch_local, args)


def _open_cache(path: Path | None) -> ResearchStore | None:
    return ResearchStore(path) if path is not None else None


def _live_web(exp: Experiment, only_sets: Sequence[str] | None = None) -> bool:
    """Must the web search answer before anything runs? True when an arm that
    stops on an outage searches the web: the legacy arm (on any set) or a v2 arm
    on a web set. An arm registered with ``outage: record`` probes before each of
    its cells instead and records a blocked one as failed."""
    names = {s.name for s in exp.sets if only_sets is None or s.name in only_sets}
    for arm in exp.arms:
        if arm.outage == "record":
            continue
        if arm.system == "legacy" and any(arm.runs_on(n) for n in names):
            return True
        if arm.system == "v2" and any(
            arm.runs_on(n) and set_mode(n) == "web" for n in names
        ):
            return True
    return False


def _browser_probe(cdp_url: str) -> Callable[[str | None], Awaitable[SearxHealth]]:
    """The per-cell web check when the arms search through a browser."""

    async def probe(_searxng_url: str | None) -> SearxHealth:
        return await require_browser(cdp_url)

    return probe


async def _cmd_experiment_run(args: argparse.Namespace) -> int:
    cache = _open_cache(args.cache)
    try:
        return await _experiment_run(args, cache)
    finally:
        if cache is not None:
            cache.close()


async def _experiment_run(args: argparse.Namespace, cache: ResearchStore | None) -> int:
    exp = load_experiment(args.file)
    if _live_web(exp, args.only_sets):
        if args.browser_cdp_url:
            health = await require_browser(args.browser_cdp_url)
            print(
                f"web search through {health.detail} at {health.url}", file=sys.stderr
            )
        else:
            health = await require_searxng(args.searxng_url)
            print(health.explain(), file=sys.stderr)
    services = Services(
        source_config=_source_config(
            args.searxng_url,
            args.pdf_extraction,
            keyless=exp.keyless,
            browser_cdp_url=args.browser_cdp_url,
        ),
        check_searxng=(
            _browser_probe(args.browser_cdp_url)
            if args.browser_cdp_url
            else require_searxng
        ),
        http_cache=cache,
        keyless=exp.keyless,
        grader=(AnswerGrader(model=exp.grader) if exp.grader is not None else None),
    )
    written = await run_experiment(
        exp,
        out_dir=args.out_dir,
        dirs=DataDirs(
            litsearch_local=args.litsearch_local_dir, simpleqa=args.simpleqa_dir
        ),
        services=services,
        progress=lambda line: print(line, file=sys.stderr),
        only_sets=args.only_sets,
        only_arms=args.arm,
        concurrency=args.concurrency,
    )
    failed = [r for r in written if r.error]
    print(
        f"{len(written)} runs recorded in {args.out_dir / 'experiments.jsonl'}, "
        f"{len(failed)} failed (concurrency {args.concurrency})",
        file=sys.stderr,
    )
    for record in failed:
        print(
            f"  {record.arm} {record.set}/{record.item_id} #{record.repeat}: {record.error}",
            file=sys.stderr,
        )
    return EXIT_ITEM_FAILED if failed else 0


def _emit(text: str, out: Path | None) -> None:
    if out is not None:
        write_atomic(out, text.encode("utf-8"))
    else:
        print(text)


async def _cmd_experiment_report(args: argparse.Namespace) -> int:
    _emit(report_from_registry(load_experiment(args.file), args.out_dir), args.out)
    return 0


async def _cmd_experiment_losses(args: argparse.Namespace) -> int:
    found = collect_losses(
        load_experiment(args.file),
        out_dir=args.out_dir,
        dirs=DataDirs(
            litsearch_local=args.litsearch_local_dir, simpleqa=args.simpleqa_dir
        ),
        arms=args.arm,
    )
    text = (
        found.model_dump_json(indent=2)
        if args.format == "json"
        else found.to_markdown(only_failed=args.only_failed)
    )
    _emit(text, args.out)
    return 0


async def _cmd_experiment(args: argparse.Namespace) -> int:
    if args.experiment_command == "run":
        return await _cmd_experiment_run(args)
    if args.experiment_command == "losses":
        return await _cmd_experiment_losses(args)
    return await _cmd_experiment_report(args)


_HANDLERS: dict[str, Callable[[argparse.Namespace], Coroutine[Any, Any, int]]] = {
    "experiment": _cmd_experiment,
    "download": _cmd_download,
}


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_HANDLERS[args.command](args))
    except (
        DatasetMissing,
        InfrastructureDown,
        SearxDown,
        RegistryMismatch,
        FileNotFoundError,
        ValueError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1
