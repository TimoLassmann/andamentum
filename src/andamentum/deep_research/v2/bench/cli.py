"""The benchmark runner: ``python -m andamentum.deep_research.v2.bench``.

    download litsearch --dir DIR [--n 30] [--seed 0]
    download synergy   --dir DIR [--reviews Donners_2021 Sep_2021 ...]
    verify-web-set
    web       --out-dir DIR --searxng-url URL --model M [--decision-model D] [--arms legacy v2] ...
    litsearch --out-dir DIR --dir DIR --model M [--arms legacy v2] [--n 30] ...
    screening --out-dir DIR --dir DIR --model M --decision-model D [--arms ...] [--max-records N]
    report FILE.json ... [--out FILE.md]

Every directory is named on the command line; nothing is read from or written to
a default location. Runs are sequential, one local-model call at a time. A run
writes ``<task>-<arm>.json`` into ``--out-dir`` after every item. API keys and
the OpenAlex contact address come from the environment (read here, after
``load_dotenv``, never in the library): ``SEMANTIC_SCHOLAR_API_KEY``,
``OPENALEX_API_KEY``, ``OPENALEX_EMAIL``. The exit status is 2 when any item
of any run failed (the failure is in the results file), else 0.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from collections.abc import Callable, Coroutine, Sequence
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Literal

import httpx
from dotenv import load_dotenv

from andamentum.core.decisions import DecisionModel
from andamentum.references import ReferenceHttp

from ..sources.base import SourceConfig
from ..sources.extraction import PDF_EXTRACTIONS, PdfExtraction
from ..types import Mode
from .arms import AnswerArm, LegacyArm, V2Arm
from .download import DatasetMissing, Downloader, write_atomic
from .experiment import Experiment, load_experiment
from .experiment_report import report_from_registry
from .experiment_run import DataDirs, InfrastructureDown, Services, run_experiment
from .concurrency import MAX_CLOUD_CONCURRENCY
from .freeze import PREFETCH_TOP_K, SnapshotMissing, freeze_set, rewarm_set
from .health import SearxDown, require_searxng
from .judging import ClaimJudge, UrlChecker
from .litsearch import DEFAULT_SEED, DEFAULT_SUBSET, download_litsearch, load_litsearch
from .registry import RegistryMismatch
from .results import (
    LitSearchRun,
    RunConfig,
    ScreeningRun,
    WebRun,
    dump_run,
    load_run,
    render_markdown,
)
from .runner import run_litsearch, run_screening, run_web
from .screening import ScreeningArm, parse_arm
from .synergy import DEFAULT_REVIEWS, download_synergy, load_review
from .web_set import WEB_QUESTIONS, fetch_page, verify_sources

AnswerArmName = Literal["legacy", "v2"]
ANSWER_ARMS: tuple[AnswerArmName, ...] = ("legacy", "v2")
SCREENING_ARMS = (
    "baseline+decision",
    "baseline+llm",
    "generic+decision",
    "generic+llm",
    "criteria+decision",
    "criteria+llm",
)
DEFAULT_SEARXNG_URL = "http://127.0.0.1:4070"
VERIFY_PAUSE_S = 1.0
EXIT_ITEM_FAILED = 2
# The bench reads PDFs from their text layer (fast); the library default is docling.
BENCH_PDF_EXTRACTION: PdfExtraction = "fast"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m andamentum.deep_research.v2.bench",
        description="Benchmark harness for deep_research v2 (spec section 9).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="fetch a dataset into a directory")
    sets = download.add_subparsers(dest="dataset", required=True)
    lit = sets.add_parser("litsearch", help="LitSearch queries and gold papers")
    lit.add_argument("--dir", type=Path, required=True)
    lit.add_argument("--n", type=int, default=DEFAULT_SUBSET)
    lit.add_argument("--seed", type=int, default=DEFAULT_SEED)
    syn = sets.add_parser("synergy", help="SYNERGY systematic-review screening sets")
    syn.add_argument("--dir", type=Path, required=True)
    syn.add_argument("--reviews", nargs="+", default=list(DEFAULT_REVIEWS))

    commands.add_parser(
        "verify-web-set",
        help="fetch each web question's source page and check it still states the answer",
    )

    def models(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--out-dir", type=Path, required=True)
        sub.add_argument("--model", help="the LLM, e.g. ollama:gemma4:31b-cloud")
        sub.add_argument(
            "--decision-model", help="a local decision model, e.g. nimble:9b"
        )
        sub.add_argument("--seed", type=int, default=DEFAULT_SEED)

    def judging(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--judge-model", help="LLM that judges claim support (not the engine's own)"
        )
        sub.add_argument(
            "--judge-decision-model", help="decision model that judges claim support"
        )
        sub.add_argument(
            "--check-links",
            action="store_true",
            help="probe each answer's links over the network (URL integrity)",
        )

    def concurrency_flag(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--concurrency",
            type=int,
            default=1,
            help=f"items processed at once, 1 to {MAX_CLOUD_CONCURRENCY}; more than 1 "
            "only when every language model is an Ollama '-cloud' model (a local "
            "model answers one request at a time). Decision-model requests stay "
            "one at a time either way",
        )

    def pdf_flag(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--pdf-extraction",
            choices=PDF_EXTRACTIONS,
            default=BENCH_PDF_EXTRACTION,
            help="how a fetched PDF becomes text: 'fast' (the PDF's text layer, "
            "no OCR; the bench default) or 'docling' (layout-aware, slow)",
        )

    web = commands.add_parser("web", help="the synthetic web question set")
    models(web)
    judging(web)
    web.add_argument(
        "--arms", nargs="+", choices=ANSWER_ARMS, default=list(ANSWER_ARMS)
    )
    web.add_argument("--searxng-url", default=DEFAULT_SEARXNG_URL)
    web.add_argument(
        "--limit", type=int, default=None, help="only the first N questions"
    )

    litsearch = commands.add_parser("litsearch", help="LitSearch scholarly retrieval")
    models(litsearch)
    litsearch.add_argument(
        "--dir", type=Path, required=True, help="the download directory"
    )
    litsearch.add_argument("--n", type=int, default=DEFAULT_SUBSET)
    litsearch.add_argument(
        "--arms", nargs="+", choices=ANSWER_ARMS, default=list(ANSWER_ARMS)
    )
    litsearch.add_argument("--mode", choices=("scholarly", "both"), default="scholarly")
    litsearch.add_argument("--searxng-url", default=DEFAULT_SEARXNG_URL)

    screening = commands.add_parser("screening", help="systematic-review screening")
    models(screening)
    screening.add_argument("--dir", type=Path, required=True)
    screening.add_argument("--reviews", nargs="+", default=list(DEFAULT_REVIEWS))
    screening.add_argument("--arms", nargs="+", choices=SCREENING_ARMS, default=None)
    screening.add_argument("--max-records", type=int, default=None)

    report = commands.add_parser("report", help="markdown tables from results files")
    report.add_argument("files", nargs="+", type=Path)
    report.add_argument("--out", type=Path, default=None)

    freeze = commands.add_parser(
        "freeze",
        help="run Plan and Search once per item and keep the result as a snapshot",
    )
    freeze.add_argument("set", choices=("litsearch", "web"))
    freeze.add_argument("--snapshots", type=Path, required=True)
    freeze.add_argument("--dir", type=Path, default=None, help="litsearch download dir")
    freeze.add_argument("--model", required=True)
    freeze.add_argument("--decision-model", default=None)
    freeze.add_argument("--n", type=int, default=None, help="subset size")
    freeze.add_argument("--seed", type=int, default=DEFAULT_SEED)
    freeze.add_argument("--searxng-url", default=DEFAULT_SEARXNG_URL)
    freeze.add_argument(
        "--prefetch-top-k",
        type=int,
        default=PREFETCH_TOP_K,
        help="fetch full text only for the best N candidates of each sub-question "
        f"by source search rank (default {PREFETCH_TOP_K}); the rest stay "
        "abstract-level, for every arm alike. Recorded in meta.json",
    )
    freeze.add_argument(
        "--rewarm",
        action="store_true",
        help="do not freeze: complete the snapshots that exist (citation hops and "
        "texts that failed, e.g. rate limited) into their own stores; frozen.json "
        "and its digest stay as they are",
    )
    concurrency_flag(freeze)
    pdf_flag(freeze)

    experiment = commands.add_parser("experiment", help="pre-registered comparisons")
    actions = experiment.add_subparsers(dest="experiment_command", required=True)
    run = actions.add_parser("run", help="run every arm of an experiment file")
    run.add_argument("file", type=Path)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--snapshots", type=Path, default=None)
    run.add_argument("--litsearch-dir", type=Path, default=None)
    run.add_argument("--synergy-dir", type=Path, default=None)
    run.add_argument("--searxng-url", default=DEFAULT_SEARXNG_URL)
    run.add_argument(
        "--only-sets",
        nargs="+",
        default=None,
        help="run only these sets of the file (the rest wait for a later run of "
        "the same command; the file and its registry are unchanged)",
    )
    run.add_argument(
        "--check-links",
        action="store_true",
        help="probe each answer's links over the network (URL integrity)",
    )
    concurrency_flag(run)
    pdf_flag(run)
    summary = actions.add_parser("report", help="the experiment's markdown report")
    summary.add_argument("file", type=Path)
    summary.add_argument("--out-dir", type=Path, required=True)
    summary.add_argument("--out", type=Path, default=None)
    return parser


def _source_config(
    searxng_url: str | None, pdf_extraction: PdfExtraction = BENCH_PDF_EXTRACTION
) -> SourceConfig:
    return SourceConfig(
        searxng_url=searxng_url,
        pdf_extraction=pdf_extraction,
        mailto=os.environ.get("OPENALEX_EMAIL"),
        openalex_api_key=os.environ.get("OPENALEX_API_KEY"),
        semantic_scholar_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
    )


def _writer(path: Path) -> Callable[[WebRun | LitSearchRun | ScreeningRun], None]:
    def write(run: WebRun | LitSearchRun | ScreeningRun) -> None:
        write_atomic(path, dump_run(run).encode("utf-8"))

    return write


def _failed(runs: Sequence[WebRun | LitSearchRun | ScreeningRun]) -> bool:
    return any(run.summary.n_errors for run in runs)


def _finish(runs: Sequence[WebRun | LitSearchRun | ScreeningRun], out_dir: Path) -> int:
    for run in runs:
        print(f"wrote {out_dir / f'{run.task}-{run.config.arm}.json'}", file=sys.stderr)
    print(render_markdown(runs))
    if _failed(runs):
        print("some items failed; see the errors in the results files", file=sys.stderr)
        return EXIT_ITEM_FAILED
    return 0


async def _open_decision(
    stack: AsyncExitStack, name: str | None
) -> DecisionModel | None:
    if not name:
        return None
    handle = await DecisionModel.open(name)
    stack.push_async_callback(handle.aclose)
    return handle


async def _judge(args: argparse.Namespace, stack: AsyncExitStack) -> ClaimJudge | None:
    if args.judge_model and args.judge_decision_model:
        raise SystemExit("give --judge-model or --judge-decision-model, not both")
    if not (args.judge_model or args.judge_decision_model):
        return None
    handle = await _open_decision(stack, args.judge_decision_model)
    return ClaimJudge(model=args.judge_model, decision_model=handle)


async def _url_checker(
    args: argparse.Namespace, stack: AsyncExitStack
) -> UrlChecker | None:
    if not args.check_links:
        return None
    http = ReferenceHttp(mailto=os.environ.get("OPENALEX_EMAIL"))
    stack.push_async_callback(http.aclose)
    return http.check_url


def _judge_label(args: argparse.Namespace) -> str | None:
    return args.judge_decision_model or args.judge_model


async def _answer_arms(
    args: argparse.Namespace, stack: AsyncExitStack, *, mode: Mode
) -> list[AnswerArm]:
    if not args.model:
        raise SystemExit("--model is required for the legacy and v2 arms")
    arms: list[AnswerArm] = []
    decision = await _open_decision(stack, args.decision_model)
    for name in args.arms:
        if name == "legacy":
            arms.append(
                LegacyArm(
                    model=args.model,
                    searxng_url=args.searxng_url,
                    decision_model=decision,
                )
            )
        else:
            arms.append(
                V2Arm(
                    mode=mode,
                    model=args.model,
                    decision_model=decision,
                    sources=_source_config(args.searxng_url),
                )
            )
    return arms


async def _cmd_web(args: argparse.Namespace) -> int:
    questions = WEB_QUESTIONS[: args.limit] if args.limit else WEB_QUESTIONS
    runs: list[WebRun] = []
    async with AsyncExitStack() as stack:
        judge = await _judge(args, stack)
        checker = await _url_checker(args, stack)
        for arm in await _answer_arms(args, stack, mode="web"):
            config = RunConfig(
                arm=arm.name,
                model=args.model,
                decision_model=args.decision_model,
                judge=_judge_label(args),
                subset_size=len(questions),
                seed=args.seed,
                started_at=time.time(),
                notes=[f"web questions W01-W{len(questions):02d}"],
            )
            runs.append(
                await run_web(
                    questions,
                    arm,
                    config=config,
                    judge=judge,
                    url_checker=checker,
                    checkpoint=_writer(args.out_dir / f"web-{arm.name}.json"),
                )
            )
    return _finish(runs, args.out_dir)


async def _cmd_litsearch(args: argparse.Namespace) -> int:
    data = load_litsearch(args.dir, n=args.n, seed=args.seed)
    runs: list[LitSearchRun] = []
    async with AsyncExitStack() as stack:
        for arm in await _answer_arms(args, stack, mode=args.mode):
            config = RunConfig(
                arm=arm.name,
                model=args.model,
                decision_model=args.decision_model,
                subset_size=len(data.items),
                seed=args.seed,
                started_at=time.time(),
                notes=[
                    f"{data.n_subset} of {data.n_queries_total} queries chosen, "
                    f"{data.n_dropped_queries} dropped for unresolved gold papers"
                ],
            )
            runs.append(
                await run_litsearch(
                    data.items,
                    arm,
                    config=config,
                    checkpoint=_writer(args.out_dir / f"litsearch-{arm.name}.json"),
                )
            )
    return _finish(runs, args.out_dir)


def _screening_arm_names(args: argparse.Namespace) -> list[str]:
    if args.arms:
        return list(args.arms)
    names = []
    for name in SCREENING_ARMS:
        kind, scorer = parse_arm(name)
        if scorer == "decision" and not args.decision_model:
            continue
        if (scorer == "llm" or kind == "criteria") and not args.model:
            continue
        names.append(name)
    if not names:
        raise SystemExit(
            "give --model and/or --decision-model: no screening arm can run"
        )
    return names


async def _cmd_screening(args: argparse.Namespace) -> int:
    reviews = [load_review(args.dir, key, seed=args.seed) for key in args.reviews]
    async with AsyncExitStack() as stack:
        handle = await _open_decision(stack, args.decision_model)
        arms: list[ScreeningArm] = []
        for name in _screening_arm_names(args):
            kind, scorer = parse_arm(name)
            arms.append(
                ScreeningArm(
                    kind=kind, scorer=scorer, model=args.model, decision_model=handle
                )
            )
        configs = {
            arm.name: RunConfig(
                arm=arm.name,
                model=args.model,
                decision_model=args.decision_model
                if arm.scorer == "decision"
                else None,
                subset_size=args.max_records or 0,
                seed=args.seed,
                started_at=time.time(),
                notes=[
                    f"reviews {', '.join(args.reviews)}; criteria compiled once per "
                    "review and shared by the criteria arms"
                ],
            )
            for arm in arms
        }
        runs = await run_screening(
            reviews,
            arms,
            configs=configs,
            max_records=args.max_records,
            compile_model=args.model,
        )
    for run in runs:
        _writer(args.out_dir / f"screening-{run.config.arm}.json")(run)
    return _finish(runs, args.out_dir)


async def _cmd_download(args: argparse.Namespace) -> int:
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        downloader = Downloader(client)
        if args.dataset == "litsearch":
            done = await download_litsearch(
                args.dir,
                downloader=downloader,
                n=args.n,
                seed=args.seed,
                s2_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
            )
            print(
                f"LitSearch: {done.n_queries_total} queries, subset of {done.n_subset}, "
                f"{done.n_gold_fetched} gold papers fetched into {args.dir}"
            )
        else:
            written = await download_synergy(
                args.dir, downloader=downloader, reviews=tuple(args.reviews)
            )
            print(
                f"SYNERGY: {len(written)} files for {len(args.reviews)} reviews in {args.dir}"
            )
    print(f"licences and checksums: {args.dir / 'MANIFEST.json'}")
    return 0


async def _cmd_verify(_args: argparse.Namespace) -> int:
    async def fetch(url: str) -> str:
        text = await fetch_page(url)
        await asyncio.sleep(VERIFY_PAUSE_S)
        return text

    checks = await verify_sources(fetch)
    bad = [c for c in checks if not c.states_answer]
    for check in checks:
        status = "ok " if check.states_answer else "BAD"
        print(f"{status} {check.question_id} {check.source_url} {check.error}".rstrip())
    return EXIT_ITEM_FAILED if bad else 0


async def _cmd_report(args: argparse.Namespace) -> int:
    runs = [load_run(path.read_text(encoding="utf-8")) for path in args.files]
    table = render_markdown(runs)
    if args.out is not None:
        write_atomic(args.out, table.encode("utf-8"))
    else:
        print(table)
    return 0


async def _cmd_freeze(args: argparse.Namespace) -> int:
    if args.rewarm:
        rewarmed = await rewarm_set(
            args.set,
            snapshots=args.snapshots,
            data_dir=args.dir,
            model=args.model,
            n=args.n,
            seed=args.seed,
            source_config=_source_config(args.searxng_url, args.pdf_extraction),
            progress=lambda line: print(line, file=sys.stderr),
            prefetch_top_k=args.prefetch_top_k,
            concurrency=args.concurrency,
        )
        for line in rewarmed.lines():
            print(line)
        return EXIT_ITEM_FAILED if rewarmed.failed or rewarmed.incomplete else 0
    async with AsyncExitStack() as stack:
        decision = await _open_decision(stack, args.decision_model)
        frozen = await freeze_set(
            args.set,
            snapshots=args.snapshots,
            data_dir=args.dir,
            model=args.model,
            decision_model=decision,
            decision_label=args.decision_model,
            n=args.n,
            seed=args.seed,
            source_config=_source_config(args.searxng_url, args.pdf_extraction),
            progress=lambda line: print(line, file=sys.stderr),
            prefetch_top_k=args.prefetch_top_k,
            concurrency=args.concurrency,
        )
    for line in frozen.lines():
        print(line)
    return EXIT_ITEM_FAILED if frozen.failed or frozen.incomplete else 0


def _live_web(exp: Experiment, only_sets: Sequence[str] | None = None) -> bool:
    """Does any arm search the web live (the legacy arm, or a live v2 arm on a web set)?"""
    names = {s.name for s in exp.sets if only_sets is None or s.name in only_sets}
    for arm in exp.arms:
        if arm.system == "legacy" and arm.runs_on("web") and "web" in names:
            return True
        if arm.system == "v2" and not arm.frozen:
            if any(arm.runs_on(n) and n != "litsearch" for n in names):
                return True
    return False


async def _cmd_experiment_run(args: argparse.Namespace) -> int:
    exp = load_experiment(args.file)
    if _live_web(exp, args.only_sets):
        health = await require_searxng(args.searxng_url)
        print(health.explain(), file=sys.stderr)
    async with AsyncExitStack() as stack:
        opened: dict[str, DecisionModel] = {}
        for name in sorted({a.decision_model for a in exp.arms if a.decision_model}):
            handle = await _open_decision(stack, name)
            if handle is not None:
                opened[name] = handle
        # the judge is part of the registered file, never a command-line choice
        judge: ClaimJudge | None = None
        judge_label: str | None = None
        judge_digest: str | None = None
        if exp.judge is not None:
            handle = await _open_decision(stack, exp.judge.decision_model)
            judge = ClaimJudge(model=exp.judge.model, decision_model=handle)
            judge_label = exp.judge.label
            judge_digest = handle.digest if handle is not None else None
        checker = await _url_checker(args, stack)
        services = Services(
            decisions=opened,
            source_config=_source_config(args.searxng_url, args.pdf_extraction),
            judge=judge,
            judge_label=judge_label,
            judge_digest=judge_digest,
            url_checker=checker,
            check_searxng=require_searxng,
        )
        written = await run_experiment(
            exp,
            out_dir=args.out_dir,
            dirs=DataDirs(
                snapshots=args.snapshots,
                litsearch=args.litsearch_dir,
                synergy=args.synergy_dir,
            ),
            services=services,
            progress=lambda line: print(line, file=sys.stderr),
            only_sets=args.only_sets,
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


async def _cmd_experiment_report(args: argparse.Namespace) -> int:
    text = report_from_registry(load_experiment(args.file), args.out_dir)
    if args.out is not None:
        write_atomic(args.out, text.encode("utf-8"))
    else:
        print(text)
    return 0


async def _cmd_experiment(args: argparse.Namespace) -> int:
    if args.experiment_command == "run":
        return await _cmd_experiment_run(args)
    return await _cmd_experiment_report(args)


_HANDLERS: dict[str, Callable[[argparse.Namespace], Coroutine[Any, Any, int]]] = {
    "freeze": _cmd_freeze,
    "experiment": _cmd_experiment,
    "download": _cmd_download,
    "verify-web-set": _cmd_verify,
    "web": _cmd_web,
    "litsearch": _cmd_litsearch,
    "screening": _cmd_screening,
    "report": _cmd_report,
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
        SnapshotMissing,
        RegistryMismatch,
        FileNotFoundError,
        ValueError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1
