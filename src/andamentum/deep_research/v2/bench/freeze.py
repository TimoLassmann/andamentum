"""Freezing: Plan and Search once per item, kept as a snapshot every arm reuses.

``bench freeze`` runs the engine's own Plan and one round of Search for each
item of a set (``frozen.capture``), then warms the snapshot's cache with the
text of the candidates a first screening round would look at, so that frozen
arms read the same pages without touching the network. One item is one folder,
``<snapshots>/<set>/<item_id>/``, holding ``frozen.json`` (the sub-questions,
queries and merged candidates), ``store.db`` with its ``blobs/`` (the source
responses and page texts) and ``meta.json`` (what the freeze did).

Rules: a snapshot is written only when capture finished with nothing degraded
and found candidates (a half-empty input would be unfair to every arm, and the
item is reported failed instead); a web set is not frozen while SearxNG is not
answering (``health.require_searxng``); a snapshot that exists is kept, so an
interrupted freeze resumes and a second freeze never silently changes inputs.
Within an item, capture and prefetch are strictly sequential; items may run
concurrently (``concurrency``, at most 4, only for a ``-cloud`` model; see
``concurrency.py``), their requests to one source or host still spaced as one
client's would be (``SharedPacing``) and their decision-model requests
serialised by ``core.decisions``' per-server lock.

What is cached (every rule is in ``meta.json``): the search results and the
citation hops of the candidates are cached for every candidate (cheap API
calls), but a candidate's full text is fetched and extracted only for the
``PREFETCH_TOP_K`` best of each sub-question by source search rank (reciprocal-
rank fusion over the sources' lists, ``source_rank_score``). Every other
candidate stays abstract-level: a frozen run that reads one finds no text in the
snapshot (``FROZEN_TEXT_REASON``) and falls back to the abstract, the same for
every arm. PDFs are read from their text layer when ``pdf_extraction`` is
``fast`` (``sources.extraction``); the extractor used per fetched URL is
recorded.

A snapshot ages with its cache: search answers live 7 days, metadata 30, page
texts 180 (``store.TTL_*``); frozen arms need only the page texts.
"""

from __future__ import annotations

import asyncio
import json
import resource
import shutil
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from pydantic import BaseModel
from pydantic_ai.models import Model

from andamentum.core.decisions import DecisionModel

from ..config import PRESETS
from ..frozen import FrozenInputs, capture
from ..sources import (
    SharedPacing,
    SourceClient,
    SourceConfig,
    SourceError,
    build_sources,
)
from ..sources.base import ExpandingSource, TextSource
from ..steps import EXPAND_PER_CALL, is_scholarly
from ..strategies import RRF_K
from ..sources.base import FROZEN_TEXT_REASON
from ..store import TTL_FULL_TEXT_S, ResearchStore, cache_key
from ..types import Candidate, Mode
from .concurrency import model_label, resolve_concurrency, run_bounded
from .health import SearxHealth, require_searxng
from .litsearch import DEFAULT_SUBSET, load_litsearch
from .planted import PlantedSource, plant_candidates
from .subsets import select_subset
from .web_set import WEB_QUESTIONS

FROZEN_FILE = "frozen.json"
STORE_FILE = "store.db"
META_FILE = "meta.json"
# Which candidates get their full text fetched and extracted: the best this many
# of each sub-question by source search rank (the same for every arm, whatever an
# arm will screen or keep). The rest stay abstract-level. The count is an option
# (`--prefetch-top-k`; None: every candidate) and is recorded in meta.json.
PREFETCH_TOP_K: int | None = 20
# Which candidates get their references and citations cached: the scholarly ones
# among the same best-by-source-rank set whose text is fetched (`prefetch_targets`),
# so one arm-independent rule decides both and the API budget is spent where the
# arms will read. A work outside that set that an arm expands from is an offline
# miss for every arm alike. (An earlier rule warmed every scholarly candidate; it
# spent OpenAlex's keyless daily budget within two or three items.)
WARM_HOP_LIMIT = EXPAND_PER_CALL
PLANTED_SOURCE_RANK_KEY = "planted"


class SnapshotMissing(RuntimeError):
    """A frozen arm needs a snapshot that has not been taken."""


def snapshot_dir(root: Path, set_name: str, item_id: str) -> Path:
    return root / set_name / item_id


def snapshot_incomplete(folder: Path) -> bool:
    """Is the snapshot's ``meta.json`` marked incomplete (a rate-limited citation hop)?"""
    path = folder / META_FILE
    if not path.exists():
        return False
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("incomplete"))
    except ValueError:
        return False


def load_frozen(folder: Path) -> FrozenInputs:
    """The snapshot in *folder*; ``SnapshotMissing`` when there is none."""
    path = folder / FROZEN_FILE
    if not path.exists():
        raise SnapshotMissing(f"no snapshot at {folder}: run `bench freeze` first")
    return FrozenInputs.load(path)


# -- planting ---------------------------------------------------------------------


def planted_frozen(base: FrozenInputs, source: PlantedSource) -> FrozenInputs:
    """The snapshot with the false page added as the top result of every sub-question.

    One candidate carries every sub-question id (the page answers the whole
    question) and the best rank in a list of its own, so rank fusion sees it
    first. The snapshot's other candidates are untouched.
    """
    ids = [s.id for s in base.subquestions]
    planted = plant_candidates({i: base.candidates_for(i) for i in ids}, source)
    candidate = next(c for c in planted[ids[0]] if c.sources == ["planted"])
    candidate = candidate.model_copy(
        update={
            "subquestion_ids": ids,
            "source_ranks": {PLANTED_SOURCE_RANK_KEY: candidate.rank or 1},
        }
    )
    return base.model_copy(update={"candidates": [candidate, *base.candidates]})


def planted_text(source: PlantedSource) -> str:
    """The page as the text extractor would hand it to the reader."""
    return f"# {source.title}\n\n{source.statement}\n"


def plant_into_store(store_path: Path, source: PlantedSource) -> None:
    """Put the false page's text where a frozen fetch of its URL finds it."""
    with ResearchStore(store_path) as store:
        store.cache_put(
            cache_key("web:text", source.url),
            source="web",
            url=source.url,
            status=200,
            body=planted_text(source).encode("utf-8"),
            ttl_s=TTL_FULL_TEXT_S,
        )


# -- prefetch ---------------------------------------------------------------------


class PrefetchReport(BaseModel):
    """What warming the cache did: how many texts were fetched, held or missing."""

    attempted: int = 0
    fetched: int = 0
    none_held: int = 0
    failed: int = 0
    errors: list[str] = []


def source_rank_score(candidate: Candidate) -> float:
    """Reciprocal-rank fusion of the candidate's places in the sources' search
    lists: the sum of ``1 / (RRF_K + rank)`` over ``source_ranks``. The search
    rank only: no screening score is involved, so the rule is the same for
    every arm."""
    return sum(1.0 / (RRF_K + rank) for rank in candidate.source_ranks.values())


def prefetch_targets(frozen: FrozenInputs, *, top_k: int | None) -> list[Candidate]:
    """The candidates whose full text is fetched: with ``top_k`` None every
    candidate; otherwise the best ``top_k`` of each sub-question by
    :func:`source_rank_score` (ties keep admission order), each work once. The
    result is in fetch order: each sub-question's best first."""
    chosen: dict[str, Candidate] = {}
    for subquestion in frozen.subquestions:
        ranked = sorted(
            frozen.candidates_for(subquestion.id), key=source_rank_score, reverse=True
        )
        for candidate in ranked if top_k is None else ranked[:top_k]:
            chosen.setdefault(candidate.work.work_id or candidate.work.title, candidate)
    return list(chosen.values())


class CitationWarmReport(BaseModel):
    """What warming the citation hops did."""

    works: int = 0
    hops_cached: int = 0
    hops_failed: int = 0
    rate_limited: int = 0
    errors: list[str] = []


def citation_targets(frozen: FrozenInputs, *, top_k: int | None) -> list[Candidate]:
    """The scholarly works whose hops are cached: those of :func:`prefetch_targets`
    (the same ``top_k``) that are scholarly, in the same order."""
    return [c for c in prefetch_targets(frozen, top_k=top_k) if is_scholarly(c.work)]


async def warm_citations(
    candidates: Sequence[Candidate], *, sources: Sequence[ExpandingSource]
) -> CitationWarmReport:
    """Ask each source for the references and citations of each candidate, so the
    answers are in the store's cache. The calls are the ones ``steps.expand``
    makes (same limit), so a frozen run finds them. They run together; the client
    keeps each source's spacing and in-flight cap, and the report lists the
    failures in candidate order."""
    report = CitationWarmReport()

    async def hop(candidate: Candidate, source: ExpandingSource, direction: str):
        follow = source.references if direction == "references" else source.citations
        try:
            await follow(candidate.work, limit=WARM_HOP_LIMIT)
        except SourceError as error:
            return error
        return None

    jobs: list[tuple[Candidate, ExpandingSource, str]] = [
        (candidate, source, direction)
        for candidate in candidates
        for source in sources
        for direction in ("references", "citations")
    ]
    report.works = len(candidates)
    outcomes = await asyncio.gather(*(hop(*job) for job in jobs))
    for (candidate, source, direction), error in zip(jobs, outcomes, strict=True):
        if error is None:
            report.hops_cached += 1
        elif error.kind == "config":
            continue
        else:
            report.hops_failed += 1
            if error.kind == "rate_limited":
                report.rate_limited += 1
            report.errors.append(
                f"{source.name} {direction} of "
                f"{candidate.work.title or candidate.work.work_id}: {error.reason}"
            )
    return report


async def prefetch_texts(
    candidates: Sequence[Candidate],
    *,
    text_sources: Sequence[TextSource],
) -> PrefetchReport:
    """Fetch each candidate's text from the first source that holds it.

    The candidates are fetched together (each tries its sources in order); the
    client keeps every source's and host's spacing, so the arXiv crawl delay
    still holds. A fetch the gate refuses or a source that fails is counted and
    named, never hidden; it does not stop the freeze (a paywalled page is
    normal). Errors are listed in candidate order.
    """
    report = PrefetchReport()

    async def one(candidate: Candidate) -> str | None:
        outcome: str | None = None
        for source in text_sources:
            try:
                fetched = await source.fetch_text(candidate.work)
            except SourceError as error:
                outcome = f"{candidate.work.url or candidate.work.title}: {error}"
                continue
            if fetched is not None:
                return "fetched"
        return outcome

    report.attempted = len(candidates)
    for outcome in await asyncio.gather(*(one(c) for c in candidates)):
        if outcome == "fetched":
            report.fetched += 1
        elif outcome is None:
            report.none_held += 1
        else:
            report.failed += 1
            report.errors.append(outcome)
    return report


# -- freezing a set ---------------------------------------------------------------


def peak_rss_mb() -> float:
    """The process's peak resident memory in MB (``ru_maxrss`` is bytes on macOS,
    kilobytes elsewhere)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


@dataclass
class FrozenItem:
    item_id: str
    status: str  # "frozen" | "kept" | "incomplete" | "failed"
    detail: str = ""
    wall_s: float = 0.0  # in flight, so it includes waiting for other items


@dataclass
class FreezeReport:
    set_name: str
    health: SearxHealth | None = None
    items: list[FrozenItem] = field(default_factory=list)
    concurrency: int = 1
    wall_s: float = 0.0
    cpu_s: float = 0.0
    peak_rss_mb: float = 0.0

    @property
    def failed(self) -> bool:
        return any(i.status == "failed" for i in self.items)

    @property
    def incomplete(self) -> bool:
        """A snapshot exists but its citation cache is missing hops (rate limited):
        ``bench freeze --rewarm`` completes it without changing its inputs."""
        return any(i.status == "incomplete" for i in self.items)

    def lines(self) -> list[str]:
        out = [f"freeze {self.set_name} (concurrency {self.concurrency})"]
        if self.health is not None:
            out.append(f"  {self.health.explain()}")
        for item in self.items:
            tail = f": {item.detail}" if item.detail else ""
            timing = (
                f" [{item.wall_s:.0f} s]"
                if item.status in ("frozen", "incomplete")
                else ""
            )
            out.append(f"  {item.status:7s} {item.item_id}{timing}{tail}")
        done = sum(1 for i in self.items if i.status in ("frozen", "kept"))
        if self.incomplete:
            out.append(
                "  incomplete items hold a snapshot whose citation cache is missing "
                "hops (the source said its budget was spent); they do not count "
                "below. After the budget resets, or with OPENALEX_API_KEY set, run "
                "`bench freeze --rewarm`"
            )
        out.append(f"  {done} of {len(self.items)} items have a snapshot")
        out.append(
            f"  process: wall {self.wall_s:.0f} s, CPU {self.cpu_s:.0f} s, "
            f"peak RSS {self.peak_rss_mb:.0f} MB, concurrency {self.concurrency}"
        )
        return out


@dataclass(frozen=True)
class FreezeItem:
    item_id: str
    question: str


def freeze_items(
    set_name: str, *, data_dir: Path | None, n: int | None, seed: int
) -> list[FreezeItem]:
    """The items of a set to freeze, the same ones ``load_items`` gives an experiment."""
    if set_name == "litsearch":
        if data_dir is None:
            raise ValueError("freezing litsearch needs --dir (the download directory)")
        data = load_litsearch(data_dir, n=n or DEFAULT_SUBSET, seed=seed)
        return [FreezeItem(i.query.query_id, i.query.query) for i in data.items]
    if set_name == "web":
        if n is None or n >= len(WEB_QUESTIONS):
            questions = list(WEB_QUESTIONS)
        else:
            wanted = set(
                select_subset([q.question_id for q in WEB_QUESTIONS], n, seed=seed)
            )
            questions = [q for q in WEB_QUESTIONS if q.question_id in wanted]
        return [FreezeItem(q.question_id, q.question) for q in questions]
    raise ValueError(f"cannot freeze the set {set_name!r}; freeze litsearch or web")


def _mode_of(set_name: str) -> Mode:
    return "scholarly" if set_name == "litsearch" else "web"


async def freeze_set(
    set_name: str,
    *,
    snapshots: Path,
    data_dir: Path | None,
    model: str | Model,
    decision_model: DecisionModel | None,
    decision_label: str | None,
    n: int | None,
    seed: int,
    source_config: SourceConfig,
    progress: Callable[[str], None] = lambda _line: None,
    prefetch_top_k: int | None = PREFETCH_TOP_K,
    concurrency: int = 1,
    health_client: httpx.AsyncClient | None = None,
    client_factory: Callable[[ResearchStore], SourceClient] | None = None,
    clock: Callable[[], float] = time.time,
) -> FreezeReport:
    """Freeze every item of *set_name* that has no snapshot yet.

    Raises ``SearxDown`` before anything is captured when the set is a web set
    and SearxNG is not answering, and ``ValueError`` for a ``concurrency`` the
    model does not allow. ``client_factory`` builds the source client of each
    item's store (offline tests give it a mock transport); the default builds
    one with ``source_config.pdf_extraction`` and spacing shared across items.
    """
    return await freeze_items_of(
        set_name,
        freeze_items(set_name, data_dir=data_dir, n=n, seed=seed),
        snapshots=snapshots,
        model=model,
        decision_model=decision_model,
        decision_label=decision_label,
        source_config=source_config,
        progress=progress,
        prefetch_top_k=prefetch_top_k,
        concurrency=concurrency,
        health_client=health_client,
        client_factory=client_factory,
        clock=clock,
    )


async def freeze_items_of(
    set_name: str,
    items: Sequence[FreezeItem],
    *,
    snapshots: Path,
    model: str | Model,
    decision_model: DecisionModel | None,
    decision_label: str | None,
    source_config: SourceConfig,
    progress: Callable[[str], None],
    prefetch_top_k: int | None,
    health_client: httpx.AsyncClient | None,
    client_factory: Callable[[ResearchStore], SourceClient] | None,
    clock: Callable[[], float],
    concurrency: int = 1,
) -> FreezeReport:
    """Freeze the given *items* of a set: the part of ``freeze_set`` after the
    items are chosen. A snapshot that exists is kept; the others are frozen with
    at most *concurrency* in flight (results in item order)."""
    concurrency = resolve_concurrency(concurrency, models=[model])
    mode = _mode_of(set_name)
    report = FreezeReport(set_name=set_name, concurrency=concurrency)
    if mode == "web":
        report.health = await require_searxng(
            source_config.searxng_url, client=health_client
        )
    pacing = SharedPacing()
    factory = client_factory or (
        lambda store: SourceClient(
            store, pdf_extraction=source_config.pdf_extraction, pacing=pacing
        )
    )

    async def freeze_item(item: FreezeItem) -> FrozenItem:
        folder = snapshot_dir(snapshots, set_name, item.item_id)
        if (folder / FROZEN_FILE).exists() and (folder / STORE_FILE).exists():
            if snapshot_incomplete(folder):
                return FrozenItem(
                    item.item_id,
                    "incomplete",
                    "kept as it is (its inputs stay fixed); `bench freeze --rewarm` "
                    "completes its citation cache",
                )
            return FrozenItem(item.item_id, "kept")
        progress(f"freezing {set_name}/{item.item_id}")
        started = time.monotonic()
        try:
            detail = await _freeze_one(
                item,
                folder=folder,
                mode=mode,
                model=model,
                decision_model=decision_model,
                decision_label=decision_label,
                source_config=source_config,
                prefetch_top_k=prefetch_top_k,
                concurrency=concurrency,
                progress=lambda line: progress(f"[{item.item_id}] {line}"),
                client_factory=factory,
                clock=clock,
            )
        except Exception as error:  # reported as a failed item, never a missing one
            return FrozenItem(
                item.item_id,
                "failed",
                f"{type(error).__name__}: {error}",
                wall_s=time.monotonic() - started,
            )
        return FrozenItem(
            item.item_id,
            "incomplete" if snapshot_incomplete(folder) else "frozen",
            detail,
            wall_s=time.monotonic() - started,
        )

    started_wall = time.monotonic()
    started_cpu = time.process_time()
    report.items = await run_bounded(
        [lambda item=item: freeze_item(item) for item in items],
        concurrency=concurrency,
    )
    report.wall_s = time.monotonic() - started_wall
    report.cpu_s = time.process_time() - started_cpu
    report.peak_rss_mb = peak_rss_mb()
    return report


async def _no_citations() -> CitationWarmReport:
    return CitationWarmReport()


async def _freeze_one(
    item: FreezeItem,
    *,
    folder: Path,
    mode: Mode,
    model: str | Model,
    decision_model: DecisionModel | None,
    decision_label: str | None,
    source_config: SourceConfig,
    prefetch_top_k: int | None,
    concurrency: int,
    progress: Callable[[str], None],
    client_factory: Callable[[ResearchStore], SourceClient],
    clock: Callable[[], float],
) -> str:
    partial = folder.with_name(folder.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    started = time.monotonic()
    try:
        with ResearchStore(partial / STORE_FILE) as store:
            client = client_factory(store)
            try:
                frozen = await capture(
                    item.question,
                    mode=mode,
                    model=model,
                    store=store,
                    decision_model=decision_model,
                    sources=source_config,
                    config=PRESETS["baseline"],
                    source_client=client,
                    progress=progress,
                    clock=clock,
                )
                if frozen.degraded:
                    raise RuntimeError(
                        "capture was degraded, so the snapshot would be an unfair "
                        "input: " + "; ".join(frozen.degraded)
                    )
                if not frozen.candidates:
                    raise RuntimeError("the search found no candidate")
                source_set = build_sources(
                    source_config, store=store, mode=mode, client=client
                )
                targets = prefetch_targets(frozen, top_k=prefetch_top_k)
                # The texts and the citation hops go to different hosts, so they
                # run together (the client spaces each source and caps its
                # in-flight requests).
                prefetch, warmed = await asyncio.gather(
                    prefetch_texts(targets, text_sources=source_set.text),
                    warm_citations(
                        citation_targets(frozen, top_k=prefetch_top_k),
                        sources=source_set.expand,
                    )
                    if mode != "web"
                    else _no_citations(),
                )
                extractions = dict(client.extractions)
            finally:
                await client.aclose()
        frozen.save(partial / FROZEN_FILE)
        prefetched = [c.work.work_id or c.work.title for c in targets]
        not_prefetched = [
            c.work.work_id or c.work.title
            for c in frozen.candidates
            if (c.work.work_id or c.work.title) not in set(prefetched)
        ]
        (partial / META_FILE).write_text(
            json.dumps(
                {
                    "item_id": item.item_id,
                    "model": model_label(model),
                    "decision_model": decision_label,
                    "digest": frozen.digest(),
                    "n_subquestions": len(frozen.subquestions),
                    "n_candidates": len(frozen.candidates),
                    "prefetch": prefetch.model_dump(mode="json"),
                    "prefetch_rule": (
                        "every candidate"
                        if prefetch_top_k is None
                        else f"best {prefetch_top_k} per sub-question by source "
                        "search rank (reciprocal-rank fusion over the sources' "
                        "lists)"
                    ),
                    "prefetch_top_k": prefetch_top_k,
                    "prefetched_works": prefetched,
                    "not_prefetched_works": not_prefetched,
                    "not_prefetched_reading": (
                        "abstract only: a frozen run that reads one of these "
                        f"finds '{FROZEN_TEXT_REASON}' (the same for every arm)"
                    ),
                    "pdf_extraction": source_config.pdf_extraction,
                    "extractors": extractions,
                    "extractor_counts": dict(
                        sorted(Counter(extractions.values()).items())
                    ),
                    "citations": warmed.model_dump(mode="json"),
                    "citations_rule": (
                        f"references and citations (limit {WARM_HOP_LIMIT}) of the "
                        "scholarly works among the prefetched ones (the same "
                        "best-by-source-rank set); a work outside it that an arm "
                        "expands from is an offline miss for every arm"
                    ),
                    "incomplete": warmed.rate_limited > 0,
                    "incomplete_reason": (
                        f"{warmed.rate_limited} citation hops were refused as rate "
                        "limited"
                        if warmed.rate_limited
                        else ""
                    ),
                    "concurrency": concurrency,
                    "item_wall_s": round(time.monotonic() - started, 1),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if folder.exists():
            shutil.rmtree(folder)
        partial.rename(folder)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return (
        f"{len(frozen.subquestions)} sub-questions, {len(frozen.candidates)} candidates, "
        f"{prefetch.fetched}/{prefetch.attempted} texts cached "
        f"({len(not_prefetched)} abstract-only), "
        f"{warmed.hops_cached} citation hops cached ({warmed.hops_failed} failed)"
    )


# -- rewarming --------------------------------------------------------------------


async def rewarm_set(
    set_name: str,
    *,
    snapshots: Path,
    data_dir: Path | None,
    model: str | Model,
    n: int | None,
    seed: int,
    source_config: SourceConfig,
    progress: Callable[[str], None] = lambda _line: None,
    prefetch_top_k: int | None = PREFETCH_TOP_K,
    concurrency: int = 1,
    client_factory: Callable[[ResearchStore], SourceClient] | None = None,
) -> FreezeReport:
    """Complete the snapshots of a set that exist: ask again for the citation hops
    and texts that failed, into each snapshot's own store.

    ``frozen.json`` and its digest are never touched, so the inputs every arm
    sees stay what they were; only the cache gains answers. A snapshot that is
    missing is reported failed (a rewarm never searches). *model* only decides
    which concurrency is allowed.
    """
    return await rewarm_items_of(
        set_name,
        freeze_items(set_name, data_dir=data_dir, n=n, seed=seed),
        snapshots=snapshots,
        model=model,
        source_config=source_config,
        progress=progress,
        prefetch_top_k=prefetch_top_k,
        concurrency=concurrency,
        client_factory=client_factory,
    )


async def rewarm_items_of(
    set_name: str,
    items: Sequence[FreezeItem],
    *,
    snapshots: Path,
    model: str | Model,
    source_config: SourceConfig,
    progress: Callable[[str], None],
    prefetch_top_k: int | None,
    concurrency: int,
    client_factory: Callable[[ResearchStore], SourceClient] | None,
) -> FreezeReport:
    """:func:`rewarm_set` for the given *items*."""
    concurrency = resolve_concurrency(concurrency, models=[model])
    mode = _mode_of(set_name)
    pacing = SharedPacing()
    factory = client_factory or (
        lambda store: SourceClient(
            store, pdf_extraction=source_config.pdf_extraction, pacing=pacing
        )
    )
    report = FreezeReport(set_name=set_name, concurrency=concurrency)

    async def rewarm(item: FreezeItem) -> FrozenItem:
        folder = snapshot_dir(snapshots, set_name, item.item_id)
        if not (folder / FROZEN_FILE).exists():
            return FrozenItem(
                item.item_id,
                "failed",
                "no snapshot to rewarm: run `bench freeze` first",
            )
        progress(f"rewarming {set_name}/{item.item_id}")
        started = time.monotonic()
        try:
            detail = await _rewarm_one(
                folder,
                mode=mode,
                source_config=source_config,
                prefetch_top_k=prefetch_top_k,
                client_factory=factory,
            )
        except Exception as error:
            return FrozenItem(
                item.item_id,
                "failed",
                f"{type(error).__name__}: {error}",
                wall_s=time.monotonic() - started,
            )
        return FrozenItem(
            item.item_id,
            "incomplete" if snapshot_incomplete(folder) else "frozen",
            detail,
            wall_s=time.monotonic() - started,
        )

    started_wall = time.monotonic()
    started_cpu = time.process_time()
    report.items = await run_bounded(
        [lambda item=item: rewarm(item) for item in items], concurrency=concurrency
    )
    report.wall_s = time.monotonic() - started_wall
    report.cpu_s = time.process_time() - started_cpu
    report.peak_rss_mb = peak_rss_mb()
    return report


async def _rewarm_one(
    folder: Path,
    *,
    mode: Mode,
    source_config: SourceConfig,
    prefetch_top_k: int | None,
    client_factory: Callable[[ResearchStore], SourceClient],
) -> str:
    frozen = load_frozen(folder)
    meta_path = folder / META_FILE
    meta = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )
    if meta.get("digest") not in (None, frozen.digest()):
        raise RuntimeError("frozen.json does not match the digest in meta.json")
    started = time.monotonic()
    with ResearchStore(folder / STORE_FILE) as store:
        client = client_factory(store)
        try:
            source_set = build_sources(
                source_config, store=store, mode=mode, client=client
            )
            targets = prefetch_targets(frozen, top_k=prefetch_top_k)
            prefetch, warmed = await asyncio.gather(
                prefetch_texts(targets, text_sources=source_set.text),
                warm_citations(
                    citation_targets(frozen, top_k=prefetch_top_k),
                    sources=source_set.expand,
                )
                if mode != "web"
                else _no_citations(),
            )
            extractions = {**meta.get("extractors", {}), **client.extractions}
        finally:
            await client.aclose()
    meta.update(
        {
            "digest": frozen.digest(),
            "prefetch": prefetch.model_dump(mode="json"),
            "extractors": extractions,
            "extractor_counts": dict(sorted(Counter(extractions.values()).items())),
            "citations": warmed.model_dump(mode="json"),
            "incomplete": warmed.rate_limited > 0,
            "incomplete_reason": (
                f"{warmed.rate_limited} citation hops were refused as rate limited"
                if warmed.rate_limited
                else ""
            ),
            "rewarmed_wall_s": round(time.monotonic() - started, 1),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return (
        f"rewarmed: {prefetch.fetched}/{prefetch.attempted} texts cached, "
        f"{warmed.hops_cached} citation hops cached ({warmed.hops_failed} failed)"
    )
