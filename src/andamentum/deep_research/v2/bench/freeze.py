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
Capture and prefetch are strictly sequential.

A snapshot ages with its cache: search answers live 7 days, metadata 30, page
texts 180 (``store.TTL_*``); frozen arms need only the page texts.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from pydantic import BaseModel
from pydantic_ai.models import Model

from andamentum.core.decisions import DecisionModel

from ..config import PRESETS
from ..frozen import FrozenInputs, capture
from ..sources import SourceClient, SourceConfig, SourceError, build_sources
from ..sources.base import ExpandingSource, TextSource
from ..steps import EXPAND_PER_CALL, is_scholarly
from ..store import TTL_FULL_TEXT_S, ResearchStore, cache_key
from ..types import Candidate, Mode
from .health import SearxHealth, require_searxng
from .litsearch import DEFAULT_SUBSET, load_litsearch
from .planted import PlantedSource, plant_candidates
from .subsets import select_subset
from .web_set import WEB_QUESTIONS

FROZEN_FILE = "frozen.json"
STORE_FILE = "store.db"
META_FILE = "meta.json"
# Which candidates get their text cached: every one (None), whatever an arm will
# screen or keep, so no arm is starved of full text by another arm's choices. A
# cap is an explicit option (`--prefetch-cap`) and is recorded in meta.json.
PREFETCH_PER_SUBQUESTION: int | None = None
# Which candidates get their references and citations cached: the first this
# many scholarly works of each sub-question in admission order, which is the
# set a baseline run screens in its first round. The rule does not depend on
# any arm's scores.
WARM_CITATIONS_PER_SUBQUESTION = 25
WARM_HOP_LIMIT = EXPAND_PER_CALL
PLANTED_SOURCE_RANK_KEY = "planted"


class SnapshotMissing(RuntimeError):
    """A frozen arm needs a snapshot that has not been taken."""


def snapshot_dir(root: Path, set_name: str, item_id: str) -> Path:
    return root / set_name / item_id


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


def prefetch_targets(
    frozen: FrozenInputs, *, per_subquestion: int | None
) -> list[Candidate]:
    """The candidates whose text is cached: with ``per_subquestion`` None every
    candidate; otherwise the first that many of each sub-question in admission
    order (the order the engine screens them in), each work once."""
    chosen: dict[str, Candidate] = {}
    for subquestion in frozen.subquestions:
        own = frozen.candidates_for(subquestion.id)
        for candidate in own if per_subquestion is None else own[:per_subquestion]:
            chosen.setdefault(candidate.work.work_id or candidate.work.title, candidate)
    return list(chosen.values())


class CitationWarmReport(BaseModel):
    """What warming the citation hops did."""

    works: int = 0
    hops_cached: int = 0
    hops_failed: int = 0
    errors: list[str] = []


def citation_targets(frozen: FrozenInputs, *, per_subquestion: int) -> list[Candidate]:
    """The scholarly works whose hops are cached: the first ``per_subquestion``
    of each sub-question in admission order, each work once."""
    chosen: dict[str, Candidate] = {}
    for subquestion in frozen.subquestions:
        scholarly = [
            c for c in frozen.candidates_for(subquestion.id) if is_scholarly(c.work)
        ]
        for candidate in scholarly[:per_subquestion]:
            chosen.setdefault(candidate.work.work_id or candidate.work.title, candidate)
    return list(chosen.values())


async def warm_citations(
    candidates: Sequence[Candidate], *, sources: Sequence[ExpandingSource]
) -> CitationWarmReport:
    """Ask each source for the references and citations of each candidate, one
    call after another, so the answers are in the store's cache. The calls are
    the ones ``steps.expand`` makes (same limit), so a frozen run finds them."""
    report = CitationWarmReport()
    for candidate in candidates:
        report.works += 1
        for source in sources:
            for direction, follow in (
                ("references", source.references),
                ("citations", source.citations),
            ):
                try:
                    await follow(candidate.work, limit=WARM_HOP_LIMIT)
                except SourceError as error:
                    if error.kind == "config":
                        continue
                    report.hops_failed += 1
                    report.errors.append(
                        f"{source.name} {direction} of "
                        f"{candidate.work.title or candidate.work.work_id}: {error.reason}"
                    )
                else:
                    report.hops_cached += 1
    return report


async def prefetch_texts(
    candidates: Sequence[Candidate],
    *,
    text_sources: Sequence[TextSource],
) -> PrefetchReport:
    """Fetch each candidate's text from the first source that holds it, one after another.

    A fetch the gate refuses or a source that fails is counted and named, never
    hidden; it does not stop the freeze (a paywalled page is normal).
    """
    report = PrefetchReport()
    for candidate in candidates:
        report.attempted += 1
        outcome: str | None = None
        for source in text_sources:
            try:
                fetched = await source.fetch_text(candidate.work)
            except SourceError as error:
                outcome = f"{candidate.work.url or candidate.work.title}: {error}"
                continue
            if fetched is not None:
                outcome = "fetched"
                break
        if outcome == "fetched":
            report.fetched += 1
        elif outcome is None:
            report.none_held += 1
        else:
            report.failed += 1
            report.errors.append(outcome)
    return report


# -- freezing a set ---------------------------------------------------------------


@dataclass
class FrozenItem:
    item_id: str
    status: str  # "frozen" | "kept" | "failed"
    detail: str = ""


@dataclass
class FreezeReport:
    set_name: str
    health: SearxHealth | None = None
    items: list[FrozenItem] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(i.status == "failed" for i in self.items)

    def lines(self) -> list[str]:
        out = [f"freeze {self.set_name}"]
        if self.health is not None:
            out.append(f"  {self.health.explain()}")
        for item in self.items:
            tail = f": {item.detail}" if item.detail else ""
            out.append(f"  {item.status:7s} {item.item_id}{tail}")
        done = sum(1 for i in self.items if i.status in ("frozen", "kept"))
        out.append(f"  {done} of {len(self.items)} items have a snapshot")
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
    prefetch_per_subquestion: int | None = PREFETCH_PER_SUBQUESTION,
    health_client: httpx.AsyncClient | None = None,
    client_factory: Callable[[ResearchStore], SourceClient] = SourceClient,
    clock: Callable[[], float] = time.time,
) -> FreezeReport:
    """Freeze every item of *set_name* that has no snapshot yet.

    Raises ``SearxDown`` before anything is captured when the set is a web set
    and SearxNG is not answering. ``client_factory`` builds the source client
    of each item's store (offline tests give it a mock transport).
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
        prefetch_per_subquestion=prefetch_per_subquestion,
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
    prefetch_per_subquestion: int | None,
    health_client: httpx.AsyncClient | None,
    client_factory: Callable[[ResearchStore], SourceClient],
    clock: Callable[[], float],
) -> FreezeReport:
    """Freeze the given *items* of a set: the part of ``freeze_set`` after the
    items are chosen."""
    mode = _mode_of(set_name)
    report = FreezeReport(set_name=set_name)
    if mode == "web":
        report.health = await require_searxng(
            source_config.searxng_url, client=health_client
        )
    for item in items:
        folder = snapshot_dir(snapshots, set_name, item.item_id)
        if (folder / FROZEN_FILE).exists() and (folder / STORE_FILE).exists():
            report.items.append(FrozenItem(item.item_id, "kept"))
            continue
        progress(f"freezing {set_name}/{item.item_id}")
        try:
            detail = await _freeze_one(
                item,
                folder=folder,
                mode=mode,
                model=model,
                decision_model=decision_model,
                decision_label=decision_label,
                source_config=source_config,
                prefetch_per_subquestion=prefetch_per_subquestion,
                progress=progress,
                client_factory=client_factory,
                clock=clock,
            )
        except Exception as error:  # reported as a failed item, never a missing one
            report.items.append(
                FrozenItem(item.item_id, "failed", f"{type(error).__name__}: {error}")
            )
            continue
        report.items.append(FrozenItem(item.item_id, "frozen", detail))
    return report


async def _freeze_one(
    item: FreezeItem,
    *,
    folder: Path,
    mode: Mode,
    model: str | Model,
    decision_model: DecisionModel | None,
    decision_label: str | None,
    source_config: SourceConfig,
    prefetch_per_subquestion: int | None,
    progress: Callable[[str], None],
    client_factory: Callable[[ResearchStore], SourceClient],
    clock: Callable[[], float],
) -> str:
    partial = folder.with_name(folder.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
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
                prefetch = await prefetch_texts(
                    prefetch_targets(frozen, per_subquestion=prefetch_per_subquestion),
                    text_sources=source_set.text,
                )
                warmed = (
                    await warm_citations(
                        citation_targets(
                            frozen, per_subquestion=WARM_CITATIONS_PER_SUBQUESTION
                        ),
                        sources=source_set.expand,
                    )
                    if mode != "web"
                    else CitationWarmReport()
                )
            finally:
                await client.aclose()
        frozen.save(partial / FROZEN_FILE)
        (partial / META_FILE).write_text(
            json.dumps(
                {
                    "item_id": item.item_id,
                    "model": model if isinstance(model, str) else str(model.model_name),
                    "decision_model": decision_label,
                    "digest": frozen.digest(),
                    "n_subquestions": len(frozen.subquestions),
                    "n_candidates": len(frozen.candidates),
                    "prefetch": prefetch.model_dump(mode="json"),
                    "prefetch_rule": (
                        "every candidate"
                        if prefetch_per_subquestion is None
                        else f"first {prefetch_per_subquestion} per sub-question"
                    ),
                    "citations": warmed.model_dump(mode="json"),
                    "citations_rule": (
                        "references and citations (limit "
                        f"{WARM_HOP_LIMIT}) of the first "
                        f"{WARM_CITATIONS_PER_SUBQUESTION} scholarly candidates of "
                        "each sub-question in admission order"
                    ),
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
        f"{prefetch.fetched}/{prefetch.attempted} texts cached, "
        f"{warmed.hops_cached} citation hops cached ({warmed.hops_failed} failed)"
    )
