"""Frozen inputs: one question's plan and search, captured once and replayed.

Comparing two configurations live mixes their difference with the planner's
and the search engines' noise (a different sub-question, an outage, a reranked
result list). A snapshot fixes those inputs: :func:`capture` runs Plan and one
round of Search exactly as the engine does (``steps.plan``, ``write_queries``,
``search``, ``admit``) and records the sub-questions, the queries, and the
merged candidates (full ``Work`` records with each source's rank). The store
it ran over holds the source responses fetched so far; the two together are
the snapshot.

:func:`capture` also asks Crossref once for the retraction status of every
candidate with a DOI (``retractions=True``), so that an offline run still
never reads a retracted work; a check that fails is listed in
``retraction_unchecked``.

``research(..., frozen=snapshot)`` takes Plan and Search from the snapshot (no
planning call, no live search) and runs everything else over an *offline*
``SourceClient``: every request is answered from the store's cache or skipped
(``ResearchResult.skipped``), never sent. Expansion uses only cached citation
responses, reading only cached texts (else the abstract the record carries),
and the link and retraction checks only cached answers. So every arm and every
repeat sees the same inputs and makes no network call.

A snapshot is one JSON file (:meth:`FrozenInputs.save`, :meth:`FrozenInputs.load`).
Abstracts and snippets under a ``no_redistribute`` licence are written as ``""``
(the ``Work`` and ``Candidate`` serialisers); the store keeps them, and a frozen
run reads them back from it (``steps.restore_licensed_text``).

``capture(plan=snapshot)`` reuses another snapshot's sub-questions and queries
(no model call of any kind): a different set of sources or a different query
form can then be tried on exactly the queries another freeze was given. What a
capture asked of each source (searches, failures, results per query, seconds)
is recorded in the snapshot as accounting; it is not part of its digest.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable, Collection
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue
from pydantic_ai.models import Model

from andamentum.core.decisions import DecisionModel

from . import steps
from .ask import Asker
from .config import DEFAULT_PRESET, PRESETS, ResearchConfig
from .sources import SourceClient, SourceConfig, SourceError, build_sources
from .store import ResearchStore
from .types import Budget, Candidate, Mode, RunStats, SubQuestion

FROZEN_VERSION = 1
# What a capture did, not what it found: left out of the digest, which names the inputs.
_ACCOUNTING = (
    "searches",
    "search_failures",
    "result_counts",
    "search_s",
    "tolerated",
)


class FrozenInputs(BaseModel):
    """One question's plan and first search, as an arm of an experiment sees them.

    ``queries`` maps a sub-question id to the queries run for it, in order;
    ``candidates`` are the merged candidates (each names its sub-questions and
    its rank in every source list it came from). ``degraded`` lists the
    failures during capture (a source down): a snapshot with any is not a fair
    input and a caller should not use it. ``models`` and ``sources`` record
    what captured it (secrets left out).
    """

    model_config = ConfigDict(frozen=True)

    version: int = FROZEN_VERSION
    question: str = Field(min_length=1)
    mode: Mode
    captured_at: float
    subquestions: list[SubQuestion] = Field(min_length=1)
    queries: dict[str, list[str]] = Field(default_factory=dict)
    candidates: list[Candidate] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
    models: dict[str, JsonValue] = Field(default_factory=dict)
    sources: dict[str, JsonValue] = Field(default_factory=dict)
    queries_per_subquestion: int = Field(default=1, ge=1)
    run_id: str = ""
    retraction_unchecked: list[str] = Field(default_factory=list)
    # Search accounting (see ``steps.SearchOutcome``): searches attempted and
    # failed per source, results returned per source and query, the seconds
    # ``steps.search`` took, and the failures of sources the capture was told
    # to tolerate (kept here, not in ``degraded``, so they are counted and
    # judged rather than making the snapshot unusable).
    searches: dict[str, int] = Field(default_factory=dict)
    search_failures: dict[str, int] = Field(default_factory=dict)
    result_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    search_s: float = 0.0
    tolerated: list[str] = Field(default_factory=list)

    def candidates_for(self, subquestion_id: str) -> list[Candidate]:
        """The candidates found for one sub-question, in capture order."""
        return [c for c in self.candidates if subquestion_id in c.subquestion_ids]

    def all_queries(self) -> list[str]:
        """Every query run, in order."""
        return [q for own in self.queries.values() for q in own]

    def digest(self) -> str:
        """sha256 of the snapshot's canonical JSON: names the inputs a run used."""
        text = json.dumps(
            self.model_dump(mode="json", exclude=set(_ACCOUNTING)),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def save(self, path: Path) -> None:
        """Write the snapshot as JSON (parents created)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> FrozenInputs:
        """Read a snapshot written by :meth:`save`; a newer version is refused."""
        data = json.loads(path.read_text(encoding="utf-8"))
        version = data.get("version") if isinstance(data, dict) else None
        if version != FROZEN_VERSION:
            raise ValueError(
                f"{path}: frozen inputs version {version!r}, this code reads "
                f"{FROZEN_VERSION}"
            )
        return cls.model_validate(data)


def _model_id(model: str | Model) -> str:
    return model if isinstance(model, str) else f"{model.system}:{model.model_name}"


async def capture(
    question: str,
    *,
    mode: Mode,
    model: str | Model | None,
    store: ResearchStore,
    decision_model: str | DecisionModel | None = None,
    budget: Budget = Budget(),
    sources: SourceConfig = SourceConfig(),
    config: ResearchConfig = PRESETS[DEFAULT_PRESET],
    source_client: SourceClient | None = None,
    progress: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.time,
    retractions: bool = True,
    plan: FrozenInputs | None = None,
    tolerate_failures: Collection[str] = (),
) -> FrozenInputs:
    """Plan *question* and run one round of search, live, and return the snapshot.

    The steps are the engine's own (``steps.plan``; ``steps.write_queries``
    with ``budget.queries_per_subq``; ``steps.search`` on every source of
    *mode*; ``steps.admit`` into *store*, which must stay open: its cache is
    half of the snapshot). ``config`` supplies only the determinism knobs (the
    model settings of the two planning calls); a snapshot serves every
    configuration. The capture is recorded as a run of the store with
    ``config.frozen_capture`` set. Failures are kept in ``degraded``, never
    hidden. ``retractions`` checks every candidate's DOI in Crossref once,
    concurrently (within the client's per-source cap), into the store's cache (a failed check is listed in
    ``retraction_unchecked``, not ``degraded``: the snapshot is still fair).

    ``plan`` is another snapshot of the same question and mode whose
    sub-questions and queries are used as they are (the planned filters follow
    from the sub-questions by code): no model is called, none is opened, and
    ``model`` may be ``None``. ``tolerate_failures`` names sources whose failed
    searches are counted (``search_failures``, ``tolerated``) instead of
    marking the snapshot degraded; every other failure still does.
    """
    if plan is None and model is None:
        raise ValueError("capture plans with a model: pass model, or plan= a snapshot")
    if plan is not None and (plan.question != question or plan.mode != mode):
        raise ValueError(
            "plan is a snapshot of another question or mode: "
            f"{plan.question!r} ({plan.mode}), not {question!r} ({mode})"
        )
    if source_client is not None and source_client.store is not store:
        raise ValueError("source_client must belong to the store passed")
    say = progress or (lambda _line: None)
    client = source_client or SourceClient(store, pdf_extraction=sources.pdf_extraction)
    handle: DecisionModel | None = None
    try:
        handle = (
            None
            if plan is not None
            else await DecisionModel.open(decision_model)
            if isinstance(decision_model, str)
            else decision_model
        )
        source_set = build_sources(sources, store=store, mode=mode, client=client)
        secretless = sources.model_dump(
            mode="json",
            exclude={
                "mailto",
                "openalex_api_key",
                "semantic_scholar_api_key",
                "local_corpus",
            },
        )
        models: dict[str, JsonValue] = {
            "model": "" if model is None else _model_id(model),
            "decision_model": handle.id if handle else None,
            "decision_digest": handle.digest if handle else None,
            "model_settings": {"temperature": config.temperature, "seed": config.seed},
        }
        run_id = store.start_run(
            question=question,
            mode=mode,
            config={
                "frozen_capture": True,
                "plan_from": plan.digest() if plan is not None else None,
                "budget": budget.model_dump(mode="json"),
                "sources": secretless,
            },
            models=models,
        )
        asker = Asker(
            model=model if model is not None else "unused",
            store=store,
            run_id=run_id,
            budget=budget,
            stats=RunStats(),
            decision_model=handle,
            model_settings=config.model_settings(),
        )
        degraded: list[str] = []
        try:
            if plan is not None:
                say("Freeze: the sub-questions and queries of another snapshot")
                subquestions = list(plan.subquestions)
                queries = {k: list(v) for k, v in plan.queries.items()}
            else:
                say("Freeze: sub-questions")
                planned = await steps.plan(question, asker=asker)
                degraded += planned.degraded
                if planned.spent:
                    degraded.append(f"budget:{planned.spent} while planning")
                subquestions = planned.subquestions
                say(f"Freeze: queries for {len(subquestions)} sub-questions")
                written = await steps.write_queries(
                    question,
                    targets=subquestions,
                    gaps={},
                    done=[],
                    per_subquestion=budget.queries_per_subq,
                    asker=asker,
                )
                degraded += written.degraded
                if written.spent:
                    degraded.append(f"budget:{written.spent} while writing queries")
                queries = written.queries
            for subquestion in subquestions:
                store.add_run_item(run_id, "subquestion", subquestion.id, subquestion)
            say(f"Freeze: search, {sum(len(q) for q in queries.values())} queries")
            began = clock()
            searched = await steps.search(
                queries,
                sources=source_set.search,
                filters={s.id: steps.filters_for(s) for s in subquestions},
                limit=sources.results_per_query,
            )
            search_s = clock() - began
            tolerated = [
                line
                for name in tolerate_failures
                for line in searched.failed_lines.get(name, [])
            ]
            degraded += [line for line in searched.degraded if line not in tolerated]
            admitted = steps.admit(searched.candidates, store=store, known={})
            unchecked: list[str] = []
            if retractions:
                dois = [c.work.doi for c in admitted.candidates.values() if c.work.doi]
                say(f"Freeze: retraction status of {len(dois)} DOIs")

                async def check(doi: str) -> str | None:
                    try:
                        await source_set.crossref.retraction_status(doi)
                    except SourceError as error:
                        return f"{doi}: {error.reason}"
                    return None

                # Concurrent calls; the client keeps Crossref's spacing and its
                # in-flight cap, and the failures stay in DOI order.
                unchecked = [
                    line
                    for line in await asyncio.gather(*(check(d) for d in dois))
                    if line is not None
                ]
        except BaseException as error:
            store.finish_run(
                run_id,
                stats=asker.stats.model_dump(mode="json"),
                degraded=[f"capture failed: {type(error).__name__}: {error}"],
            )
            raise
        store.finish_run(
            run_id, stats=asker.stats.model_dump(mode="json"), degraded=degraded
        )
        return FrozenInputs(
            question=question,
            mode=mode,
            captured_at=clock(),
            subquestions=subquestions,
            queries=queries,
            candidates=list(admitted.candidates.values()),
            degraded=steps.unique(degraded),
            models=models,
            sources=secretless,
            queries_per_subquestion=budget.queries_per_subq,
            run_id=run_id,
            retraction_unchecked=unchecked,
            searches=dict(searched.searches),
            search_failures=dict(searched.failures),
            result_counts={k: dict(v) for k, v in searched.result_counts.items()},
            search_s=search_s,
            tolerated=tolerated,
        )
    finally:
        if isinstance(decision_model, str) and handle is not None:
            await handle.aclose()
        if source_client is None:
            await client.aclose()
