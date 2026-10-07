"""The systems under test, behind one small interface.

An answer arm takes a question and returns an :class:`ArmOutput`: the answer
text, the claims with the evidence behind them, the links, the ranked works,
and what the run cost. Two arms exist: the legacy ``run_research`` (web only)
and v2's ``research()``. Both are mapped onto the same record, so every metric
is computed by one piece of code from the same fields.

The v2 arm calls ``research()`` with ``store=None`` by default, so nothing
persists between questions and every question pays for its own searches.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field
from pydantic_ai.models import Model

from andamentum.core.decisions import DecisionModel
from andamentum.deep_research.models import ResearchResult as LegacyResult
from andamentum.deep_research.orchestrator import run_research
from andamentum.references import Work
from andamentum.references.works import canonical_url

from ..config import PRESETS, ResearchConfig
from ..frozen import FrozenInputs
from ..graph import research
from ..sources.base import SourceClient, SourceConfig
from ..sources.local_corpus import LocalCorpusSource
from ..store import ResearchStore
from ..dense import Embedder
from ..types import _FINDING_REFERENCE, Budget, Mode, ResearchResult, RunLog
from .judging import ClaimEvidence, best_window
from .metrics import Cost, extract_urls

MIN_SENTENCE_WORDS = 4
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\(])")
_LIST_MARKER = re.compile(r"^(?:[-*]|\d+[.)])\s+")
_CITATION = re.compile(r"\s*\[\s*F\d+(?:\s*[,;]\s*F\d+)*\s*\]")


class ArmOutput(BaseModel):
    """One answer, reduced to what the metrics read."""

    answer_text: str
    claims: list[ClaimEvidence] = Field(default_factory=list)
    n_unsourced: int = Field(
        default=0, description="Claims with nothing to check them against."
    )
    urls: list[str] = Field(default_factory=list)
    works: list[Work] = Field(
        default_factory=list, description="The kept works, in the arm's rank order."
    )
    degraded: list[str] = Field(default_factory=list)
    stop_reason: str = ""
    cost: Cost
    reference: bool = Field(
        default=False,
        description=(
            "A retrieval reference (BM25 over a corpus): ``works`` is its whole ranked "
            "list, there is no screening and no result, and a gold paper outside it "
            "was never retrieved."
        ),
    )
    result: ResearchResult | None = Field(
        default=None,
        description=(
            "v2 only: the run's evidence table (passages, criteria and the log "
            "left out), which the stability and planted-source measures read."
        ),
    )
    log: RunLog | None = Field(
        default=None,
        exclude=True,
        description=(
            "v2 only: the run log, kept out of every dump (the registry would "
            "grow by megabytes); an experiment writes it as files of its own."
        ),
    )


class AnswerArm(Protocol):
    """A system that answers a question."""

    name: str
    # True when ``works`` are web pages (a page title wraps a paper's), not works.
    returns_pages: bool

    async def run(self, question: str) -> ArmOutput: ...


class ExperimentArm(AnswerArm, Protocol):
    """An answer arm an experiment can run on frozen inputs.

    ``frozen`` is the snapshot to replay (None: run live) and ``store`` the
    path of the run's own working store; the legacy pipeline accepts both and
    refuses a snapshot.
    """

    async def run(  # type: ignore[override]
        self,
        question: str,
        *,
        frozen: FrozenInputs | None = None,
        store: Path | None = None,
    ) -> ArmOutput: ...


# -- legacy -------------------------------------------------------------------------


def _web_work(url: str, title: str) -> Work | None:
    """A page as a ``Work``, or None for a URL that is not one."""
    try:
        return Work(url=url, title=title)
    except ValueError:
        return None


def map_legacy(result: LegacyResult, *, wall_s: float) -> ArmOutput:
    """The legacy result as an :class:`ArmOutput`.

    A claim is an evidence item (else a key finding); its evidence is the window
    of a fetched page that shares most with it. An evidence item whose source
    page was never fetched has nothing to check against and is counted
    ``unsourced``. The legacy pipeline counts no tokens or calls: those costs
    are None.
    """
    output = result.output
    pages = {(canonical_url(p.url) or p.url): p for p in result.fetched_pages}
    claims: list[ClaimEvidence] = []
    unsourced = 0
    if output.evidence_items:
        for item in output.evidence_items:
            page = pages.get(canonical_url(item.source_url) or item.source_url)
            if page is None:
                unsourced += 1
                continue
            claims.append(
                ClaimEvidence(
                    claim=item.finding,
                    evidence=best_window(item.finding, [page.content]),
                )
            )
    else:
        contents = [p.content for p in result.fetched_pages]
        for finding in output.key_findings:
            if not contents:
                unsourced += 1
                continue
            claims.append(
                ClaimEvidence(claim=finding, evidence=best_window(finding, contents))
            )
    answer = "\n".join([output.evidence_summary, *output.key_findings])
    cited = [*output.sources, *(i.source_url for i in output.evidence_items)]
    urls = list(dict.fromkeys([*cited, *extract_urls(answer)]))
    titles = {key: p.title for key, p in pages.items()}
    works: list[Work] = []
    seen: set[str] = set()
    for url in [*output.sources, *pages]:
        key = canonical_url(url) or url
        if key in seen:
            continue
        seen.add(key)
        work = _web_work(url, titles.get(key, ""))
        if work is not None:
            works.append(work)
    degraded = [output.degraded_reason] if output.degraded else []
    return ArmOutput(
        answer_text=answer,
        claims=claims,
        n_unsourced=unsourced,
        urls=urls,
        works=works,
        degraded=degraded,
        stop_reason=f"iterations:{result.iterations}",
        cost=Cost(wall_s=wall_s),
    )


class LegacyArm:
    """The legacy ``run_research`` pipeline, web only."""

    name = "legacy"
    returns_pages = True

    def __init__(
        self,
        *,
        model: str,
        searxng_url: str,
        decision_model: DecisionModel | None = None,
        max_iterations: int = 3,
        max_results: int = 10,
        max_pages: int = 5,
    ) -> None:
        self.model = model
        self.searxng_url = searxng_url
        self.decision_model = decision_model
        self.max_iterations = max_iterations
        self.max_results = max_results
        self.max_pages = max_pages

    async def run(
        self,
        question: str,
        *,
        frozen: FrozenInputs | None = None,
        store: Path | None = None,
    ) -> ArmOutput:
        if frozen is not None:
            raise ValueError(
                "the legacy pipeline cannot take frozen inputs; run it live"
            )
        started = time.monotonic()
        result = await run_research(
            question,
            model=self.model,
            searxng_url=self.searxng_url,
            max_iterations=self.max_iterations,
            max_results=self.max_results,
            max_pages=self.max_pages,
            decision_model=self.decision_model,
        )
        return map_legacy(result, wall_s=time.monotonic() - started)


# -- v2 -----------------------------------------------------------------------------


def report_claims(result: ResearchResult) -> tuple[list[ClaimEvidence], int]:
    """Each cited sentence of the report with the quotes of the findings it cites.

    Returns ``(claims, n_unsourced)``. A sentence that cites no finding, or a
    finding that does not exist, has nothing to check against and is counted
    unsourced; headings, table rows and fragments of fewer than
    ``MIN_SENTENCE_WORDS`` words are not claims.
    """
    quotes = {f.finding_id: f.quote for f in result.findings}
    claims: list[ClaimEvidence] = []
    unsourced = 0
    for line in result.report_markdown.splitlines():
        text = _LIST_MARKER.sub("", line.strip())
        if not text or text.startswith(("#", "|", ">")):
            continue
        for sentence in _SENTENCE_BREAK.split(text):
            plain = _CITATION.sub("", sentence).strip()
            if len(plain.split()) < MIN_SENTENCE_WORDS:
                continue
            ids = [
                token
                for group in _FINDING_REFERENCE.findall(sentence)
                for token in re.findall(r"F\d+", group)
            ]
            if not ids or any(i not in quotes for i in ids):
                unsourced += 1
                continue
            claims.append(
                ClaimEvidence(claim=plain, evidence="\n".join(quotes[i] for i in ids))
            )
    return claims, unsourced


def map_v2(result: ResearchResult) -> ArmOutput:
    """The v2 result as an :class:`ArmOutput`."""
    claims, unsourced = report_claims(result)
    by_id = {w.work_id: w for w in result.works}
    cited_links = [
        f"https://doi.org/{w.doi}" if w.doi else w.url
        for w in (by_id[i] for i in result.cited_work_ids() if i in by_id)
        if w.doi or w.url
    ]
    urls = list(
        dict.fromkeys(
            [*extract_urls(result.report_markdown), *(u for u in cited_links if u)]
        )
    )
    stats = result.stats
    return ArmOutput(
        answer_text=result.report_markdown,
        claims=claims,
        n_unsourced=unsourced,
        urls=urls,
        works=list(result.works),
        degraded=list(result.degraded),
        stop_reason=result.stop_reason,
        cost=Cost(
            wall_s=stats.wall_s,
            llm_calls=stats.llm_calls,
            input_tokens=stats.input_tokens,
            output_tokens=stats.output_tokens,
            decision_calls=stats.decision_calls,
        ),
        result=result.model_copy(
            update={"passages": [], "criteria": {}, "log": RunLog(run_id=result.run_id)}
        ),
        log=result.log,
    )


class V2Arm:
    """v2's ``research()`` in one mode.

    ``decision_model`` is a name (opened and closed by each run) or an opened
    handle (shared by every run). ``store`` defaults to ``None``: nothing
    persists between questions. ``source_client`` is for offline tests (it
    must belong to ``store``). ``config`` is the strategy preset (the
    baseline unless one is given). A run may be given ``frozen`` inputs (Plan and Search
    from a snapshot) and the ``store`` path that holds the snapshot's cache;
    both override the arm's own for that run. ``http_cache`` is a shared
    read-through HTTP cache for the live runs (not for a frozen one, which
    reads only its snapshot). ``shared_decisions`` is a store the runs share
    their decision answers through (``research(shared_decisions=)``).
    ``embedding_model`` goes to a configuration that ranks by dense
    similarity (the ``hybrid`` preset needs one).
    """

    returns_pages = False

    def __init__(
        self,
        *,
        mode: Mode,
        model: str | Model,
        sources: SourceConfig,
        decision_model: str | DecisionModel | None = None,
        budget: Budget | None = None,
        store: Path | ResearchStore | None = None,
        source_client: SourceClient | None = None,
        config: ResearchConfig = PRESETS["baseline"],
        name: str | None = None,
        http_cache: ResearchStore | None = None,
        shared_decisions: ResearchStore | None = None,
        embedding_model: str | Embedder | None = None,
    ) -> None:
        self.name = name or f"v2-{mode}"
        self.embedding_model = embedding_model
        self.http_cache = http_cache
        self.shared_decisions = shared_decisions
        self.mode: Mode = mode
        self.model = model
        self.decision_model = decision_model
        self.sources = sources
        self.budget = budget or Budget()
        self.store = store
        self.source_client = source_client
        self.config = config

    async def run(
        self,
        question: str,
        *,
        frozen: FrozenInputs | None = None,
        store: Path | None = None,
    ) -> ArmOutput:
        use_store = store if store is not None else self.store
        # a frozen run reads only its snapshot, and an open store brings its own cache
        cache = (
            self.http_cache
            if frozen is None and not isinstance(use_store, ResearchStore)
            else None
        )
        result = await research(
            question,
            mode=self.mode,
            model=self.model,
            decision_model=self.decision_model,
            embedding_model=self.embedding_model,
            store=use_store,
            budget=self.budget,
            sources=self.sources,
            config=self.config,
            frozen=frozen,
            source_client=self.source_client,
            http_cache=cache,
            shared_decisions=self.shared_decisions,
        )
        return map_v2(result)


# -- reference ------------------------------------------------------------------------


class Bm25Arm:
    """BM25 over a local corpus with the raw question as the query: the
    retrieval reference of a local-corpus set. No model, no network, no
    screening; the answer is the first ``top_k`` records of the corpus' BM25
    ranking, in rank order. It is what plain lexical retrieval achieves, so an
    arm that spends language-model calls on query writing and screening can be
    read against it."""

    returns_pages = False

    def __init__(self, *, corpus_dir: Path, top_k: int, name: str = "bm25-question"):
        self.name = name
        self.top_k = top_k
        self._source = LocalCorpusSource(corpus_dir=corpus_dir)

    async def run(
        self,
        question: str,
        *,
        frozen: FrozenInputs | None = None,
        store: Path | None = None,
    ) -> ArmOutput:
        if frozen is not None:
            raise ValueError(
                "the BM25 reference reads the corpus; it takes no snapshot"
            )
        started = time.monotonic()
        found = await self._source.search(question, limit=self.top_k)
        return ArmOutput(
            answer_text="",
            works=[c.work for c in found],
            stop_reason="bm25",
            cost=Cost(
                wall_s=time.monotonic() - started,
                llm_calls=0,
                input_tokens=0,
                output_tokens=0,
                decision_calls=0,
            ),
            reference=True,
        )
