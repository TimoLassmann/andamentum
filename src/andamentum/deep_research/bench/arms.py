"""The systems under test, behind one small interface.

An answer arm takes a question and returns an :class:`ArmOutput`: the answer
text, the claims with the evidence behind them, the links, the ranked works,
and what the run cost. Three systems are mapped onto this record, so every
metric is computed by one piece of code from the same fields:

- ``research``: the engine's ``research()`` (``ResearchArm``);
- ``finder``: ``finder.find_works`` with or without citation following
  (``FinderArm``); the question is the sentence;
- ``s2_search``: one plain Semantic Scholar search with the question's text,
  its top results in the order the service gave (``SemanticScholarArm``).

Every arm runs with a fresh in-memory store (``store=None``), so nothing
persists between questions and every question pays for its own searches.
"""

from __future__ import annotations

import re
import time
from typing import Protocol

from pydantic import BaseModel, Field
from pydantic_ai.models import Model

from andamentum.references import Work

from ..engine import research
from ..finder import FINDER_SOURCES, FinderResult, find_works
from ..search import search
from ..sources.base import SourceClient, SourceConfig
from ..sources.semantic_scholar import SemanticScholarSource
from ..store import ResearchStore
from ..types import _FINDING_REFERENCE, Budget, Mode, ResearchResult, RunLog
from .judging import ClaimEvidence
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
    result: ResearchResult | None = Field(
        default=None,
        description=(
            "Research arms only: the run's evidence table (passages and the log left out)."
        ),
    )
    finder: FinderResult | None = Field(
        default=None,
        description=(
            "Finder arms only: what each step kept (work ids of ``works``; the "
            "result's own works list and log left out)."
        ),
    )
    log: RunLog | None = Field(
        default=None,
        exclude=True,
        description=(
            "Research arms only: the run log, kept out of every dump (the registry would "
            "grow by megabytes); an experiment writes it as files of its own."
        ),
    )


class AnswerArm(Protocol):
    """A system that answers a question."""

    name: str

    async def run(self, question: str) -> ArmOutput: ...


# -- the research arm ------------------------------------------------------------


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


def map_research(result: ResearchResult) -> ArmOutput:
    """The research result as an :class:`ArmOutput`."""
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
        ),
        result=result.model_copy(
            update={"passages": [], "log": RunLog(run_id=result.run_id)}
        ),
        log=result.log,
    )


class ResearchArm:
    """The engine's ``research()`` in one mode.

    The run keeps nothing between questions (``store=None``). ``http_cache`` is
    a shared read-through HTTP cache for the runs.
    """

    def __init__(
        self,
        *,
        mode: Mode,
        model: str | Model,
        sources: SourceConfig,
        budget: Budget | None = None,
        http_cache: ResearchStore | None = None,
        name: str | None = None,
    ) -> None:
        self.name = name or f"research-{mode}"
        self.mode: Mode = mode
        self.model = model
        self.sources = sources
        self.budget = budget or Budget()
        self.http_cache = http_cache

    async def run(self, question: str) -> ArmOutput:
        result = await research(
            question,
            mode=self.mode,
            model=self.model,
            store=None,
            budget=self.budget,
            sources=self.sources,
            http_cache=self.http_cache,
        )
        return map_research(result)


# -- the finder arm ----------------------------------------------------------------


def map_finder(result: FinderResult) -> ArmOutput:
    """The finder's result as an :class:`ArmOutput`."""
    stats = result.stats
    return ArmOutput(
        answer_text="",
        works=list(result.works),
        degraded=list(result.degraded),
        cost=Cost(
            wall_s=stats.wall_s,
            llm_calls=stats.llm_calls,
            input_tokens=stats.input_tokens,
            output_tokens=stats.output_tokens,
        ),
        finder=result.model_copy(update={"works": []}),
        log=result.log,
    )


class FinderArm:
    """``find_works`` on the question as the sentence (no paragraph).

    The finder's own sources (``finder.FINDER_SOURCES``: Semantic Scholar,
    OpenAlex and Europe PMC searched, Semantic Scholar's citation links) with
    the keys and address of *sources*.
    """

    def __init__(
        self,
        *,
        model: str | Model,
        follow_citations: bool,
        sources: SourceConfig,
        http_cache: ResearchStore | None = None,
        name: str | None = None,
    ) -> None:
        self.name = name or ("finder" if follow_citations else "finder-no-follow")
        self.model = model
        self.follow_citations = follow_citations
        self.sources = sources.model_copy(
            update={
                "scholarly": FINDER_SOURCES.scholarly,
                "expand_with": FINDER_SOURCES.expand_with,
            }
        )
        self.http_cache = http_cache

    async def run(self, question: str) -> ArmOutput:
        result = await find_works(
            question,
            model=self.model,
            follow_citations=self.follow_citations,
            sources=self.sources,
            http_cache=self.http_cache,
        )
        return map_finder(result)


# -- the plain Semantic Scholar arm -------------------------------------------------

S2_TOP = 20


class SemanticScholarArm:
    """One Semantic Scholar search with the question's text, the top ``S2_TOP``
    in the service's order. No model. A failed search is recorded in
    ``degraded`` in ``search.search``'s words (the bench reads it as an outage)."""

    def __init__(
        self,
        *,
        sources: SourceConfig,
        http_cache: ResearchStore | None = None,
        top: int = S2_TOP,
        name: str | None = None,
    ) -> None:
        self.name = name or "s2-search"
        self.sources = sources
        self.http_cache = http_cache
        self.top = top

    async def run(self, question: str) -> ArmOutput:
        started = time.monotonic()
        store = ResearchStore.in_memory(shared_cache=self.http_cache)
        client = SourceClient(store)
        try:
            source = SemanticScholarSource(
                client=client, api_key=self.sources.semantic_scholar_api_key
            )
            found = await search(
                [question],
                subquestion_id="s2",
                sources=[source],
                filters=None,
                limit=self.top,
            )
        finally:
            await client.aclose()
            store.close()
        ordered = sorted(
            found.candidates, key=lambda c: c.source_ranks.get(source.name, 0)
        )
        return ArmOutput(
            answer_text="",
            works=[c.work for c in ordered][: self.top],
            degraded=found.degraded,
            cost=Cost(wall_s=time.monotonic() - started, llm_calls=0),
        )
