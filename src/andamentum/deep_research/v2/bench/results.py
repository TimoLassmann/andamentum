"""What a benchmark run records, how it is summarised, and the markdown table.

One file per run (``WebRun``, ``LitSearchRun`` or ``ScreeningRun``): the
configuration, every item with its metrics, and a summary computed from the
items. An item that raised is kept with its error and counts against the arm
(a missed answer, zero recall), never dropped, so a crashing arm cannot look
better than a working one. ``render_markdown`` puts any number of run files
side by side.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

from ..types import Criterion
from .metrics import (
    DEFAULT_KS,
    ClaimSupport,
    Cost,
    RetrievalScore,
    ScreeningScore,
    UrlIntegrity,
    mean_or_none,
    total_cost,
)


class RunConfig(BaseModel):
    """How a run was set up, so a table can state what it compares."""

    arm: str
    model: str | None = None
    decision_model: str | None = None
    judge: str | None = None
    subset_size: int = Field(ge=0)
    seed: int = 0
    started_at: float = 0.0
    notes: list[str] = Field(default_factory=list)


# -- web ----------------------------------------------------------------------------


class WebItem(BaseModel):
    question_id: str
    question: str
    expected: str
    correct: bool = False
    answer_excerpt: str = ""
    claim_support: ClaimSupport | None = None
    links: UrlIntegrity | None = None
    cost: Cost | None = None
    stop_reason: str = ""
    degraded: list[str] = Field(default_factory=list)
    error: str | None = None


class WebSummary(BaseModel):
    n: int
    n_errors: int
    accuracy: float | None
    claim_precision: float | None = Field(
        description="Supported / judged claims, pooled over the answers; None if not measured."
    )
    n_judged: int
    n_unsourced: int
    link_integrity: float | None = Field(
        description="ok / (ok + dead) links, pooled; None if not measured."
    )
    n_links: int
    n_degraded: int
    cost: Cost


class WebRun(BaseModel):
    task: Literal["web"] = "web"
    config: RunConfig
    items: list[WebItem]
    summary: WebSummary


def summarise_web(items: Sequence[WebItem]) -> WebSummary:
    judged = [i.claim_support for i in items if i.claim_support is not None]
    links = [i.links for i in items if i.links is not None]
    n_judged = sum(c.n_judged for c in judged)
    settled = sum(link.ok + link.dead for link in links)
    return WebSummary(
        n=len(items),
        n_errors=sum(1 for i in items if i.error),
        accuracy=mean_or_none([1.0 if i.correct else 0.0 for i in items]),
        claim_precision=(
            sum(c.n_supported for c in judged) / n_judged if n_judged else None
        ),
        n_judged=n_judged,
        n_unsourced=sum(c.n_unsourced for c in judged),
        link_integrity=(sum(link.ok for link in links) / settled if settled else None),
        n_links=sum(link.n for link in links),
        n_degraded=sum(1 for i in items if i.degraded),
        cost=total_cost([i.cost for i in items if i.cost is not None]),
    )


# -- litsearch ----------------------------------------------------------------------


class LitSearchResultItem(BaseModel):
    query_id: str
    query: str
    n_gold: int
    retrieval: RetrievalScore | None = None
    cost: Cost | None = None
    stop_reason: str = ""
    degraded: list[str] = Field(default_factory=list)
    error: str | None = None


class LitSearchSummary(BaseModel):
    n: int
    n_errors: int
    ks: list[int]
    recall_at: dict[int, float | None]
    found_any: float | None = Field(
        description="Share of queries with at least one gold paper in the result."
    )
    kept_precision: float | None = Field(
        description="Mean over queries that kept something."
    )
    mean_kept_n: float | None
    n_degraded: int
    cost: Cost


class LitSearchRun(BaseModel):
    task: Literal["litsearch"] = "litsearch"
    config: RunConfig
    items: list[LitSearchResultItem]
    summary: LitSearchSummary


def summarise_litsearch(
    items: Sequence[LitSearchResultItem], *, ks: Sequence[int] = DEFAULT_KS
) -> LitSearchSummary:
    scored = [i.retrieval for i in items if i.retrieval is not None]
    return LitSearchSummary(
        n=len(items),
        n_errors=sum(1 for i in items if i.error),
        ks=list(ks),
        # an errored query has no retrieval: it counts as recall 0
        recall_at={
            k: mean_or_none(
                [(i.retrieval.recall_at[k] if i.retrieval else 0.0) for i in items]
            )
            for k in ks
        },
        found_any=mean_or_none(
            [
                1.0
                if i.retrieval and any(r is not None for r in i.retrieval.gold_ranks)
                else 0.0
                for i in items
            ]
        ),
        kept_precision=mean_or_none(
            [s.kept_precision for s in scored if s.kept_precision is not None]
        ),
        mean_kept_n=mean_or_none([float(s.kept_n) for s in scored]),
        n_degraded=sum(1 for i in items if i.degraded),
        cost=total_cost([i.cost for i in items if i.cost is not None]),
    )


# -- screening ----------------------------------------------------------------------


class ScreeningItem(BaseModel):
    review_key: str
    question: str
    n_records: int
    n_records_total: int
    score: ScreeningScore | None = None
    criteria: list[Criterion] | None = None
    cost: Cost | None = None
    error: str | None = None


class ScreeningSummary(BaseModel):
    n_reviews: int
    n_errors: int
    n_records: int
    prevalence: float | None
    kept_recall: float | None
    kept_precision: float | None
    work_saved_kept: float | None
    work_saved_at_95: float | None
    wss_at_95: float | None
    cost: Cost


class ScreeningRun(BaseModel):
    task: Literal["screening"] = "screening"
    config: RunConfig
    items: list[ScreeningItem]
    summary: ScreeningSummary


def summarise_screening(items: Sequence[ScreeningItem]) -> ScreeningSummary:
    """Macro means over the reviews that were scored."""
    scored = [i.score for i in items if i.score is not None]

    def mean(values: list[float]) -> float | None:
        return mean_or_none(values)

    return ScreeningSummary(
        n_reviews=len(items),
        n_errors=sum(1 for i in items if i.error),
        n_records=sum(i.n_records for i in items),
        prevalence=mean([s.prevalence for s in scored]),
        kept_recall=mean([s.kept_recall for s in scored]),
        kept_precision=mean(
            [s.kept_precision for s in scored if s.kept_precision is not None]
        ),
        work_saved_kept=mean([s.work_saved_kept for s in scored]),
        work_saved_at_95=mean([s.work_saved_at_95 for s in scored]),
        wss_at_95=mean([s.wss_at_95 for s in scored]),
        cost=total_cost([i.cost for i in items if i.cost is not None]),
    )


BenchRun = Annotated[WebRun | LitSearchRun | ScreeningRun, Field(discriminator="task")]
_RUN_ADAPTER: TypeAdapter[WebRun | LitSearchRun | ScreeningRun] = TypeAdapter(BenchRun)


def load_run(text: str) -> WebRun | LitSearchRun | ScreeningRun:
    """Parse one results file."""
    return _RUN_ADAPTER.validate_json(text)


def dump_run(run: WebRun | LitSearchRun | ScreeningRun) -> str:
    return run.model_dump_json(indent=2)


# -- markdown -------------------------------------------------------------------------


def _fmt(value: float | None, *, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _count(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _models(config: RunConfig) -> str:
    parts = [
        f"{label} {value}"
        for label, value in (
            ("model", config.model),
            ("decision model", config.decision_model),
            ("judge", config.judge),
        )
        if value
    ]
    return ", ".join(parts) or "no model"


def _web_table(runs: list[WebRun]) -> str:
    rows = [
        [
            r.config.arm,
            str(r.summary.n),
            str(r.summary.n_errors),
            _fmt(r.summary.accuracy),
            f"{_fmt(r.summary.claim_precision)} ({r.summary.n_judged} judged, "
            f"{r.summary.n_unsourced} unsourced)",
            f"{_fmt(r.summary.link_integrity)} ({r.summary.n_links} links)",
            _fmt(
                r.summary.cost.wall_s / r.summary.n if r.summary.n else None, digits=1
            ),
            _count(r.summary.cost.llm_calls),
            _count(r.summary.cost.decision_calls),
        ]
        for r in runs
    ]
    return _table(
        [
            "arm",
            "questions",
            "errors",
            "answer correct",
            "claim-support precision",
            "URL integrity",
            "s per question",
            "LLM calls",
            "decision calls",
        ],
        rows,
    )


def _litsearch_table(runs: list[LitSearchRun]) -> str:
    ks = sorted({k for r in runs for k in r.summary.ks})
    rows = [
        [
            r.config.arm,
            str(r.summary.n),
            str(r.summary.n_errors),
            *[_fmt(r.summary.recall_at.get(k)) for k in ks],
            _fmt(r.summary.found_any),
            _fmt(r.summary.kept_precision, digits=3),
            _fmt(r.summary.mean_kept_n, digits=1),
            _fmt(
                r.summary.cost.wall_s / r.summary.n if r.summary.n else None, digits=1
            ),
        ]
        for r in runs
    ]
    return _table(
        [
            "arm",
            "queries",
            "errors",
            *[f"recall@{k}" for k in ks],
            "any gold found",
            "kept-set precision",
            "kept works",
            "s per query",
        ],
        rows,
    )


def _screening_table(runs: list[ScreeningRun]) -> str:
    rows = [
        [
            r.config.arm,
            str(r.summary.n_reviews),
            str(r.summary.n_errors),
            str(r.summary.n_records),
            _fmt(r.summary.prevalence),
            _fmt(r.summary.kept_recall),
            _fmt(r.summary.kept_precision),
            _fmt(r.summary.work_saved_kept),
            _fmt(r.summary.work_saved_at_95),
            _fmt(r.summary.wss_at_95),
            _count(r.summary.cost.llm_calls),
            _count(r.summary.cost.decision_calls),
        ]
        for r in runs
    ]
    return _table(
        [
            "arm",
            "reviews",
            "errors",
            "records",
            "prevalence",
            "kept recall",
            "kept precision",
            "work saved (kept)",
            "work saved @95% recall",
            "WSS@95",
            "LLM calls",
            "decision calls",
        ],
        rows,
    )


def _setup_lines(runs: Sequence[WebRun | LitSearchRun | ScreeningRun]) -> str:
    lines = []
    for run in runs:
        note = f"; {'; '.join(run.config.notes)}" if run.config.notes else ""
        lines.append(
            f"- `{run.config.arm}`: {_models(run.config)}; subset {run.config.subset_size}, "
            f"seed {run.config.seed}{note}"
        )
    return "\n".join(lines)


def render_markdown(runs: Sequence[WebRun | LitSearchRun | ScreeningRun]) -> str:
    """The tables of every task present in *runs*, then each run's setup."""
    web = [r for r in runs if isinstance(r, WebRun)]
    lit = [r for r in runs if isinstance(r, LitSearchRun)]
    screening = [r for r in runs if isinstance(r, ScreeningRun)]
    sections: list[str] = []
    if web:
        sections.append("## Web questions\n\n" + _web_table(web))
    if lit:
        sections.append("## LitSearch retrieval\n\n" + _litsearch_table(lit))
    if screening:
        sections.append(
            "## Systematic-review screening\n\n" + _screening_table(screening)
        )
    if not sections:
        raise ValueError("no runs to report")
    sections.append("## Setup\n\n" + _setup_lines(runs))
    return "\n\n".join(sections) + "\n"
