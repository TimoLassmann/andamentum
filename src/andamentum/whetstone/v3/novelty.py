"""Novelty check for v3 — 3-stage deterministic pipeline.

Split from v2's single-node-with-tool shape into three explicit graph
nodes (FlagNoveltyTargets → RunNoveltySearches → JudgeNovelty). The
agent-with-tool shape v2 used worked but had three problems:

  * variable per-run cost (agent decided how many tool calls to make)
  * opaque audit trail (agent's tool-use was not separately observable)
  * difficult to benchmark (no reproducible flag-set + budget)

The three-node split gives reproducible flag-set, bounded search budget,
and explicit cost shape: 1 LLM call (extract) + N deep_research calls
(one per target, max novelty_target_cap=8 by default) + 0 LLM calls in
the pure adapter (judge_novelty).

Findings flow into ``state.findings`` as regular ``Finding`` objects
with ``category="novelty"`` and ``criterion="novelty"``. Renderers,
Gate, and Synthesise need zero novelty-awareness.

Only contradicted claims (is_novel=False) become findings; the v2
"silence on confirmation" invariant is preserved.

No on-disk cache. The v2 ``~/.cache/whetstone/novelty/`` cache is
intentionally dropped — per the project rule against hidden home-dir
state. Repeated runs re-query deep_research; users who want caching
should run novelty at a coarser cadence than the full review loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal, cast

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from andamentum.core.models import resolve_model

from .model import Section
from .review import Finding

logger = logging.getLogger("andamentum.whetstone.v3")


_DEFAULT_TARGET_CAP: int = 8
_DEFAULT_SEARCH_DEPTH: int = 2


# ── Extractor agent I/O ─────────────────────────────────────────────────────


class NoveltyTarget(BaseModel):
    """One novelty claim extracted from the manuscript for verification."""

    claim_text: str = Field(
        description=(
            "Verbatim novelty claim from the manuscript — the sentence "
            "asserting first/novel/unique. Max ~300 chars; paraphrase if "
            "longer."
        ),
        max_length=400,
    )
    short_summary: str = Field(
        description=(
            "One-sentence summary of WHAT is claimed novel. Used as the "
            "search query, so it should read as a self-contained claim."
        )
    )
    why_load_bearing: str = Field(
        description=(
            "One sentence on why verifying this claim matters — e.g. "
            "abstract claim, results headline, contribution statement."
        )
    )
    origin_section_id: str | None = None


class NoveltyTargetList(BaseModel):
    targets: list[NoveltyTarget] = Field(default_factory=list)


_EXTRACTOR_PROMPT = (
    "You are reading a draft manuscript to identify the EXPLICIT NOVELTY "
    "CLAIMS the author is staking — sentences where the author asserts "
    "they have done something new.\n\n"
    "A novelty claim is a sentence where the author tells the reader "
    "that this work is the FIRST to do X, presents a NOVEL Y, introduces "
    "the first Z, or otherwise asserts originality. Not every assertion "
    "is a novelty claim — only the ones the author would defend if asked "
    "'what's new about this paper?'\n\n"
    "Return 3-5 of the most LOAD-BEARING novelty claims (the ones the "
    "rest of the paper depends on). For each:\n"
    "  - claim_text: the verbatim sentence (≤300 chars, paraphrased if "
    "longer)\n"
    "  - short_summary: one self-contained sentence about WHAT is novel "
    "(becomes the search query)\n"
    "  - why_load_bearing: one sentence on why verifying matters\n\n"
    "If the manuscript stakes no novelty claims, return an empty list. "
    "Returning an empty list is honest; padding with weak claims would "
    "waste downstream LLM calls."
)


# ── Per-target evidence + verdict ──────────────────────────────────────────


class SimilarWorkRef(BaseModel):
    """One piece of prior work surfaced by the novelty search."""

    title: str
    url: str
    relevance: Literal["direct", "partial", "tangential"] = "partial"
    summary: str = ""


class NoveltyEvidence(BaseModel):
    """One target's deep_research result, normalised. ``error`` is set
    when the per-target check crashed; the rest of the pipeline treats
    that target as 'no signal' (verdict skipped, no Finding emitted)."""

    target: NoveltyTarget
    is_novel: bool = True
    confidence: float = 0.0
    assessment: str = ""
    similar_work: list[SimilarWorkRef] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    search_queries_used: list[str] = Field(default_factory=list)
    error: str | None = None


class NoveltyVerdict(BaseModel):
    """Per-target verdict ready for Finding adaptation. ``severity`` is
    None when ``is_novel=True`` (no Finding will be emitted)."""

    target: NoveltyTarget
    is_novel: bool
    severity: Literal["minor", "moderate", "major"] | None
    confidence_band: Literal["low", "medium", "high"]
    rationale: str


# ── Node 1: extract targets ────────────────────────────────────────────────


async def flag_novelty_targets(
    sections: list[Section],
    source: str,
    *,
    agent_model: str,
    cap: int = _DEFAULT_TARGET_CAP,
) -> list[NoveltyTarget]:
    """One LLM call → up to ``cap`` NoveltyTarget objects. Returns [] on
    agent crash (per-run failure isolation; novelty being unavailable is
    a soft fail, not a pipeline crash)."""
    if not source.strip():
        return []
    try:
        agent: Agent[None, NoveltyTargetList] = Agent(
            resolve_model(agent_model),
            output_type=NoveltyTargetList,
            instructions=_EXTRACTOR_PROMPT,
        )
        result = await agent.run(
            f"MANUSCRIPT:\n--- BEGIN ---\n{source}\n--- END ---\n\n"
            f"Extract up to {cap} of the most load-bearing novelty claims."
        )
        raw = cast(NoveltyTargetList, result.output).targets[:cap]
    except Exception as exc:
        logger.warning(
            "[v3.novelty] target extractor crashed (%s); skipping novelty check",
            exc,
        )
        return []

    # Anchor each target to its origin section by short_summary substring
    # match (best-effort — used only to scope the finding's
    # sections_involved field; missing anchors are not an error).
    enriched: list[NoveltyTarget] = []
    for target in raw:
        origin_id: str | None = None
        for sec in sections:
            if target.claim_text and target.claim_text[:60] in sec.text:
                origin_id = sec.id
                break
        enriched.append(target.model_copy(update={"origin_section_id": origin_id}))
    logger.info("[v3.novelty] %d targets extracted", len(enriched))
    return enriched


# ── Node 2: search per target ──────────────────────────────────────────────


async def _check_one_target(
    target: NoveltyTarget,
    *,
    agent_model: str,
    search_depth: int,
) -> NoveltyEvidence:
    """Per-target deep_research call. Failures become NoveltyEvidence(
    error=...) — never raise — so the gather() loop never aborts."""
    try:
        from andamentum.deep_research import run_novelty_check

        report = await run_novelty_check(
            claim=target.short_summary,
            model=agent_model,
            search_depth=search_depth,
            verbose=False,
        )
    except Exception as exc:
        logger.warning(
            "[v3.novelty] target '%s...' deep_research crashed: %s",
            target.short_summary[:60],
            exc,
        )
        return NoveltyEvidence(target=target, error=str(exc))

    similar = [
        SimilarWorkRef(
            title=getattr(sw, "title", "(untitled)"),
            url=getattr(sw, "url", ""),
            relevance=_normalise_relevance(getattr(sw, "relevance", None)),
            summary=getattr(sw, "summary", "")[:400],
        )
        for sw in getattr(report, "similar_work", [])[:5]
    ]
    return NoveltyEvidence(
        target=target,
        is_novel=bool(getattr(report, "is_novel", True)),
        confidence=float(getattr(report, "confidence", 0.0)),
        assessment=str(getattr(report, "assessment", ""))[:600],
        similar_work=similar,
        sources=list(getattr(report, "sources", []))[:20],
        search_queries_used=list(getattr(report, "search_queries_used", [])),
    )


def _normalise_relevance(value) -> Literal["direct", "partial", "tangential"]:
    label = str(value).lower() if value is not None else "partial"
    if "direct" in label:
        return "direct"
    if "tangent" in label:
        return "tangential"
    return "partial"


async def run_novelty_searches(
    targets: list[NoveltyTarget],
    *,
    agent_model: str,
    search_depth: int = _DEFAULT_SEARCH_DEPTH,
) -> list[NoveltyEvidence]:
    """Parallel deep_research over targets, with per-target failure
    isolation. Order of results matches the order of ``targets``."""
    if not targets:
        return []
    return list(
        await asyncio.gather(
            *[
                _check_one_target(t, agent_model=agent_model, search_depth=search_depth)
                for t in targets
            ]
        )
    )


# ── Node 3: judge + adapt ──────────────────────────────────────────────────


def judge_novelty(evidence: list[NoveltyEvidence]) -> list[NoveltyVerdict]:
    """Pure. severity scaling: confidence ≥0.7 → major / high; ≥0.4 →
    moderate / medium; else minor / low. is_novel=True → severity None
    (no Finding will be emitted by verdicts_to_findings)."""
    verdicts: list[NoveltyVerdict] = []
    for ev in evidence:
        if ev.error is not None:
            # The check crashed — no verdict to report.
            continue
        if ev.is_novel:
            verdicts.append(
                NoveltyVerdict(
                    target=ev.target,
                    is_novel=True,
                    severity=None,
                    confidence_band="medium",
                    rationale="",
                )
            )
            continue
        if ev.confidence >= 0.7:
            severity, band = "major", "high"
        elif ev.confidence >= 0.4:
            severity, band = "moderate", "medium"
        else:
            severity, band = "minor", "low"
        similar_summary = ""
        if ev.similar_work:
            lines = "\n".join(
                f"  - {sw.title} ({sw.relevance}): {sw.summary[:200]}"
                for sw in ev.similar_work[:3]
            )
            similar_summary = f"\n\nSimilar work found:\n{lines}"
        rationale = (
            f'The author claims: "{ev.target.claim_text[:200]}". '
            f"Literature search ({band}-confidence) suggests this is not "
            f"novel. {ev.assessment[:300]}"
            f"{similar_summary}"
        )
        verdicts.append(
            NoveltyVerdict(
                target=ev.target,
                is_novel=False,
                severity=cast(Literal["minor", "moderate", "major"], severity),
                confidence_band=cast(Literal["low", "medium", "high"], band),
                rationale=rationale,
            )
        )
    return verdicts


def verdicts_to_findings(verdicts: list[NoveltyVerdict]) -> list[Finding]:
    """Adapt to ``v3.review.Finding`` (the type the rest of the v3 graph
    works in). The existing ``synth.py:_to_wfinding`` adapter then
    converts these into the public ``whetstone.schemas.Finding`` shape
    the renderers consume — no novelty-specific code needs to leak into
    Gate, Synthesise, or any renderer. Drops is_novel=True verdicts
    (silence on confirmation — v2 invariant)."""
    findings: list[Finding] = []
    for v in verdicts:
        if v.is_novel or v.severity is None:
            continue
        # `Finding(quote=...)` carries the claim's verbatim text; `span`
        # stays None (no source-anchor object for novelty findings — the
        # verdict is about external evidence, not a passage in the
        # source). The synthetic criterion name "Novelty" routes via the
        # existing _to_wfinding mapping (category="novelty",
        # title="Novelty", rationale=issue) without renderer changes.
        findings.append(
            Finding(
                criterion="Novelty",
                issue=v.rationale,
                quote=v.target.claim_text,
                severity=v.severity,
                span=None,
            )
        )
    return findings
