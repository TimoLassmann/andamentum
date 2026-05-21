"""Node: ReconcileClaims — cross-section claim substantiation.

Recovers the global reasoning a whole-document foundation model gives. Runs
after AuthorQuestions, before Consolidate (PRD:
docs/plans/2026-05-21-whetstone-claim-substantiation-prd.md).

  MAP    — extract the contribution claims each reviewable section makes about
           the work itself (every claim verbatim-anchored; unanchorable claims
           dropped, so the digest is hallucination-free).
  VERIFY — for each claim, read the FULL TEXT of the document's body sections
           and judge whether it is substantiated by the work's own data OR a
           citation. Support is a reasoning task, so we re-read the text rather
           than rely on any similarity shortcut.
  EMIT   — claims nothing substantiates become findings (low confidence,
           carrying the model's reason), flowing through Consolidate.

Reference/boilerplate sections (per the section classifier, recorded on the
state by CriticalRead) are excluded — no claims are extracted from them and
they don't form review context.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pydantic_graph import BaseNode, GraphRunContext

from ..agents import (
    ClaimSupport,
    SectionClaims,
    build_pydantic_ai_agent,
)
from ..agents.section_kinds import classify_section_kind
from ..anchoring import anchor_quote
from ..deps import ReviewDeps
from ..schemas import Finding, Quote, ReviewResult
from ..state import ReviewState

if TYPE_CHECKING:
    from ..structural.types import SectionRef
    from .consolidate import Consolidate


logger = logging.getLogger("andamentum.whetstone")

_MAX_CONCURRENT_EXTRACT = 4
_MAX_CONCURRENT_VERIFY = 6
# Section kinds whose full text forms the evidence context a claim is checked
# against (where results/data/citations live). Falls back to all reviewable
# sections when none of these are present (e.g. a non-academic document).
_EVIDENCE_KINDS = frozenset({"results", "methods", "discussion", "conclusion", "introduction", "abstract"})
# Flood safety: never emit more than this many substantiation findings from one
# run. These are low-confidence; the point is a useful signal, not burying the
# author. Logged loudly when hit.
_MAX_UNSUBSTANTIATED = 30


@dataclass
class _AnchoredClaim:
    text: str
    quote: Quote
    has_citation: bool
    section_id: str


@dataclass
class ReconcileClaims(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Find document-level unsubstantiated contribution claims."""

    async def run(
        self, ctx: GraphRunContext[ReviewState, ReviewDeps]
    ) -> "Consolidate":
        ctx.state.current_phase = "reconcile_claims"
        sections = _reviewable(ctx.state)
        if not sections:
            from .consolidate import Consolidate

            return Consolidate()

        # ── MAP: contribution claims per section, anchored ─────────────────
        claims = await _extract_claims(ctx.deps, sections)
        logger.info("[reconcile] %d contribution claim(s) extracted", len(claims))
        if not claims:
            from .consolidate import Consolidate

            return Consolidate()

        # ── VERIFY: each claim against the body sections' full text ────────
        evidence_context = _evidence_context(sections)
        unsupported = await _verify(ctx.deps, claims, evidence_context)
        ctx.state.llm_calls += len(claims)
        logger.info(
            "[reconcile] %d unsubstantiated claim(s) of %d",
            len(unsupported),
            len(claims),
        )

        # ── EMIT ───────────────────────────────────────────────────────────
        if len(unsupported) > _MAX_UNSUBSTANTIATED:
            logger.warning(
                "[reconcile] %d unsubstantiated claims exceeds cap %d — "
                "emitting the first %d",
                len(unsupported),
                _MAX_UNSUBSTANTIATED,
                _MAX_UNSUBSTANTIATED,
            )
            unsupported = unsupported[:_MAX_UNSUBSTANTIATED]
        for claim, reason in unsupported:
            ctx.state.challenged_findings.append(_to_finding(claim, reason))

        from .consolidate import Consolidate

        return Consolidate()


# ── Section selection ─────────────────────────────────────────────────────


def _reviewable(state: ReviewState) -> list["SectionRef"]:
    """Reviewable sections, reusing the classifier labels CriticalRead stored.
    Falls back to all sections if classification hasn't run."""
    ids = state.reviewable_section_ids
    if ids is None:
        return list(state.sections)
    return [s for s in state.sections if s.id in ids]


def _evidence_context(sections: list["SectionRef"]) -> str:
    """Full text of the body sections a claim is checked against."""
    body = [s for s in sections if classify_section_kind(s.title) in _EVIDENCE_KINDS]
    chosen = body or sections
    return "\n\n".join(
        f"=== {s.title} ===\n{s.text}" for s in chosen
    )


# ── MAP ─────────────────────────────────────────────────────────────────


async def _extract_claims(
    deps: ReviewDeps, sections: list["SectionRef"]
) -> list[_AnchoredClaim]:
    """Extract + anchor contribution claims from every reviewable section.
    Unanchorable claims are dropped (hallucination-free)."""
    sem = asyncio.Semaphore(_MAX_CONCURRENT_EXTRACT)

    async def one(section: "SectionRef") -> list[_AnchoredClaim]:
        async with sem:
            try:
                out = await _extract_section_claims(deps, section)
            except Exception as exc:
                logger.warning(
                    "[reconcile] claim extract for %s crashed: %s", section.id, exc
                )
                return []
        claims: list[_AnchoredClaim] = []
        for rc in out.claims:
            q = anchor_quote(rc.quote, section.text, section.id)
            if q is None:
                continue  # drop unanchorable
            claims.append(
                _AnchoredClaim(
                    text=rc.text,
                    quote=q,
                    has_citation=rc.has_citation,
                    section_id=section.id,
                )
            )
        return claims

    results = await asyncio.gather(*[one(s) for s in sections])
    return [c for cs in results for c in cs]


async def _extract_section_claims(
    deps: ReviewDeps, section: "SectionRef"
) -> SectionClaims:
    prompt = f"""SECTION ID: {section.id}
SECTION TITLE: {section.title}

SECTION TEXT — extract contribution claims; quote VERBATIM:
--- BEGIN ---
{section.text}
--- END ---"""
    agent = build_pydantic_ai_agent("digest_extractor", deps.model)
    result = await agent.run(prompt)
    return cast(SectionClaims, result.output)


# ── VERIFY ────────────────────────────────────────────────────────────────


async def _verify(
    deps: ReviewDeps,
    claims: list[_AnchoredClaim],
    evidence_context: str,
) -> list[tuple[_AnchoredClaim, str]]:
    """Verify each claim against the body text. Returns (claim, reason) for the
    unsubstantiated ones. A citation on the claim substantiates it outright."""
    sem = asyncio.Semaphore(_MAX_CONCURRENT_VERIFY)

    async def judge(claim: _AnchoredClaim) -> tuple[_AnchoredClaim, str] | None:
        if claim.has_citation:
            return None  # citation substantiates (data AND/OR citation)
        async with sem:
            try:
                verdict = await _verify_claim(deps, claim.text, evidence_context)
            except Exception as exc:
                logger.warning("[reconcile] verify crashed: %s — keeping", exc)
                return None  # don't flag on an LLM hiccup
        if verdict.supported:
            return None
        return claim, verdict.reason

    results = await asyncio.gather(*[judge(c) for c in claims])
    return [r for r in results if r is not None]


async def _verify_claim(
    deps: ReviewDeps, claim: str, evidence_context: str
) -> ClaimSupport:
    prompt = f"""CLAIM (made by the document about its own contribution):
{claim}

DOCUMENT BODY (its substantive sections, full text):
{evidence_context}

Is the claim substantiated by the document's own data or a citation?"""
    agent = build_pydantic_ai_agent("claim_support", deps.model)
    result = await agent.run(prompt)
    return cast(ClaimSupport, result.output)


# ── Emit ────────────────────────────────────────────────────────────────


def _to_finding(claim: _AnchoredClaim, reason: str) -> Finding:
    detail = f" {reason}" if reason else ""
    return Finding(
        title="Possibly unsubstantiated claim",
        severity="moderate",
        confidence="low",
        rationale=(
            f"This claim may not be supported by evidence or a citation in the "
            f"document: “{claim.text}”.{detail} Flagged by a cross-section "
            f"scan — confirm whether support exists."
        ),
        quotes=[claim.quote],
        sections_involved=[claim.section_id],
        source="investigate",
        perspective="claim_substantiation",
        category="substantiation",
    )
