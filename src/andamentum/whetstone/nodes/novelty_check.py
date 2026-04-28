"""Node: NoveltyCheck — verify the manuscript's novelty claims.

Reads the harvested markdown, extracts 3-5 explicit novelty claims via
``novelty_claim_extractor``, then routes each to deep_research's
``check_novelty`` for literature-overlap discovery. Each NoveltyReport
is adapted to a Finding so the result flows through the same renderers
as everything else.

Cost: 1 extraction call + (3-5) × deep_research runs + (3-5) ×
assessment calls. Roughly 10-15 LLM calls + N web fetches per run.
Disabled by default (``state.check_novelty=False``); the user opts in
via the ``check_novelty`` keyword on ``review_document`` or the
``--check-novelty`` CLI flag.

Caching: NoveltyReports are cached on disk per
``sha256(claim_short_summary)`` so re-running on the same draft is
cheap. Default cache dir: ``~/.cache/whetstone/novelty/``. Overridden
by ``state.novelty_cache_dir``.

Failure mode: per-claim try/except — a single claim's check failing
(network error, deep_research crash) doesn't abort the whole node. The
failed claim is logged and skipped; the remaining claims proceed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic_graph import BaseNode, GraphRunContext

from ..agents import (
    NoveltyClaim,
    NoveltyClaimList,
    build_pydantic_ai_agent,
)
from ..deps import ReviewDeps
from ..schemas import Finding, ReviewResult
from ..state import ReviewState

if TYPE_CHECKING:
    from .edit_sections import EditSections


logger = logging.getLogger("andamentum.whetstone")


@dataclass
class NoveltyCheck(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Verify the manuscript's novelty claims via deep_research."""

    async def run(
        self, ctx: GraphRunContext[ReviewState, ReviewDeps]
    ) -> "EditSections":
        ctx.state.current_phase = "novelty_check"

        # Pass-through if not opted in. Cheap when the flag is off.
        if not ctx.state.check_novelty:
            from .edit_sections import EditSections

            return EditSections()

        if not ctx.state.markdown.strip():
            logger.warning("[novelty_check] empty manuscript — skipping")
            from .edit_sections import EditSections

            return EditSections()

        try:
            claims = await _extract_claims(ctx.deps, ctx.state.markdown)
            ctx.state.llm_calls += 1
        except Exception as exc:
            logger.warning("[novelty_check] claim extraction crashed: %s", exc)
            from .edit_sections import EditSections

            return EditSections()

        if not claims.claims:
            logger.info("[novelty_check] no novelty claims extracted")
            from .edit_sections import EditSections

            return EditSections()

        logger.info(
            "[novelty_check] checking %d novelty claim(s) via deep_research",
            len(claims.claims),
        )

        cache_dir = (
            ctx.state.novelty_cache_dir
            if ctx.state.novelty_cache_dir is not None
            else Path.home() / ".cache" / "whetstone" / "novelty"
        )

        async def check_one(claim: NoveltyClaim) -> Finding | None:
            try:
                report_dict = await _check_one_claim(
                    claim=claim,
                    deps=ctx.deps,
                    cache_dir=cache_dir,
                    search_depth=ctx.state.novelty_search_depth,
                )
            except Exception as exc:
                logger.warning(
                    "[novelty_check] '%s...' crashed: %s",
                    claim.short_summary[:60],
                    exc,
                )
                return None
            return _report_to_finding(claim, report_dict)

        results = await asyncio.gather(
            *[check_one(c) for c in claims.claims], return_exceptions=False
        )
        for finding in results:
            if finding is not None:
                ctx.state.findings.append(finding)

        logger.info(
            "[novelty_check] done — %d novelty finding(s) added",
            sum(1 for r in results if r is not None),
        )

        from .edit_sections import EditSections

        return EditSections()


# ── Claim extraction ───────────────────────────────────────────────────


async def _extract_claims(deps: ReviewDeps, markdown: str) -> NoveltyClaimList:
    prompt = (
        "Extract 3-5 explicit novelty claims from the manuscript below.\n\n"
        "MANUSCRIPT:\n"
        "--- BEGIN ---\n"
        f"{markdown}\n"
        "--- END ---"
    )
    agent = build_pydantic_ai_agent("novelty_claim_extractor", deps.model)
    result = await agent.run(prompt)
    return cast(NoveltyClaimList, result.output)


# ── Per-claim novelty check ────────────────────────────────────────────


async def _check_one_claim(
    *,
    claim: NoveltyClaim,
    deps: ReviewDeps,
    cache_dir: Path,
    search_depth: int,
) -> dict[str, Any]:
    """Return a NoveltyReport-like dict for one claim, hitting cache when possible."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(claim.short_summary.encode("utf-8")).hexdigest()[:16]
    cache_path = cache_dir / f"{cache_key}.json"

    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception as exc:
            logger.warning("[novelty_check] cache read failed for %s: %s", cache_key, exc)
            # Fall through and recompute

    # Lazy import deep_research so v2 doesn't pay the import cost when novelty is off
    from andamentum.deep_research import check_novelty
    from andamentum.deep_research.orchestrator import run_research

    async def research_fn(*, query: str, max_iterations: int, verbose: bool) -> dict[str, Any]:
        result = await run_research(
            query=query,
            max_iterations=max_iterations,
            model=deps.model,
            verbose=verbose,
        )
        return {"output": result.output}

    async def assess_fn(claim_text, evidence_summary, key_findings, sources):
        from ..agents.novelty_assessor import NoveltyAssessment as NA  # type: ignore[attr-defined]
        from ..agents import build_pydantic_ai_agent as build

        prompt = (
            f"CLAIM: {claim_text}\n\n"
            f"EVIDENCE_SUMMARY:\n{evidence_summary}\n\n"
            f"KEY_FINDINGS:\n"
            + "\n".join(f"  - {f}" for f in key_findings)
            + "\n\nSOURCES:\n"
            + "\n".join(f"  - {s}" for s in sources)
        )
        agent = build("novelty_assessor", deps.model)
        result = await agent.run(prompt)
        return cast(NA, result.output)

    report = await check_novelty(
        claim=claim.short_summary,
        research_fn=research_fn,
        assess_fn=assess_fn,
        search_depth=search_depth,
        verbose=False,
    )

    # Convert dataclass to plain dict for JSON serialisation
    report_dict = asdict(report)
    # Fix Relevance enum → string for JSON
    for w in report_dict.get("similar_work", []):
        if hasattr(w.get("relevance"), "value"):
            w["relevance"] = w["relevance"].value

    try:
        cache_path.write_text(json.dumps(report_dict, indent=2))
    except Exception as exc:
        logger.warning("[novelty_check] cache write failed: %s", exc)

    return report_dict


# ── Adapter: NoveltyReport → Finding ───────────────────────────────────


def _report_to_finding(
    claim: NoveltyClaim, report: dict[str, Any]
) -> Finding | None:
    """Adapt a deep_research NoveltyReport (as dict) to a v2 Finding.

    Only surfaces findings that contradict the claim — i.e. cases where
    deep_research confidently reports prior work exists. Strong-novelty
    findings (deep_research confirms novelty) are not surfaced as
    Findings: they're a no-op for the author.
    """
    is_novel = report.get("is_novel", True)
    confidence = float(report.get("confidence", 0.5))
    assessment = report.get("assessment", "")
    similar_work = report.get("similar_work", [])

    # No finding if the claim looks novel.
    if is_novel:
        return None

    # Severity scales with confidence: high-confidence "not novel" is major
    if confidence >= 0.7:
        severity: str = "major"
        sev_label = "high-confidence"
    elif confidence >= 0.4:
        severity = "moderate"
        sev_label = "moderate-confidence"
    else:
        severity = "minor"
        sev_label = "low-confidence"

    similar_summary = ""
    if similar_work:
        first_three = similar_work[:3]
        similar_summary = "\n\nSimilar work found:\n" + "\n".join(
            f"  - {w.get('title', '?')} ({w.get('relevance', '?')}): "
            f"{w.get('summary', '')[:200]}"
            for w in first_three
        )

    return Finding(
        title=f"Novelty claim contradicted by prior work: {claim.short_summary[:60]}",
        severity=severity,  # type: ignore[arg-type]
        confidence={"high": "high", "medium": "medium", "low": "low"}.get(
            sev_label.split("-", 1)[0], "medium"
        ),  # type: ignore[arg-type]
        rationale=(
            f"The author claims: \"{claim.claim_text[:200]}\". "
            f"Literature search ({sev_label}) suggests this is not novel. "
            f"{assessment[:300]}"
            f"{similar_summary}"
        ),
        sections_involved=[],
        source="investigate",
        category="novelty",
    )
