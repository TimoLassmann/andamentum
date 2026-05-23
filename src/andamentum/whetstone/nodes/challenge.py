"""Node: Challenge.

For each finding above the minor severity threshold, ask the
``challenge_agent`` to refute it. Withdrawals drop the finding;
weakenings reduce its confidence; standings keep it as is.

Runs in parallel — independent findings can be challenged concurrently.
Disabled if ``state.challenge_enabled`` is False (caller passed
``challenge=False`` to ``review_document``).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_graph import BaseNode, GraphRunContext

from ..agents import build_pydantic_ai_agent, ChallengeVerdict
from ..deps import ReviewDeps
from ..schemas import Finding, ReviewResult
from ..state import ReviewState

if TYPE_CHECKING:
    from .author_questions import AuthorQuestions


logger = logging.getLogger("andamentum.whetstone")


# Severities that get challenged. Minor findings are kept as is — they're
# already low-stakes, not worth a refutation pass.
_CHALLENGEABLE_SEVERITIES = {"moderate", "major"}

# Concurrency cap on parallel challenge calls. Dropped from 6 → 2 to
# avoid stale-connection / NAT-table saturation against the OpenAI edge
# (waves of `Connection error` in batches of exactly N parallel calls).
_MAX_CONCURRENT_CHALLENGES = 2


@dataclass
class Challenge(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Refute high-severity findings, in parallel."""

    async def run(
        self, ctx: GraphRunContext[ReviewState, ReviewDeps]
    ) -> "AuthorQuestions":
        ctx.state.current_phase = "challenge"

        if not ctx.state.challenge_enabled:
            logger.info("[challenge] disabled — skipping refutation pass")
            ctx.state.challenged_findings = list(ctx.state.findings)
            from .author_questions import AuthorQuestions

            return AuthorQuestions()

        challengeable_idx = [
            i
            for i, f in enumerate(ctx.state.findings)
            if f.severity in _CHALLENGEABLE_SEVERITIES
        ]
        if not challengeable_idx:
            logger.info("[challenge] no challengeable findings (need moderate+ severity)")
            ctx.state.challenged_findings = list(ctx.state.findings)
            from .author_questions import AuthorQuestions

            return AuthorQuestions()

        logger.info(
            "[challenge] refuting %d finding(s) (concurrency=%d)",
            len(challengeable_idx),
            _MAX_CONCURRENT_CHALLENGES,
        )

        sections_by_id = {s.id: s for s in ctx.state.sections}
        sem = asyncio.Semaphore(_MAX_CONCURRENT_CHALLENGES)

        async def challenge_one(idx: int) -> tuple[int, ChallengeVerdict | None]:
            async with sem:
                try:
                    verdict = await _run_challenge(
                        ctx.deps, ctx.state.findings[idx], sections_by_id
                    )
                    return idx, verdict
                except Exception as exc:
                    # Loud-fail-safe: a challenge call that errors out
                    # leaves the finding intact (we'd rather keep a true
                    # finding than silently drop it on an LLM hiccup).
                    logger.warning(
                        "[challenge] finding[%d] crashed: %s — keeping intact",
                        idx,
                        exc,
                    )
                    return idx, None

        results = await asyncio.gather(
            *[challenge_one(i) for i in challengeable_idx]
        )
        ctx.state.llm_calls += sum(1 for _, v in results if v is not None)

        # Apply verdicts. Build a fresh list to avoid mutating in flight.
        verdict_by_idx = {idx: verdict for idx, verdict in results if verdict is not None}
        challenged: list[Finding] = []
        stood = weakened = withdrawn = 0
        for i, finding in enumerate(ctx.state.findings):
            verdict = verdict_by_idx.get(i)
            if verdict is None:
                challenged.append(finding)
                continue
            if verdict.verdict == "stand":
                challenged.append(_with_source(finding, "challenged"))
                stood += 1
                logger.info('[challenge] stand    — "%s"', finding.title)
            elif verdict.verdict == "weaken":
                weakened_finding = _weaken(finding, verdict.reason)
                challenged.append(weakened_finding)
                weakened += 1
                logger.info(
                    '[challenge] weaken   — "%s" (%s→%s): %s',
                    finding.title,
                    finding.confidence,
                    weakened_finding.confidence,
                    verdict.reason or "(no reason given)",
                )
            else:  # "withdraw" → finding is dropped; do not append.
                withdrawn += 1
                logger.info(
                    '[challenge] withdraw — "%s": %s',
                    finding.title,
                    verdict.reason or "(no reason given)",
                )
        ctx.state.challenged_findings = challenged
        logger.info(
            "[challenge] done — %d stood, %d weakened, %d withdrawn",
            stood,
            weakened,
            withdrawn,
        )

        from .author_questions import AuthorQuestions

        return AuthorQuestions()


# ── Helpers ─────────────────────────────────────────────────────────────


async def _run_challenge(
    deps: ReviewDeps,
    finding: Finding,
    sections_by_id,
) -> ChallengeVerdict:
    """One challenge call against one finding."""
    cited_sections = [
        sections_by_id[sid]
        for sid in finding.sections_involved
        if sid in sections_by_id
    ]
    sections_text = "\n\n".join(
        f"--- BEGIN {s.id} ({s.title}) ---\n{s.text}\n--- END {s.id} ---"
        for s in cited_sections
    )
    quotes_block = "\n".join(f"  • {q.text!r} (in {q.section_id})" for q in finding.quotes)

    prompt = f"""FINDING TO CHALLENGE:
title:      {finding.title}
severity:   {finding.severity}
confidence: {finding.confidence}
rationale:  {finding.rationale}

QUOTES the finding cites:
{quotes_block or "  (none)"}

CITED SECTIONS (full text):
{sections_text or "(no cited sections found in document)"}

Verdict: stand | weaken | withdraw. Default to "stand" unless evidence refutes."""

    agent = build_pydantic_ai_agent("challenge", deps.model)
    result = await agent.run(prompt)
    from typing import cast

    return cast(ChallengeVerdict, result.output)


def _weaken(finding: Finding, reason: str) -> Finding:
    """Lower the finding's confidence by one tier.

    The challenge agent's reasoning is internal deliberation — it is logged
    for traceability but deliberately NOT appended to the rationale, which
    is the reviewer-facing comment body.
    """
    new_confidence = {"high": "medium", "medium": "low", "low": "low"}[finding.confidence]
    logger.debug("[challenge] weakened %s — %s", finding.id, reason)
    return finding.model_copy(
        update={
            "confidence": new_confidence,  # type: ignore[arg-type]
            "source": "challenged",
        }
    )


def _with_source(finding: Finding, source: str) -> Finding:
    return finding.model_copy(update={"source": source})  # type: ignore[arg-type]
