"""One view across several reviewers.

Running four reviewers over a draft produces four reports and, near enough,
four times the reading. Most of it overlaps: two reviewers reach the same
sentence from different directions and say much the same thing in different
words. Reading all four to notice that is work the tool should have done.

So a run with more than one reviewer ends with this: **one call that says what
they agreed on, where they disagreed, and what only one of them saw.** It is
not a fifth review and it does not judge the document — it judges the reviews.

Three things only, and each of them needs more than one reviewer to exist at
all:

*Agreed* is the most valuable output of a multi-reviewer run and the hardest to
spot by eye, because the same defect is phrased differently by each reviewer
that finds it. Two independent reviewers reaching the same passage is the
strongest signal available here, and it is what should be fixed first.

*Disagreed* is surfaced and never resolved. One reviewer calling a claim
overstated while another says the contribution is undersold is a real tension
in the draft, and choosing between them is the author's judgement, not a
model's. Presenting both with their reasons is the honest move; picking a
winner would hide that a choice was made.

*Only one saw it* keeps the summary honest. Without it a reader who stops after
the summary loses every finding that exactly one reviewer raised — which on a
four-reviewer run is most of them.

What it must NOT do is restate every finding. A summary that reproduces its
inputs leaves the reader with the same pages of overlapping issues plus a layer
on top, which is the problem it was added to solve. The per-reviewer reports
are still in the document, below this, for anyone who wants them.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from andamentum.core.agents import AgentDefinition, build_pydantic_ai_agent
from andamentum.core.models import resolve_model

from ..schemas import ReviewResult

logger = logging.getLogger("andamentum.whetstone.v3")


class AgreedPoint(BaseModel):
    """One thing more than one reviewer raised independently."""

    issue: str = Field(description="The shared point, in one sentence.")
    raised_by: list[str] = Field(
        default_factory=list,
        description="The reviewers who raised it, by their identity.",
    )
    quote: str = Field(
        default="", description="A verbatim passage from the draft it concerns."
    )


class Disagreement(BaseModel):
    """Two reviewers pulling in opposite directions on one point."""

    about: str = Field(description="What they disagree about, in one sentence.")
    positions: list[str] = Field(
        default_factory=list,
        description="One line per side, each naming the reviewer who holds it.",
    )


class ReviewerReconciliation(BaseModel):
    """The output of one reconciliation call. Flat, so small models fill it."""

    verdict: str = Field(
        default="",
        description=(
            "Two or three sentences: what the reviewers collectively make of "
            "this draft, and what to do first."
        ),
    )
    agreed: list[AgreedPoint] = Field(default_factory=list)
    disagreements: list[Disagreement] = Field(default_factory=list)
    raised_by_one: list[str] = Field(
        default_factory=list,
        description=(
            "Points exactly one reviewer raised that still matter, one line "
            "each, each naming its reviewer."
        ),
    )


_PROMPT = """You are reconciling several independent reviews of ONE draft into a single view.

You are NOT reviewing the document. You are reading what the reviewers said and
reporting where they converge, where they conflict, and what only one of them
noticed. Do not add findings of your own.

Each reviewer read the draft with a different remit, so the same defect often
appears in two reviews in different words. Recognising that is your main job.

# agreed
Points more than one reviewer raised. Match on WHAT IS WRONG and WHERE, not on
wording — "the conclusion outruns the data" and "this claim is not supported by
the reported results" about the same sentence are one point, not two. Name
every reviewer who raised it. These come first because independent agreement is
the strongest signal in the whole run.

# disagreements
Points where reviewers pull in opposite directions — one says a claim is
overstated, another says the contribution is undersold. Give both positions
with the reviewer who holds each. DO NOT choose between them and do not soften
either side: which one is right is the author's judgement, and presenting both
is the answer.

# raised_by_one
Points exactly one reviewer raised that still deserve attention. One line each,
naming the reviewer. This is what stops a reader who only reads your summary
from losing most of the review.

# verdict
Two or three sentences. What the reviewers collectively make of the draft, and
what to fix first.

Be short. You are replacing several pages of overlapping reports with one page;
restating every finding defeats the point, and the full reports are still there
for anyone who wants them."""


def _brief_findings(result: ReviewResult, *, limit: int = 12) -> str:
    lines: list[str] = []
    for finding in result.findings[:limit]:
        quote = finding.quotes[0].text[:120] if finding.quotes else ""
        lines.append(f"  - [{finding.severity}] {finding.title}: {finding.rationale}")
        if quote:
            lines.append(f'    concerning: "{quote}"')
    if len(result.findings) > limit:
        lines.append(f"  - (+{len(result.findings) - limit} more)")
    return "\n".join(lines) or "  (no findings)"


async def reconcile_reviewers(
    reviews: list[tuple[str, ReviewResult]],
    *,
    model: str,
) -> ReviewerReconciliation | None:
    """One call over several reviewers' results.

    ``reviews`` is ``(reviewer identity, result)`` in the order they ran.
    Returns ``None`` on a crash — a failed reconciliation must not cost you the
    reviews themselves, which are already written into the document.
    """
    if len(reviews) < 2:
        return None

    blocks = [
        f"## {identity}\n\n{(result.summary or '').strip()[:1500]}\n\n"
        f"Findings:\n{_brief_findings(result)}"
        for identity, result in reviews
    ]
    prompt = (
        f"{len(reviews)} reviewers read this draft, in this order: "
        f"{', '.join(identity for identity, _ in reviews)}.\n\n"
        + "\n\n".join(blocks)
    )

    defn = AgentDefinition(
        name="v3_reconcile",
        prompt=_PROMPT,
        output_model=ReviewerReconciliation,
        retries=2,
        output_retries=3,
    )
    try:
        agent = build_pydantic_ai_agent(defn, resolve_model(model))
        res = await agent.run(prompt)
    except Exception as exc:
        logger.warning(
            "[v3.reconcile] reconciliation failed (%s) — the per-reviewer "
            "reports stand on their own",
            exc,
        )
        return None

    from ._metrics import bump_from_result

    bump_from_result(res)
    out = res.output
    logger.info(
        "[v3.reconcile] %d reviewer(s) → %d agreed, %d disagreement(s), "
        "%d raised by one",
        len(reviews),
        len(out.agreed),
        len(out.disagreements),
        len(out.raised_by_one),
    )
    return out
