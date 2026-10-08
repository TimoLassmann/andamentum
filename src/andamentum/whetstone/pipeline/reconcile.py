"""One view across several reviewers.

Running four reviewers over a draft produces four reports and, near enough,
four times the reading. Most of it overlaps: two reviewers reach the same
sentence from different directions and say much the same thing in different
words. Reading all four to notice that is work the tool should have done.

So a run with more than one reviewer ends with this: **one call that says what
they agreed on and where they disagreed.** It is not a fifth review and it does
not judge the document — it judges the reviews.

Two things only, and each of them needs more than one reviewer to exist at
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

*Agreed* is also what removes the duplicates. Each agreed point names the
findings it groups by id, and :func:`combine_reviews` merges each group into
ONE finding — the most severe member, with the other reviewers named on it —
so a point three reviewers raised is one comment in the margin, not three. One
call decides both the summary line and the merge, so the two cannot disagree
about what counts as the same point.

What it must NOT do is restate every finding. The merged findings follow the
summary in the same report, so a summary that lists them again gives the reader
every point twice. That is why there is no "raised by one reviewer" list any
more: those findings are in the report, once, tagged with their reviewer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import BaseModel, Field

from andamentum.core.agents import AgentDefinition, build_pydantic_ai_agent
from andamentum.core.models import resolve_model

from ..schemas import Finding, ReviewResult

logger = logging.getLogger("andamentum.whetstone.v3")


class AgreedPoint(BaseModel):
    """One thing more than one reviewer raised independently."""

    issue: str = Field(description="The shared point, in one plain sentence.")
    finding_ids: list[str] = Field(
        default_factory=list,
        description=(
            "The ids of every finding that makes this point, as shown in "
            "brackets in the input (e.g. ['R1.2', 'R3.1']). They must come "
            "from at least two different reviewers."
        ),
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


_PROMPT = """You are reconciling several independent reviews of ONE draft into a single view.

You are NOT reviewing the document. You are reading what the reviewers said and
reporting where they converge and where they conflict. Do not add findings of
your own.

Each reviewer read the draft with a different remit, so the same defect often
appears in two reviews in different words. Recognising that is your main job:
every finding you group is shown to the author ONCE instead of once per
reviewer.

# agreed
Points more than one reviewer raised. Match on WHAT IS WRONG and WHERE, not on
wording — "the conclusion outruns the data" and "this claim is not supported by
the reported results" about the same sentence are one point, not two. For each
point, list the ids of EVERY finding that makes it (the bracketed ids in the
input, e.g. R1.2). A point needs findings from at least two different
reviewers. Two findings about the same sentence that raise DIFFERENT problems
are not one point — leave them apart. When in doubt, do not group. A finding
belongs to at most one point.

# disagreements
Points where reviewers pull in opposite directions — one says a claim is
overstated, another says the contribution is undersold. Give both positions
with the reviewer who holds each. DO NOT choose between them and do not soften
either side: which one is right is the author's judgement, and presenting both
is the answer.

# verdict
Two or three plain sentences to the author: what the reviewers collectively
make of the draft, and what to fix first.

Be short. Every finding is listed in full after your summary, so do not restate
them."""


def finding_ref(*, reviewer: int, finding: int) -> str:
    """The id a finding carries in the reconciliation prompt — ``R2.3`` is
    the third finding of the second reviewer (both 1-based, as a reader
    counts)."""
    return f"R{reviewer + 1}.{finding + 1}"


def _brief_findings(result: ReviewResult, *, reviewer: int) -> str:
    """Every finding of one reviewer, each with its id.

    All of them, not the first dozen: a finding the call never sees can never
    be grouped, so a cap here is a cap on how many duplicates can be removed.
    """
    lines: list[str] = []
    for index, finding in enumerate(result.findings):
        ref = finding_ref(reviewer=reviewer, finding=index)
        lines.append(
            f"  [{ref}] ({finding.severity}) {finding.title}: {finding.rationale}"
        )
        if finding.quotes:
            quote = finding.quotes[0]
            lines.append(
                f"    concerning ({quote.section_id}, characters "
                f'{quote.char_start}-{quote.char_end}): "{quote.text[:160]}"'
            )
    return "\n".join(lines) or "  (no findings)"


async def reconcile_reviewers(
    reviews: list[tuple[str, ReviewResult]],
    *,
    model: str,
) -> ReviewerReconciliation | None:
    """One call over several reviewers' results.

    ``reviews`` is ``(reviewer identity, result)`` in the order they ran.
    Returns ``None`` on a crash — a failed reconciliation costs the summary and
    the merge, never the reviews themselves: :func:`combine_reviews` then keeps
    every finding, and says in the summary that duplicates may remain.
    """
    if len(reviews) < 2:
        return None

    blocks = [
        f"## Reviewer {index + 1}: {identity}\n\n"
        f"{(result.summary or '').strip()[:1500]}\n\n"
        f"Findings:\n{_brief_findings(result, reviewer=index)}"
        for index, (identity, result) in enumerate(reviews)
    ]
    prompt = (
        f"{len(reviews)} reviewers read this draft, in this order: "
        f"{', '.join(identity for identity, _ in reviews)}.\n\n" + "\n\n".join(blocks)
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
            "[v3.reconcile] reconciliation failed (%s) — every reviewer's "
            "findings are kept unmerged",
            exc,
        )
        return None

    from ._metrics import bump_from_result

    bump_from_result(res)
    out = res.output
    logger.info(
        "[v3.reconcile] %d reviewer(s) → %d agreed point(s), %d disagreement(s)",
        len(reviews),
        len(out.agreed),
        len(out.disagreements),
    )
    return out


# ── the merge (deterministic) ───────────────────────────────────────────────

_SEVERITY_RANK = {"major": 3, "moderate": 2, "minor": 1}


@dataclass(frozen=True)
class AgreedGroup:
    """An agreed point, resolved against the findings it names.

    ``raised_by`` is derived from the findings' own reviewers rather than
    taken from the model, so the summary cannot credit a reviewer whose
    finding is not in the group.
    """

    issue: str
    raised_by: list[str]
    finding: Finding


def _merge(members: list[tuple[int, str, Finding]]) -> Finding:
    """One finding for a group: the most severe member, others named on it.

    Ties go to the reviewer that ran first. The member's own text is kept — its
    problem statement and its fix were written against its own quote, and a
    rewritten sentence covering every member would be a new, unverified
    finding. The other members' passages are kept as extra quotes, so a point
    two reviewers anchored in two places still lists both.
    """
    primary_index, primary_reviewer, primary = max(
        members, key=lambda m: (_SEVERITY_RANK.get(m[2].severity, 0), -m[0])
    )
    others: list[str] = []
    for _, reviewer, _finding in members:
        if reviewer != primary_reviewer and reviewer not in others:
            others.append(reviewer)
    quotes = list(primary.quotes)
    seen = {(q.char_start, q.char_end) for q in quotes}
    for _, _, finding in sorted(members, key=lambda m: m[0]):
        for quote in finding.quotes:
            if (quote.char_start, quote.char_end) not in seen:
                seen.add((quote.char_start, quote.char_end))
                quotes.append(quote)
    sections = list(
        dict.fromkeys(s for _, _, f in members for s in f.sections_involved)
    )
    return primary.model_copy(
        update={
            "perspective": primary_reviewer,
            "corroborated_by": others,
            "quotes": quotes,
            "sections_involved": sections,
        }
    )


def merge_agreed(
    reviews: list[tuple[str, ReviewResult]],
    reconciliation: ReviewerReconciliation | None,
) -> tuple[list[Finding], list[AgreedGroup]]:
    """Every reviewer's findings, with each agreed point merged into one.

    Returns ``(findings, groups)``. ``findings`` holds every finding exactly
    once — a merged one where the reconciliation grouped it, the reviewer's own
    otherwise — each tagged with its reviewer in ``perspective``. It keeps the
    order the reviewers ran in, a merged finding taking the place of its first
    member.

    The model's grouping is checked, not trusted. An id that names no finding
    is ignored; a finding already claimed by an earlier point stays with that
    point; and a point left with findings from fewer than two reviewers is not
    agreement and merges nothing. Each case is logged, because a grouping that
    quietly shrinks is a duplicate the reader will still see.
    """
    by_ref: dict[str, tuple[int, str, Finding]] = {}
    order: list[str] = []
    for r, (identity, result) in enumerate(reviews):
        for i, finding in enumerate(result.findings):
            ref = finding_ref(reviewer=r, finding=i)
            by_ref[ref] = (len(order), identity, finding)
            order.append(ref)

    claimed: dict[str, int] = {}  # ref → index into `groups`
    groups: list[AgreedGroup] = []
    for point in reconciliation.agreed if reconciliation is not None else []:
        refs = []
        for raw in point.finding_ids:
            ref = raw.strip().strip("[]").upper()
            if ref not in by_ref:
                logger.warning("[v3.reconcile] agreed point names no finding %r", raw)
            elif ref in claimed:
                logger.info("[v3.reconcile] %s is already in another point", ref)
            elif ref not in refs:
                refs.append(ref)
        members = [by_ref[ref] for ref in refs]
        reviewers = list(dict.fromkeys(identity for _, identity, _ in members))
        if len(reviewers) < 2:
            logger.info(
                "[v3.reconcile] point %r has findings from %d reviewer(s) — not merged",
                point.issue[:60],
                len(reviewers),
            )
            continue
        for ref in refs:
            claimed[ref] = len(groups)
        groups.append(
            AgreedGroup(issue=point.issue, raised_by=reviewers, finding=_merge(members))
        )

    findings: list[Finding] = []
    placed: set[int] = set()
    for ref in order:
        _, identity, finding = by_ref[ref]
        if ref not in claimed:
            findings.append(finding.model_copy(update={"perspective": identity}))
            continue
        group = claimed[ref]
        if group not in placed:
            placed.add(group)
            findings.append(groups[group].finding)
    if groups:
        logger.info(
            "[v3.reconcile] %d finding(s) → %d after merging %d agreed point(s)",
            len(order),
            len(findings),
            len(groups),
        )
    return findings, groups


# ── the combined result ─────────────────────────────────────────────────────


def _synopsis(summary: str) -> str:
    """The synopsis paragraph of one reviewer's own summary, or ``""``."""
    from .synth import synopsis_of

    return synopsis_of(summary)


def reconciliation_markdown(
    reconciliation: ReviewerReconciliation | None,
    *,
    reviews: list[tuple[str, ReviewResult]],
    groups: list[AgreedGroup],
) -> str:
    """The summary of a multi-reviewer run, as markdown.

    Rendered as markdown rather than as a new atom kind so all three renderers
    carry it without knowing it exists — the summary is prose in every one of
    them already.

    Four parts, each saying something the findings list below it cannot: the
    verdict, what more than one reviewer raised (so it is fixed first), where
    they disagree, and one line per reviewer on how it read the draft. A
    reviewer that ran on a document type it does not fit is named here as
    well — a single reviewer's report carries that warning in its header, and
    a combined report has no per-reviewer header to carry it.
    """
    out: list[str] = []
    if reconciliation is None:
        out += [
            "Several reviewers ran, and reconciling them failed. Every "
            "finding is listed below with its reviewer, but a point more "
            "than one of them raised may appear more than once.",
            "",
        ]
    elif reconciliation.verdict:
        out += [reconciliation.verdict.strip(), ""]
    if groups:
        out.append(
            f"**Raised by more than one reviewer ({len(groups)})** — fix these "
            "first. Each is listed once below."
        )
        out.append("")
        for group in groups:
            out.append(f"- {group.issue} *({', '.join(group.raised_by)})*")
        out.append("")
    if reconciliation is not None and reconciliation.disagreements:
        out.append(
            f"**Where the reviewers disagree ({len(reconciliation.disagreements)})** "
            "— both positions are given because choosing between them is your "
            "judgement, not the tool's."
        )
        out.append("")
        for clash in reconciliation.disagreements:
            out.append(f"- {clash.about}")
            for position in clash.positions:
                out.append(f"  - {position}")
        out.append("")
    readings = [(identity, _synopsis(result.summary)) for identity, result in reviews]
    if any(text for _, text in readings):
        out.append("**How each reviewer read the draft**")
        out.append("")
        for identity, text in readings:
            if text:
                out.append(f"- *{identity}:* {text}")
        out.append("")
    warnings = [
        (identity, result.reviewer_header.fit_warning)
        for identity, result in reviews
        if result.reviewer_header is not None and result.reviewer_header.fit_warning
    ]
    for identity, warning in warnings:
        out.append(f"*{identity}: {warning}*")
        out.append("")
    return "\n".join(out).strip()


def combine_reviews(
    reviews: list[tuple[str, ReviewResult]],
    *,
    reconciliation: ReviewerReconciliation | None,
) -> tuple[ReviewResult, list[AgreedGroup]]:
    """Several reviewers' results as one, for every renderer.

    Findings are merged by :func:`merge_agreed` and tagged with their reviewer.
    The summary becomes :func:`reconciliation_markdown`. Novelty claims come
    from the first reviewer only; the draft's assertions do not change because
    a second reviewer read it.

    Every round read the same text — no round edits in a list — so every
    finding's offsets address the one document all of them read, and the
    combined result renders against it directly, the .docx included.
    """
    findings, groups = merge_agreed(reviews, reconciliation)
    first, last = reviews[0][1], reviews[-1][1]
    edits = [e for _, res in reviews for e in res.edits]
    combined = last.model_copy(
        update={
            "summary": reconciliation_markdown(
                reconciliation, reviews=reviews, groups=groups
            ),
            # No single reviewer's block: this result is all of them.
            "reviewer_header": None,
            "findings": findings,
            "edits": edits,
            "declined_findings": [
                d for _, res in reviews for d in res.declined_findings
            ],
            "author_questions": [q for _, res in reviews for q in res.author_questions],
            "failed_criteria": [
                f"{identity}: {name}"
                for identity, res in reviews
                for name in res.failed_criteria
            ],
            "novelty_claims": first.novelty_claims,
            "novelty_checked": first.novelty_checked,
            "metrics": last.metrics.model_copy(
                update={
                    "llm_calls": sum(r.metrics.llm_calls for _, r in reviews),
                    "edits_count": len(edits),
                }
            ),
        }
    )
    return combined, groups
