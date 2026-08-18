"""Order findings by importance. Nothing is discarded here.

This stage used to drop any finding whose passage overlapped a more severe
one's, on the reasoning that two comments on one sentence is noise. Measured on
two real documents, that lost about a fifth of the review:

    FINDINGS.md      9 kept, 2 dropped
    CONSTITUTION.md 11 kept, 2 dropped

and every single drop was CROSS-CRITERION. Not one was a duplicate, because
:mod:`.consolidate` runs first and has already merged the findings that made
the same point — so everything reaching here had been judged by a prior stage
to be a DIFFERENT observation that happened to touch the same sentence. The
losses were the specific ones:

    dropped  [Evaluations/moderate]  define how AUC/ECE are computed
    kept     [Story/major]           the novelty claims are not adequately supported

The second is broader; the first is what an author can act on this afternoon.
Severity is not a reason to bin an actionable point raised from another angle.

Nor is there a technical reason to. Overlapping comments were once fragile, and
are not: :func:`~andamentum.whetstone.docx.surgery.split_at_boundaries` cuts
every note boundary before any is placed, precisely so two comments on one
sentence are ordinary. Verified — two overlapping notes both anchor, each
bracketing its own passage, neither clipped.

So each finding keeps its own comment on its own words, with its own severity
and criterion. Where two criteria pull in opposite directions — Story calling a
claim overstated while Significance says the contribution is undersold —
surfacing the tension is what a multi-perspective review is FOR. Resolving it
would mean picking a side, and that is the author's job, not this function's.

What remains needs no judgement at all.

*Exact repeats go.* Two findings with the same span AND the same wording are
one observation reported twice, and dropping the second decides nothing.
:mod:`.consolidate` normally catches these, but it is an LLM call and returns
its input unchanged on any error — so without a deterministic backstop here, a
consolidate failure would put nine identical comments on one sentence. The test
of redundancy is textual identity, never similarity: judging whether two
differently-worded points are "really the same" is the judgement this stage
declines to make.

*Then ordering*, which costs nothing and helps a reader: most severe first,
then by criterion so one perspective's points sit together.
"""

from __future__ import annotations

from .review import Finding

_SEVERITY_RANK = {"major": 3, "moderate": 2, "minor": 1}


def _identity(finding: Finding) -> tuple:
    """What makes two findings the same observation, not merely a similar one."""
    span = finding.span
    return (
        span.section_id if span else None,
        span.start if span else None,
        span.end if span else None,
        finding.issue.strip(),
    )


def gate_and_aggregate(findings: list[Finding]) -> list[Finding]:
    """Drop exact repeats, then order by severity and criterion.

    Findings that merely overlap are both kept — see the module docstring.
    """
    ordered = sorted(
        findings,
        key=lambda f: (-_SEVERITY_RANK.get(f.severity, 0), f.criterion),
    )
    seen: set[tuple] = set()
    kept: list[Finding] = []
    for finding in ordered:
        identity = _identity(finding)
        if identity in seen:
            continue
        seen.add(identity)
        kept.append(finding)
    return kept
