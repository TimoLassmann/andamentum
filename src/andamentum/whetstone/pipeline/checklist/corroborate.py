"""Deterministic co-detection cross-linking — two probes, one anchor, both kept.

This is the checklist-mode counterpart to :mod:`..consolidate`, and it does the
opposite thing on purpose.

Review mode consolidates near-duplicates through a stochastic agent call,
because the author asked for help with a draft and wants one comment instead of
nine. Checklist mode must not, for two reasons:

1. **The fixed slot structure is the deliverable.** The mode exists so that two
   runs of the same document — or run *n* against run *n+1* of a revision —
   produce comparable output. An LLM merger makes the finding count move
   between runs on an unchanged document, and a count that drifts for reasons
   unrelated to the draft destroys exactly the comparability being bought.

2. **Co-location is signal.** A passage that trips more than one check is worth
   the author's attention before one that trips a single check. Merging the two
   findings into one comment hides that the passage was reached twice.

   Note what this deliberately does NOT claim. Two probes landing on the same
   sentence have not *agreed* about anything — they have different instructions,
   so the ordinary case is one sentence carrying two different defects, not two
   readings of one defect. Deciding whether two findings make the same point
   requires comparing their prose, which is a judgment call, which is an LLM
   call, which is the nondeterminism reason 1 exists to avoid. So this module
   reports the thing it can establish deterministically — these checks landed on
   the same passage — and leaves "do they say the same thing" alone. Review mode
   is where that question gets answered, by an agent, on purpose.

So nothing here merges, ranks, rewrites or drops. Every finding that arrives
leaves, with the probe ids of the other checks that reached the same passage
recorded in ``corroborated_by`` for the renderer and the coverage table to
surface.

**Only presence findings corroborate.** Agreement needs a shared anchor, and an
absence finding has none: it names what it could not find, in whatever words
its own probe used. Two independently-prompted probes would have to emit the
same ``missing_item`` string for a match to fire, which does not happen in
practice — so an absence cross-link is machinery for a case that will not
occur. A fuzzier match is not the fix either: a false cross-link claims two
probes agree on a claim that by construction has no quote backing it, which is
the one place inflated agreement does real damage.
"""

from __future__ import annotations

import logging

from ..model import Span
from .probes import ProbeFinding

logger = logging.getLogger("andamentum.whetstone.pipeline")


# A pair of spans counts as the same passage only when they share at least this
# fraction of the SHORTER span. Bare interval overlap is too loose: a probe that
# quotes a whole paragraph would otherwise be recorded against every probe that
# quoted any sentence inside it, and on a short document that links most of the
# findings to most of the others (measured: 11 of 16 on a five-paragraph draft).
# Scoring against the shorter span is what makes the test symmetric — the
# question is "is the small quote essentially inside the big one", not "how much
# of the big one did the small one cover", which no threshold answers well.
_SAME_PASSAGE_MIN_SHARE = 0.5


def _same_passage(a: Span, b: Span) -> bool:
    """True when two located spans cover substantially the same text.

    Requires the same section and an intersection of at least
    ``_SAME_PASSAGE_MIN_SHARE`` of the shorter span. Two spans that merely abut
    — one ending exactly where the next begins — share nothing and are adjacent
    sentences, not one passage.

    The relation is symmetric, so the two parameters are genuinely positional:
    transposing them cannot change the answer.
    """
    if a.section_id != b.section_id:
        return False
    overlap = min(a.end, b.end) - max(a.start, b.start)
    if overlap <= 0:
        return False
    shorter = min(a.end - a.start, b.end - b.start)
    # A zero-length span cannot be substantially covered by anything; treat it
    # as no match rather than dividing by zero.
    if shorter <= 0:
        return False
    return overlap / shorter >= _SAME_PASSAGE_MIN_SHARE


def corroborate(findings: list[ProbeFinding]) -> list[ProbeFinding]:
    """Cross-link findings from different probes that landed on the same passage.

    Each member of a co-located pair gets the other's ``probe_id`` appended to
    its ``corroborated_by``. **Both findings are kept.** Nothing is merged,
    reordered or dropped — see the module docstring for why that is the whole
    point rather than an omission.

    "The same passage" means: both findings are presence-shaped, and their
    located spans sit in the same section sharing at least half the shorter span
    (:func:`_same_passage`). It does NOT mean the two findings make the same
    point — establishing that needs their prose compared, which this mode
    deliberately will not spend a model call on. Absence findings never link:
    they have no anchor to share (module docstring).

    A finding is never cross-linked with another from the same probe. A probe's
    quota already ranks its own findings against each other, so a self-link
    carries no information — it would simply add a partner to every finding of
    every multi-finding probe and make the signal meaningless.

    Pure and total: the input list is not mutated (findings are returned as
    copies with the new field set), input order is preserved, and no LLM is
    involved. Cost is O(n²) in the number of findings, which is correct rather
    than merely tolerable here — a bank run produces tens of findings, and an
    interval index would add a second representation of "same anchor" to keep
    in sync with :func:`_same_passage` for no measurable gain.
    """
    if len(findings) < 2:
        return list(findings)

    # Findings that reached this stage without a usable anchor cannot
    # participate. That should not happen — .verify drops presence findings
    # whose quote will not locate — so it is reported rather than passed over.
    unanchored = [f for f in findings if f.shape == "presence" and f.span is None]
    if unanchored:
        logger.warning(
            "[corroborate] %d presence finding(s) have no span and cannot be "
            "corroborated (verification should have located or dropped them): %s",
            len(unanchored),
            ", ".join(sorted({f.probe_id for f in unanchored})),
        )

    partners: list[set[str]] = [set() for _ in findings]
    pair_count = 0

    for i, first in enumerate(findings):
        if first.shape != "presence" or first.span is None:
            continue
        for j in range(i + 1, len(findings)):
            second = findings[j]
            if second.shape != "presence" or second.span is None:
                continue
            if first.probe_id == second.probe_id:
                continue
            if not _same_passage(first.span, second.span):
                continue
            partners[i].add(second.probe_id)
            partners[j].add(first.probe_id)
            pair_count += 1

    linked = 0
    result: list[ProbeFinding] = []
    for finding, found in zip(findings, partners, strict=True):
        if not found:
            result.append(finding.model_copy())
            continue
        # Union with whatever was already there, drop any self reference, then
        # sort. Reproducibility is the product of this mode: the same findings
        # must serialise byte-identically on every run, and set iteration order
        # is not a guarantee anyone should be relying on for that.
        merged = tuple(
            sorted((set(finding.corroborated_by) | found) - {finding.probe_id})
        )
        result.append(finding.model_copy(update={"corroborated_by": merged}))
        linked += 1

    if pair_count:
        logger.info(
            "[corroborate] %d co-detection pair(s) across %d/%d finding(s); "
            "all kept",
            pair_count,
            linked,
            len(findings),
        )
    return result
