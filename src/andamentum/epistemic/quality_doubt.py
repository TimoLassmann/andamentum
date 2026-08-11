"""Second opinions that gate a TMS evidence DESTROY (llm_judge, FAST path).

**One rule, one module: nothing in the pipeline may erase a piece of evidence
on the strength of a single sampled model output.** Exactly two places would
otherwise do that, and both now come through here. Four OTHER sites set
``Evidence.invalidated``, and none of them rests on a sampled judgment —
they are deterministic consequences of state, so there is nothing for a
second opinion to check: a provider returning no results
(``operations/evidence.py``), empty extracted content at the quality cliff
(same file), the loser of a cross-provider dedup (``dedupe_evidence.py``),
and the TMS demotion cascade (``operations/belief_maintenance.py``).

DESTROY SITE 1 — the near-zero quality cliff.
``ExtractEvidenceOperation._score_evidence`` Path 2 blends four agent-produced
floats into one quality score and, below ``0.10``, invalidates the evidence
outright (a Truth-Maintenance-System destroy). That is a **categorical cliff
driven by a single sampled point value**: one unlucky draw on four
free-floating numbers and a piece of real evidence is erased, with nothing
downstream able to tell erasure from absence. Gated by
``confirm_low_quality``.

DESTROY SITE 2 — the ``evidence_corrupted`` sentinel.
``ScrutiniseClaimOperation._handle_issues`` invalidates evidence when
``epistemic_identify_single_issue`` returns the ``"evidence_corrupted"``
issue type. That is ONE sampled categorical from a small model — the same
defect class as site 1, one layer up: no distribution, no entropy, nothing
downstream able to tell "the source was garbled" from "the sampler picked
the first branch of the prompt's classification rules". Gated by
``confirm_corruption``.

This module is the gate on both cliffs. It asks ``llm_judge.judge_score`` —
the system's uncertainty-carrying judge — whether the proposition that would
justify the destroy actually holds, and returns the verdict together with the
DERIVED doubt and ``needs_review`` signals. The caller destroys the evidence
only when that second opinion confidently agrees.

WHAT SURVIVAL MEANS DIFFERS BETWEEN THE TWO SITES, and the difference is not
cosmetic. At site 1 the retained item keeps its near-zero ``quality_score``,
so it stays heavily down-weighted in ``scrutiny_weight._item_masses`` and in
the counting posterior: attenuation, not destruction — the project's "process
flags filter inquiry, not output" doctrine, applied at the evidence layer.
At site 2 NOTHING is attenuated. The retained item keeps its full quality
score and contributes its full directional mass; the concern is recorded as a
NON-BLOCKING ``Uncertainty`` rather than dropped, and that caveat changes no
weight, no verdict and no posterior. Measured A/B against a control where the
sentinel never fired: identical ``scrutiny_verdict``, ``scrutiny_doubt``,
stage and ``quality_score``, one extra non-blocking uncertainty.

That asymmetry is inherited rather than chosen here.
``adapters.adapt_identify_single_issue`` derives ``reversal_test`` as
``issue_type in {"unknown", "contradiction"}``, so ``evidence_corrupted``
never downgrades a source's weight. Adding it to that set WOULD make a
declined destroy attenuate — a real design decision about how much a single
sampled sentinel may move a verdict, deliberately left open rather than
smuggled in behind a gate whose only job was to stop an ungated erasure.

TWO CRITERIA, ONE PER SITE, because the two destroys rest on two different
propositions: "there is substantive content here at all" (site 1) versus
"the text is intact as retrieved" (site 2). They are NOT interchangeable — a
perfectly substantive paper can arrive mojibake-mangled, and an intact page
can be pure navigation chrome. Everything else is shared: one call, one
criterion, one ``QualitySecondOpinion`` record, one decision rule
(``overall == "fails" and not needs_review``), assembled by one private
helper so the two gates cannot drift apart.

FAST, NOT PANEL. ``model`` is a bare ``str``, so ``judge_score`` takes its
single-judge, temperature-0 path: one call, reproducible. A panel would be K
sequential calls on a tail path that fires once per near-zero item, and its
contract is an *agreement* poll rather than a *reproducible gate* — the wrong
shape for a destroy/retain decision that must give the same answer on a re-run.

ONE CRITERION, deliberately. ``judge_score`` fires one model call per
criterion and rolls them up equal-weight. Mapping the four quality axes to
four criteria would be a 4x cost blowup here AND would silently replace the
deliberate 0.35 / 0.25 / 0.25 / 0.15 blend the agent scorer applies, with an
equal-weight mean nobody chose.

WHERE llm_judge IS USED IN epistemic — FOUR SITES, ONE SHAPE. This module
was the first; there are now four, and all four are the same shape (FAST
path, exactly one ``Criterion``, gated so they fire only on judgments that
are already borderline, entropy-blind, or about to destroy something):

1. **this module, ``confirm_low_quality``** — the near-zero-quality TMS
   destroy gate (per evidence item, tail path only);
2. **this module, ``confirm_corruption``** — the ``evidence_corrupted`` TMS
   destroy gate (per flagged item, tail path only);
3. ``direction_opinion`` — an independent CONFIRMATION of a one-hot,
   directional evidence judgment, during scrutiny (bounded per pass and per
   claim);
4. ``scrutiny_opinion`` — an independent read on a BORDERLINE deterministic
   evidence weight, per claim.

Sites 1 and 2 are the only two that can BLOCK an action; sites 3 and 4 are
suspend-only and cannot change a verdict, a label, a mass or a stage.

CONVERSION vs. CONFIRMATION — the distinction that keeps the argument below
true. ``judge_evidence`` is STILL NOT CONVERTED, for exactly the original
reasons: it forces a four-step scope analysis and enforces ``in_scope`` ↔
argmax over ``{supports, contradicts, no_bearing}`` via a retry-triggering
cross-field validator, none of which the generic three-label score schema can
express. Replacing it would be a regression dressed as an upgrade. What site
(3) adds is a layer ABOVE it: the primary judge has already committed to a
direction, and "does this text genuinely support [/contradict] the claim" IS
a meets/partial/fails question. The scope analysis is not re-done and not
lost — it is the proposition being confirmed. Site (4) is the same move at
the claim level, on a question ("is this enough, on-scope evidence to
establish the claim") that is natively a quality axis. Site (2) is the same
move on the ``evidence_corrupted`` sentinel: the identifier agent has already
committed to "this is garbled", and "is this text intact as retrieved" IS a
meets/partial/fails question.

What all four share, and what a fifth site would have to share, is that the
judgment being put to llm_judge has NO DOMAIN INVARIANT TO LOSE.

This module mutates nothing and raises rather than degrading: a judge failure
propagates to ``_execute_op``, which records it as a loud quarantine.

Architecture: Layer 1 (framework-agnostic; async driver + pure assembly).
"""

from __future__ import annotations

from dataclasses import dataclass

from andamentum.llm_judge import Criterion, judge_score
from andamentum.llm_judge.signals import normalize_three

# Exactly one criterion — see the module docstring for why not four.
QUALITY_CRITERIA: list[Criterion] = [
    Criterion(
        name="substantive_content",
        description=(
            "The text contains substantive, checkable content that could bear "
            "on a scientific claim — rather than being empty, boilerplate, an "
            "error or paywall page, or navigation chrome."
        ),
    )
]

# Exactly one criterion, and a DIFFERENT proposition from the one above: this
# gate answers "did the retrieval arrive intact", not "is there anything of
# substance here". Reusing QUALITY_CRITERIA would ask the wrong question in
# both directions — a mojibake-mangled trial report is substantive and
# unusable, a pristine cookie banner is intact and worthless.
CORRUPTION_CRITERIA: list[Criterion] = [
    Criterion(
        name="text_is_intact",
        description=(
            "The source text is intact and readable as retrieved — coherent, "
            "in one language and encoding, and complete enough to interpret "
            "— rather than garbled, encoding-mangled, truncated mid-thought, "
            "or an interleaving of unrelated fragments."
        ),
    )
]


@dataclass(frozen=True)
class QualitySecondOpinion:
    """One judge's read on the proposition a TMS destroy would rest on.

    Shared by both destroy gates: at site 1 the proposition is "this source
    has substantive content", at site 2 "this source arrived intact". One
    record, so the two gates cannot grow two vocabularies for the same
    question.

    ``overall`` is the LABEL to decide on; ``expected_score`` is the continuous
    companion for ranking (llm_judge's documented rule: label for decisions,
    scalar for ranking). ``doubt`` and ``needs_review`` are DERIVED by
    ``llm_judge``, never asked of the model — ``needs_review`` already folds in
    ``doubt >= NEEDS_REVIEW_DOUBT_THRESHOLD``, which is why this module
    introduces no numeric threshold of its own.
    """

    distribution: list[float]
    overall: str
    expected_score: float
    doubt: float
    needs_review: bool
    summary: str

    @property
    def confidently_fails(self) -> bool:
        """THE DECISION RULE FOR BOTH DESTROY GATES, defined once.

        A destroy proceeds only when the judge says the proposition FAILS
        *and* is not itself flagged for review. The label decides;
        ``needs_review`` already folds in llm_judge's own
        ``doubt >= NEEDS_REVIEW_DOUBT_THRESHOLD``, so neither gate invents a
        numeric cutoff, and neither can drift from the other.
        """
        return self.overall == "fails" and not self.needs_review


async def _one_criterion_opinion(
    *,
    content: str,
    criteria: list[Criterion],
    context: str,
    model: str,
) -> QualitySecondOpinion:
    """The single shared call both destroy gates make.

    Exactly one ``Criterion`` in, exactly one model call out, FAST path (a
    bare ``str`` model), and one assembly of the derived signals. Private so
    the two public gates are the only entry points and the shape cannot be
    re-implemented a third time.
    """
    result = await judge_score(
        content,
        criteria=criteria,
        context=context,
        model=model,
    )
    row = result.per_criterion[0]
    return QualitySecondOpinion(
        distribution=normalize_three(row.meets, row.partial, row.fails),
        overall=result.overall,
        expected_score=result.expected_score,
        doubt=result.doubt,
        needs_review=result.needs_review,
        summary=row.reasoning,
    )


async def confirm_low_quality(
    *,
    content: str,
    source_ref: str,
    claim_statement: str,
    agent_score: float,
    model: str,
) -> QualitySecondOpinion:
    """Ask one judge whether ``content`` really is substanceless.

    Every parameter is keyword-only: ``content`` / ``source_ref`` /
    ``claim_statement`` / ``model`` are four same-typed strings, and
    transposing any pair would silently judge the wrong text against the
    wrong context.

    Args:
        content: The extracted source text the agent scored near zero.
        source_ref: The source identifier, surfaced to the judge as context.
        claim_statement: The claim this evidence was gathered for, so
            "could bear on a scientific claim" is asked about the right claim.
        agent_score: The agent's blended quality score, included in the
            context purely as provenance — the judge is NOT told to agree
            with it, only what prompted the second look.
        model: A single pydantic-ai model id. A ``str`` (not a sequence) is
            required: it selects llm_judge's FAST path.

    Returns:
        The judge's verdict, with derived doubt / needs_review.

    Raises:
        Whatever ``judge_score`` raises. This function never swallows a
        failure into a default answer — a silently-defaulted second opinion
        would be worse than the point value it replaces.
    """
    context = (
        f"This text was retrieved as evidence for the claim: {claim_statement or '(none)'}\n"
        f"Source: {source_ref}\n"
        f"An automated scorer rated its quality {agent_score:.3f} (near zero), "
        f"which would normally discard it. Judge the text on its own terms."
    )
    return await _one_criterion_opinion(
        content=content,
        criteria=QUALITY_CRITERIA,
        context=context,
        model=model,
    )


async def confirm_corruption(
    *,
    content: str,
    source_ref: str,
    claim_statement: str,
    issue_description: str,
    model: str,
) -> QualitySecondOpinion:
    """Ask one judge whether ``content`` really did arrive corrupted.

    The caller destroys the evidence only when
    ``QualitySecondOpinion.confidently_fails`` — i.e. the judge says the text
    is NOT intact and is not itself unsure. Note the polarity: the criterion
    is stated AFFIRMATIVELY ("the text is intact"), so ``fails`` means
    "corrupted", agreeing with the sentinel that prompted the second look.

    Every parameter is keyword-only: ``content`` / ``source_ref`` /
    ``claim_statement`` / ``issue_description`` / ``model`` are five
    same-typed strings, and transposing any pair would silently judge the
    wrong text against the wrong context.

    Args:
        content: The extracted source text the identifier agent flagged.
        source_ref: The source identifier, surfaced to the judge as context.
        claim_statement: The claim this evidence was gathered for, so the
            judge knows what the text was retrieved to speak to.
        issue_description: What the identifier agent said was wrong,
            included as provenance — the judge is NOT told to agree with it,
            only what prompted the second look. A judge told "an automated
            check flagged this" and shown the text can disagree; the
            deterministic verdict layer never sees this string.
        model: A single pydantic-ai model id. A ``str`` (not a sequence) is
            required: it selects llm_judge's FAST path. A one-element LIST
            would silently select PANEL at sampling temperature —
            ``_is_panel`` keys on TYPE, not length — which is the wrong
            contract for a destroy gate that must be reproducible.

    Returns:
        The judge's verdict, with derived doubt / needs_review.

    Raises:
        Whatever ``judge_score`` raises. This function never swallows a
        failure into a default answer — a silently-defaulted second opinion
        would be worse than the single sampled categorical it replaces.
    """
    context = (
        f"This text was retrieved as evidence for the claim: {claim_statement or '(none)'}\n"
        f"Source: {source_ref}\n"
        f"An automated check flagged it as corrupted — "
        f"{issue_description or '(no detail given)'} — which would normally "
        f"discard it. Judge the text on its own terms."
    )
    return await _one_criterion_opinion(
        content=content,
        criteria=CORRUPTION_CRITERIA,
        context=context,
        model=model,
    )


__all__ = [
    "CORRUPTION_CRITERIA",
    "QUALITY_CRITERIA",
    "QualitySecondOpinion",
    "confirm_corruption",
    "confirm_low_quality",
]
