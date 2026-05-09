"""Canonical thresholds for the epistemic system.

This module is the *single source of truth* for every decision-relevant
numeric threshold. Each constant is named after the philosophical
commitment it encodes, and every use site across the codebase imports
from here. The goal is twofold:

1. **Manuscript-defensibility.** A reviewer reading this one file
   can see every threshold the system uses and the principled basis
   for each value.
2. **Consistency.** Values that previously appeared as bare numerics
   (e.g. ``0.7`` for "survived" and ``0.6`` for "challenged" in
   different files) are unified — when a single concept is checked
   in several places, every site references the same constant.

The thresholds are grouped by the theoretical framework that
motivates them:

- **Popper / Lakatos** — adversarial balance breakpoints (refuted,
  contested, survived, suspicious).
- **Peirce** — bound on iterative inquiry cycling per claim.
- **Output-layer provenance** — confidence penalties applied when a
  process flag (cycle cap, retrieval failure) flags that the inquiry
  did not converge cleanly.
- **Verdict labelling** — directional and decisive posterior
  breakpoints used by combiner and synthesis-demand gates.

Adding a new threshold? Put it here, with a docstring that names
its theoretical basis and where it's read. Avoid bare numerics in
decision logic across the rest of ``epistemic/``.

Architecture: Layer 1 (framework-agnostic, pure constants).
"""

from __future__ import annotations

# ── Adversarial balance (Popper / Lakatos) ─────────────────────────
#
# ``Claim.adversarial_balance`` is a [0, 1] score reflecting how the
# claim survives an active adversarial search: 1.0 means no
# counterevidence found, 0.0 means counterevidence dominates.
#
# The principled three-band reading:
#
#   * **Refuted** (Popper): below ``ADVERSARIAL_REFUTED_THRESHOLD``,
#     adversarial evidence dominates and the claim cannot be held.
#     This is genuine falsification — the claim must demote/abandon.
#   * **Contested** (Lakatos): between REFUTED and SURVIVED, the
#     claim has substantive counter-evidence but isn't decisively
#     refuted. Cannot promote past PROVISIONAL; must remain hedged.
#   * **Survived** (Popperian corroboration): above
#     ``ADVERSARIAL_SURVIVED_THRESHOLD``, the claim has withstood a
#     genuine adversarial challenge — necessary (though not
#     sufficient) for ROBUST/ACTIONABLE stages.
#
# The thresholds are symmetric around 0.5 with ±0.2 distance, giving
# a contested middle band of width 0.4. The width is deliberately
# wide: small differences around 0.5 don't license decisive
# directional commitments. Narrowing the band would license more
# false-positive "refute" / "survive" calls.
#
# ``ADVERSARIAL_SUSPICIOUS_THRESHOLD`` is a meta-diagnostic, not a
# decision threshold: balances above 0.95 suggest the adversarial
# search itself may have been insufficient (no genuine challenge
# found). This is an indicator for human review, not for
# auto-decisions.

ADVERSARIAL_REFUTED_THRESHOLD: float = 0.3
"""Below this balance, the claim is Popper-refuted by adversarial
search. Read by: stage demotion, posterior calculation, reporters
that label refuted claims."""

ADVERSARIAL_SURVIVED_THRESHOLD: float = 0.7
"""At or above this balance, the claim has survived adversarial
challenge — Popperian corroboration. Required for promotion past
PROVISIONAL stage. Read by: stage gates, refire-skip logic,
synthesis writer guard rules, reporters."""

ADVERSARIAL_SUSPICIOUS_THRESHOLD: float = 0.95
"""Above this balance the search itself is suspicious — no
adversary should be this convinced. Read by: adversarial-balance
interpreter for diagnostic flags."""

FRAMING_TIE_SATURATION_GAP: float = (
    ADVERSARIAL_SURVIVED_THRESHOLD - ADVERSARIAL_REFUTED_THRESHOLD
)
"""Width of the framing-tie CONTESTED band, in loveliness-gap units.

Mirrors the width of the adversarial CONTESTED band — the same
"contested zone width" concept applied at the IBE / loveliness layer
rather than the Popperian survival layer. Derived from the canonical
adversarial thresholds rather than declared independently so that
revising the adversarial band automatically moves this in lockstep.

When the chosen IBE candidate's loveliness exceeds the best opposing
candidate's by at least this gap, the framing-tie cap is 1.0 (no
dampening). At a perfect tie (gap = 0), the cap is 0.5 (severe
dampening — the abductive chain has no principled tie-breaker
between opposing coherent explanations). Linear in between.

Read by: ``operations.integration._framing_tie_cap``."""


# ── Peirce cycling (one cap, three loops) ──────────────────────────
#
# Three distinct iterative loops in the graph all bound their
# re-entry count by the same number. Peirce's framework is the
# common motivation: inquiry should be bounded — at some point the
# system declares "we have circled this enough" and accepts the
# current state, even if not fully resolved. The three loops are:
#
#   1. **Investigation attempts.** A HYPOTHESIS claim that fails
#      scrutiny gets up to ``PEIRCE_CYCLE_CAP`` rounds of fresh
#      evidence-gathering before being abandoned (graph/nodes.py).
#   2. **Scrutiny↔resolve cycle.** A claim oscillating between
#      Scrutinize and ResolveUncertainties hits the cap and is
#      flagged ``cycle_capped=True`` (graph/nodes.py).
#   3. **Nested uncertainty resolution.** ResolveUncertainties
#      recursing on uncertainties spawned by the prior round bounds
#      its depth at the cap (graph/nodes.py).
#
# All three are conceptually the same Peircean "fix belief in
# bounded inquiry" commitment, so they share one constant. Tuning
# the cap is one decision, not three.
#
# Operational caps that are NOT Peirce-grounded
# (``MAX_VALIDATION_ROUNDS`` for the writer-validator loop) keep their
# own names and homes.

PEIRCE_CYCLE_CAP: int = 3
"""Bound on iterative re-entry per claim, applied uniformly across
the three Peirce-grounded loops (investigation, scrutiny-resolve,
uncertainty-depth). Read by: graph/nodes.py at all three sites."""


# ── IBE chain internal agreement (Reichenbach, applied to LLM-stochastic IBE) ──
#
# The IBE chain is four sequential LLM operations
# (Enumerate → Loveliness → Likeliness → Select), each at non-zero
# temperature. The final argmax step is sensitive to score noise: when
# the chosen and best-opposing candidates have similar combined
# scores, different runs of the same chain on the same claim can
# commit to opposite verdicts. The framing-tie cap dampens
# ``integrated_confidence`` to compensate, but argmax is discrete —
# the verdict's *direction* can still flip.
#
# ``IBE_AGREEMENT_K_DEFAULT`` is the number of independent IBE chain
# runs whose verdicts must agree on direction before
# ``integrated_assessment`` is committed for a claim. When the K runs
# don't agree, ``integrated_assessment`` falls back to
# ``insufficient`` — the same Reichenbach-style "agreement across
# independent samples" commitment that motivates the K=2 provider
# tournament. K=2 is the minimum sample size at which "agreement"
# carries any information; higher K trades more LLM calls for stricter
# agreement.
#
# The default lives here; the actual K used by a graph run is read
# from ``EpistemicGraphState.ibe_agreement_k`` so callers can override
# per-run (e.g. K=20 for an important interactive query).

IBE_AGREEMENT_K_DEFAULT: int = 2
"""Number of independent IBE chain runs whose verdicts must agree
before ``integrated_assessment`` is committed. K=1 disables the
agreement check (legacy single-run behaviour). K=2 is the minimum
sample size that detects disagreement and is the canonical default —
parallels ``RESEARCH_MODE_PROVIDER_K`` at the provider-tournament
layer. Read by: ``operations.integration.SelectBestExplanationOperation``
via ``EpistemicGraphState.ibe_agreement_k``."""


# ── Confidence penalties (output-layer provenance) ─────────────────
#
# When a process flag fires (the inquiry didn't converge cleanly),
# the verdict it produced is provisional. Surface that signal with
# reduced weight rather than discarding it (the previous
# all-capped → 0.5 anti-pattern). The penalty is applied
# multiplicatively to confidence in the integration path, and as a
# pull-toward-neutral on the aggregated posterior in the counting
# fallback path.
#
# Both penalties take the same value (0.7) because they encode the
# same epistemic claim: "inquiry was forced to terminate before
# reaching its natural conclusion; the verdict carries directional
# signal but should weigh less than a verdict from a converged
# inquiry."

CYCLE_CAP_CONFIDENCE_PENALTY: float = 0.7
"""Multiplier applied to confidence (or pull-toward-neutral on
counting posterior) when at least one contributing claim is
``cycle_capped``. See
docs/superpowers/plans/2026-05-04-confidence-honest-aggregation.md."""

RETRIEVAL_FAILED_CONFIDENCE_PENALTY: float = 0.7
"""Pull-toward-neutral on aggregated posterior when the inquiry's
``state.retrieval_failed`` flag fired. Same shape, same value as
CYCLE_CAP_CONFIDENCE_PENALTY (the two stack multiplicatively when
both apply: a doubly-provisional verdict gets 0.7 × 0.7 = 0.49 on
distance from 0.5)."""


# ── Posterior breakpoints ──────────────────────────────────────────
#
# Two distinct decisions read the aggregated posterior:
#
#   * **Verdict labelling** (combiner, _verdict_label): map a
#     posterior to "supports" / "contradicts" / "insufficient". The
#     directional breakpoint is wider than 0.5 to keep the label
#     calibrated with the underlying evidence strength.
#   * **Decisive-enough-to-skip-more-inquiry** (CheckSynthesisDemand
#     Gate 4): a posterior so far from 0.5 that further
#     investigation is unlikely to change the headline. The
#     threshold is asymmetric in stakes (false negatives on
#     "decisive" trigger more LLM cost; false positives ship a
#     wrong-confident answer), so it sits well above the labelling
#     breakpoint.

POSTERIOR_DIRECTIONAL_BREAKPOINT: float = 0.66
"""Posteriors above this map to "supports"; below ``1 -
POSTERIOR_DIRECTIONAL_BREAKPOINT`` map to "contradicts"; otherwise
"insufficient". Read by: graph/combination._verdict_label."""

POSTERIOR_DECISIVE_THRESHOLD: float = 0.85
"""Posteriors at or above this (or at or below ``1 - threshold``)
are decisive — CheckSynthesisDemand Gate 4 will not loop back for
more investigation. Read by: graph/nodes.py:CheckSynthesisDemand."""


# ── Convergence (Reichenbach common-cause / Mill's methods) ────────
#
# Multi-domain convergence: a claim that holds across genuinely
# independent evidence pools is more credible than the same claim
# holding in one pool. This is Reichenbach's common-cause
# principle — agreement across causally-independent sources is
# evidence the agreement isn't artefactual.
#
# ``convergence_detector._determine_verdict`` maps a *strength*
# score (computed from cluster count, inter-domain distance, and
# representative quality) to one of {NO_EVIDENCE, SINGLE_DOMAIN,
# PARTIAL, CONVERGENT}. The CONVERGENT label is load-bearing: it
# triggers the fast-path-to-IBE in
# ``graph/nodes.py:RunVerification`` — a SUPPORTED claim with at
# least one CONVERGENT sibling skips ``ResolveUncertainties`` and
# goes straight to integration.
#
# Thresholds here are kept at their pre-2026-05-05 values
# (no behaviour change vs. the previous bare numerics). The point
# of naming them is that the manuscript can refer to them by
# name and the reader can see them all in one place; tuning is a
# separate decision.

CONVERGENCE_STRONG_THRESHOLD: float = 0.7
"""Strength score at or above which the convergence verdict is
CONVERGENT. Below: PARTIAL. Read by:
``convergence_detector._determine_verdict``. Load-bearing — gates
the IBE fast-path."""

CONVERGENCE_INTRA_DIVERSITY_THRESHOLD: float = 0.5
"""Minimum fraction of within-cluster pairs that must be judged
methodologically independent for the cluster to count as
diverse. Read by:
``convergence_detector._compute_independence_checks``."""

CONVERGENCE_INTER_DOMAIN_DISTANCE_LOW: float = 0.3
"""Below this average inter-domain distance, clusters are too
similar to count as truly independent domains and the
``shared_error_modes`` weakness is flagged. Read by:
``convergence_detector.assess_quality``."""


__all__ = [
    "ADVERSARIAL_REFUTED_THRESHOLD",
    "ADVERSARIAL_SURVIVED_THRESHOLD",
    "ADVERSARIAL_SUSPICIOUS_THRESHOLD",
    "FRAMING_TIE_SATURATION_GAP",
    "PEIRCE_CYCLE_CAP",
    "IBE_AGREEMENT_K_DEFAULT",
    "CYCLE_CAP_CONFIDENCE_PENALTY",
    "RETRIEVAL_FAILED_CONFIDENCE_PENALTY",
    "POSTERIOR_DIRECTIONAL_BREAKPOINT",
    "POSTERIOR_DECISIVE_THRESHOLD",
    "CONVERGENCE_STRONG_THRESHOLD",
    "CONVERGENCE_INTRA_DIVERSITY_THRESHOLD",
    "CONVERGENCE_INTER_DOMAIN_DISTANCE_LOW",
]
