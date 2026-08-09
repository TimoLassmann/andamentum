"""The one structured verdict an epistemic run produces.

Before this module the system decided its verdict in five independent
places — ``graph/combination._verdict_label``, ``confidence.compute_posterior``,
the writer agent's prose, ``report_generator``'s re-parse of that prose,
and ``audit_report._normalised_verdict`` — and nothing forced them to
agree. The HTML report could print a headline that contradicted the
per-claim badges in the same document.

``EpistemicVerdict`` is that decision, once. It stores the minimum
(the posterior report, the per-claim entries, the provenance counts)
and DERIVES everything a reader wants (``posterior``, ``confidence``,
``direction``, ``label``) as computed fields, so a second number can
never drift from the first. It is built inside the graph by
``CombineClaimVerdicts`` (via ``confidence.build_verdict``), persisted
on the frozen ``Snapshot``, and RENDERED — never re-decided — by the
synthesis prose, the CLI, the offline report and the HTML audit report.

``verdict_label`` here is the ONLY place in the package that compares a
posterior against ``POSTERIOR_DIRECTIONAL_BREAKPOINT``
(``tests/test_verdict_single_source.py`` pins that by source scan).
``display_label`` maps a *direction* onto the five-label user-facing
vocabulary and never re-thresholds — that is what removes the old
``>`` (combiner) vs ``>=`` (renderer) boundary disagreement.

Leaf module by construction: it imports ``pydantic``, ``..thresholds``
and ``.cap_evaluation`` and nothing else, so ``entities/snapshot.py``
and ``graph/state.py`` can type their fields against it without an
import cycle.

Architecture: Layer 1 (framework-agnostic, pure data + pure functions)
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, computed_field

from ..thresholds import POSTERIOR_DIRECTIONAL_BREAKPOINT
from .cap_evaluation import CapEvaluation

# ── The closed display vocabulary ────────────────────────────────────
#
# Five labels, chosen so their lowercased form does NOT match the
# ``[data-value="supported"]`` / ``[data-value="contradicts"]`` CSS
# rules in typeset — badges render neutral and the WORD does the
# signalling. Moved here from ``audit_report`` so the renderer reads
# the vocabulary instead of owning it.
VERDICT_CONFIRMED = "Confirmed"
VERDICT_CONFIRMED_REFINED = "Confirmed with refinement"
VERDICT_REFUTED = "Refuted"
VERDICT_INCONCLUSIVE = "Inconclusive"
VERDICT_INSUFFICIENT = "Insufficient evidence"

# The directional band, named once. The renderer imports THESE to print
# its legend rather than importing the threshold itself — so the source
# scan in ``tests/test_verdict_single_source.py`` keeps holding and the
# legend can never drift from the comparison it describes.
DIRECTIONAL_HI = POSTERIOR_DIRECTIONAL_BREAKPOINT
DIRECTIONAL_LO = 1.0 - POSTERIOR_DIRECTIONAL_BREAKPOINT


def verdict_label(p: float) -> str:
    """Map a posterior to a directional verdict label.

    Uses ``POSTERIOR_DIRECTIONAL_BREAKPOINT`` from
    ``epistemic.thresholds`` (canonical value 0.66) and its mirror
    ``1 - threshold`` for the contradicts side. Symmetric around 0.5.

    THE SINGLE THRESHOLD READ. Every posterior→label decision in the
    package goes through this function; ``display_label`` below maps
    the *result* rather than comparing the number a second time. A
    sixth derivation is what produced the disagreeing-headline defect
    this module exists to remove.
    """
    if p > DIRECTIONAL_HI:
        return "supports"
    if p < DIRECTIONAL_LO:
        return "contradicts"
    return "insufficient"


def display_label(
    direction: str,
    terminal_state: str = "completed",
    *,
    has_supports_refined: bool = False,
) -> str:
    """Map a *direction* onto the five-label display vocabulary.

    Never re-thresholds — it consumes the direction ``verdict_label``
    already decided. The one extra rule it applies is the terminal-state
    override, preserved verbatim from the renderer it replaces: an
    inquiry that did not complete (``retrieval_failed`` /
    ``oscillation_detected``) reports "Insufficient evidence" whatever
    the number says, because in that state the posterior is not a
    directional answer.
    """
    if terminal_state != "completed":
        return VERDICT_INSUFFICIENT
    if direction == "supports":
        return VERDICT_CONFIRMED_REFINED if has_supports_refined else VERDICT_CONFIRMED
    if direction == "contradicts":
        return VERDICT_REFUTED
    if direction == "insufficient":
        return VERDICT_INCONCLUSIVE
    # "no_data" / "union" — there is no scalar verdict to display.
    return VERDICT_INSUFFICIENT


class PosteriorReport(BaseModel):
    """Posterior probability P(Y) for a yes/no-style research objective.

    Primary signal is the abduction agent's verdict (mode="abductive").
    Counting is reported as a diagnostic field and only drives the
    posterior when integration didn't run (mode="counting_fallback").
    Only meaningful for verificatory and predictive questions; other
    question types return None from compute_posterior().

    Lives here (rather than in ``confidence``) so ``EpistemicVerdict``
    can embed it without the entity layer importing the repository.
    ``confidence`` re-exports the name, so
    ``from .confidence import PosteriorReport`` keeps working and the
    benchmark harness's ``model_dump`` shape is unchanged.
    """

    posterior: float = Field(description="P(Y) in [0.0, 1.0]")
    log_odds: int = Field(description="Effective log-odds from posterior")
    supporting_count: float = Field(
        description=(
            "Diagnostic: total weighted supporting evidence across active "
            "claims. Each representative contributes "
            "``1 + log(corroboration_count)``. Drives the posterior only "
            "in counting_fallback mode (when no claim received an "
            "integration verdict)."
        )
    )
    contradicting_count: float = Field(
        description=(
            "Diagnostic: total weighted contradicting evidence across "
            "active claims. Same weighting as supporting_count."
        )
    )
    counting_posterior: float = Field(
        description=(
            "Diagnostic: P(Y) implied by counting alone "
            "(sigmoid(supporting - contradicting)). For comparison with "
            "the abduction-driven posterior; not used when the abduction "
            "verdict is available."
        )
    )
    integration_verdict: str | None = Field(
        default=None,
        description="Integration assessment: 'supports', 'contradicts', 'insufficient', or None",
    )
    integration_confidence: float | None = Field(
        default=None,
        description="Integration confidence 0.0-1.0, or None if not run",
    )
    mode: str = Field(
        default="abductive",
        description=(
            "'abductive' when the posterior follows the integration "
            "verdict; 'counting_fallback' when no claim received an "
            "integration verdict and the posterior is the sigmoid of "
            "weighted counts."
        ),
    )
    terminal_state: Literal["completed", "retrieval_failed", "oscillation_detected"] = (
        Field(
            default="completed",
            description=(
                "How the investigation terminated. 'completed' for normal runs; "
                "'retrieval_failed' when evidence extraction kept returning empty "
                "content, meaning the posterior is based on insufficient data; "
                "'oscillation_detected' when one or more claims hit the "
                "scrutinise/resolve cycle cap before converging — the posterior "
                "is 0.5 (genuinely uncertain) and the inquiry did not fix belief."
            ),
        )
    )
    suspended_claim_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Claims whose IBE direction the concordance tripwire flagged as "
            "contradicted by BOTH independent signals. Their directional "
            "verdict was suspended (treated as neutral 0.5) in the "
            "aggregation — suspend-only, never flipped. Additive field: it "
            "surfaces a decision compute_posterior already made internally "
            "so ``ClaimVerdict.suspended`` can read it instead of re-running "
            "the tripwire."
        ),
    )
    objective_id: str
    question_type: str
    explanation: str


class EvidenceRef(BaseModel):
    """One evidence item's directional contribution to a claim's verdict.

    The same items, with the same eligibility filter and the same
    weighting, that produced the claim's counting mass — so a reader
    following the verdict back to its sources sees exactly what was
    counted (CONSTITUTION rule 6: don't silently drop data).
    """

    evidence_id: str
    source_ref: str = ""
    stance: Literal["supports", "contradicts"]
    mass: float = Field(
        description=(
            "Weighted directional contribution: "
            "``(1 + log(corroboration_count)) * p_stance``."
        )
    )
    quality_score: Optional[float] = None


class ClaimVerdict(BaseModel):
    """One claim's contribution to the objective-level verdict."""

    claim_id: str
    statement: str = ""
    sub_investigation_id: Optional[str] = None
    stage: str = ""
    integrated_assessment: Optional[str] = None
    integrated_confidence: Optional[float] = None
    posterior: Optional[float] = Field(
        default=None,
        description=(
            "Per-claim posterior from ``graph.combination._claim_posterior`` — "
            "so the cycle-cap penalty and the concordance suspension are "
            "already applied. None when the claim carries no integration "
            "verdict."
        ),
    )
    cycle_capped: bool = False
    abandoned: bool = False
    suspended: bool = Field(
        default=False,
        description="The concordance tripwire suspended this claim's direction.",
    )
    supporting_evidence: list[EvidenceRef] = Field(default_factory=list)
    refuting_evidence: list[EvidenceRef] = Field(default_factory=list)
    n_no_bearing: int = 0
    cap_evaluation: Optional[CapEvaluation] = Field(
        default=None,
        description=(
            "The claim's persisted cap-firing snapshot, copied onto the "
            "run-level object so a renderer never has to reach back to the "
            "Claim entity. CapEvaluation is a PERSISTED SNAPSHOT with an "
            "entity lifetime; this object is a RUN-LEVEL AGGREGATE rebuilt "
            "from the live DB on every pass through CombineClaimVerdicts. "
            "They are deliberately different things — carrying the snapshot "
            "here is what keeps the aggregate complete."
        ),
    )
    terminal_state: str = Field(
        default="completed",
        description=(
            "The objective-level terminal state, carried per claim so the "
            "badge label and the headline label cannot disagree about "
            "whether the inquiry completed."
        ),
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def direction(self) -> str:
        """'supports' | 'contradicts' | 'insufficient' | 'no_data'."""
        if self.posterior is None:
            return "no_data"
        return verdict_label(self.posterior)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def label(self) -> str:
        """Display label for this claim's badge."""
        return display_label(
            self.direction,
            self.terminal_state,
            has_supports_refined=self.integrated_assessment == "supports_refined",
        )


class VerdictProvenance(BaseModel):
    """How the verdict was reached — everything ``PosteriorReport`` does not carry.

    These counts are the routing substrate: ``CheckCompletion`` picks its
    exit from ``n_adjudicated`` / ``n_claims_total`` / ``n_abandoned`` /
    ``retrieval_failed``, and ``CheckSynthesisDemand``'s gates read
    ``combination_rule`` and ``n_no_verdict``. Deriving both the routing
    and the rendered headline from one object is what makes the
    writer/aggregator coherence contract structural rather than a
    maintained convention.
    """

    retrieval_failed: bool = False
    combination_rule: Optional[str] = Field(
        default=None,
        description=(
            "AND / OR / WEIGHTED_AND / UNION, or None for verify-mode and "
            "open-research runs that have no decomposition rule."
        ),
    )
    n_claims_total: int = 0
    n_active_claims: int = 0
    n_abandoned: int = 0
    n_cycle_capped: int = 0
    n_adjudicated: int = Field(
        default=0,
        description=(
            "Active claims carrying an ``integrated_assessment``. This is the "
            "system's canonical 'did we adjudicate?' signal — the same "
            "condition compute_posterior's no-certified-verdict gate uses."
        ),
    )
    n_no_verdict: int = Field(
        default=0,
        description="Active claims with no integration verdict (n_active - n_adjudicated).",
    )
    n_orphan: int = 0
    combination_explanation: str = ""


class EpistemicVerdict(BaseModel):
    """The single decision record for one epistemic run.

    Stores the minimum; derives the rest. ``posterior`` / ``confidence`` /
    ``explanation`` read straight off the embedded ``PosteriorReport``
    rather than being copied, so there is no second number that can
    drift.
    """

    objective_id: str
    posterior_report: Optional[PosteriorReport] = None
    claims: list[ClaimVerdict] = Field(default_factory=list)
    provenance: VerdictProvenance = Field(default_factory=VerdictProvenance)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def posterior(self) -> Optional[float]:
        return self.posterior_report.posterior if self.posterior_report else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence(self) -> Optional[float]:
        return (
            self.posterior_report.integration_confidence
            if self.posterior_report
            else None
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def explanation(self) -> str:
        return self.posterior_report.explanation if self.posterior_report else ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def terminal_state(self) -> str:
        return self.posterior_report.terminal_state if self.posterior_report else "completed"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def direction(self) -> str:
        """'supports' | 'contradicts' | 'insufficient' | 'no_data' | 'union'."""
        if self.posterior_report is None:
            return "no_data"
        if (self.provenance.combination_rule or "").upper() == "UNION":
            return "union"
        return verdict_label(self.posterior_report.posterior)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def label(self) -> str:
        """The user-facing display label for this run."""
        has_refined = any(
            c.integrated_assessment == "supports_refined"
            for c in self.claims
            if not c.abandoned
        )
        return display_label(
            self.direction,
            self.terminal_state,
            has_supports_refined=has_refined,
        )


def render_verdict_line(verdict: Optional[EpistemicVerdict]) -> str:
    """The one-line machine-readable headline, rendered from the object.

    Code-owned, not prompt-owned: the writer agent produces the answer
    prose, this function produces the line the benchmark judge and the
    report headline read. A writer that says the opposite of the object
    cannot move this string.
    """
    if verdict is None:
        return VERDICT_INSUFFICIENT
    parts = [verdict.label]
    p = verdict.posterior
    if p is not None:
        parts.append(f"posterior {p:.3f}")
    prov = verdict.provenance
    noun = "claim" if prov.n_active_claims == 1 else "claims"
    parts.append(f"{prov.n_adjudicated} of {prov.n_active_claims} {noun} adjudicated")
    if prov.n_cycle_capped:
        parts.append(f"{prov.n_cycle_capped} cycle-capped")
    if prov.retrieval_failed:
        parts.append("retrieval failed")
    return f"{parts[0]} — " + " · ".join(parts[1:]) if len(parts) > 1 else parts[0]
