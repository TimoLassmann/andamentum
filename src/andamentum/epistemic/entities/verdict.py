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
``display_label`` maps a *direction* onto the six-label user-facing
vocabulary and never re-thresholds — that is what removes the old
``>`` (combiner) vs ``>=`` (renderer) boundary disagreement. It is a
total function of the direction: no terminal-state override, no second
comparison, so the objective headline and the per-claim badges in one
document cannot say different things about the same number.

Leaf module by construction: it imports ``pydantic``, ``..thresholds``,
``.cap_evaluation`` and ``.judge_opinion`` and nothing else, so
``entities/snapshot.py`` and ``graph/state.py`` can type their fields
against it without an import cycle.

Architecture: Layer 1 (framework-agnostic, pure data + pure functions)
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, computed_field

from ..thresholds import POSTERIOR_DIRECTIONAL_BREAKPOINT
from .cap_evaluation import CapEvaluation
from .judge_opinion import JudgeOpinion

# ── The closed display vocabulary ────────────────────────────────────
#
# Six labels, chosen so their lowercased form does NOT match the
# ``[data-value="supported"]`` / ``[data-value="contradicts"]`` CSS
# rules in typeset — badges render neutral and the WORD does the
# signalling. Moved here from ``audit_report`` so the renderer reads
# the vocabulary instead of owning it.
VERDICT_CONFIRMED = "Confirmed"
VERDICT_CONFIRMED_REFINED = "Confirmed with refinement"
VERDICT_REFUTED = "Refuted"
VERDICT_INCONCLUSIVE = "Inconclusive"
VERDICT_INSUFFICIENT = "Insufficient evidence"
# The sixth, added 2026-08-09. Some runs have no SCALAR objective verdict
# by construction — an open-research question type that
# ``compute_posterior`` declines to score, or a UNION decomposition whose
# rule declares the sub-investigations are facets rather than a value to
# combine. Both used to collapse onto "Insufficient evidence", which is a
# different (and false) statement: it says the evidence did not decide,
# while the per-claim badges in the same document said "Confirmed". That
# is the headline-vs-badge disagreement this module exists to remove, and
# on the open-research arm it was a regression — the previous headline at
# least answered the question.
VERDICT_PER_CLAIM = "No single verdict"

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


def display_label(direction: str, *, has_supports_refined: bool = False) -> str:
    """Map a *direction* onto the six-label display vocabulary.

    A TOTAL FUNCTION OF THE DIRECTION, and nothing else. It never
    re-thresholds (it consumes the direction ``verdict_label`` already
    decided), and — since 2026-08-09 — it no longer applies a
    terminal-state override.

    Why the override went. It was lifted verbatim from the HTML renderer
    this function replaces, where it only ever coloured a report. Moving
    it here put it on the artefact's code-owned ``> **Verdict:**`` line,
    i.e. on the string the benchmark harness writes to ``synthesis.md``
    — and there it contradicted the router. ``CheckCompletion``
    deliberately sends ``retrieval_failed`` + at least one adjudicated
    claim to the DIRECTIONAL terminal ("retrieval_failed is annotated in
    prose but does not suppress it"), so that run's posterior,
    ``artefact_type``, prose and routing all say directional while the
    headline alone said "Insufficient evidence".

    The terminal state is still reported — as a CAVEAT rather than as a
    verdict: ``render_verdict_line`` appends "retrieval failed" to the
    same line, and ``audit_report._confidence_body`` renders the state
    explicitly. For ``oscillation_detected`` the override was redundant
    anyway: ``compute_posterior`` forces the posterior to 0.5 in that
    state, so the direction is already "insufficient".

    Directions and their labels:

    * ``supports`` / ``contradicts`` — Confirmed (or Confirmed with
      refinement) / Refuted.
    * ``insufficient`` — a scalar verdict exists and sits inside the
      neutral band: Inconclusive.
    * ``no_verdict`` (claim level) — this claim was never adjudicated:
      Insufficient evidence.
    * ``no_data`` / ``union`` (objective level) — there is no single
      scalar verdict BY CONSTRUCTION, and the per-claim results are the
      answer: "No single verdict".
    """
    if direction == "supports":
        return VERDICT_CONFIRMED_REFINED if has_supports_refined else VERDICT_CONFIRMED
    if direction == "contradicts":
        return VERDICT_REFUTED
    if direction == "insufficient":
        return VERDICT_INCONCLUSIVE
    if direction in ("no_data", "union"):
        return VERDICT_PER_CLAIM
    # "no_verdict", plus anything a future direction adds: the honest
    # default is that nothing was adjudicated here.
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

    # ── How reliable was the judgment behind this contribution? ──────
    #
    # TWO NAMES, NEVER ONE BARE ``doubt``. An Evidence carries two
    # different doubts — how unsure the direction judge was, and how
    # unsure the quality second opinion was — and a single unnamed
    # field would invite conflating them.
    #
    # ``None`` means NOT MEASURED, never 0.0. Adversarial
    # counter-evidence is permanently distribution-less by
    # construction, so coercing absence to zero would read as
    # certainty the system never established.
    judgment_distribution: Optional[list[float]] = Field(
        default=None,
        description=(
            "The direction judge's verbalized belief vector "
            "[supports, contradicts, no_bearing] for this item, or None "
            "when none was captured."
        ),
    )
    judgment_doubt: Optional[float] = Field(
        default=None,
        description=(
            "Normalized entropy of ``judgment_distribution`` — how unsure "
            "the DIRECTION judge was. None when not measured."
        ),
    )
    quality_doubt: Optional[float] = Field(
        default=None,
        description=(
            "Normalized entropy of the llm_judge quality second opinion — "
            "how unsure THAT judge was that the source has substance. None "
            "on the (overwhelming) majority of items, where no near-zero "
            "quality score ever triggered the gate."
        ),
    )
    direction_opinion: Optional[JudgeOpinion] = Field(
        default=None,
        description=(
            "An independent judge's confirmation of this item's direction, "
            "when one was run (one-hot, directional, inside the per-claim "
            "budget). Carries its own doubt and an explicit ``dissents`` "
            "flag. None when no confirmation ran — which is not the same as "
            "a confirmation that agreed."
        ),
    )


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

    # ── How reliable was the reading behind this claim's verdict? ────
    #
    # A doubt signal the verdict object cannot express is a signal no
    # consumer can read. These four carry the scrutiny-time measurements
    # onto the run-level object, where the renderers and any machine
    # reader can reach them without going back to the Claim entity.
    scrutiny_verdict: Optional[str] = Field(
        default=None,
        description=(
            "The claim's scrutiny verdict ('pass' / 'fail' / "
            "'needs_resolution'). Carried because without it the doubt "
            "below is un-interpretable: doubt ABOUT WHAT READING."
        ),
    )
    scrutiny_doubt: Optional[float] = Field(
        default=None,
        description=(
            "Directional-mass-weighted mean entropy of the evidence "
            "judgments that produced ``scrutiny_verdict``. None when "
            "unmeasured — never 0.0."
        ),
    )
    scrutiny_opinion: Optional[JudgeOpinion] = Field(
        default=None,
        description=(
            "An independent judge's read on whether the evidence establishes "
            "the claim, taken only when the deterministic weight was "
            "borderline. None otherwise."
        ),
    )
    judgment_dissent_count: int = Field(
        default=0,
        description=(
            "How many of this claim's load-bearing evidence directions an "
            "independent judge contradicted, in the pass that produced the "
            "current verdict. 0 means 'none dissented' OR 'none were "
            "confirmed' — the confirmations that ran are visible on the "
            "evidence refs' ``direction_opinion``."
        ),
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def direction(self) -> str:
        """'supports' | 'contradicts' | 'insufficient' | 'no_verdict'.

        ``no_verdict`` — not ``no_data`` — because at the CLAIM level a
        missing posterior means this claim was never adjudicated, which
        reads as "Insufficient evidence". At the OBJECTIVE level the same
        absence means something different (the question admits no scalar
        verdict at all), so the two levels use different words and
        ``display_label`` can map each correctly without a mode flag.
        """
        if self.posterior is None:
            return "no_verdict"
        return verdict_label(self.posterior)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def label(self) -> str:
        """Display label for this claim's badge."""
        return display_label(
            self.direction,
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
        description=(
            "Claims in the COMBINATION SCOPE with no integration verdict — "
            "the decomposition-matched, non-abandoned subset when the "
            "objective has a decomposition, otherwise every active claim. "
            "Deliberately a narrower scope than ``n_adjudicated`` (which is "
            "over all active claims): this is what CheckSynthesisDemand's "
            "stranded-claims gate routes on, and widening it to include "
            "orphan claims would change routing and LLM cost on decomposed "
            "runs. Orphans are surfaced by ``n_orphan`` instead."
        ),
    )
    n_orphan: int = 0
    n_high_doubt_claims: int = Field(
        default=0,
        description=(
            "Active claims whose scrutiny doubt was MEASURED at or above "
            "``JUDGMENT_DOUBT_ESCALATION_THRESHOLD``."
        ),
    )
    n_doubt_unmeasured_claims: int = Field(
        default=0,
        description=(
            "Active claims carrying no measured scrutiny doubt at all. Its "
            "companion above is meaningless without it: a bare "
            "``n_high_doubt_claims == 0`` reads as 'all measured, all sure', "
            "when it may mean 'nothing was measured'."
        ),
    )
    judgment_doubt_escalated: bool = Field(
        default=False,
        description=(
            "The run spent its one judgment-doubt escalation "
            "(``CheckSynthesisDemand`` Gate 0). Lives on graph state, not the "
            "DB, so ``build_verdict`` receives it as a parameter — the same "
            "shape and the same reason as ``retrieval_failed``."
        ),
    )
    combination_explanation: str = ""
    suspended_judgment: bool = Field(
        default=False,
        description=(
            "The run was routed to the FALLIBILISM TERMINAL — "
            "``SynthesizeInsufficient`` — so the system is declining to give a "
            "directional answer, and every surface must say so.\n\n"
            "This lives on the object because the terminal is the one piece of "
            "provenance a renderer cannot re-derive from the counts: a run with "
            "posterior 0.5 and one adjudicated claim looks identical to a "
            "suspended one. Without it, each caller had to know out-of-band "
            "which of the two renderers to use — ``operations/synthesis.py`` "
            "knew (it IS the insufficient operation) while ``audit_report.py`` "
            "and ``cli_handlers.py`` did not, and both rendered "
            "'Inconclusive — posterior 0.500' as the headline of a document "
            "whose body said 'No posterior — no IBE-certified verdict.' A "
            "self-contradicting artefact is precisely what the one-verdict-"
            "object change exists to make impossible, so the flag is on the "
            "object and ``render_verdict_line`` dispatches on it."
        ),
    )


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
        """'supports' | 'contradicts' | 'insufficient' | 'no_data' | 'union'.

        ``no_data`` and ``union`` are the two states in which the run has
        NO scalar objective verdict by construction — ``compute_posterior``
        declines to score this question type, or the decomposition's UNION
        rule declares the sub-investigations are facets rather than a value
        to combine. They are not "the evidence did not decide"; the claims
        may all be adjudicated. ``display_label`` maps both to
        "No single verdict".
        """
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
        return display_label(self.direction, has_supports_refined=has_refined)


def _verdict_detail_parts(verdict: EpistemicVerdict) -> list[str]:
    """The provenance clauses that follow the label on a verdict line.

    Shared by both renderers so the two terminals cannot drift in the
    detail they report — only in the label they are entitled to print.
    """
    parts: list[str] = []
    direction = verdict.direction
    p = verdict.posterior
    if direction == "union":
        # The UNION rule declines a scalar by construction. ``compute_posterior``
        # still falls through to its confidence-weighted average and produces
        # a number, but printing it beside "No single verdict" is precisely
        # the self-contradictory line this module exists to prevent.
        parts.append("see per-claim results")
    elif direction == "no_data":
        parts.append("see per-claim results")
    elif p is not None:
        parts.append(f"posterior {p:.3f}")
    prov = verdict.provenance
    noun = "claim" if prov.n_active_claims == 1 else "claims"
    parts.append(f"{prov.n_adjudicated} of {prov.n_active_claims} {noun} adjudicated")
    if prov.n_cycle_capped:
        parts.append(f"{prov.n_cycle_capped} cycle-capped")
    if prov.retrieval_failed:
        # A CAVEAT, not a verdict. ``display_label`` deliberately no
        # longer suppresses the direction for a non-completed terminal
        # state — see its docstring.
        parts.append("retrieval failed")
    return parts


def render_verdict_line(verdict: Optional[EpistemicVerdict]) -> str:
    """The one-line machine-readable headline, rendered from the object.

    Code-owned, not prompt-owned: the writer agent produces the answer
    prose, this function produces the line the benchmark judge and the
    report headline read. A writer that says the opposite of the object
    cannot move this string.
    """
    if verdict is None:
        return VERDICT_INSUFFICIENT
    # The object knows which terminal the graph chose, so every caller gets
    # the right headline without having to know. Previously this dispatch
    # lived in the caller and only ONE of the three callers performed it.
    if verdict.provenance.suspended_judgment:
        return render_insufficient_verdict_line(verdict)
    parts = _verdict_detail_parts(verdict)
    return f"{verdict.label} — " + " · ".join(parts) if parts else verdict.label


def render_insufficient_verdict_line(verdict: Optional[EpistemicVerdict]) -> str:
    """The verdict line for the FALLIBILISM TERMINAL specifically.

    ``SynthesizeInsufficientReportOperation`` is the system declaring, in
    a typed way (``Artefact.artefact_type == "insufficient"``, a document
    titled "Insufficient Evidence"), that it is not giving a directional
    answer. Its headline must say the same thing.

    Without this, a run routed to that terminal by
    ``CheckSynthesisDemand._maybe_loop_back`` — every claim cycle-capped
    and every ``integrated_assessment`` equal to "insufficient" — rendered
    ``Inconclusive — posterior 0.500 · 1 of 1 claim adjudicated`` inside a
    document whose body says the investigation reached no integration
    verdict. That is a fresh instance of the disagreeing-headline class,
    manufactured by the renderer.

    This is NOT a second verdict derivation: the label is fixed by the
    terminal the graph chose, and every provenance clause still comes
    from the same object through ``_verdict_detail_parts``. The routing
    guarantees the direction is never ``supports``/``contradicts`` here —
    ``CheckCompletion`` sends adjudicated runs to the directional
    terminal, and the loop-back guard only falls through when no active
    claim carries a directional verdict.
    """
    if verdict is None:
        return VERDICT_INSUFFICIENT
    parts = _verdict_detail_parts(verdict)
    return f"{VERDICT_INSUFFICIENT} — " + " · ".join(parts) if parts else VERDICT_INSUFFICIENT
