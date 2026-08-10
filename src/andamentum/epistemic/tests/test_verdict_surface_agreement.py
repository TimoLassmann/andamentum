"""Every surface must render the same terminal — the one-verdict-object thesis.

Regression pins for the SEAM-1 defect found by the cross-cutting fragmentation
audit (2026-08-09). CHANGE 2 gave the system one typed ``EpistemicVerdict`` and
made synthesis / the CLI / the audit report renderers of it — but the terminal
the graph chose was NOT on the object, so each caller had to know out-of-band
which of the two renderers to use. ``operations/synthesis.py`` knew (it *is* the
insufficient operation); ``audit_report.py`` and ``cli_handlers.py`` did not, and
both produced::

    Inconclusive — posterior 0.500 · 1 of 1 claim adjudicated     <- headline
    No posterior — no IBE-certified verdict.                       <- body

inside a single document. A self-contradicting artefact is exactly what the
one-verdict-object change exists to make impossible, so ``suspended_judgment``
now lives on ``VerdictProvenance`` and ``render_verdict_line`` dispatches on it.

These tests pin the PROPERTY (all surfaces agree), not the wording, so a future
change to the phrasing does not have to touch them — but a future change that
re-introduces caller-side dispatch will fail them.
"""

from __future__ import annotations

from andamentum.epistemic.entities.verdict import (
    VERDICT_INSUFFICIENT,
    EpistemicVerdict,
    PosteriorReport,
    VerdictProvenance,
    render_insufficient_verdict_line,
    render_verdict_line,
)


def _report() -> PosteriorReport:
    """A mid posterior — the value that made the two cases indistinguishable."""
    return PosteriorReport(
        posterior=0.5,
        # int, not 0.0: `PosteriorReport.log_odds` is declared int and the
        # producer rounds (`confidence.py`: `round(math.log(p/(1-p)))`) — it is
        # a coarse reporting quantity, not the raw log-odds.
        log_odds=0,
        supporting_count=1,
        contradicting_count=1,
        counting_posterior=0.5,
        objective_id="obj",
        question_type="empirical",
        explanation="fixture",
    )


def _suspended_verdict() -> EpistemicVerdict:
    """A verdict shaped exactly like the failing case: a mid posterior and an
    adjudicated claim, so the counts alone are indistinguishable from a
    genuinely inconclusive directional run."""
    return EpistemicVerdict(
        objective_id="obj-1",
        posterior_report=_report(),
        provenance=VerdictProvenance(
            n_claims_total=1,
            n_active_claims=1,
            n_adjudicated=1,
            suspended_judgment=True,
        ),
    )


def _directional_verdict() -> EpistemicVerdict:
    return EpistemicVerdict(
        objective_id="obj-2",
        posterior_report=_report(),
        provenance=VerdictProvenance(
            n_claims_total=1,
            n_active_claims=1,
            n_adjudicated=1,
            suspended_judgment=False,
        ),
    )


def test_suspended_verdict_renders_the_insufficient_line_from_the_generic_entry_point():
    """The bug: a caller reaching for the generic renderer got a directional
    headline. The generic renderer must now dispatch on the object itself."""
    v = _suspended_verdict()
    assert render_verdict_line(v) == render_insufficient_verdict_line(v)


def test_suspended_verdict_headline_never_claims_a_direction():
    """The LABEL is what SEAM 1 was about: a suspended run must not be
    headlined as a directional finding.

    Scope note: the posterior clause DOES still appear here, and that is
    deliberate — ``render_insufficient_verdict_line`` pins the label to the
    terminal while letting every provenance clause come from the same object
    via ``_verdict_detail_parts``, so there is exactly one derivation. This
    test therefore pins the label, not the absence of a number. Whether a
    suspended headline should quote a posterior at all is a separate design
    question, deliberately not decided here."""
    line = render_verdict_line(_suspended_verdict())
    assert line.startswith(VERDICT_INSUFFICIENT), (
        f"suspended run headlined as a directional verdict: {line!r}"
    )
    assert "Inconclusive" not in line


def test_directional_breakpoint_maps_to_insufficient_at_the_boundary():
    """B1 from the adversarial audit: the documented semantics — *exactly*
    0.66 and *exactly* 0.34 are insufficient, not directional — was unpinned.
    Flipping ``>`` to ``>=`` in verdict_label passed the entire suite.

    This is the test that fails under that mutation."""
    from andamentum.epistemic.entities.verdict import (
        DIRECTIONAL_HI,
        DIRECTIONAL_LO,
        verdict_label,
    )

    hi = DIRECTIONAL_HI
    lo = DIRECTIONAL_LO
    assert verdict_label(hi) == "insufficient", (
        f"posterior exactly {hi} must be insufficient, got {verdict_label(hi)!r}"
    )
    assert verdict_label(lo) == "insufficient", (
        f"posterior exactly {lo} must be insufficient, got {verdict_label(lo)!r}"
    )
    # And the first values that ARE directional, so the test pins a boundary
    # rather than merely a wide band.
    assert verdict_label(hi + 1e-9) == "supports"
    assert verdict_label(lo - 1e-9) == "contradicts"


def test_directional_verdict_is_unaffected():
    """The dispatch must be narrow — an ordinary run keeps its detail parts,
    otherwise the fix would have flattened every headline in the system."""
    line = render_verdict_line(_directional_verdict())
    assert line != render_insufficient_verdict_line(_directional_verdict())
    assert "0.500" in line


def test_the_two_verdicts_differ_only_by_the_terminal_flag():
    """Guards the premise of this whole test file. If counts alone could tell
    the two apart, `suspended_judgment` would be redundant and a renderer could
    re-derive the terminal — so this asserts they genuinely cannot."""
    a = _suspended_verdict().model_dump()
    b = _directional_verdict().model_dump()
    a["objective_id"] = b["objective_id"] = "same"
    a["provenance"].pop("suspended_judgment")
    b["provenance"].pop("suspended_judgment")
    assert a == b


def test_synthesize_insufficient_node_declares_it_writes_the_verdict():
    """P6 contract: the node stamps state.verdict, so it must declare that.
    A node mutating undeclared state is the class of bug the contract exists
    to catch."""
    from andamentum.epistemic.graph.nodes import SynthesizeInsufficient

    assert "verdict" in SynthesizeInsufficient.writes
