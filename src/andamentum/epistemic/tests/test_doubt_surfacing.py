"""Doubt is measured, carried — and now rendered to a reader.

CHANGE 1 put an uncertainty signal on the two highest-leverage
judgments. A signal no reader can see cannot inform a reader's trust, so
these tests pin the render path: the phrasing is single-sourced, it
speaks only when a measurement actually raised something, and an
unmeasured judgment never renders as a certain one.

No DB, no LLM, no HTTP — in-memory ``EpistemicVerdict`` / ``ReportData``
throughout.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

from rich.console import Console

from andamentum.epistemic import cli_handlers
from andamentum.epistemic.audit_report import _evidence_line, build_audit_report
from andamentum.epistemic.entities.verdict import (
    ClaimVerdict,
    EpistemicVerdict,
    PosteriorReport,
    VerdictProvenance,
)
from andamentum.epistemic.report_data import (
    DOUBT_SUSPICION_CAVEAT,
    ClaimSummary,
    ConfidenceScores,
    EvidenceSummary,
    InvestigationStats,
    ReportData,
    doubt_is_suspicious,
    format_doubt,
    judgment_reliability_line,
)
from andamentum.epistemic.thresholds import JUDGMENT_DOUBT_ESCALATION_THRESHOLD

_ABOVE = JUDGMENT_DOUBT_ESCALATION_THRESHOLD + 0.2
_BELOW = JUDGMENT_DOUBT_ESCALATION_THRESHOLD - 0.2


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _verdict(
    *,
    posterior: float | None = 0.85,
    terminal_state: str = "completed",
    claims: list[ClaimSummary] | None = None,
    n_high_doubt_claims: int = 0,
    n_doubt_unmeasured_claims: int = 0,
    judgment_doubt_escalated: bool = False,
) -> EpistemicVerdict:
    claims = claims or []
    report = (
        PosteriorReport(
            posterior=posterior,
            log_odds=0,
            supporting_count=0.0,
            contradicting_count=0.0,
            counting_posterior=posterior,
            objective_id="obj-test",
            question_type="verificatory",
            explanation="test",
            terminal_state=terminal_state,  # type: ignore[arg-type]
        )
        if posterior is not None
        else None
    )
    verdict = EpistemicVerdict(
        objective_id="obj-test",
        posterior_report=report,
        claims=[
            ClaimVerdict(
                claim_id=c.claim_id,
                statement=c.statement,
                stage=c.stage,
                posterior=posterior,
                # ``verdict_doubt`` is DERIVED on ClaimVerdict (it returns the
                # cap's doubt for a cycle-capped claim, scrutiny's otherwise).
                # These fixtures are not capped, so seeding the settable
                # ``scrutiny_doubt`` is what makes the property return it.
                scrutiny_doubt=c.verdict_doubt,
            )
            for c in claims
        ],
        provenance=VerdictProvenance(
            n_claims_total=len(claims),
            n_active_claims=len(claims),
            n_adjudicated=len(claims),
            n_high_doubt_claims=n_high_doubt_claims,
            n_doubt_unmeasured_claims=n_doubt_unmeasured_claims,
            judgment_doubt_escalated=judgment_doubt_escalated,
        ),
    )
    labels = {cv.claim_id: cv.label for cv in verdict.claims}
    for c in claims:
        c.verdict_label = labels.get(c.claim_id, "")
    return verdict


def _claim(
    claim_id: str = "c1", *, verdict_doubt: float | None = None
) -> ClaimSummary:
    return ClaimSummary(
        claim_id=claim_id,
        statement=f"Statement for {claim_id}",
        scope="general",
        assumptions=[],
        stage="supported",
        verdict_doubt=verdict_doubt,
    )


def _report_data(
    *,
    verdict: EpistemicVerdict,
    claims: list[ClaimSummary] | None = None,
    evidence: list[EvidenceSummary] | None = None,
) -> ReportData:
    evidence = evidence or []
    return ReportData(
        research_question="Does X cause Y?",
        clarified_question="Does X cause Y in adults?",
        investigation_date=datetime(2026, 8, 10),
        model_used="openai:gpt-5.4-nano",
        database_name="test",
        direct_answer="The evidence supports the claim.",
        question_type="verificatory",
        verdict_object=verdict,
        claims=claims or [],
        evidence=evidence,
        stats=InvestigationStats(total_evidence=len(evidence)),
        confidence_scores=ConfidenceScores(
            posterior=verdict.posterior,
            posterior_question_type="verificatory",
            terminal_state=verdict.terminal_state,
        ),
    )


def _confidence_row(atoms: list[dict[str, Any]]) -> str:
    items = [a for a in atoms if a.get("kind") == "items"][0]
    return next(
        e["body"] for e in items["entries"] if e["label"] == "How confident are we?"
    )


def _glance_section(atoms: list[dict[str, Any]]) -> str:
    return [a for a in atoms if a.get("heading") == "Evidence at a glance"][0][
        "content"
    ]


# ──────────────────────────────────────────────────────────────────────
# The shared vocabulary
# ──────────────────────────────────────────────────────────────────────


class TestDoubtVocabulary:
    def test_unmeasured_doubt_never_renders_as_a_number(self) -> None:
        """PINS: ``None`` must not collapse onto ``0.00``.

        Mutation that breaks it: ``f"{doubt or 0:.2f}"`` — which turns
        "we never checked" into "the judge was completely sure", the one
        misreading the whole signal exists to prevent.
        """
        assert format_doubt(None) == "not measured"
        assert "0.00" not in format_doubt(None)

    def test_a_measured_doubt_above_the_cutoff_is_marked_as_such(self) -> None:
        """A bare ``0.70`` means nothing to a reader who does not know
        the cutoff. PINS: the marking, and the one cutoff it uses.

        Mutation that breaks it: dropping the ``(unsure)`` marker, or
        marking every measured value.
        """
        assert format_doubt(_ABOVE).startswith(f"{_ABOVE:.2f}")
        assert "unsure" in format_doubt(_ABOVE)
        assert "unsure" not in format_doubt(_BELOW)

    def test_the_cutoff_is_the_packages_one_doubt_cutoff(self) -> None:
        """PINS: the render path must not grow a private threshold —
        ``test_doubt_scale_single_source`` keeps that one equal to
        ``llm_judge``'s, and a second one here would escape it.

        Mutation that breaks it: hardcoding ``0.6`` in the renderer.
        """
        assert doubt_is_suspicious(JUDGMENT_DOUBT_ESCALATION_THRESHOLD)
        assert not doubt_is_suspicious(JUDGMENT_DOUBT_ESCALATION_THRESHOLD - 1e-9)
        assert not doubt_is_suspicious(None)

    def test_the_line_is_silent_when_no_count_raised_anything(self) -> None:
        """Speak-only-on-a-signal, the same rule the per-claim gate-trace
        note follows.

        Mutation that breaks it: rendering the line unconditionally,
        which trains a reader to read the (usually reassuring) sentence
        as a clean bill of health.
        """
        assert judgment_reliability_line(_verdict(claims=[_claim()])) is None
        assert judgment_reliability_line(None) is None

    def test_the_line_reports_the_unmeasured_count_beside_the_high_one(self) -> None:
        """PINS: a bare "0 claims were unsure" must never stand alone —
        it reads as "all measured, all sure" when it may mean "nothing
        was measured".

        Mutation that breaks it: dropping the ``n_doubt_unmeasured_claims``
        clause, or gating the whole line on ``n_high_doubt_claims``.
        """
        line = judgment_reliability_line(
            _verdict(
                claims=[_claim("c1"), _claim("c2")],
                n_doubt_unmeasured_claims=2,
            )
        )
        assert line is not None
        assert "2 of 2 carried no doubt measurement at all" in line

    def test_the_line_names_every_signal_that_fired_and_always_caveats(self) -> None:
        """PINS: all three provenance signals reach the reader, and the
        honesty clause travels with them.

        Mutation that breaks it: dropping the escalation clause, or
        rendering the counts without ``DOUBT_SUSPICION_CAVEAT`` — a
        number with no caveat implies a precision doubt does not have.
        """
        line = judgment_reliability_line(
            _verdict(
                claims=[_claim("c1"), _claim("c2"), _claim("c3")],
                n_high_doubt_claims=1,
                n_doubt_unmeasured_claims=2,
                judgment_doubt_escalated=True,
            )
        )
        assert line is not None
        assert "1 of 3 active claims" in line
        assert f"{JUDGMENT_DOUBT_ESCALATION_THRESHOLD:.2f}" in line
        assert "2 of 3 carried no doubt measurement" in line
        assert "one doubt-triggered extra round" in line
        assert DOUBT_SUSPICION_CAVEAT in line

    def test_the_caveat_says_low_doubt_proves_nothing(self) -> None:
        """The standing clause is not decoration: doubt is a
        suspicion-raiser in one direction only.

        Mutation that breaks it: softening the caveat into "low doubt
        indicates a reliable reading".
        """
        assert "suspicion-raiser" in DOUBT_SUSPICION_CAVEAT
        assert "not evidence that the reading is correct" in DOUBT_SUSPICION_CAVEAT


# ──────────────────────────────────────────────────────────────────────
# The HTML audit report
# ──────────────────────────────────────────────────────────────────────


class TestAuditReportSurfacesDoubt:
    def test_reliability_sits_in_the_confidence_row(self) -> None:
        """Doubt belongs next to the confidence it qualifies.

        Mutation that breaks it: dropping the append in
        ``_confidence_body``, or moving the sentence to a section of its
        own where a reader has to go and correlate it with the number.
        """
        data = _report_data(
            verdict=_verdict(claims=[_claim()], n_high_doubt_claims=1),
            claims=[_claim()],
        )
        body = _confidence_row(build_audit_report(data))
        assert "Probability the claim is true" in body
        assert "Judgment reliability:" in body
        assert DOUBT_SUSPICION_CAVEAT in body

    def test_reliability_survives_a_run_with_no_posterior(self) -> None:
        """The terminal-state branches return early. PINS: reliability
        is appended after them, not inside the posterior branch — a run
        that ended in oscillation is precisely one whose reader wants
        the reliability picture.

        Mutation that breaks it: putting the append inside the
        ``p is not None`` path.
        """
        data = _report_data(
            verdict=_verdict(
                terminal_state="uncertified",
                claims=[_claim()],
                n_doubt_unmeasured_claims=1,
            ),
            claims=[_claim()],
        )
        body = _confidence_row(build_audit_report(data))
        assert "IBE-certified verdict" in body
        assert "Judgment reliability:" in body

    def test_the_confidence_row_is_unchanged_when_nothing_was_raised(self) -> None:
        """PINS: the quiet case stays quiet.

        Mutation that breaks it: an unconditional reliability sentence,
        which adds noise to every report and dilutes the one case that
        matters.
        """
        data = _report_data(verdict=_verdict(claims=[_claim()]), claims=[_claim()])
        body = _confidence_row(build_audit_report(data))
        assert "Judgment reliability:" not in body

    def test_per_claim_table_carries_the_doubt_beside_the_verdict(self) -> None:
        """The per-claim table is where a reader decides how much of each
        verdict to believe, so ``ClaimSummary.scrutiny_doubt`` renders
        there rather than staying a carried-but-unread field.

        Mutation that breaks it: dropping the column, or dropping the
        ``ClaimSummary.scrutiny_doubt`` read and rendering a constant.
        """
        claims = [
            _claim("c1", verdict_doubt=_ABOVE),
            _claim("c2", verdict_doubt=None),
        ]
        data = _report_data(verdict=_verdict(claims=claims), claims=claims)
        text = _glance_section(build_audit_report(data))
        assert "| Claim | Verdict | Judgment doubt |" in text
        assert f"{_ABOVE:.2f} (unsure)" in text
        # The unmeasured claim says so in its own cell.
        assert "not measured" in text

    def test_the_doubt_column_is_explained_where_it_is_shown(self) -> None:
        """A bare number in a table column is uninterpretable.

        Mutation that breaks it: rendering the column without the
        footnote, leaving a reader to guess whether high is good.

        The footnote must describe what ``Claim.scrutiny_doubt`` actually
        is — the mass-weighted mean of each judgment's OWN entropy, i.e.
        how internally unsure the individual judges were. It is not
        inter-judge agreement: two confidently opposed judges both score
        near 0, which is exactly why ``judgment_dissent_count`` exists as
        a separate signal. An "the judges were of one mind" reading of a
        ~0 cell is the inversion this assertion pins out.
        """
        claims = [_claim("c1", verdict_doubt=_BELOW)]
        data = _report_data(verdict=_verdict(claims=claims), claims=claims)
        text = _glance_section(build_audit_report(data))
        assert "how internally unsure the individual evidence judgments" in text
        assert "It is NOT agreement between judges" in text
        assert "of one mind" not in text
        assert DOUBT_SUSPICION_CAVEAT in text

    def test_an_unsure_evidence_judgment_is_annotated_on_its_own_line(self) -> None:
        """Item-level doubt, spoken only above the cutoff.

        Mutation that breaks it: annotating every item (noise on every
        line), or never annotating (``EvidenceSummary.judgment_doubt``
        goes back to being carried and unread), or reinstating the
        empty-``judgment_reasoning`` branch that returned the bare
        reference and discarded every parenthetical with it.
        """
        unsure = EvidenceSummary(
            evidence_id="e1",
            source_type="pubmed",
            source_ref="PMID:12345678",
            extracted_content="x",
            support_judgment="supports",
            judgment_reasoning="The trial reports a benefit.",
            judgment_doubt=_ABOVE,
        )
        sure = EvidenceSummary(
            evidence_id="e2",
            source_type="pubmed",
            source_ref="PMID:87654321",
            extracted_content="x",
            support_judgment="supports",
            judgment_reasoning="The trial reports a benefit.",
            judgment_doubt=_BELOW,
        )
        unmeasured = EvidenceSummary(
            evidence_id="e3",
            source_type="pubmed",
            source_ref="PMID:11111111",
            extracted_content="x",
            support_judgment="supports",
            judgment_reasoning="The trial reports a benefit.",
        )
        # ``judgment_reasoning`` is Optional and the annotation must not
        # depend on it: an unsure item with no judge sentence is exactly
        # the item a reader most needs flagged.
        unsure_silent = EvidenceSummary(
            evidence_id="e4",
            source_type="pubmed",
            source_ref="PMID:22222222",
            extracted_content="x",
            support_judgment="supports",
            judgment_reasoning=None,
            judgment_doubt=_ABOVE,
        )
        assert "the direction judge was unsure" in _evidence_line(unsure)
        assert f"{_ABOVE:.2f}" in _evidence_line(unsure)
        assert "unsure" not in _evidence_line(sure)
        assert "unsure" not in _evidence_line(unmeasured)
        silent_line = _evidence_line(unsure_silent)
        assert "the direction judge was unsure" in silent_line
        assert f"{_ABOVE:.2f}" in silent_line
        assert "PMID:22222222" in silent_line


# ──────────────────────────────────────────────────────────────────────
# The CLI
# ──────────────────────────────────────────────────────────────────────


def _cli_output(verdict: EpistemicVerdict | None, monkeypatch) -> str:
    buffer = io.StringIO()
    monkeypatch.setattr(
        cli_handlers,
        "console",
        Console(file=buffer, width=200, no_color=True, highlight=False),
    )
    cli_handlers._print_ask_results(
        "Does X cause Y?",
        [],
        [],
        [],
        {"synthesis": {"summary": "An answer."}},
        False,
        "none",
        None,
        verdict=verdict,
    )
    return buffer.getvalue()


class TestCliSurfacesDoubt:
    def test_high_doubt_is_visible_beside_the_verdict(self, monkeypatch) -> None:
        """PINS: the CLI verdict output reports reliability, not just the
        label and the counts.

        Mutation that breaks it: deleting the reliability print — which
        is the state the CLI was in before this change, with the signal
        measured, persisted, and shown to nobody.
        """
        out = _cli_output(
            _verdict(claims=[_claim()], n_high_doubt_claims=1),
            monkeypatch,
        )
        assert "Verdict:" in out
        assert "Judgment reliability:" in out
        assert "suspicion-raiser" in out

    def test_the_cli_is_quiet_when_nothing_was_raised(self, monkeypatch) -> None:
        """Mutation that breaks it: an unconditional line."""
        out = _cli_output(_verdict(claims=[_claim()]), monkeypatch)
        assert "Verdict:" in out
        assert "Judgment reliability:" not in out

    def test_the_two_surfaces_say_the_same_sentence(self, monkeypatch) -> None:
        """ONE measurement, ONE wording.

        The whole point of putting the phrasing in ``report_data`` is
        that the HTML report and the CLI cannot describe the same
        numbers in two different sets of words — the failure mode the
        verdict object was introduced to end, reappearing one field
        down.

        Mutation that breaks it: either surface formatting the counts
        itself.
        """
        verdict = _verdict(
            claims=[_claim("c1"), _claim("c2")],
            n_high_doubt_claims=1,
            n_doubt_unmeasured_claims=1,
        )
        sentence = judgment_reliability_line(verdict)
        assert sentence is not None
        html = _confidence_row(
            build_audit_report(
                _report_data(verdict=verdict, claims=[_claim("c1"), _claim("c2")])
            )
        )
        cli = _cli_output(verdict, monkeypatch)
        assert sentence in html
        # Rich wraps at the console width, so compare on collapsed
        # whitespace rather than the literal line breaks.
        assert " ".join(sentence.split()) in " ".join(cli.split())
