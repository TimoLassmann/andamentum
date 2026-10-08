"""Tests for the deterministic coverage table (pure, no model).

The property the module is written to protect: nothing that failed to check is
reported as a bare ``0``. ``0`` says the defect was looked for and is not
there, so lending that cell to a probe that crashed (``not run``) or to one
whose findings were all discarded by verification (``0 (n unverifiable)``)
tells the author the document is clean of something nobody established.
"""

from __future__ import annotations

from andamentum.whetstone.v3.checklist.probes import Probe, ProbeFinding
from andamentum.whetstone.v3.checklist.summary import coverage_table


def _probe(probe_id: str, title: str, *, quota: int | None = None) -> Probe:
    return Probe(id=probe_id, title=title, instruction="Find the defect.", quota=quota)


def _finding(
    probe_id: str,
    *,
    severity: str = "moderate",
    corroborated_by: tuple[str, ...] = (),
) -> ProbeFinding:
    return ProbeFinding(
        probe_id=probe_id,
        issue="an issue",
        quote=f"quote-{probe_id}-{severity}",
        severity=severity,  # type: ignore[arg-type]
        corroborated_by=corroborated_by,
    )


_BANK = [
    _probe("unmotivated_choice", "Unmotivated choices"),
    _probe("term_drift", "Drifting terms"),
    _probe("open_slot", "Biggest remaining problem"),
]


def _row(table: str, title: str) -> str:
    return next(
        line
        for line in table.splitlines()
        if line.startswith("|") and line.split("|")[1].strip() == title
    )


def _row_titles(table: str) -> list[str]:
    rows = [
        line.split("|")[1].strip()
        for line in table.splitlines()
        if line.startswith("|")
    ]
    return [t for t in rows if t not in {"Check", "---"}]


def test_a_crashed_probe_reads_not_run_and_never_zero() -> None:
    """The module's central correctness property — see the module docstring."""
    table = coverage_table(
        _BANK, [], failed_probe_ids=["term_drift"], unverifiable_counts={}
    )

    row = _row(table, "Drifting terms")
    assert "not run" in row
    assert "0" not in row  # a crashed probe must never be reported as a count


def test_a_probe_that_ran_and_found_nothing_reads_zero() -> None:
    table = coverage_table(_BANK, [], failed_probe_ids=[], unverifiable_counts={})

    cells = [c.strip() for c in _row(table, "Drifting terms").split("|")]
    assert "0" in cells
    assert "not run" not in cells


def test_a_probe_with_findings_reports_its_count_and_severity_spread() -> None:
    findings = [
        _finding("term_drift", severity="major"),
        _finding("term_drift", severity="minor"),
        _finding("term_drift", severity="major"),
    ]
    table = coverage_table(_BANK, findings, failed_probe_ids=[], unverifiable_counts={})

    row = _row(table, "Drifting terms")
    assert "| 3 |" in row
    assert "2 major, 1 minor" in row  # fixed severity order, not count order


def test_rows_follow_bank_order_not_findings_order() -> None:
    findings = [_finding("open_slot"), _finding("unmotivated_choice")]
    table = coverage_table(_BANK, findings, failed_probe_ids=[], unverifiable_counts={})

    assert _row_titles(table) == [
        "Unmotivated choices",
        "Drifting terms",
        "Biggest remaining problem",
    ]


def test_incomplete_coverage_warning_appears_when_a_probe_failed() -> None:
    table = coverage_table(
        _BANK, [], failed_probe_ids=["term_drift"], unverifiable_counts={}
    )

    assert "**Coverage is incomplete.**" in table
    assert "Drifting terms" in table.split("**Coverage is incomplete.**")[1]


def test_no_incomplete_coverage_warning_when_every_probe_ran() -> None:
    table = coverage_table(
        _BANK, [_finding("term_drift")], failed_probe_ids=[], unverifiable_counts={}
    )

    assert "Coverage is incomplete" not in table


def test_corroboration_is_surfaced_when_present() -> None:
    findings = [
        _finding("unmotivated_choice", corroborated_by=("term_drift",)),
        _finding("term_drift", corroborated_by=("unmotivated_choice",)),
    ]
    table = coverage_table(_BANK, findings, failed_probe_ids=[], unverifiable_counts={})

    assert "more than one check flagged" in table
    assert "term_drift, unmotivated_choice" in table  # both checks named


def test_a_probe_whose_findings_were_all_discarded_never_reads_a_bare_zero() -> None:
    """The second way an empty row can lie — see the module docstring.

    A probe that returned three paraphrased quotes loses all three at the
    hallucination gate. Rendering that as ``0`` tells the author the defect was
    looked for and is not there, when nothing was established at all.
    """
    table = coverage_table(
        _BANK, [], failed_probe_ids=[], unverifiable_counts={"term_drift": 3}
    )

    row = _row(table, "Drifting terms")
    assert "0 (3 unverifiable)" in row
    # …and the reader is told what that means, not left to decode the cell.
    assert "were discarded" in table
    assert "Drifting terms" in table.split("were discarded")[0]


def test_probe_titles_containing_commas_stay_readable_in_the_prose() -> None:
    """Titles are sentences, and several shipped ones contain commas.

    Comma-joining them produced an unparseable run-on in a real run — "Not so
    for 2 of them — Information needed to act, and absent, Aims without
    answers, results without aims reported instances…" — where a reader cannot
    tell how many checks are named. Quoting makes the boundaries visible.
    """
    bank = [
        _probe("missing_to_act", "Information needed to act, and absent"),
        _probe("unanswered_aim", "Aims without answers, results without aims"),
    ]
    table = coverage_table(
        bank,
        [],
        failed_probe_ids=[],
        unverifiable_counts={"missing_to_act": 2, "unanswered_aim": 1},
    )

    assert '"Information needed to act, and absent"' in table
    assert '"Aims without answers, results without aims"' in table
    # Joined as prose, so the two titles read as two items rather than four.
    assert '" and "' in table


def test_the_looked_for_and_absent_claim_is_qualified_when_findings_were_lost() -> None:
    """The prose must not assert a clean bill of health it cannot support."""
    clean = coverage_table(_BANK, [], failed_probe_ids=[], unverifiable_counts={})
    assert "which is a result: that defect was looked for and is not there" in clean

    lossy = coverage_table(
        _BANK, [], failed_probe_ids=[], unverifiable_counts={"term_drift": 1}
    )
    assert "which is a result: that defect was looked for and is not there" not in lossy
    assert "usually a result" in lossy


def test_partial_loss_is_shown_alongside_the_surviving_count() -> None:
    table = coverage_table(
        _BANK,
        [_finding("term_drift")],
        failed_probe_ids=[],
        unverifiable_counts={"term_drift": 2},
    )

    assert "1 (2 unverifiable)" in _row(table, "Drifting terms")
    # The probe DID report something, so the qualifying sentence is not needed.
    assert "usually a result" not in table


def test_a_drop_count_for_a_crashed_probe_does_not_alter_its_not_run_row() -> None:
    """``not run`` is authoritative: a crashed probe checked nothing at all."""
    table = coverage_table(
        _BANK,
        [],
        failed_probe_ids=["term_drift"],
        unverifiable_counts={"term_drift": 2},
    )

    row = _row(table, "Drifting terms")
    assert row == "| Drifting terms | not run | not run | no |"


def test_a_quota_is_rendered_so_a_saturated_row_is_visible_as_saturated() -> None:
    """``3`` and ``3 of 3`` mean different things to an author.

    The second says the check returned everything it was allowed to return, so
    the document may hold more instances it was never asked for.
    """
    bank = [_probe("term_drift", "Drifting terms", quota=3)]
    findings = [_finding("term_drift") for _ in range(3)]
    table = coverage_table(bank, findings, failed_probe_ids=[], unverifiable_counts={})

    assert "| 3 of 3 |" in _row(table, "Drifting terms")
    assert "may be hiding more" in table


def test_an_unsaturated_quota_row_does_not_claim_it_may_be_hiding_more() -> None:
    bank = [_probe("term_drift", "Drifting terms", quota=3)]
    table = coverage_table(
        bank, [_finding("term_drift")], failed_probe_ids=[], unverifiable_counts={}
    )

    assert "| 1 of 3 |" in _row(table, "Drifting terms")
    assert "may be hiding more" not in table


def test_corroboration_is_silent_when_nothing_was_co_detected() -> None:
    table = coverage_table(
        _BANK, [_finding("term_drift")], failed_probe_ids=[], unverifiable_counts={}
    )

    assert "more than one check flagged" not in table
