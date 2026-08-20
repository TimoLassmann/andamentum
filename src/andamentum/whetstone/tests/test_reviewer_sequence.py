"""Several reviewers in one invocation.

``--reviewer claims,methods`` runs them IN SEQUENCE over one unchanged text;
no round edits. With ``--editor`` (or ``editor`` in the list) one editor pass
follows the last reviewer and rewrites against all of their findings at once —
see ``test_final_editor.py``.

A run with more than one reviewer ends with a reconciliation, which is likewise
not a flag — asking for several opinions and being handed four reports to
cross-reference by hand is the problem it solves.
"""

from __future__ import annotations

from types import SimpleNamespace as N

import pytest

from andamentum.whetstone import cli as cli_mod
from andamentum.whetstone.v3.reviewers import PROFILES


# ── parsing ─────────────────────────────────────────────────────────────────


def test_one_name_is_a_list_of_one() -> None:
    assert cli_mod._parse_reviewers("claims") == ["claims"]


def test_order_is_preserved_because_order_is_significant() -> None:
    """The list is a SEQUENCE: the reports stack in the order the reviewers
    ran, so claims,methods and methods,claims do not produce the same file."""
    assert cli_mod._parse_reviewers("methods,claims") == ["methods", "claims"]
    assert cli_mod._parse_reviewers("claims,methods") == ["claims", "methods"]


def test_whitespace_around_names_is_tolerated() -> None:
    assert cli_mod._parse_reviewers(" claims , methods ") == ["claims", "methods"]


def test_an_unknown_name_names_itself_and_the_alternatives(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_mod._parse_reviewers("claims,statistician")
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "statistician" in err
    for name in PROFILES:
        assert name in err


def test_a_repeated_reviewer_is_refused_not_collapsed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each reviewer is a full review, so a repeat costs a full review. That
    makes it a typo every time rather than something to silently de-duplicate."""
    with pytest.raises(SystemExit) as exc:
        cli_mod._parse_reviewers("claims,methods,claims")
    assert exc.value.code == 1
    assert "more than once" in capsys.readouterr().err


def test_an_empty_value_is_refused() -> None:
    with pytest.raises(SystemExit):
        cli_mod._parse_reviewers("  ,  ")


# ── the reconciliation, rendered ────────────────────────────────────────────


def _rec(**kw):
    base = dict(verdict="", agreed=[], disagreements=[], raised_by_one=[])
    base.update(kw)
    return N(**base)


def test_agreement_comes_first_and_names_who_raised_it() -> None:
    """Independent agreement is the strongest signal a multi-reviewer run
    produces, and the hardest to spot by eye because each reviewer phrases the
    same defect differently."""
    out = cli_mod._reconciliation_markdown(
        _rec(
            verdict="The draft overclaims.",
            agreed=[
                N(issue="The conclusion outruns the data",
                  raised_by=["Claims reviewer", "Devil's advocate"],
                  quote="model size is not the main driver")
            ],
        )
    )
    assert out.startswith("The draft overclaims.")
    assert "Raised by more than one reviewer (1)" in out
    assert "Claims reviewer, Devil's advocate" in out
    assert '"model size is not the main driver"' in out


def test_a_quote_is_not_wrapped_twice() -> None:
    """Models often return the passage already quoted."""
    out = cli_mod._reconciliation_markdown(
        _rec(agreed=[N(issue="x", raised_by=["A"], quote='"already quoted"')])
    )
    assert '""' not in out


def test_both_sides_of_a_disagreement_survive() -> None:
    """Choosing between two reviewers is the author's judgement. Presenting one
    side would hide that a choice had been made."""
    out = cli_mod._reconciliation_markdown(
        _rec(
            disagreements=[
                N(about="whether the claim is overstated",
                  positions=["Claims reviewer: overstated",
                             "Framing reviewer: undersold"])
            ]
        )
    )
    assert "Claims reviewer: overstated" in out
    assert "Framing reviewer: undersold" in out
    assert "your judgement" in out


def test_single_reviewer_findings_are_kept() -> None:
    """Without this a reader who stops at the summary loses every finding
    exactly one reviewer raised — most of them, on a four-reviewer run."""
    out = cli_mod._reconciliation_markdown(
        _rec(raised_by_one=["Something only methods saw — Methods reviewer"])
    )
    assert "Raised by one reviewer (1)" in out
    assert "only methods saw" in out
