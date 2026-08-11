"""Tests for the probe vocabulary — the shape contract everything else assumes.

Pure and deterministic: no agent, no model, no I/O. The invariants under test
are the ones the rest of the pipeline is allowed to rely on without re-checking
— that a finding is exactly one shape, that an absence claim is falsifiable,
and that a tag is either in the probe's closed set or gone.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from andamentum.whetstone.v3.checklist.probes import (
    Probe,
    ProbeFinding,
    normalise_tag,
)


def _probe(**overrides: object) -> Probe:
    fields: dict[str, object] = {
        "id": "unmotivated_choice",
        "title": "Choices made without a stated reason",
        "instruction": "Find the 3 choices made without a stated reason.",
    }
    fields.update(overrides)
    return Probe(**fields)  # type: ignore[arg-type]


# ── ProbeFinding: the shape invariant ───────────────────────────────────────


def test_presence_finding_with_a_quote_is_accepted() -> None:
    finding = ProbeFinding(probe_id="term_drift", issue="drifts", quote="the model")
    assert finding.shape == "presence"


def test_absence_finding_with_item_and_terms_is_accepted() -> None:
    finding = ProbeFinding(
        probe_id="missing_to_act",
        issue="no sample size given",
        missing_item="sample size",
        search_terms=("participants", "n ="),
    )
    assert finding.shape == "absence"


def test_quote_and_missing_item_together_are_rejected() -> None:
    """A finding is one shape or the other — never both."""
    with pytest.raises(ValidationError) as exc:
        ProbeFinding(
            probe_id="missing_to_act",
            issue="both shapes at once",
            quote="the cohort was small",
            missing_item="sample size",
            search_terms=("participants",),
        )
    message = str(exc.value)
    assert "ProbeFinding[missing_to_act]" in message  # error names the probe
    assert "never both" in message


def test_neither_quote_nor_missing_item_is_rejected() -> None:
    """A finding with no anchor of either kind cannot be verified at all."""
    with pytest.raises(ValidationError) as exc:
        ProbeFinding(probe_id="open_slot", issue="something is wrong")
    message = str(exc.value)
    assert "ProbeFinding[open_slot]" in message
    assert "neither a quote nor a missing_item" in message


def test_absence_without_search_terms_is_rejected() -> None:
    """Without search terms an absence claim is unfalsifiable — i.e. a guess."""
    with pytest.raises(ValidationError) as exc:
        ProbeFinding(
            probe_id="unanswered_aim",
            issue="the stated aim is never answered",
            missing_item="answer to the stated aim",
        )
    message = str(exc.value)
    assert "ProbeFinding[unanswered_aim]" in message
    assert "search_terms" in message


# ── normalise_tag ───────────────────────────────────────────────────────────


def test_normalise_tag_returns_the_canonical_member_on_exact_match() -> None:
    assert normalise_tag("population", ("population", "setting")) == "population"


def test_normalise_tag_ignores_case_and_surrounding_whitespace() -> None:
    assert normalise_tag("  PopUlation \n", ("population", "setting")) == "population"


def test_normalise_tag_drops_a_tag_outside_the_allowed_set() -> None:
    """Wrong tags are dropped, not repaired — the drop rate is the signal."""
    assert normalise_tag("timescale", ("population", "setting")) == ""


def test_normalise_tag_returns_empty_when_the_probe_has_no_tags() -> None:
    assert normalise_tag("population", ()) == ""


# ── Probe: quota semantics and the id pattern ───────────────────────────────


def test_allows_empty_is_true_exactly_when_quota_is_none() -> None:
    assert _probe(quota=None).allows_empty is True
    assert _probe(quota=3).allows_empty is False
    assert _probe(quota=1).allows_empty is False


@pytest.mark.parametrize("bad_id", ["Bad-Id", "1x", ""])
def test_probe_id_pattern_rejects_non_snake_case_ids(bad_id: str) -> None:
    """Ids are join keys for corroboration and the coverage table."""
    with pytest.raises(ValidationError):
        _probe(id=bad_id)
