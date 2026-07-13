"""Fast + panel ``judge_score`` tests with a FAKE ``elicit`` layer — fully
offline, no network, no live Ollama.

Patches ``andamentum.llm_judge.elicit.elicit_criterion_score`` directly (the
documented monkeypatch seam) — ``panel.py`` calls it as ``elicit.<name>``,
so patching the attribute on the ``elicit`` module object is visible to
every caller regardless of how many modules imported ``elicit``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from andamentum.llm_judge import Criterion, judge_score, signals
from andamentum.llm_judge import elicit as elicit_module

CRITERIA = [
    Criterion(name="correctness", description="Is it correct?"),
    Criterion(name="clarity", description="Is it clear?"),
]


def _dist(reasoning: str, meets: int, partial: int, fails: int):
    return elicit_module._CriterionDist(
        reasoning=reasoning, meets=meets, partial=partial, fails=fails
    )


# ── FAST (single judge) ───────────────────────────────────────────────────


async def test_judge_score_fast_assembles_in_order_and_derives_signals(monkeypatch):
    scripted = {
        "correctness": _dist("looks right", 80, 15, 5),
        "clarity": _dist("clear enough", 60, 30, 10),
    }

    async def fake_elicit(output, criterion, context, *, model, temperature):
        assert temperature == elicit_module.FAST_TEMPERATURE
        return scripted[criterion.name]

    mock = AsyncMock(side_effect=fake_elicit)
    monkeypatch.setattr(elicit_module, "elicit_criterion_score", mock)

    result = await judge_score(
        "some output", criteria=CRITERIA, model="anthropic:claude-haiku-4-5"
    )

    # Assembled in the order criteria were given.
    assert [c.criterion for c in result.per_criterion] == ["correctness", "clarity"]
    assert result.per_criterion[0].reasoning == "looks right"
    assert result.judges is None

    d1 = signals.normalize_three(80, 15, 5)
    d2 = signals.normalize_three(60, 30, 10)
    mean = signals.mean_distributions([d1, d2])
    expected_verdict = signals.argmax_label(
        mean, signals.SCORE_LABELS, signals.SCORE_TIEBREAK
    )

    assert result.overall == expected_verdict
    assert result.confidence == pytest.approx(max(mean))
    assert result.doubt == pytest.approx(signals.normalized_entropy(mean))
    assert mock.await_count == len(CRITERIA)


async def test_judge_score_fast_needs_review_follows_doubt_threshold(monkeypatch):
    # A near-uniform distribution -> high doubt -> needs_review True even
    # though there is only one judge (no split possible in fast mode).
    scripted = {"correctness": _dist("torn", 34, 33, 33)}

    async def fake_elicit(output, criterion, context, *, model, temperature):
        return scripted[criterion.name]

    monkeypatch.setattr(
        elicit_module, "elicit_criterion_score", AsyncMock(side_effect=fake_elicit)
    )

    result = await judge_score(
        "ambiguous output",
        criteria=[Criterion(name="correctness", description="Is it correct?")],
        model="anthropic:claude-haiku-4-5",
    )
    assert result.doubt >= signals.NEEDS_REVIEW_DOUBT_THRESHOLD
    assert result.needs_review is True


async def test_judge_score_fast_low_doubt_no_review(monkeypatch):
    scripted = {"correctness": _dist("confident", 95, 3, 2)}

    async def fake_elicit(output, criterion, context, *, model, temperature):
        return scripted[criterion.name]

    monkeypatch.setattr(
        elicit_module, "elicit_criterion_score", AsyncMock(side_effect=fake_elicit)
    )

    result = await judge_score(
        "clear output",
        criteria=[Criterion(name="correctness", description="Is it correct?")],
        model="anthropic:claude-haiku-4-5",
    )
    assert result.doubt < signals.NEEDS_REVIEW_DOUBT_THRESHOLD
    assert result.needs_review is False


async def test_judge_score_default_criteria_used(monkeypatch):
    from andamentum.llm_judge import DEFAULT_CRITERIA

    async def fake_elicit(output, criterion, context, *, model, temperature):
        return _dist("fine", 100, 0, 0)

    monkeypatch.setattr(
        elicit_module, "elicit_criterion_score", AsyncMock(side_effect=fake_elicit)
    )

    result = await judge_score("some output", model="anthropic:claude-haiku-4-5")

    assert [c.criterion for c in result.per_criterion] == [
        c.name for c in DEFAULT_CRITERIA
    ]
    assert result.overall == "meets"


# ── PANEL ─────────────────────────────────────────────────────────────────

_SINGLE_CRITERION = [Criterion(name="correctness", description="Is it correct?")]


async def test_judge_score_panel_split_returns_majority_and_needs_review(monkeypatch):
    scripted = {
        "m1": _dist("meets, confident", 90, 5, 5),
        "m2": _dist("meets, confident", 85, 10, 5),
        "m3": _dist("partial, confident", 10, 80, 10),
    }

    async def fake_elicit(output, criterion, context, *, model, temperature):
        assert temperature == elicit_module.PANEL_SAMPLING_TEMPERATURE
        return scripted[model]

    mock = AsyncMock(side_effect=fake_elicit)
    monkeypatch.setattr(elicit_module, "elicit_criterion_score", mock)

    result = await judge_score(
        "some output", criteria=_SINGLE_CRITERION, model=["m1", "m2", "m3"]
    )

    assert result.overall == "meets"  # 2-1 majority
    assert result.needs_review is True  # split panel
    assert result.judges is not None
    assert [j.model for j in result.judges] == ["m1", "m2", "m3"]
    assert [j.verdict for j in result.judges] == ["meets", "meets", "partial"]

    # Panel confidence is the least convinced judge's belief in the verdict
    # the PANEL REPORTS ('meets') — NOT the minimum of each judge's confidence
    # in its own verdict. The dissenter m3 was 0.80 confident in 'partial',
    # but gave 'meets' only 0.10; publishing 0.80 as the panel's confidence in
    # 'meets' would attach a dissenter's belief in one label to a different
    # label entirely. 0.10 is the honest number: someone on this panel thinks
    # 'meets' is nearly wrong.
    masses_on_meets = [
        signals.mass_on(
            signals.normalize_three(90, 5, 5), signals.SCORE_LABELS, "meets"
        ),
        signals.mass_on(
            signals.normalize_three(85, 10, 5), signals.SCORE_LABELS, "meets"
        ),
        signals.mass_on(
            signals.normalize_three(10, 80, 10), signals.SCORE_LABELS, "meets"
        ),
    ]
    assert result.confidence == pytest.approx(min(masses_on_meets))
    assert result.confidence == pytest.approx(0.10)
    # The per-judge votes still report each judge's confidence in ITS OWN
    # verdict — that is the per-judge fact a caller reading judges[] wants.
    assert result.judges[2].verdict == "partial"
    assert result.judges[2].confidence == pytest.approx(0.80)
    assert mock.await_count == 3  # one call per judge, one criterion each


async def test_judge_score_panel_unanimous_low_doubt_no_review(monkeypatch):
    scripted = {
        "m1": _dist("meets", 90, 5, 5),
        "m2": _dist("meets", 88, 7, 5),
        "m3": _dist("meets", 92, 4, 4),
    }

    async def fake_elicit(output, criterion, context, *, model, temperature):
        return scripted[model]

    monkeypatch.setattr(
        elicit_module, "elicit_criterion_score", AsyncMock(side_effect=fake_elicit)
    )

    result = await judge_score(
        "some output", criteria=_SINGLE_CRITERION, model=["m1", "m2", "m3"]
    )

    assert result.overall == "meets"
    assert result.needs_review is False
    expected_confidences = [
        signals.max_mass(signals.normalize_three(90, 5, 5)),
        signals.max_mass(signals.normalize_three(88, 7, 5)),
        signals.max_mass(signals.normalize_three(92, 4, 4)),
    ]
    assert result.confidence == pytest.approx(min(expected_confidences))


async def test_judge_score_panel_unanimous_high_doubt_still_flagged(monkeypatch):
    # All three judges agree the verdict is 'meets', but each is internally
    # torn (near-uniform distribution) -> doubt alone triggers needs_review
    # despite unanimity.
    scripted = {
        "m1": _dist("torn but leaning meets", 40, 32, 28),
        "m2": _dist("torn but leaning meets", 38, 33, 29),
        "m3": _dist("torn but leaning meets", 36, 34, 30),
    }

    async def fake_elicit(output, criterion, context, *, model, temperature):
        return scripted[model]

    monkeypatch.setattr(
        elicit_module, "elicit_criterion_score", AsyncMock(side_effect=fake_elicit)
    )

    result = await judge_score(
        "ambiguous output", criteria=_SINGLE_CRITERION, model=["m1", "m2", "m3"]
    )

    assert result.overall == "meets"
    assert result.needs_review is True
    assert result.doubt >= signals.NEEDS_REVIEW_DOUBT_THRESHOLD


async def test_judge_score_single_element_list_is_panel_path(monkeypatch):
    # A one-element LIST is an explicit request for the panel code path
    # (per the module docstring / `_normalize_models` contract), even
    # though there is only one judge: panel sampling temperature, a
    # `judges` list of length 1, not the fast path's `judges=None`.
    scripted = {"m1": _dist("meets", 90, 5, 5)}

    async def fake_elicit(output, criterion, context, *, model, temperature):
        assert temperature == elicit_module.PANEL_SAMPLING_TEMPERATURE
        return scripted[model]

    monkeypatch.setattr(
        elicit_module, "elicit_criterion_score", AsyncMock(side_effect=fake_elicit)
    )

    result = await judge_score("some output", criteria=_SINGLE_CRITERION, model=["m1"])

    assert result.judges is not None
    assert len(result.judges) == 1
    assert result.judges[0].model == "m1"
    assert result.needs_review is False  # unanimous (trivially, N=1) + low doubt


async def test_judge_score_panel_per_criterion_is_pooled_mean(monkeypatch):
    scripted = {
        "m1": _dist("meets", 80, 10, 10),
        "m2": _dist("meets", 60, 30, 10),
    }

    async def fake_elicit(output, criterion, context, *, model, temperature):
        return scripted[model]

    monkeypatch.setattr(
        elicit_module, "elicit_criterion_score", AsyncMock(side_effect=fake_elicit)
    )

    result = await judge_score(
        "some output", criteria=_SINGLE_CRITERION, model=["m1", "m2"]
    )

    only = result.per_criterion[0]
    assert only.criterion == "correctness"
    # Pooled mean of [80,10,10] and [60,30,10] normalised -> [70,20,10].
    assert only.meets == pytest.approx(70, abs=1)
    assert only.partial == pytest.approx(20, abs=1)
    assert only.fails == pytest.approx(10, abs=1)
    assert "panel mean" in only.reasoning.lower()


async def test_panel_per_criterion_rows_sum_to_exactly_100(monkeypatch):
    """The sum-to-100 contract CriterionScore documents, asserted where a
    consumer actually reads it: on the assembled panel result. Three judges
    landing on a flat mean used to render as 33/33/33 = 99."""
    thirds = {
        "correctness": _dist("a", 100, 0, 0),
        "clarity": _dist("a", 100, 0, 0),
    }
    seconds = {
        "correctness": _dist("b", 0, 100, 0),
        "clarity": _dist("b", 0, 100, 0),
    }
    thirds_c = {
        "correctness": _dist("c", 0, 0, 100),
        "clarity": _dist("c", 0, 0, 100),
    }
    by_model = {"m1": thirds, "m2": seconds, "m3": thirds_c}

    async def fake_elicit(output, criterion, context, *, model, temperature):
        return by_model[model][criterion.name]

    monkeypatch.setattr(
        elicit_module, "elicit_criterion_score", AsyncMock(side_effect=fake_elicit)
    )

    result = await judge_score("out", criteria=CRITERIA, model=["m1", "m2", "m3"])
    for row in result.per_criterion:
        assert row.meets + row.partial + row.fails == 100, (
            f"{row.criterion} row sums to "
            f"{row.meets + row.partial + row.fails}, not 100"
        )


async def test_fast_path_publishes_the_normalized_row_not_the_models_raw_triple(
    monkeypatch,
):
    """A small model reading meets/partial/fails as three INDEPENDENT
    confidences returns e.g. 60/50/30 — every field in range, no constraint
    violated, nothing raises. The fast path used to copy that straight into
    the public CriterionScore, so a consumer reading meets/100 got 0.60 while
    the verdict had actually been derived from 60/140 = 0.43. The panel path
    normalized and reported 43 for the identical judge. One judge, two
    contracts, and the discrepancy is invisible."""

    async def fake_elicit(output, criterion, context, *, model, temperature):
        return _dist("independent confidences", 60, 50, 30)

    monkeypatch.setattr(
        elicit_module, "elicit_criterion_score", AsyncMock(side_effect=fake_elicit)
    )

    fast = await judge_score("out", criteria=_SINGLE_CRITERION, model="m")
    row = fast.per_criterion[0]
    assert row.meets + row.partial + row.fails == 100
    assert row.meets == 43  # 60/140, not 60

    # And the fast path must agree with the panel path for the same judge.
    panel = await judge_score("out", criteria=_SINGLE_CRITERION, model=["m", "m"])
    prow = panel.per_criterion[0]
    assert (row.meets, row.partial, row.fails) == (prow.meets, prow.partial, prow.fails)


async def test_fast_path_row_still_matches_a_well_behaved_judge_exactly(monkeypatch):
    """Normalisation must be a no-op when the model does what it was asked."""

    async def fake_elicit(output, criterion, context, *, model, temperature):
        return _dist("well behaved", 80, 15, 5)

    monkeypatch.setattr(
        elicit_module, "elicit_criterion_score", AsyncMock(side_effect=fake_elicit)
    )
    result = await judge_score("out", criteria=_SINGLE_CRITERION, model="m")
    row = result.per_criterion[0]
    assert (row.meets, row.partial, row.fails) == (80, 15, 5)
