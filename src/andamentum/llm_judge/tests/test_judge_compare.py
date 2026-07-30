"""Fast + panel ``judge_compare`` tests with a FAKE ``elicit`` layer — fully
offline. Validates both-orders execution, canonicalisation, and the
order-consistency gate.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from andamentum.llm_judge import judge_compare, signals
from andamentum.llm_judge import elicit as elicit_module

OUTPUT_A = "response A text"
OUTPUT_B = "response B text"


def _pair(reasoning: str, r1: int, tie: int, r2: int):
    return elicit_module._PairwiseDist(
        reasoning=reasoning, response_1_better=r1, tie=tie, response_2_better=r2
    )


# ── FAST (single judge) ───────────────────────────────────────────────────


async def test_judge_compare_fast_runs_both_orders_and_is_position_neutral(monkeypatch):
    # A decisive, order-consistent pair: 'a' wins clearly under both orders,
    # and the resulting order-averaged distribution is skewed enough to
    # stay under NEEDS_REVIEW_DOUBT_THRESHOLD (a 3-way histogram's entropy
    # climbs quickly, so this needs a fairly one-sided distribution).
    scripted = {
        "AB": _pair("a looks better", 85, 5, 10),
        "BA": _pair("a still looks better", 10, 5, 85),
    }

    async def fake_elicit(
        output_1, output_2, context, criteria, *, model, order, temperature
    ):
        assert temperature == elicit_module.FAST_TEMPERATURE
        return scripted[order]

    mock = AsyncMock(side_effect=fake_elicit)
    monkeypatch.setattr(elicit_module, "elicit_pairwise", mock)

    result = await judge_compare(OUTPUT_A, OUTPUT_B, model="anthropic:claude-haiku-4-5")

    assert mock.await_count == 2
    calls = mock.call_args_list
    # Position-neutrality: the caller passes output_a/output_b in the SAME
    # argument position for both calls — only the `order` keyword differs.
    assert calls[0].args[0] == OUTPUT_A
    assert calls[0].args[1] == OUTPUT_B
    assert calls[0].kwargs["order"] == "AB"
    assert calls[1].args[0] == OUTPUT_A
    assert calls[1].args[1] == OUTPUT_B
    assert calls[1].kwargs["order"] == "BA"

    canon_ab = signals.canonicalize(scripted["AB"].to_row(), "AB")
    canon_ba = signals.canonicalize(scripted["BA"].to_row(), "BA")
    avg = signals.order_average(canon_ab, canon_ba)
    assert result.winner == signals.compare_verdict(avg)
    assert result.winner == "a"
    assert result.order_consistent is True
    assert result.needs_review is False
    assert result.judges is None
    assert result.reasoning == "a looks better"  # AB-order reasoning surfaced


async def test_judge_compare_fast_order_flip_sets_needs_review(monkeypatch):
    # AB says 'a' wins; BA (after canonicalisation) says 'b' wins -> flip.
    scripted = {
        "AB": _pair("a wins", 80, 10, 10),
        "BA": _pair("response 1 (=B here) wins", 80, 10, 10),
    }

    async def fake_elicit(
        output_1, output_2, context, criteria, *, model, order, temperature
    ):
        return scripted[order]

    monkeypatch.setattr(
        elicit_module, "elicit_pairwise", AsyncMock(side_effect=fake_elicit)
    )

    result = await judge_compare(OUTPUT_A, OUTPUT_B, model="anthropic:claude-haiku-4-5")

    assert result.order_consistent is False
    assert result.needs_review is True


# ── PANEL ─────────────────────────────────────────────────────────────────


async def test_judge_compare_panel_split_returns_majority_and_needs_review(monkeypatch):
    scripted = {
        ("m1", "AB"): _pair("a wins", 80, 10, 10),
        ("m1", "BA"): _pair("a wins", 10, 10, 80),  # consistent 'a'
        ("m2", "AB"): _pair("a wins", 75, 15, 10),
        ("m2", "BA"): _pair("a wins", 15, 10, 75),  # consistent 'a'
        ("m3", "AB"): _pair("b wins", 10, 10, 80),
        ("m3", "BA"): _pair("b wins", 80, 10, 10),  # consistent 'b'
    }

    async def fake_elicit(
        output_1, output_2, context, criteria, *, model, order, temperature
    ):
        assert temperature == elicit_module.PANEL_SAMPLING_TEMPERATURE
        return scripted[(model, order)]

    mock = AsyncMock(side_effect=fake_elicit)
    monkeypatch.setattr(elicit_module, "elicit_pairwise", mock)

    result = await judge_compare(OUTPUT_A, OUTPUT_B, model=["m1", "m2", "m3"])

    assert mock.await_count == 6  # 3 models x 2 orders
    assert result.winner == "a"  # 2-1 majority
    assert result.needs_review is True  # split panel
    assert result.order_consistent is True  # every judge agreed with itself
    assert result.judges is not None
    assert [j.model for j in result.judges] == ["m1", "m2", "m3"]
    assert [j.verdict for j in result.judges] == ["a", "a", "b"]
    assert all(j.order_consistent for j in result.judges)


async def test_judge_compare_panel_order_flip_in_one_judge_flags_panel(monkeypatch):
    scripted = {
        ("m1", "AB"): _pair("a wins", 80, 10, 10),
        ("m1", "BA"): _pair("a wins", 10, 10, 80),  # consistent 'a'
        ("m2", "AB"): _pair("a wins", 80, 10, 10),
        ("m2", "BA"): _pair("inconsistent", 80, 10, 10),  # canon flips to 'b'
    }

    async def fake_elicit(
        output_1, output_2, context, criteria, *, model, order, temperature
    ):
        return scripted[(model, order)]

    monkeypatch.setattr(
        elicit_module, "elicit_pairwise", AsyncMock(side_effect=fake_elicit)
    )

    result = await judge_compare(OUTPUT_A, OUTPUT_B, model=["m1", "m2"])

    assert result.order_consistent is False  # m2 flipped -> panel flag False
    assert result.needs_review is True
    judges = result.judges
    assert judges is not None
    assert judges[0].order_consistent is True
    assert judges[1].order_consistent is False


async def test_judge_compare_single_element_list_is_panel_path(monkeypatch):
    # A one-element LIST is an explicit request for the panel code path,
    # even with only one judge: panel sampling temperature, a `judges`
    # list of length 1, not the fast path's `judges=None`.
    scripted = {
        ("m1", "AB"): _pair("a wins clearly", 90, 5, 5),
        ("m1", "BA"): _pair("a wins clearly", 5, 5, 90),
    }

    async def fake_elicit(
        output_1, output_2, context, criteria, *, model, order, temperature
    ):
        assert temperature == elicit_module.PANEL_SAMPLING_TEMPERATURE
        return scripted[(model, order)]

    monkeypatch.setattr(
        elicit_module, "elicit_pairwise", AsyncMock(side_effect=fake_elicit)
    )

    result = await judge_compare(OUTPUT_A, OUTPUT_B, model=["m1"])

    assert result.judges is not None
    assert len(result.judges) == 1
    assert result.judges[0].model == "m1"
    assert result.winner == "a"


async def test_judge_compare_panel_confidence_is_min_doubt_is_max(monkeypatch):
    scripted = {
        ("m1", "AB"): _pair("a wins clearly", 90, 5, 5),
        ("m1", "BA"): _pair("a wins clearly", 5, 5, 90),
        ("m2", "AB"): _pair("a wins narrowly", 55, 25, 20),
        ("m2", "BA"): _pair("a wins narrowly", 20, 25, 55),
    }

    async def fake_elicit(
        output_1, output_2, context, criteria, *, model, order, temperature
    ):
        return scripted[(model, order)]

    monkeypatch.setattr(
        elicit_module, "elicit_pairwise", AsyncMock(side_effect=fake_elicit)
    )

    result = await judge_compare(OUTPUT_A, OUTPUT_B, model=["m1", "m2"])

    canon1 = signals.order_average(
        signals.canonicalize(scripted[("m1", "AB")].to_row(), "AB"),
        signals.canonicalize(scripted[("m1", "BA")].to_row(), "BA"),
    )
    canon2 = signals.order_average(
        signals.canonicalize(scripted[("m2", "AB")].to_row(), "AB"),
        signals.canonicalize(scripted[("m2", "BA")].to_row(), "BA"),
    )
    expected_confidence = min(signals.max_mass(canon1), signals.max_mass(canon2))
    expected_doubt = max(
        signals.normalized_entropy(canon1), signals.normalized_entropy(canon2)
    )
    assert result.confidence == pytest.approx(expected_confidence)
    assert result.doubt == pytest.approx(expected_doubt)
    assert result.winner == "a"


# ═══════════════════════════════════════════════════════════════════════
# Regression: a hung panel must never award the win to output_a
# ═══════════════════════════════════════════════════════════════════════


async def test_a_hung_compare_panel_reports_tie_not_the_first_output(monkeypatch):
    """Two judges, each internally order-consistent, one voting 'a' and one
    voting 'b'. The panel is hung. It used to report winner='a' — i.e. the
    output the caller happened to pass FIRST — while also reporting
    order_consistent=True, which reads as "A wins and both orders agreed".
    Harvest preference labels from split panels and every one of them carries
    a systematic preference for position 1."""
    # m1 prefers whichever response is shown first... no: m1 genuinely prefers
    # A (it says so under both presentation orders); m2 genuinely prefers B.
    scripted = {
        ("m1", "AB"): _pair("A is better", 90, 5, 5),
        ("m1", "BA"): _pair("A is better", 5, 5, 90),
        ("m2", "AB"): _pair("B is better", 5, 5, 90),
        ("m2", "BA"): _pair("B is better", 90, 5, 5),
    }

    async def fake_elicit(o1, o2, context, criteria, *, model, order, temperature):
        return scripted[(model, order)]

    monkeypatch.setattr(
        elicit_module, "elicit_pairwise", AsyncMock(side_effect=fake_elicit)
    )

    result = await judge_compare(OUTPUT_A, OUTPUT_B, model=["m1", "m2"])

    assert result.winner == "tie", "a hung panel must not crown the first output"
    assert result.needs_review is True
    assert result.judges is not None
    assert [j.verdict for j in result.judges] == ["a", "b"]


async def test_a_hung_compare_panel_is_symmetric_under_swapping_the_outputs(
    monkeypatch,
):
    """The strongest statement of the no-position-bias property: swapping the
    two candidate outputs must not change the verdict of a hung panel."""

    def _run(prefers_first: str, prefers_second: str):
        scripted = {
            (prefers_first, "AB"): _pair("first is better", 90, 5, 5),
            (prefers_first, "BA"): _pair("first is better", 5, 5, 90),
            (prefers_second, "AB"): _pair("second is better", 5, 5, 90),
            (prefers_second, "BA"): _pair("second is better", 90, 5, 5),
        }

        async def fake_elicit(o1, o2, context, criteria, *, model, order, temperature):
            return scripted[(model, order)]

        return AsyncMock(side_effect=fake_elicit)

    monkeypatch.setattr(elicit_module, "elicit_pairwise", _run("m1", "m2"))
    forward = await judge_compare(OUTPUT_A, OUTPUT_B, model=["m1", "m2"])

    monkeypatch.setattr(elicit_module, "elicit_pairwise", _run("m2", "m1"))
    swapped = await judge_compare(OUTPUT_A, OUTPUT_B, model=["m1", "m2"])

    assert forward.winner == swapped.winner == "tie"


async def test_compare_confidence_is_the_mass_on_the_reported_winner(monkeypatch):
    """A judge that splits its belief evenly between a and b reports 'tie'.
    Confidence must be the mass on 'tie' — not the raw maximum, which sits on
    'a' and 'b' and would claim high confidence in a verdict that holds almost
    none of the belief."""

    async def fake_elicit(o1, o2, context, criteria, *, model, order, temperature):
        return _pair("evenly split", 45, 10, 45)

    monkeypatch.setattr(
        elicit_module, "elicit_pairwise", AsyncMock(side_effect=fake_elicit)
    )

    result = await judge_compare(OUTPUT_A, OUTPUT_B, model="m")
    assert result.winner == "tie"
    assert result.confidence == pytest.approx(0.10)
    assert result.needs_review is True


# ── expected_preference: the continuous companion to `winner` ─────────────


async def test_judge_compare_fast_exposes_expected_preference(monkeypatch):
    scripted = {
        "AB": _pair("a looks better", 85, 5, 10),
        "BA": _pair("a still looks better", 10, 5, 85),
    }

    async def fake_elicit(
        output_1, output_2, context, criteria, *, model, order, temperature
    ):
        return scripted[order]

    monkeypatch.setattr(
        elicit_module, "elicit_pairwise", AsyncMock(side_effect=fake_elicit)
    )

    result = await judge_compare(OUTPUT_A, OUTPUT_B, model="anthropic:claude-haiku-4-5")

    # E[preference for A] off the order-averaged histogram — a wins, so > 0.5.
    canon_ab = signals.canonicalize(scripted["AB"].to_row(), "AB")
    canon_ba = signals.canonicalize(scripted["BA"].to_row(), "BA")
    avg = signals.order_average(canon_ab, canon_ba)
    expected = signals.expectation(avg, signals.COMPARE_EXPECTATION_WEIGHTS)
    assert result.expected_preference == pytest.approx(expected)
    assert result.winner == "a"
    assert result.expected_preference > 0.5


async def test_judge_compare_expected_preference_symmetric_under_swap(monkeypatch):
    """Swapping which output is A must mirror expected_preference around 0.5 —
    the position-neutrality the both-orders machinery guarantees, now visible
    on the continuous signal too."""
    scripted = {
        "AB": _pair("a wins", 80, 10, 10),
        "BA": _pair("a wins", 10, 10, 80),
    }

    async def fake_elicit(
        output_1, output_2, context, criteria, *, model, order, temperature
    ):
        # Depends only on order, not on which text is A — so swapping A/B
        # inputs yields the mirror-image histogram.
        return scripted[order]

    monkeypatch.setattr(
        elicit_module, "elicit_pairwise", AsyncMock(side_effect=fake_elicit)
    )

    forward = await judge_compare(OUTPUT_A, OUTPUT_B, model="m")
    swapped_scripted = {
        "AB": _pair("b wins", 10, 10, 80),
        "BA": _pair("b wins", 80, 10, 10),
    }

    async def fake_elicit_swapped(
        output_1, output_2, context, criteria, *, model, order, temperature
    ):
        return swapped_scripted[order]

    monkeypatch.setattr(
        elicit_module, "elicit_pairwise", AsyncMock(side_effect=fake_elicit_swapped)
    )
    swapped = await judge_compare(OUTPUT_A, OUTPUT_B, model="m")

    assert forward.expected_preference == pytest.approx(
        1.0 - swapped.expected_preference
    )


async def test_judge_compare_panel_expected_preference_pools_the_judges(monkeypatch):
    scripted = {
        ("m1", "AB"): _pair("a wins", 80, 10, 10),
        ("m1", "BA"): _pair("a wins", 10, 10, 80),
        ("m2", "AB"): _pair("a wins", 75, 15, 10),
        ("m2", "BA"): _pair("a wins", 15, 10, 75),
        ("m3", "AB"): _pair("b wins", 10, 10, 80),
        ("m3", "BA"): _pair("b wins", 80, 10, 10),
    }

    async def fake_elicit(
        output_1, output_2, context, criteria, *, model, order, temperature
    ):
        return scripted[(model, order)]

    monkeypatch.setattr(
        elicit_module, "elicit_pairwise", AsyncMock(side_effect=fake_elicit)
    )

    result = await judge_compare(OUTPUT_A, OUTPUT_B, model=["m1", "m2", "m3"])

    # Pooled mean of each judge's order-averaged canonical histogram.
    avgs = []
    for m in ("m1", "m2", "m3"):
        canon_ab = signals.canonicalize(scripted[(m, "AB")].to_row(), "AB")
        canon_ba = signals.canonicalize(scripted[(m, "BA")].to_row(), "BA")
        avgs.append(signals.order_average(canon_ab, canon_ba))
    expected = signals.expectation(
        signals.mean_distributions(avgs), signals.COMPARE_EXPECTATION_WEIGHTS
    )
    assert result.expected_preference == pytest.approx(expected)
