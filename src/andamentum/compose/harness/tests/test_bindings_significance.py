"""Significance flips read from the text around a changed number."""

from __future__ import annotations

from andamentum.compose.harness.bindings.significance import Change, find_flips


def change(text: str, old: str, new: str) -> Change:
    start = text.index(old)
    return Change(
        start=start,
        end=start + len(old),
        old=float(old),
        new=float(new),
        old_text=old,
        new_text=new,
    )


def test_p_crossing_alpha_flips_either_way():
    text = "The effect was small (p = 0.03) overall."
    (flip,) = find_flips(text, [change(text, "0.03", "0.07")])
    assert flip.kind == "p"
    text = "The effect was small (p = 0.07) overall."
    assert len(find_flips(text, [change(text, "0.07", "0.03")])) == 1


def test_p_staying_on_one_side_does_not_flip():
    text = "The effect was small (p = 0.03) overall."
    assert find_flips(text, [change(text, "0.03", "0.04")]) == []
    assert find_flips(text, [change(text, "0.03", "0.001")]) == []


def test_a_number_that_is_not_a_p_value_never_flips():
    text = "The mean difference was 0.03 units."
    assert find_flips(text, [change(text, "0.03", "0.07")]) == []


def test_p_with_less_than_and_capital_letter():
    text = "Significant (P < 0.04) in the pilot."
    assert len(find_flips(text, [change(text, "0.04", "0.06")])) == 1


def test_ci_gaining_zero_flips():
    text = "The difference was 0.5 (95% CI [0.2, 0.8])."
    (flip,) = find_flips(text, [change(text, "0.2", "-0.1")])
    assert flip.kind == "ci" and "now contains 0" in flip.message


def test_ci_losing_one_for_a_ratio_flips():
    text = "The hazard ratio was 1.2 (95% CI (0.9, 1.6))."
    (flip,) = find_flips(text, [change(text, "0.9", "1.1")])
    assert "no longer contains 1" in flip.message


def test_a_ci_shifting_without_crossing_is_not_a_flip():
    text = "The difference was 0.5 (95% CI [0.2, 0.8])."
    assert find_flips(text, [change(text, "0.2", "0.3")]) == []


def test_a_changed_limit_is_judged_against_the_unchanged_one_in_the_text():
    text = "The difference was 0.5 (95% CI [-0.2, 0.8])."
    # the upper limit changes below zero; the lower stays -0.2: the interval no
    # longer contains 0
    (flip,) = find_flips(text, [change(text, "0.8", "-0.1")])
    assert "no longer contains 0" in flip.message
