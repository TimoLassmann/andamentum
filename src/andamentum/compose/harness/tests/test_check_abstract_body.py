from __future__ import annotations

from andamentum.compose.harness.checks.abstract_body import AbstractBodyCheck

from ._helpers import CTX, make_view


def run(view):
    return AbstractBodyCheck().run(view, CTX)


def doc(abstract: str, body: str):
    return make_view(
        ("Abstract", 1),
        (abstract, None),
        ("Results", 1),
        (body, None),
    )


def test_no_abstract_no_diagnostics():
    assert run(make_view(("Intro", 1), ("We found 45%.", None))) == []


def test_matching_numbers_pass():
    assert (
        run(
            doc(
                "Mean age was 45.2 years; 30% responded.",
                "Age was 45.2 years. 30 percent responded.",
            )
        )
        == []
    )


def test_missing_number_is_error_per_number():
    out = run(
        doc("We enrolled 120 patients; 33% improved.", "We enrolled 120 patients.")
    )
    assert [d.severity for d in out] == ["error"]
    assert out[0].evidence == ["33%"]
    assert out[0].paragraph_id == "p1"
    assert (out[0].start, out[0].end) == (26, 29)


def test_percent_must_match_percent():
    out = run(doc("Response was 45%.", "We enrolled 45 patients."))
    assert len(out) == 1


def test_unit_must_match():
    out = run(doc("Dose was 5 mg.", "A total of 5 days passed."))
    assert len(out) == 1
    assert run(doc("Dose was 5 mg.", "Dose: 5 mg daily.")) == []


def test_years_and_glued_digits_skipped():
    assert run(doc("In 2019, COVID-19 and H2O mattered.", "Nothing numeric.")) == []


def test_thousands_and_negative():
    assert (
        run(doc("n = 1,200 and change -3.5.", "Of 1200 people, change was -3.5.")) == []
    )
    assert len(run(doc("change -3.5.", "change was 3.5."))) == 1


def test_significant_phrase():
    out = run(
        doc("Treatment significantly reduced pain.", "Pain fell by a small amount.")
    )
    assert [d.severity for d in out] == ["warning"]
    assert (
        run(
            doc(
                "Treatment significantly reduced pain.",
                "The reduction was significant.",
            )
        )
        == []
    )


def test_negated_significant_needs_negated_body():
    out = run(
        doc("There was no significant difference.", "The effect was significant.")
    )
    assert len(out) == 1
    assert (
        run(doc("There was no significant difference.", "It was not significant."))
        == []
    )


def test_abstract_style_paragraph():
    view = make_view(("We found 77 cases.", None, "Abstract"), ("Body.", None))
    assert len(run(view)) == 1


def test_body_numbers_exclude_abstract_itself():
    view = make_view(
        ("Abstract", 1), ("We saw 9 cases.", None), ("Methods", 1), ("Text.", None)
    )
    assert len(run(view)) == 1
