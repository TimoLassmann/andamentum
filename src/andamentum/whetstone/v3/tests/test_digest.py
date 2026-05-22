"""Tests for the deterministic digest helpers + document-model assembly."""

from __future__ import annotations

from andamentum.whetstone.v3.digest import (
    build_document_model,
    find_citation_markers,
    gist_for,
    has_citation,
)
from andamentum.whetstone.v3.model import Claim, Section, Span


def _section(text: str, *, id="s1", title="S", start=0) -> Section:
    return Section(id=id, title=title, text=text, start=start, end=start + len(text))


def test_find_citation_markers_numeric_pandoc_authoryear() -> None:
    text = "As shown [12] and [3, 4] and [@smith2020] and (Jones et al., 2019)."
    found = find_citation_markers(text)
    assert "[12]" in found
    assert "[3, 4]" in found
    assert "[@smith2020]" in found
    assert "(Jones et al., 2019)" in found


def test_has_citation() -> None:
    assert has_citation("recovers 4x more associations [12]") is True
    assert has_citation("the system is robust to noise") is False


def test_gist_is_first_sentence_without_heading() -> None:
    s = _section("## Methods\n\nWe trained the model on data. Then we evaluated it.")
    g = gist_for(s)
    assert g == "We trained the model on data."


def test_build_document_model_sets_has_citation_and_facets() -> None:
    src = "We recover 4x more [12]. The system is robust."
    sec = _section(src)
    claims = [
        Claim(
            id="c1",
            quote="We recover 4x more [12].",
            span=Span(section_id="s1", start=0, end=24),
        ),
        Claim(
            id="c2",
            quote="The system is robust.",
            span=Span(section_id="s1", start=25, end=46),
        ),
    ]
    model = build_document_model(src, [sec], claims)
    by_id = {c.id: c for c in model.claims}
    assert by_id["c1"].has_citation is True
    assert by_id["c2"].has_citation is False
    assert len(model.gists) == 1 and model.gists[0].section_id == "s1"
    assert any(c.marker == "[12]" for c in model.citations)
    assert model.source == src
