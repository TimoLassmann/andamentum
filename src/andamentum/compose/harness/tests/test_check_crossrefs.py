from __future__ import annotations

from andamentum.compose.harness.checks.crossrefs import CrossrefsCheck

from ._helpers import CTX, field, make_view


def run(view):
    return CrossrefsCheck().run(view, CTX)


def test_registered():
    from andamentum.compose.harness.kernel import select

    assert [c.spec.id for c in select(ids=["crossrefs"])] == ["crossrefs"]


def test_clean_document():
    view = make_view(
        ("As Figure 1 and Table 2 show.", None),
        ("Figure 1. A caption.", None),
        ("Table 2: another.", None),
    )
    assert run(view) == []


def test_missing_figure_is_error_when_other_captions_exist():
    view = make_view(("See Figure 3.", None), ("Figure 1. Cap.", None))
    (d,) = run(view)
    assert d.severity == "error"
    assert d.paragraph_id == "p0"
    assert (d.start, d.end) == (4, 12)


def test_no_captions_at_all_is_warning():
    (d,) = run(make_view(("See Table 1.", None)))
    assert d.severity == "warning"


def test_caption_style_counts():
    view = make_view(("See Figure 2.", None), ("Figure 2", None, "Caption"))
    assert run(view) == []


def test_caption_paragraph_not_a_mention():
    view = make_view(("Figure 5. Shows Figure 5 itself.", None))
    assert run(view) == []


def test_crossref_target_missing():
    view = make_view(
        ("text", None),
        fields=[
            field("crossref", "word", data={"target": "_Ref1", "target_exists": False}),
            field("crossref", "word", data={}),
            field("crossref", "word", data={"target": "_Ref2", "target_exists": True}),
            field("crossref", "word", data={"target": "_Ref3"}),
        ],
    )
    out = run(view)
    assert [d.severity for d in out] == ["error", "error"]
    assert "does not exist" in out[0].message
    assert "no target" in out[1].message


def test_unparsed_citation_field_and_missing_bibliography():
    view = make_view(
        ("text", None),
        fields=[
            field("citation", "zotero", data={}),
            field("citation", "endnote", data={"id": 1}),
            field("citation", "none", data={}),
        ],
    )
    out = run(view)
    messages = [d.message for d in out]
    assert any("zotero citation field has no parseable data" in m for m in messages)
    assert any("no bibliography field" in m for m in messages)
    assert len(out) == 2


def test_bibliography_present():
    view = make_view(
        ("text", None),
        fields=[
            field("citation", "zotero", data={"items": []} | {"k": 1}),
            field("bibliography", "zotero"),
        ],
    )
    assert run(view) == []
