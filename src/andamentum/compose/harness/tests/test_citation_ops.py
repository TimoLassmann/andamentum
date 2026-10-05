"""InsertCitation and UpdateBibliography: Zotero fields, plain text, the reference list."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from lxml import etree  # type: ignore

from andamentum.compose.harness.docx.fields import classify_field
from andamentum.compose.harness.docx.ooxml import w
from andamentum.compose.harness.docx.reader import open_docx
from andamentum.compose.harness.docx.writer import DocxAdapter
from andamentum.compose.harness.kernel import (
    CitationItem,
    CommentOp,
    DocumentView,
    InsertCitation,
    Transaction,
    UpdateBibliography,
)
from andamentum.compose.harness.kernel.csl import (
    CitationStyle,
    bibliography_entry,
    citation_text,
    detect_style,
    work_key,
)
from andamentum.compose.harness.md.adapter import MarkdownAdapter
from andamentum.compose.harness.spike import make_spike_docx
from andamentum.compose.harness.spike.make import (
    build_docx,
    complex_field,
    paragraph,
    run,
)
from andamentum.compose.harness.spike.survival import SOFFICE, libreoffice_roundtrip

AUTHOR = "andamentum compose"
ABSTRACT = "p2-10000003"
INTRO = "p4-10000005"

SMITH = {
    "id": "smith2021",
    "type": "article-journal",
    "title": "Weekly salt logging in invented clinics",
    "container-title": "Journal of Invented Results",
    "volume": "12",
    "issue": "3",
    "page": "45-67",
    "author": [
        {"family": "Smith", "given": "Jane Q."},
        {"family": "Jones", "given": "Kim"},
    ],
    "issued": {"date-parts": [[2021]]},
    "DOI": "10.5555/saltlog.smith",
}
LEE = {
    "id": "lee2019",
    "type": "article-journal",
    "title": "Sodium diaries across invented cohorts",
    "container-title": "Invented Nutrition Review",
    "author": [{"family": "Lee", "given": "Min"}],
    "issued": {"date-parts": [[2019]]},
    "DOI": "10.5555/saltlog.lee",
}
DOE = {
    "id": "doe2020",
    "title": "Sodium and salt logging in invented cohorts",
    "author": [{"family": "Doe", "given": "Jane"}],
    "issued": {"date-parts": [[2020]]},
    "DOI": "10.9999/spike.001",
}


def _item(csl: dict, **kwargs) -> CitationItem:
    return CitationItem(csl=csl, **kwargs)


def _txn(*ops) -> Transaction:
    return Transaction(ops=list(ops), author=AUTHOR, reason="test", source="test")


def _text(view: DocumentView, paragraph_id: str) -> str:
    return next(p.text for p in view.paragraphs if p.id == paragraph_id)


def _root(path: Path):
    with zipfile.ZipFile(path) as archive:
        return etree.fromstring(archive.read("word/document.xml"))


@pytest.fixture
def spike(tmp_path: Path) -> Path:
    return make_spike_docx(tmp_path / "spike.docx")


@pytest.fixture
def adapter() -> DocxAdapter:
    return DocxAdapter()


def _end_of(view: DocumentView, paragraph_id: str) -> int:
    return len(_text(view, paragraph_id)) - 1  # before the final full stop


# ------------------------------------------------------------------ the formatter


def test_detect_style_reads_comma_numbering_and_brackets():
    assert detect_style(["(Doe, 2020)", "(Roe & Poe, 2019)"]) == CitationStyle(
        comma=True, ampersand=True
    )
    assert detect_style(["(Doe 2020)"]) == CitationStyle()
    assert detect_style(["[1]", "[2, 3]"]).kind == "numeric"
    assert detect_style(["(1)", "(2)"]).open == "("
    assert detect_style(["1", "2,3"]) == CitationStyle(
        kind="numeric", open="[", close="]"
    )
    assert detect_style([]) == CitationStyle()


def test_citation_text_author_year_forms():
    style = CitationStyle()
    smith, lee = _item(SMITH), _item(LEE)
    assert citation_text([smith], style) == "(Smith and Jones 2021)"
    assert citation_text([smith, lee], style) == "(Smith and Jones 2021; Lee 2019)"
    three = {**SMITH, "author": SMITH["author"] + [{"family": "Wu"}]}
    assert citation_text([_item(three)], style) == "(Smith et al. 2021)"
    commas = CitationStyle(comma=True, ampersand=True)
    assert citation_text([smith], commas) == "(Smith & Jones, 2021)"
    located = _item(LEE, locator="12", prefix="see", suffix="for details")
    assert citation_text([located], style) == "(see Lee 2019, p. 12 for details)"
    assert (
        citation_text([_item(LEE, locator="12-14")], style) == "(Lee 2019, pp. 12-14)"
    )
    nameless = {"title": "An anonymous report", "issued": {"date-parts": [[2018]]}}
    assert citation_text([_item(nameless)], style) == "(An anonymous report 2018)"


def test_citation_text_numeric_needs_numbers():
    style = CitationStyle(kind="numeric", open="[", close="]")
    numbers = {work_key(SMITH): 2, work_key(LEE): 5}
    assert citation_text([_item(SMITH), _item(LEE)], style, numbers=numbers) == "[2, 5]"
    with pytest.raises(ValueError):
        citation_text([_item(SMITH)], style)


def test_bibliography_entry_forms():
    apa = bibliography_entry(SMITH, CitationStyle(comma=True))
    assert apa == (
        "Smith, J. Q., & Jones, K. (2021). Weekly salt logging in invented clinics. "
        "Journal of Invented Results, 12(3), 45–67. https://doi.org/10.5555/saltlog.smith"
    )
    numeric = bibliography_entry(
        SMITH, CitationStyle(kind="numeric", open="[", close="]"), number=3
    )
    assert numeric.startswith("[3] Smith JQ, Jones K. Weekly salt logging")
    assert "2021;12(3):45–67" in numeric and numeric.endswith(
        "doi:10.5555/saltlog.smith"
    )
    dotted = bibliography_entry(
        LEE, CitationStyle(kind="numeric", open="(", close=")"), number=1
    )
    assert dotted.startswith("1. Lee M.")
    with pytest.raises(ValueError):
        bibliography_entry(LEE, CitationStyle(kind="numeric"))


# ------------------------------------------------------------------ Zotero mode


def test_citation_in_a_zotero_document_is_a_tracked_zotero_field(
    spike: Path, adapter: DocxAdapter, tmp_path: Path
):
    before = spike.read_bytes()
    view = adapter.open(spike)
    at = _end_of(view, ABSTRACT)
    result = adapter.apply(
        view,
        _txn(InsertCitation(paragraph_id=ABSTRACT, at=at, items=[_item(SMITH)])),
        output=tmp_path / "out.docx",
    )
    assert result.refused == [] and result.applied == 1 and result.untouched.ok
    assert spike.read_bytes() == before
    after = adapter.open(result.output)
    # The document's own style is "(Doe, 2020)": a comma, "and" between two authors.
    assert _text(after, ABSTRACT).endswith("intake(Smith and Jones, 2021).")
    token = next(
        f
        for f in after.fields
        if f.paragraph_id == ABSTRACT and f.kind == "citation" and f.manager == "zotero"
    )
    assert token.display == "(Smith and Jones, 2021)"
    cited = token.data["citationItems"][0]  # type: ignore[index]
    assert cited["itemData"]["title"] == SMITH["title"]
    assert cited["itemData"]["DOI"] == SMITH["DOI"]
    assert "uris" not in cited  # the work did not come from Zotero
    assert token.data["properties"]["formattedCitation"] == token.display  # type: ignore[index]
    # A tracked insertion by the author, the whole field inside one w:ins.
    root = _root(result.output)
    inserted = [
        e
        for e in root.iter(w("ins"))
        if e.get(w("author")) == AUTHOR and e.find(f".//{w('instrText')}") is not None
    ]
    assert len(inserted) == 1
    assert len(inserted[0].findall(f".//{w('fldChar')}")) == 3


def test_a_zotero_item_carries_its_uri_and_locator(
    spike: Path, adapter: DocxAdapter, tmp_path: Path
):
    uri = "http://zotero.org/users/local/abc/items/SMITH001"
    view = adapter.open(spike)
    result = adapter.apply(
        view,
        _txn(
            InsertCitation(
                paragraph_id=ABSTRACT,
                at=_end_of(view, ABSTRACT),
                items=[
                    _item(SMITH, uri=uri, locator="7", prefix="see", suffix="and refs")
                ],
            )
        ),
        output=tmp_path / "out.docx",
    )
    after = adapter.open(result.output)
    token = [
        f for f in after.fields if f.paragraph_id == ABSTRACT and f.manager == "zotero"
    ][-1]
    cited = token.data["citationItems"][0]  # type: ignore[index]
    assert cited["uris"] == [uri]
    assert (cited["locator"], cited["prefix"], cited["suffix"]) == (
        "7",
        "see",
        "and refs",
    )
    assert token.display == "(see Smith and Jones, 2021, p. 7 and refs)"


def test_a_uri_makes_a_zotero_field_even_in_a_document_without_any(
    tmp_path: Path, adapter: DocxAdapter
):
    path = build_docx(
        tmp_path / "plain.docx",
        paragraph(run("Salt matters (Doe 2020).")) + paragraph(run("More text here.")),
    )
    view = adapter.open(path)
    uri = "http://zotero.org/users/1/items/ABC"
    result = adapter.apply(
        view,
        _txn(InsertCitation(paragraph_id="p1", at=14, items=[_item(LEE, uri=uri)])),
        output=tmp_path / "out.docx",
    )
    after = adapter.open(result.output)
    assert [
        f.manager
        for f in after.fields
        if f.kind == "citation" and f.manager == "zotero"
    ] == ["zotero"]
    assert _text(after, "p1") == "More text here(Lee 2019)."


def test_two_citations_in_one_paragraph_each_become_a_field(
    spike: Path, adapter: DocxAdapter, tmp_path: Path
):
    view = adapter.open(spike)
    text = _text(view, ABSTRACT)
    first = text.index("participants")
    result = adapter.apply(
        view,
        _txn(
            InsertCitation(paragraph_id=ABSTRACT, at=first, items=[_item(SMITH)]),
            InsertCitation(paragraph_id=ABSTRACT, at=len(text) - 1, items=[_item(LEE)]),
        ),
        output=tmp_path / "out.docx",
    )
    assert result.applied == 2 and result.refused == []
    after = adapter.open(result.output)
    shown = [
        f.display
        for f in after.fields
        if f.paragraph_id == ABSTRACT and f.manager == "zotero"
    ]
    assert shown == ["(Smith and Jones, 2021)", "(Lee, 2019)"]
    assert "48 (Smith and Jones, 2021)participants" in _text(
        after, ABSTRACT
    )  # `at` is verbatim: no space is added


def test_a_comment_after_a_citation_in_the_same_transaction_lands_on_its_text(
    spike: Path, adapter: DocxAdapter, tmp_path: Path
):
    view = adapter.open(spike)
    text = _text(view, ABSTRACT)
    start = text.index("four weeks")
    result = adapter.apply(
        view,
        _txn(
            InsertCitation(
                paragraph_id=ABSTRACT, at=text.index("participants"), items=[_item(LEE)]
            ),
            CommentOp(
                paragraph_id=ABSTRACT, start=start, end=start + 10, text="Check this."
            ),
        ),
        output=tmp_path / "out.docx",
    )
    assert result.refused == []
    comment = next(
        c for c in adapter.open(result.output).comments if c.text == "Check this."
    )
    assert (
        _text(adapter.open(result.output), ABSTRACT)[comment.start : comment.end]
        == "four weeks"
    )


def test_a_citation_inside_a_field_result_is_refused_and_counted(
    spike: Path, adapter: DocxAdapter, tmp_path: Path
):
    view = adapter.open(spike)
    inside = next(
        f for f in view.fields if f.manager == "zotero" and f.kind == "citation"
    )
    result = adapter.apply(
        view,
        _txn(
            InsertCitation(paragraph_id=INTRO, at=inside.start + 3, items=[_item(LEE)]),
            InsertCitation(
                paragraph_id=ABSTRACT, at=_end_of(view, ABSTRACT), items=[_item(LEE)]
            ),
        ),
        output=tmp_path / "out.docx",
    )
    assert result.applied == 1
    assert len(result.refused) == 1 and "field result" in result.refused[0]


def test_a_numeric_zotero_document_gets_numbers(tmp_path: Path, adapter: DocxAdapter):
    def instruction(item: dict, shown: str) -> str:
        payload = {
            "citationID": "x",
            "properties": {
                "formattedCitation": shown,
                "plainCitation": shown,
                "noteIndex": 0,
            },
            "citationItems": [{"id": item["id"], "itemData": item}],
            "schema": "x",
        }
        return "ADDIN ZOTERO_ITEM CSL_CITATION " + json.dumps(payload)

    path = build_docx(
        tmp_path / "numeric.docx",
        paragraph(
            run("Intake is high ")
            + complex_field(instruction(DOE, "[1]"), "[1]")
            + run(".")
        )
        + paragraph(run("A second paragraph of text.")),
    )
    view = adapter.open(path)
    result = adapter.apply(
        view,
        _txn(
            InsertCitation(paragraph_id="p1", at=26, items=[_item(SMITH), _item(DOE)]),
        ),
        output=tmp_path / "out.docx",
    )
    after = adapter.open(result.output)
    # Numbers follow first appearance: Doe is already 1, Smith becomes 2.
    assert _text(after, "p1") == "A second paragraph of text[2, 1]."


# ------------------------------------------------------------- Zotero bibliography


def test_update_bibliography_extends_the_zotero_field_with_missing_works_only(
    spike: Path, adapter: DocxAdapter, tmp_path: Path
):
    view = adapter.open(spike)
    result = adapter.apply(
        view,
        _txn(
            InsertCitation(
                paragraph_id=ABSTRACT,
                at=_end_of(view, ABSTRACT),
                items=[_item(SMITH), _item(DOE)],
            ),
            UpdateBibliography(),
        ),
        output=tmp_path / "out.docx",
    )
    assert result.refused == [] and result.applied == 2
    after = adapter.open(result.output)
    bibliography = next(f for f in after.fields if f.kind == "bibliography")
    assert "Weekly salt logging in invented clinics" in bibliography.display
    assert (
        bibliography.display.count("Sodium and salt logging in invented cohorts") == 1
    )
    assert "Meal diaries in invented trials" in bibliography.display  # untouched entry
    root = _root(result.output)
    appended = [
        e
        for e in root.iter(w("ins"))
        if e.get(w("author")) == AUTHOR and e.find(f".//{w('br')}") is not None
    ]
    assert len(appended) == 1
    # Still one field: begin, separate, end around the result.
    kinds = [
        c.get(w("fldCharType"))
        for c in root.iter(w("fldChar"))
        if c.getparent().getparent().tag != w("ins")
    ]
    assert kinds.count("begin") == kinds.count("end")


def test_update_bibliography_with_nothing_missing_changes_nothing(
    spike: Path, adapter: DocxAdapter, tmp_path: Path
):
    view = adapter.open(spike)
    result = adapter.apply(
        view, _txn(UpdateBibliography()), output=tmp_path / "out.docx"
    )
    assert result.applied == 1 and result.refused == []
    assert [p.text for p in adapter.open(result.output).paragraphs] == [
        p.text for p in view.paragraphs
    ]


def test_a_zotero_document_without_a_bibliography_gets_a_field_under_a_heading(
    tmp_path: Path, adapter: DocxAdapter
):
    payload = {
        "citationID": "x",
        "properties": {
            "formattedCitation": "(Doe 2020)",
            "plainCitation": "(Doe 2020)",
            "noteIndex": 0,
        },
        "citationItems": [{"id": "d", "itemData": DOE}],
        "schema": "x",
    }
    path = build_docx(
        tmp_path / "nobib.docx",
        paragraph(run("Introduction"), style="Heading1")
        + paragraph(
            run("Salt matters ")
            + complex_field(
                "ADDIN ZOTERO_ITEM CSL_CITATION " + json.dumps(payload), "(Doe 2020)"
            )
            + run(".")
        )
        + paragraph(run("Closing words.")),
    )
    view = adapter.open(path)
    result = adapter.apply(
        view,
        _txn(
            InsertCitation(paragraph_id="p2", at=13, items=[_item(LEE)]),
            UpdateBibliography(),
        ),
        output=tmp_path / "out.docx",
    )
    assert result.refused == []
    after = adapter.open(result.output)
    heading = after.paragraphs[-2]
    assert (heading.text, heading.heading_level) == ("References", 1)
    bibliography = next(f for f in after.fields if f.kind == "bibliography")
    assert bibliography.manager == "zotero"
    assert bibliography.display.index("Doe, J.") < bibliography.display.index("Lee, M.")
    assert "Sodium diaries across invented cohorts" in bibliography.display
    assert "Sodium and salt logging in invented cohorts" in bibliography.display
    assert after.paragraphs[-1].id == bibliography.paragraph_id
    # The new field parses as Zotero's bibliography field.
    root = _root(result.output)
    instructions = [t.text for t in root.iter(w("instrText"))]
    assert any(
        classify_field(i or "")[:2] == ("bibliography", "zotero") for i in instructions
    )


# ----------------------------------------------------------------------- plain text


def _plain_document(
    tmp_path: Path, *, numbered: bool = False, references: bool = True
) -> Path:
    cite = "[1]" if numbered else "(Doe 2020)"
    body = paragraph(run("Introduction"), style="Heading1") + paragraph(
        run(f"Salt matters {cite} and so does logging.")
    )
    body += paragraph(run("Closing paragraph of the draft."))
    if references:
        body += paragraph(run("References"), style="Heading1")
        if numbered:
            body += paragraph(
                run("[1] Doe J. Sodium and salt logging in invented cohorts. 2020.")
            )
        else:
            body += paragraph(
                run("Doe, J. (2020). Sodium and salt logging in invented cohorts.")
            )
            body += paragraph(run("Zed, A. (2018). Zinc studies in invented diets."))
    return build_docx(tmp_path / "plain.docx", body)


def test_plain_document_gets_plain_citation_text_and_an_alphabetical_entry(
    tmp_path: Path, adapter: DocxAdapter
):
    path = _plain_document(tmp_path)
    view = adapter.open(path)
    result = adapter.apply(
        view,
        _txn(
            InsertCitation(
                paragraph_id="p2",
                at=len("Closing paragraph of the draft"),
                items=[_item(SMITH), _item(LEE)],
            ),
            UpdateBibliography(),
        ),
        output=tmp_path / "out.docx",
    )
    assert result.refused == [] and result.applied == 2
    after = adapter.open(result.output)
    texts = [p.text for p in after.paragraphs]
    assert texts[2] == "Closing paragraph of the draft(Smith and Jones 2021; Lee 2019)."
    # No Zotero field was written.
    assert not [f for f in after.fields if f.manager == "zotero"]
    references = texts[texts.index("References") + 1 :]
    assert len(references) == 4
    assert references[0].startswith("Doe, J. (2020)")
    assert references[1].startswith("Lee, M. (2019)")  # between Doe and Zed
    assert references[2].startswith("Smith, J. Q., & Jones, K. (2021)")
    assert references[3].startswith("Zed, A.")
    root = _root(result.output)
    assert root.find(f".//{w('ins')}") is not None  # tracked


def test_plain_document_without_a_references_section_gets_one(
    tmp_path: Path, adapter: DocxAdapter
):
    path = _plain_document(tmp_path, references=False)
    view = adapter.open(path)
    result = adapter.apply(
        view,
        _txn(
            InsertCitation(
                paragraph_id="p1", at=len("Salt matters"), items=[_item(LEE)]
            ),
            UpdateBibliography(),
        ),
        output=tmp_path / "out.docx",
    )
    after = adapter.open(result.output)
    heading, entry = after.paragraphs[-2:]
    assert (heading.text, heading.heading_level) == ("References", 1)
    assert entry.text.startswith(
        "Lee, M. (2019). Sodium diaries across invented cohorts."
    )
    assert entry.text.endswith("https://doi.org/10.5555/saltlog.lee")


def test_a_refused_citation_leaves_nothing_in_the_bibliography(
    tmp_path: Path, adapter: DocxAdapter
):
    path = _plain_document(tmp_path, references=False)
    view = adapter.open(path)
    result = adapter.apply(
        view,
        _txn(
            InsertCitation(paragraph_id="p1", at=999, items=[_item(LEE)]),
            UpdateBibliography(),
        ),
        output=tmp_path / "out.docx",
    )
    assert len(result.refused) == 1 and "range ends past" in result.refused[0]
    after = adapter.open(result.output)
    assert [p.text for p in after.paragraphs] == [p.text for p in view.paragraphs]
    assert "References" not in [p.text for p in after.paragraphs]


def test_only_the_applied_citations_reach_the_bibliography(
    tmp_path: Path, adapter: DocxAdapter
):
    path = _plain_document(tmp_path, references=False)
    view = adapter.open(path)
    result = adapter.apply(
        view,
        _txn(
            InsertCitation(paragraph_id="p1", at=999, items=[_item(SMITH)]),
            InsertCitation(
                paragraph_id="p1", at=len("Salt matters"), items=[_item(LEE)]
            ),
            UpdateBibliography(),
        ),
        output=tmp_path / "out.docx",
    )
    assert len(result.refused) == 1 and result.applied == 2
    after = adapter.open(result.output)
    entries = [p.text for p in after.paragraphs]
    assert any(t.startswith("Lee, M. (2019)") for t in entries)
    assert not any("Smith" in t for t in entries)


def test_plain_numeric_document_continues_the_numbering(
    tmp_path: Path, adapter: DocxAdapter
):
    path = _plain_document(tmp_path, numbered=True)
    view = adapter.open(path)
    result = adapter.apply(
        view,
        _txn(
            InsertCitation(
                paragraph_id="p1", at=len("Salt matters [1]"), items=[_item(SMITH)]
            ),
            InsertCitation(
                paragraph_id="p1",
                at=len("Salt matters [1] and so does logging"),
                items=[_item(DOE)],
            ),
            UpdateBibliography(),
        ),
        output=tmp_path / "out.docx",
    )
    assert result.refused == []
    texts = [p.text for p in adapter.open(result.output).paragraphs]
    # Smith is new (2); Doe is already listed as 1 and is not listed again.
    assert texts[1] == "Salt matters [1][2] and so does logging[1]."
    entries = texts[texts.index("References") + 1 :]
    assert len(entries) == 2
    assert entries[1].startswith("[2] Smith JQ, Jones K.")


def test_a_second_update_bibliography_in_a_transaction_adds_nothing_more(
    tmp_path: Path, adapter: DocxAdapter
):
    path = _plain_document(tmp_path)
    view = adapter.open(path)
    result = adapter.apply(
        view,
        _txn(
            InsertCitation(paragraph_id="p1", at=12, items=[_item(LEE)]),
            UpdateBibliography(),
            UpdateBibliography(),
        ),
        output=tmp_path / "out.docx",
    )
    after = adapter.open(result.output)
    assert sum(1 for p in after.paragraphs if p.text.startswith("Lee, M.")) == 1


def test_the_citation_ops_validate_their_arguments():
    with pytest.raises(ValueError):
        InsertCitation(paragraph_id="p0", at=0, items=[])
    with pytest.raises(ValueError):
        InsertCitation(paragraph_id="p0", at=-1, items=[_item(LEE)])


def test_markdown_adapter_names_what_it_cannot_do(tmp_path: Path):
    adapter = MarkdownAdapter()
    view = DocumentView(
        path=tmp_path,
        project_id=None,
        paragraphs=[],
        anchors=[],
        fields=[],
        comments=[],
    )
    for op in (
        InsertCitation(paragraph_id="s01:0", at=0, items=[_item(LEE)]),
        UpdateBibliography(),
    ):
        with pytest.raises(NotImplementedError):
            adapter.apply(view, _txn(op), output=tmp_path / "out")


@pytest.mark.skipif(not SOFFICE, reason="LibreOffice not installed")
def test_libreoffice_keeps_the_citation_field(
    spike: Path, adapter: DocxAdapter, tmp_path: Path
):
    view = adapter.open(spike)
    result = adapter.apply(
        view,
        _txn(
            InsertCitation(
                paragraph_id=ABSTRACT, at=_end_of(view, ABSTRACT), items=[_item(SMITH)]
            ),
            UpdateBibliography(),
        ),
        output=tmp_path / "out.docx",
    )
    converted = open_docx(libreoffice_roundtrip(result.output, workdir=tmp_path / "lo"))
    assert "(Smith and Jones, 2021)" in _text(converted, converted.paragraphs[2].id)
