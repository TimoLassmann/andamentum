"""The docx reader, on synthetic documents built from raw WordprocessingML."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from andamentum.compose.harness.docx.anchors import (
    HARNESS_NS,
    anchor_tag,
    parse_anchor_tag,
    project_id_part_xml,
    read_project_id,
)
from andamentum.compose.harness.docx.reader import DocxReadError, open_docx, parse_docx
from andamentum.compose.harness.spike import make
from andamentum.compose.harness.spike.make import (
    build_docx,
    comment_part,
    comments_extended_part,
    complex_field,
    content_control,
    paragraph,
    run,
)


@pytest.fixture
def spike(tmp_path: Path) -> Path:
    return make.make_spike_docx(tmp_path / "spike.docx")


def _text(view, paragraph_id: str) -> str:
    return next(p.text for p in view.paragraphs if p.id == paragraph_id)


def test_paragraph_ids_are_stable_across_reopens(spike: Path):
    first = open_docx(spike)
    second = open_docx(spike)
    assert [p.id for p in first.paragraphs] == [p.id for p in second.paragraphs]
    assert first.paragraphs[0].id == "p0-10000001"


def test_paragraph_id_without_word_para_id_is_the_index(tmp_path: Path):
    path = build_docx(
        tmp_path / "a.docx", paragraph(run("One")) + paragraph(run("Two"))
    )
    assert [p.id for p in open_docx(path).paragraphs] == ["p0", "p1"]


def test_text_has_changes_accepted(spike: Path):
    view = open_docx(spike)
    methods = _text(view, "p6-10000007")
    assert (
        methods
        == "Participants logged every meal using the SaltLog app. Intake is summarised in Table 1."
    )
    assert " on paper" not in view.text()


def test_headings_styles_and_sections(spike: Path):
    view = open_docx(spike)
    assert [(p.style, p.heading_level) for p in view.paragraphs[:2]] == [
        ("Title", None),
        ("heading 1", 1),
    ]
    assert [name for name, _ in view.sections()] == [
        "",
        "Abstract",
        "Introduction",
        "Methods",
        "References",
    ]


def test_heading_level_from_direct_outline_level(tmp_path: Path):
    body = '<w:p><w:pPr><w:outlineLvl w:val="2"/></w:pPr>' + run("Deep") + "</w:p>"
    view = open_docx(build_docx(tmp_path / "a.docx", body))
    assert view.paragraphs[0].heading_level == 3


def test_anchors_address_the_bound_text(spike: Path):
    view = open_docx(spike)
    texts = {a.id: _text(view, a.paragraph_id)[a.start : a.end] for a in view.anchors}
    assert texts == {
        make.SPIKE_NUMBER_ANCHOR: make.SPIKE_NUMBER_TEXT,
        make.SPIKE_CLAIM_ANCHOR: make.SPIKE_CLAIM_TEXT,
    }
    assert {a.id: a.kind for a in view.anchors} == {
        make.SPIKE_NUMBER_ANCHOR: "number",
        make.SPIKE_CLAIM_ANCHOR: "claim",
    }


def test_foreign_content_controls_are_not_anchors(tmp_path: Path):
    other = (
        '<w:sdt><w:sdtPr><w:tag w:val="date-picker"/></w:sdtPr><w:sdtContent>'
        + run("x")
        + "</w:sdtContent></w:sdt>"
    )
    untagged = "<w:sdt><w:sdtPr/><w:sdtContent>" + run("y") + "</w:sdtContent></w:sdt>"
    view = open_docx(build_docx(tmp_path / "a.docx", paragraph(other + untagged)))
    assert view.anchors == []
    assert view.text() == "xy"


def test_unknown_alias_reads_as_span(tmp_path: Path):
    body = paragraph(content_control("a1", "mystery", run("hello")))
    view = open_docx(build_docx(tmp_path / "a.docx", body))
    assert [(a.id, a.kind) for a in view.anchors] == [("a1", "span")]


def test_empty_content_control_is_a_point(tmp_path: Path):
    body = paragraph(run("ab") + content_control("a1", "span", "") + run("cd"))
    view = open_docx(build_docx(tmp_path / "a.docx", body))
    assert (view.anchors[0].start, view.anchors[0].end) == (2, 2)


def test_block_level_content_control_anchors_its_first_paragraph(tmp_path: Path):
    body = (
        paragraph(run("before"))
        + '<w:sdt><w:sdtPr><w:alias w:val="claim"/><w:tag w:val="am:blk"/></w:sdtPr><w:sdtContent>'
        + paragraph(run("whole paragraph"))
        + "</w:sdtContent></w:sdt>"
    )
    view = open_docx(build_docx(tmp_path / "a.docx", body))
    (anchor,) = view.anchors
    assert (anchor.id, anchor.kind, anchor.paragraph_id) == ("blk", "claim", "p1")
    assert (anchor.start, anchor.end) == (0, len("whole paragraph"))


def test_fields_are_read_with_display_text_in_place(spike: Path):
    view = open_docx(spike)
    by_key = {(f.kind, f.manager): f for f in view.fields if f.manager != "none"}
    zotero = by_key[("citation", "zotero")]
    assert zotero.display == "(Doe, 2020)"
    assert _text(view, zotero.paragraph_id)[zotero.start : zotero.end] == "(Doe, 2020)"
    items = zotero.data["citationItems"]
    assert isinstance(items, list) and items[0]["uris"] == [make.SPIKE_ZOTERO_ITEM]
    endnote = by_key[("citation", "endnote")]
    assert endnote.data["cites"][0]["doi"] == "10.9999/spike.002"  # type: ignore[index]
    assert by_key[("crossref", "word")].data["target"] in {"_Ref100", "Table"}


def test_bibliography_field_spans_paragraphs(spike: Path):
    view = open_docx(spike)
    (bibliography,) = [f for f in view.fields if f.kind == "bibliography"]
    assert bibliography.manager == "zotero"
    assert bibliography.paragraph_id == "p9-1000000A"
    assert bibliography.display.count("\n") == 1
    assert bibliography.display.endswith("Journal of Invented Results.")
    assert bibliography.end == len(_text(view, bibliography.paragraph_id))


def test_fields_come_in_document_order(spike: Path):
    view = open_docx(spike)
    positions = [
        (int(f.paragraph_id.split("-")[0][1:]), f.start)
        for f in view.fields
        if f.manager != "none"
    ]
    assert positions == sorted(positions)


def test_simple_field(tmp_path: Path):
    body = paragraph(
        run("See ")
        + '<w:fldSimple w:instr=" REF _Ref9 \\h ">'
        + run("Figure 2")
        + "</w:fldSimple>"
        + run(".")
    )
    view = open_docx(build_docx(tmp_path / "a.docx", body))
    (field,) = view.fields
    assert (field.kind, field.display, field.start, field.end) == (
        "crossref",
        "Figure 2",
        4,
        12,
    )
    assert field.data["target"] == "_Ref9"


def test_malformed_citation_field_is_still_a_token(tmp_path: Path):
    body = paragraph(
        run("x ")
        + complex_field('ADDIN ZOTERO_ITEM CSL_CITATION {"broken": ', "(A, 2020)")
    )
    view = open_docx(build_docx(tmp_path / "a.docx", body))
    (field,) = view.fields
    assert field.manager == "zotero" and "parse_error" in field.data


def test_instruction_split_over_several_runs_is_joined(tmp_path: Path):
    parts = ["ADDIN ZOTERO_ITEM CSL_CIT", 'ATION {"citationItems": ', "[]}"]
    body = paragraph(
        '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        + "".join(
            f'<w:r><w:instrText xml:space="preserve">{p.replace(chr(34), "&quot;")}</w:instrText></w:r>'
            for p in parts
        )
        + '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        + run("(B, 2021)")
        + '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
    )
    view = open_docx(build_docx(tmp_path / "a.docx", body))
    (field,) = view.fields
    assert field.data == {"citationItems": []}


def test_nested_fields_are_both_reported(tmp_path: Path):
    body = paragraph(
        '<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText>ADDIN OUTER</w:instrText></w:r>'
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        + complex_field("PAGEREF _Ref1", "7")
        + '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
    )
    view = open_docx(build_docx(tmp_path / "a.docx", body))
    assert [f.manager for f in view.fields] == ["none", "word"]
    assert [f.display for f in view.fields] == ["7", "7"]


def test_comments_and_reply_thread(spike: Path):
    view = open_docx(spike)
    parent, reply = view.comments
    assert (parent.author, parent.text, parent.parent_id) == (
        "Reviewer A",
        make.SPIKE_COMMENT_TEXT,
        None,
    )
    assert (reply.author, reply.parent_id) == ("Author", parent.id)
    claim = _text(view, parent.paragraph_id)[parent.start : parent.end]
    assert claim == make.SPIKE_CLAIM_TEXT
    assert parent.date == "2026-10-01T09:00:00Z"
    assert not parent.resolved


def test_resolved_flag_comes_from_comments_extended(tmp_path: Path):
    body = paragraph(
        '<w:commentRangeStart w:id="0"/>'
        + run("word")
        + '<w:commentRangeEnd w:id="0"/><w:r><w:commentReference w:id="0"/></w:r>'
    )
    path = build_docx(
        tmp_path / "a.docx",
        body,
        comments_xml=comment_part(
            [(0, "A", "2026-10-01T09:00:00Z", "ok?", "AA000001")]
        ),
        comments_extended_xml=comments_extended_part([("AA000001", None, True)]),
    )
    (comment,) = open_docx(path).comments
    assert comment.resolved


def test_comment_without_extended_part_has_no_parent(tmp_path: Path):
    body = paragraph(
        '<w:commentRangeStart w:id="0"/>'
        + run("word")
        + '<w:commentRangeEnd w:id="0"/><w:r><w:commentReference w:id="0"/></w:r>'
    )
    path = build_docx(
        tmp_path / "a.docx",
        body,
        comments_xml=comment_part(
            [(0, "A", "2026-10-01T09:00:00Z", "ok?", "AA000001")]
        ),
    )
    (comment,) = open_docx(path).comments
    assert comment.parent_id is None and (comment.start, comment.end) == (0, 4)


def test_point_comment_has_an_empty_range(tmp_path: Path):
    body = paragraph(
        run("ab") + '<w:r><w:commentReference w:id="0"/></w:r>' + run("cd")
    )
    path = build_docx(
        tmp_path / "a.docx",
        body,
        comments_xml=comment_part(
            [(0, "A", "2026-10-01T09:00:00Z", "here", "AA000001")]
        ),
    )
    (comment,) = open_docx(path).comments
    assert (comment.start, comment.end) == (2, 2)


def test_comment_with_no_anchor_in_the_body_is_marked_unanchored(tmp_path: Path):
    path = build_docx(
        tmp_path / "a.docx",
        paragraph(run("text")),
        comments_xml=comment_part(
            [(0, "A", "2026-10-01T09:00:00Z", "orphan", "AA000001")]
        ),
    )
    (comment,) = open_docx(path).comments
    assert (comment.paragraph_id, comment.start, comment.end) == ("", 0, 0)


def test_project_id(spike: Path, tmp_path: Path):
    assert open_docx(spike).project_id == make.SPIKE_PROJECT_ID
    assert (
        open_docx(build_docx(tmp_path / "a.docx", paragraph(run("x")))).project_id
        is None
    )


def test_project_id_helpers(tmp_path: Path):
    assert parse_anchor_tag(anchor_tag("c7")) == "c7"
    assert parse_anchor_tag("other") is None and parse_anchor_tag("am:") is None
    path = build_docx(tmp_path / "a.docx", paragraph(run("x")), project_id="pid-1")
    with zipfile.ZipFile(path) as archive:
        assert read_project_id(archive) == "pid-1"
    assert HARNESS_NS.encode() in project_id_part_xml("pid-1")


def test_table_paragraphs_are_included(tmp_path: Path):
    table = "<w:tbl><w:tr><w:tc>" + paragraph(run("cell")) + "</w:tc></w:tr></w:tbl>"
    view = open_docx(
        build_docx(
            tmp_path / "a.docx",
            paragraph(run("before")) + table + paragraph(run("after")),
        )
    )
    assert [p.text for p in view.paragraphs] == ["before", "cell", "after"]


def test_deleted_field_code_is_ignored(tmp_path: Path):
    body = paragraph(
        run("a")
        + '<w:del w:id="1" w:author="x" w:date="2026-10-01T00:00:00Z"><w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        '<w:r><w:instrText>REF gone</w:instrText></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:del>'
        + run("b")
    )
    view = open_docx(build_docx(tmp_path / "a.docx", body))
    assert view.fields == [] and view.text() == "ab"


def test_plain_text_citations_are_found_by_the_references_scanner(tmp_path: Path):
    body = (
        paragraph(
            run(
                "Salt matters [1] and (3) is a list item. Later work (Smith et al., 2019) agreed."
            )
        )
        + paragraph(run("References"))
        + paragraph(run("1. Doe J. Invented. 2020."))
    )
    view = open_docx(build_docx(tmp_path / "a.docx", body))
    plain = [f for f in view.fields if f.manager == "none"]
    assert [f.display for f in plain] == ["[1]", "(Smith et al., 2019)"]
    assert plain[0].data["numbers"] == [1]
    first = plain[0]
    assert _text(view, first.paragraph_id)[first.start : first.end] == "[1]"


def test_plain_scan_does_not_double_count_a_field_citation(spike: Path):
    plain = [f for f in open_docx(spike).fields if f.manager == "none"]
    assert plain == []


def test_superscript_numbers_are_candidates(tmp_path: Path):
    body = paragraph(
        run("Sodium is high")
        + '<w:r><w:rPr><w:vertAlign w:val="superscript"/></w:rPr><w:t>2</w:t></w:r>'
        + run(" in adults.")
    )
    view = open_docx(build_docx(tmp_path / "a.docx", body))
    plain = [f for f in view.fields if f.manager == "none"]
    assert [(f.display, f.data["source"]) for f in plain] == [("2", "superscript")]


def test_parse_docx_exposes_elements_that_line_up_with_the_view(spike: Path):
    parsed = parse_docx(spike)
    assert len(parsed.elements) == len(parsed.view.paragraphs) == len(parsed.atoms)
    for atoms, paragraph_ in zip(parsed.atoms, parsed.view.paragraphs, strict=True):
        assert "".join(a.char for a in atoms) == paragraph_.text


def test_not_a_docx_raises(tmp_path: Path):
    bad = tmp_path / "bad.docx"
    bad.write_bytes(b"not a zip")
    with pytest.raises(DocxReadError):
        open_docx(bad)


def test_reading_never_modifies_the_file(spike: Path):
    before = spike.read_bytes()
    open_docx(spike)
    assert spike.read_bytes() == before


def test_text_moved_away_is_not_part_of_the_accepted_text(tmp_path: Path):
    body = paragraph(
        run("Kept. ")
        + '<w:moveFrom w:id="1" w:author="A" w:date="2026-01-01T00:00:00Z">'
        + run("Moved away. ")
        + "</w:moveFrom>"
        + '<w:moveTo w:id="2" w:author="A" w:date="2026-01-01T00:00:00Z">'
        + run("Moved here. ")
        + "</w:moveTo>"
        + run("End.")
    )
    view = open_docx(build_docx(tmp_path / "move.docx", body))
    assert view.paragraphs[0].text == "Kept. Moved here. End."


def test_a_text_box_written_twice_is_read_once(tmp_path: Path):
    box = run("BOX")
    body = paragraph(
        run("Host ")
        + "<mc:AlternateContent>"
        + f"<mc:Choice Requires=\"wps\">{box}</mc:Choice>"
        + f"<mc:Fallback>{box}</mc:Fallback>"
        + "</mc:AlternateContent>"
        + run(" text.")
    )
    view = open_docx(build_docx(tmp_path / "box.docx", body))
    assert view.paragraphs[0].text == "Host BOX text."
