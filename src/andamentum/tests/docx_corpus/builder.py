"""Synthetic ``.docx`` fixtures for both docx engines' tests.

Each fixture builder takes a directory and returns the path of the ``.docx``
it wrote; the file name is the fixture name. Everything is invented (Jane Doe,
Apollo, SaltLog). Read-back helpers: :func:`accepted_text` (the text with every
tracked change accepted) and :func:`document_xml` (the raw ``word/document.xml``).

Raw XML pieces and the package writer live in :mod:`._xml`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ._xml import (
    accepted_text,
    document_xml,
    _CT,
    _REL,
    _SUPERSCRIPT,
    _bookmark,
    _comment_part,
    _comments_extended_part,
    _comments_ids_part,
    _content_control,
    _del,
    _field,
    _footnote_part,
    _ins,
    _para,
    _run,
    _NUMBERING,
    _simple_field,
    _table,
    _write_package,
)

__all__ = [
    "ALL_FIXTURES",
    "FORMATTED_RUNS_TEXT",
    "HYPERLINK_TARGET",
    "LIST_ITEMS",
    "TABLE_CELLS",
    "TRACKED_ACCEPTED_TEXT",
    "accepted_text",
    "document_xml",
    "existing_comments",
    "existing_tracked",
    "fields_and_controls",
    "formatted_runs",
    "headed_sections",
    "hyperlink_and_footnote",
    "plain_two_paragraphs",
    "table_and_list",
]

# Expected read-back values, shared with test_builder.py.
FORMATTED_RUNS_TEXT = "The Saline flows slowly from H2O into red tubes."
TRACKED_ACCEPTED_TEXT = "Participants logged every meal in the SaltLog app."
HYPERLINK_TARGET = "https://example.org/saltlog"
TABLE_CELLS = (("Week", "Sodium (g)"), ("1", "2.1"))
LIST_ITEMS = ("Recruit participants.", "Log every meal.", "Analyse weekly means.")


def _doc(tmp_dir: Path, name: str) -> Path:
    return tmp_dir / f"{name}.docx"


def plain_two_paragraphs(tmp_dir: Path) -> Path:
    """Two Normal paragraphs, one run each."""
    body = _para(
        _run("Jane Doe recorded the first SaltLog reading at Apollo.")
    ) + _para(_run("The second paragraph follows on from the first."))
    return _write_package(
        _doc(tmp_dir, "plain_two_paragraphs"), body, parts={}, overrides=[], doc_rels=[]
    )


_FORMATTED_RUNS: tuple[tuple[str, str], ...] = (
    ("The ", ""),
    ("Sali", "<w:b/>"),  # mid-word boundary: "Sali" bold + "ne" plain
    ("ne", ""),
    (" flows ", ""),
    ("slowly", "<w:i/>"),
    (" from H", ""),
    ("2", _SUPERSCRIPT),
    ("O into ", ""),
    ("red", '<w:color w:val="C00000"/>'),
    (" tubes.", ""),
)


def formatted_runs(tmp_dir: Path) -> Path:
    """One paragraph split into bold, italic, superscript and coloured runs."""
    body = _para("".join(_run(t, p) for t, p in _FORMATTED_RUNS))
    return _write_package(
        _doc(tmp_dir, "formatted_runs"), body, parts={}, overrides=[], doc_rels=[]
    )


_HEADED: tuple[tuple[str | None, str], ...] = (
    ("Title", "SaltLog Pilot Report"),
    ("Heading1", "Introduction"),
    (None, "Sodium intake exceeds guidelines in the invented cohort."),
    ("Heading2", "Participants"),
    (None, "Jane Doe enrolled twelve invented adults at Apollo."),
    ("Heading1", "Results"),
    (None, "Intake fell by a fixed margin over four weeks."),
    ("Heading1", "Bibliography"),
    (None, "Doe, J. (2020). Sodium and salt logging in invented cohorts."),
)


def headed_sections(tmp_dir: Path) -> Path:
    """Title, Heading1 and Heading2 with body text, and a bibliography heading."""
    body = "".join(_para(_run(text), style=style) for style, text in _HEADED)
    return _write_package(
        _doc(tmp_dir, "headed_sections"), body, parts={}, overrides=[], doc_rels=[]
    )


_LINK_REL = "rIdLink1"


def hyperlink_and_footnote(tmp_dir: Path) -> Path:
    """External hyperlink, bookmark, one footnote reference, footnotes.xml."""
    link = (
        f'<w:hyperlink r:id="{_LINK_REL}" w:history="1">'
        f"{_run('SaltLog site')}</w:hyperlink>"
    )
    first = (
        _run("Read the ")
        + link
        + _run(" before the ")
        + _bookmark(1, "PilotStart", _run("pilot begins"))
        + _run(".")
        + '<w:r><w:footnoteReference w:id="1"/></w:r>'
    )
    footnote = (
        '<w:footnote w:id="1"><w:p><w:r><w:footnoteRef/></w:r>'
        f"{_run(' Invented footnote: the pilot ran at two sites.')}</w:p></w:footnote>"
    )
    return _write_package(
        _doc(tmp_dir, "hyperlink_and_footnote"),
        _para(first) + _para(_run("Second paragraph, without links.")),
        parts={"word/footnotes.xml": _footnote_part(footnote)},
        overrides=[("/word/footnotes.xml", f"{_CT}.footnotes+xml")],
        doc_rels=[
            (_LINK_REL, f"{_REL}/hyperlink", HYPERLINK_TARGET),
            ("rIdFootnotes", f"{_REL}/footnotes", "footnotes.xml"),
        ],
    )


def existing_tracked(tmp_dir: Path) -> Path:
    """A paragraph with a w:ins and a w:del by a named author."""
    author = "Jane Doe"
    body = _para(
        _run("Participants logged ")
        + _ins(author, "every meal", 101)
        + _del(author, " on paper", 102)
        + _run(" in the SaltLog app.")
    )
    return _write_package(
        _doc(tmp_dir, "existing_tracked"), body, parts={}, overrides=[], doc_rels=[]
    )


def existing_comments(tmp_dir: Path) -> Path:
    """Two comments on overlapping ranges and one reply (commentsExtended)."""
    # Comment 0 (Reviewer A) and its reply 2 share a range; comment 1 overlaps it.
    starts = '<w:commentRangeStart w:id="0"/><w:commentRangeStart w:id="2"/>'
    body = _para(
        starts
        + _run("Daily logging ")
        + '<w:commentRangeStart w:id="1"/>'
        + _run("lowered reported sodium intake")
        + '<w:commentRangeEnd w:id="0"/><w:commentRangeEnd w:id="2"/>'
        + _run(" in the pilot.")
        + '<w:commentRangeEnd w:id="1"/>'
        + "".join(f'<w:r><w:commentReference w:id="{i}"/></w:r>' for i in (0, 1, 2))
    )
    comments = [
        (0, "Reviewer A", "Is this causal?", "20000001"),
        (1, "Jane Doe", "Define the sodium unit.", "20000002"),
        (2, "Author", "No, associative; I will reword.", "20000003"),
    ]
    return _write_package(
        _doc(tmp_dir, "existing_comments"),
        body,
        parts={
            "word/comments.xml": _comment_part(comments),
            "word/commentsExtended.xml": _comments_extended_part(
                [("20000001", None), ("20000002", None), ("20000003", "20000001")]
            ),
            "word/commentsIds.xml": _comments_ids_part(
                [
                    ("20000001", "3A000001"),
                    ("20000002", "3A000002"),
                    ("20000003", "3A000003"),
                ]
            ),
        },
        overrides=[
            ("/word/comments.xml", f"{_CT}.comments+xml"),
            ("/word/commentsExtended.xml", f"{_CT}.commentsExtended+xml"),
            ("/word/commentsIds.xml", f"{_CT}.commentsIds+xml"),
        ],
        doc_rels=[
            ("rIdComments", f"{_REL}/comments", "comments.xml"),
            (
                "rIdCommentsEx",
                "http://schemas.microsoft.com/office/2011/relationships/commentsExtended",
                "commentsExtended.xml",
            ),
            (
                "rIdCommentsIds",
                "http://schemas.microsoft.com/office/2016/09/relationships/commentsIds",
                "commentsIds.xml",
            ),
        ],
    )


def table_and_list(tmp_dir: Path) -> Path:
    """A 2x2 table followed by a numbered list (numbering.xml, numId 1)."""
    list_ppr = '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>'
    body = (
        _para(_run("Intake by week:"))
        + _table(TABLE_CELLS)
        + "".join(_para(_run(item), ppr=list_ppr) for item in LIST_ITEMS)
    )
    return _write_package(
        _doc(tmp_dir, "table_and_list"),
        body,
        parts={"word/numbering.xml": _NUMBERING},
        overrides=[("/word/numbering.xml", f"{_CT}.numbering+xml")],
        doc_rels=[("rIdNumbering", f"{_REL}/numbering", "numbering.xml")],
    )


def fields_and_controls(tmp_dir: Path) -> Path:
    """REF field to a bookmark, a complex field, a tagged content control, a fldSimple."""
    caption = _para(
        _bookmark(200, "_Ref200", _run("Table ") + _field("SEQ Table \\* ARABIC", "2"))
        + _run(": intake by week."),
        style="Caption",
    )
    control = _content_control("am:claim-1", "claim", _run("Sodium intake fell."), 1001)
    reference = _para(
        _run("See ")
        + _field("REF _Ref200 \\h", "Table 2")
        + _run(" on page ")
        + _field("PAGE", "1")
        + _run(" and ")
        + control
        + _run(" in ")
        + _simple_field(" REF _Ref200 \\h ", "Table 2")
        + _run(".")
    )
    return _write_package(
        _doc(tmp_dir, "fields_and_controls"),
        caption + reference,
        parts={},
        overrides=[],
        doc_rels=[],
    )


ALL_FIXTURES: tuple[Callable[[Path], Path], ...] = (
    plain_two_paragraphs,
    formatted_runs,
    headed_sections,
    hyperlink_and_footnote,
    existing_tracked,
    existing_comments,
    table_and_list,
    fields_and_controls,
)
