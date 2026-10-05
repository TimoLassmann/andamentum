"""Build synthetic ``.docx`` files from raw WordprocessingML.

``build_docx`` writes a minimal valid package around a body (and optional
comment parts, project id), so tests and the spike control every element:
python-docx cannot write content controls, fields, threaded comments or custom
XML parts. ``make_spike_docx`` is the M0 spike document: an invented
manuscript ("SaltLog pilot") carrying one of everything the harness depends on.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from ..docx.anchors import HARNESS_NS, project_id_part_xml

SPIKE_PROJECT_ID = "spike-project-0001"

# What make_spike_docx writes; the survival checks compare a round-tripped file to it.
SPIKE_NUMBER_ANCHOR = "spike-number"
SPIKE_CLAIM_ANCHOR = "spike-claim"
SPIKE_NUMBER_TEXT = "48"
SPIKE_CLAIM_TEXT = "Daily logging lowered reported sodium intake."
SPIKE_ZOTERO_ITEM = "http://zotero.org/users/local/abc/items/SPIKE001"
SPIKE_COMMENT_TEXT = "Is this causal?"
SPIKE_REPLY_TEXT = "No, associative; I will reword."
SPIKE_INSERTION_TEXT = " using the SaltLog app"

_NS = (
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
    'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
    'xmlns:w15="http://schemas.microsoft.com/office/word/2012/wordml" '
    'xmlns:w16cid="http://schemas.microsoft.com/office/word/2016/wordml/cid" '
    'mc:Ignorable="w14 w15 w16cid"'
)
_XML = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'

_STYLES = f"""{_XML}<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="1"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="caption"/><w:basedOn w:val="Normal"/></w:style>
</w:styles>"""

_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_CT = "application/vnd.openxmlformats-officedocument.wordprocessingml"


def build_docx(
    path: Path,
    body_xml: str,
    *,
    comments_xml: str | None = None,
    comments_extended_xml: str | None = None,
    comments_ids_xml: str | None = None,
    project_id: str | None = None,
) -> Path:
    """Write a ``.docx`` whose body is *body_xml* (the children of ``w:body``).

    The comment arguments are the complete part XML, namespace declarations
    included (use :func:`comment_part` / :func:`comments_extended_part`).
    """
    overrides = [
        ("/word/document.xml", f"{_CT}.document.main+xml"),
        ("/word/styles.xml", f"{_CT}.styles+xml"),
    ]
    doc_rels = [("rIdStyles", f"{_REL}/styles", "styles.xml")]
    parts: dict[str, str | bytes] = {
        "word/document.xml": (
            f"{_XML}<w:document {_NS}><w:body>{body_xml}"
            '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/></w:sectPr></w:body></w:document>'
        ),
        "word/styles.xml": _STYLES,
    }
    if comments_xml is not None:
        parts["word/comments.xml"] = comments_xml
        overrides.append(("/word/comments.xml", f"{_CT}.comments+xml"))
        doc_rels.append(("rIdComments", f"{_REL}/comments", "comments.xml"))
    if comments_extended_xml is not None:
        parts["word/commentsExtended.xml"] = comments_extended_xml
        overrides.append(("/word/commentsExtended.xml", f"{_CT}.commentsExtended+xml"))
        doc_rels.append(
            (
                "rIdCommentsEx",
                "http://schemas.microsoft.com/office/2011/relationships/commentsExtended",
                "commentsExtended.xml",
            )
        )
    if comments_ids_xml is not None:
        parts["word/commentsIds.xml"] = comments_ids_xml
        overrides.append(("/word/commentsIds.xml", f"{_CT}.commentsIds+xml"))
        doc_rels.append(
            (
                "rIdCommentsIds",
                "http://schemas.microsoft.com/office/2016/09/relationships/commentsIds",
                "commentsIds.xml",
            )
        )
    if project_id is not None:
        parts["customXml/item1.xml"] = project_id_part_xml(project_id)
        parts["customXml/itemProps1.xml"] = (
            f'{_XML}<ds:datastoreItem ds:itemID="{{5A6E3B1C-0000-4000-8000-00000000A0A0}}" '
            'xmlns:ds="http://schemas.openxmlformats.org/officeDocument/2006/customXml">'
            f'<ds:schemaRefs><ds:schemaRef ds:uri="{HARNESS_NS}"/></ds:schemaRefs></ds:datastoreItem>'
        )
        parts["customXml/_rels/item1.xml.rels"] = (
            f'{_XML}<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{_REL}/customXmlProps" Target="itemProps1.xml"/></Relationships>'
        )
        overrides.append(
            (
                "/customXml/itemProps1.xml",
                "application/vnd.openxmlformats-officedocument.customXmlProperties+xml",
            )
        )
        doc_rels.append(("rIdCustom", f"{_REL}/customXml", "../customXml/item1.xml"))

    parts["[Content_Types].xml"] = (
        f'{_XML}<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        + "".join(f'<Override PartName="{n}" ContentType="{t}"/>' for n, t in overrides)
        + "</Types>"
    )
    parts["_rels/.rels"] = (
        f'{_XML}<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'<Relationship Id="rId1" Type="{_REL}/officeDocument" Target="word/document.xml"/></Relationships>'
    )
    parts["word/_rels/document.xml.rels"] = (
        f'{_XML}<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(
            f'<Relationship Id="{i}" Type="{t}" Target="{g}"/>' for i, t, g in doc_rels
        )
        + "</Relationships>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        # [Content_Types].xml first, as Word writes it.
        archive.writestr("[Content_Types].xml", parts.pop("[Content_Types].xml"))
        for name, data in parts.items():
            archive.writestr(name, data)
    return path


# ---------------------------------------------------------------- XML pieces


def run(text: str) -> str:
    return f'<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def paragraph(
    content: str, *, style: str | None = None, para_id: str | None = None
) -> str:
    properties = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    attribute = f' w14:paraId="{para_id}"' if para_id else ""
    return f"<w:p{attribute}>{properties}{content}</w:p>"


def complex_field(instruction: str, result: str) -> str:
    """A complex field: begin, instruction, separate, result run, end."""
    return (
        '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        f'<w:r><w:instrText xml:space="preserve">{escape(instruction)}</w:instrText></w:r>'
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        f"{run(result)}"
        '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
    )


def content_control(
    anchor_id: str, kind: str, content: str, *, hidden: bool = True
) -> str:
    """A run-level content control tagged ``am:<anchor_id>``."""
    appearance = '<w15:appearance w15:val="hidden"/>' if hidden else ""
    return (
        f'<w:sdt><w:sdtPr><w:alias w:val="{kind}"/><w:tag w:val="am:{anchor_id}"/>'
        f"{appearance}</w:sdtPr><w:sdtContent>{content}</w:sdtContent></w:sdt>"
    )


def comment_part(comments: list[tuple[int, str, str, str, str]]) -> str:
    """``comments.xml`` from ``(id, author, date, text, w14:paraId)`` tuples."""
    body = "".join(
        f'<w:comment w:id="{cid}" w:author={quoteattr(author)} w:date="{date}" w:initials="X">'
        f'<w:p w14:paraId="{para_id}"><w:r><w:annotationRef/></w:r>{run(text)}</w:p></w:comment>'
        for cid, author, date, text, para_id in comments
    )
    return f"{_XML}<w:comments {_NS}>{body}</w:comments>"


def comments_extended_part(entries: list[tuple[str, str | None, bool]]) -> str:
    """``commentsExtended.xml`` from ``(paraId, paraIdParent, done)`` tuples."""
    body = "".join(
        f'<w15:commentEx w15:paraId="{pid}"'
        + (f' w15:paraIdParent="{parent}"' if parent else "")
        + f' w15:done="{1 if done else 0}"/>'
        for pid, parent, done in entries
    )
    return f"{_XML}<w15:commentsEx {_NS}>{body}</w15:commentsEx>"


def comments_ids_part(entries: list[tuple[str, str]]) -> str:
    """``commentsIds.xml`` from ``(paraId, durableId)`` tuples."""
    body = "".join(
        f'<w16cid:commentId w16cid:paraId="{pid}" w16cid:durableId="{did}"/>'
        for pid, did in entries
    )
    return f"{_XML}<w16cid:commentsIds {_NS}>{body}</w16cid:commentsIds>"


# ------------------------------------------------------------- spike document


def _zotero_instruction() -> str:
    csl = {
        "citationID": "SPIKE-CIT-1",
        "properties": {
            "formattedCitation": "(Doe, 2020)",
            "plainCitation": "(Doe, 2020)",
            "noteIndex": 0,
        },
        "citationItems": [
            {
                "id": SPIKE_ZOTERO_ITEM,
                "uris": [SPIKE_ZOTERO_ITEM],
                "itemData": {
                    "id": SPIKE_ZOTERO_ITEM,
                    "type": "article-journal",
                    "title": "Sodium and salt logging in invented cohorts",
                    "container-title": "Journal of Invented Results",
                    "author": [{"family": "Doe", "given": "Jane"}],
                    "issued": {"date-parts": [[2020]]},
                    "DOI": "10.9999/spike.001",
                },
            }
        ],
        "schema": "https://github.com/citation-style-language/schema/raw/master/csl-citation.json",
    }
    return f"ADDIN ZOTERO_ITEM CSL_CITATION {json.dumps(csl)}"


def _endnote_instruction() -> str:
    return (
        " ADDIN EN.CITE <EndNote><Cite><Author>Roe</Author><Year>2019</Year><RecNum>12</RecNum>"
        "<DisplayText>(Roe, 2019)</DisplayText><record><rec-number>12</rec-number>"
        "<contributors><authors><author>Roe, Richard</author></authors></contributors>"
        "<titles><title>Meal diaries in invented trials</title></titles>"
        "<dates><year>2019</year></dates>"
        "<electronic-resource-num>10.9999/spike.002</electronic-resource-num></record></Cite></EndNote>"
    )


def make_spike_docx(path: Path) -> Path:
    """Write the M0 spike document to *path*."""
    bibliography = (
        '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        '<w:r><w:instrText xml:space="preserve">ADDIN ZOTERO_BIBL {"uncited":[],"omitted":[],"custom":[]} CSL_BIBLIOGRAPHY</w:instrText></w:r>'
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        + run(
            "Doe, J. (2020). Sodium and salt logging in invented cohorts. Journal of Invented Results."
        )
    )
    body = "".join(
        [
            paragraph(
                run("SaltLog: a pilot of meal-by-meal salt logging"),
                style="Title",
                para_id="10000001",
            ),
            paragraph(run("Abstract"), style="Heading1", para_id="10000002"),
            paragraph(
                run("In the SaltLog pilot, ")
                + content_control(SPIKE_NUMBER_ANCHOR, "number", run(SPIKE_NUMBER_TEXT))
                + run(" participants logged meals for four weeks. ")
                + '<w:commentRangeStart w:id="0"/><w:commentRangeStart w:id="1"/>'
                + content_control(SPIKE_CLAIM_ANCHOR, "claim", run(SPIKE_CLAIM_TEXT))
                + '<w:commentRangeEnd w:id="0"/><w:commentRangeEnd w:id="1"/>'
                + '<w:r><w:commentReference w:id="0"/></w:r><w:r><w:commentReference w:id="1"/></w:r>',
                para_id="10000003",
            ),
            paragraph(run("Introduction"), style="Heading1", para_id="10000004"),
            paragraph(
                run("Sodium intake exceeds guidelines in most adults ")
                + complex_field(_zotero_instruction(), "(Doe, 2020)")
                + run(" and meal diaries are common ")
                + complex_field(_endnote_instruction(), "(Roe, 2019)")
                + run("."),
                para_id="10000005",
            ),
            paragraph(run("Methods"), style="Heading1", para_id="10000006"),
            paragraph(
                run("Participants logged every meal")
                + f'<w:ins w:id="101" w:author="Co-author" w:date="2026-10-01T10:00:00Z">{run(SPIKE_INSERTION_TEXT)}</w:ins>'
                + '<w:del w:id="102" w:author="Co-author" w:date="2026-10-01T10:00:00Z"><w:r><w:delText xml:space="preserve"> on paper</w:delText></w:r></w:del>'
                + run(". Intake is summarised in ")
                + complex_field("REF _Ref100 \\h", "Table 1")
                + run("."),
                para_id="10000007",
            ),
            paragraph(
                '<w:bookmarkStart w:id="100" w:name="_Ref100"/>'
                + run("Table ")
                + complex_field("SEQ Table \\* ARABIC", "1")
                + '<w:bookmarkEnd w:id="100"/>'
                + run(": mean sodium intake by week."),
                style="Caption",
                para_id="10000008",
            ),
            paragraph(run("References"), style="Heading1", para_id="10000009"),
            paragraph(bibliography, para_id="1000000A"),
            paragraph(
                run(
                    "Roe, R. (2019). Meal diaries in invented trials. Journal of Invented Results."
                )
                + '<w:r><w:fldChar w:fldCharType="end"/></w:r>',
                para_id="1000000B",
            ),
        ]
    )
    return build_docx(
        path,
        body,
        comments_xml=comment_part(
            [
                (
                    0,
                    "Reviewer A",
                    "2026-10-01T09:00:00Z",
                    SPIKE_COMMENT_TEXT,
                    "20000001",
                ),
                (1, "Author", "2026-10-01T09:30:00Z", SPIKE_REPLY_TEXT, "20000002"),
            ]
        ),
        comments_extended_xml=comments_extended_part(
            [("20000001", None, False), ("20000002", "20000001", False)]
        ),
        comments_ids_xml=comments_ids_part(
            [("20000001", "3A000001"), ("20000002", "3A000002")]
        ),
        project_id=SPIKE_PROJECT_ID,
    )
