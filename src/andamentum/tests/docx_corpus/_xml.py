"""Raw WordprocessingML pieces and the package writer behind the corpus.

python-docx cannot write tracked changes, fields, content controls, comment
threads or footnotes, so every part is written by hand here. Fixture builders
live in :mod:`.builder`; this module only builds XML strings and zip files.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_W = f"{{{W_NS}}}"
_XML = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
_CT = "application/vnd.openxmlformats-officedocument.wordprocessingml"
_NS = (
    f'xmlns:w="{W_NS}" xmlns:r="{R_NS}" '
    'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
    'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
    'xmlns:w15="http://schemas.microsoft.com/office/word/2012/wordml" '
    'xmlns:w16cid="http://schemas.microsoft.com/office/word/2016/wordml/cid" '
    'mc:Ignorable="w14 w15 w16cid"'
)
_STYLES = f"""{_XML}<w:styles xmlns:w="{W_NS}">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="1"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="caption"/><w:basedOn w:val="Normal"/></w:style>
</w:styles>"""
_SUPERSCRIPT = '<w:vertAlign w:val="superscript"/>'
_DATE = "2026-10-01T10:00:00Z"

# Expected read-back values, shared with test_builder.py.
FORMATTED_RUNS_TEXT = "The Saline flows slowly from H2O into red tubes."
TRACKED_ACCEPTED_TEXT = "Participants logged every meal in the SaltLog app."
HYPERLINK_TARGET = "https://example.org/saltlog"


# ----------------------------------------------------------- package writer


def _content_types(overrides: list[tuple[str, str]]) -> str:
    return (
        f'{_XML}<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        + "".join(f'<Override PartName="{n}" ContentType="{t}"/>' for n, t in overrides)
        + "</Types>"
    )


def _rel(rid: str, rtype: str, target: str) -> str:
    mode = ' TargetMode="External"' if "://" in target else ""
    return f'<Relationship Id="{rid}" Type="{rtype}" Target={quoteattr(target)}{mode}/>'


def _write_package(
    path: Path,
    body_xml: str,
    *,
    parts: dict[str, str],
    overrides: list[tuple[str, str]],
    doc_rels: list[tuple[str, str, str]],
) -> Path:
    """Write a minimal package. ``parts`` maps part names to complete XML."""
    base_overrides = [
        ("/word/document.xml", f"{_CT}.document.main+xml"),
        ("/word/styles.xml", f"{_CT}.styles+xml"),
    ]
    base_rels = [("rIdStyles", f"{_REL}/styles", "styles.xml")]
    files = {
        "[Content_Types].xml": _content_types(base_overrides + overrides),
        "_rels/.rels": (
            f'{_XML}<Relationships xmlns="{_PKG_REL}">'
            f'<Relationship Id="rId1" Type="{_REL}/officeDocument" Target="word/document.xml"/>'
            "</Relationships>"
        ),
        "word/_rels/document.xml.rels": (
            f'{_XML}<Relationships xmlns="{_PKG_REL}">'
            + "".join(_rel(*r) for r in base_rels + doc_rels)
            + "</Relationships>"
        ),
        "word/document.xml": (
            f"{_XML}<w:document {_NS}><w:body>{body_xml}"
            '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/></w:sectPr></w:body></w:document>'
        ),
        "word/styles.xml": _STYLES,
        **parts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():  # [Content_Types].xml first, as Word writes it
            archive.writestr(name, data)
    return path


# ---------------------------------------------------------------- XML pieces


def _run(text: str, props: str = "") -> str:
    rpr = f"<w:rPr>{props}</w:rPr>" if props else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def _para(content: str, *, style: str | None = None, ppr: str = "") -> str:
    style_xml = f'<w:pStyle w:val="{style}"/>' if style else ""
    props = style_xml + ppr
    return f"<w:p>{'<w:pPr>' + props + '</w:pPr>' if props else ''}{content}</w:p>"


def _field(instruction: str, result: str) -> str:
    """A complex field: begin, instruction, separate, result, end."""
    return (
        '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        f'<w:r><w:instrText xml:space="preserve">{escape(instruction)}</w:instrText></w:r>'
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        f"{_run(result)}"
        '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
    )


def _simple_field(instruction: str, result: str) -> str:
    return f"<w:fldSimple w:instr={quoteattr(instruction)}>{_run(result)}</w:fldSimple>"


def _content_control(tag: str, alias: str, content: str, sdt_id: int) -> str:
    return (
        f'<w:sdt><w:sdtPr><w:alias w:val="{alias}"/><w:tag w:val="{tag}"/>'
        f'<w:id w:val="{sdt_id}"/></w:sdtPr><w:sdtContent>{content}</w:sdtContent></w:sdt>'
    )


def _ins(author: str, text: str, change_id: int) -> str:
    return (
        f'<w:ins w:id="{change_id}" w:author={quoteattr(author)} w:date="{_DATE}">'
        f"{_run(text)}</w:ins>"
    )


def _del(author: str, text: str, change_id: int) -> str:
    return (
        f'<w:del w:id="{change_id}" w:author={quoteattr(author)} w:date="{_DATE}">'
        f'<w:r><w:delText xml:space="preserve">{escape(text)}</w:delText></w:r></w:del>'
    )


def _bookmark(bookmark_id: int, name: str, content: str) -> str:
    return (
        f'<w:bookmarkStart w:id="{bookmark_id}" w:name="{name}"/>{content}'
        f'<w:bookmarkEnd w:id="{bookmark_id}"/>'
    )


def _footnote_part(entries: str) -> str:
    separators = (
        '<w:footnote w:type="separator" w:id="-1"><w:p><w:r><w:separator/></w:r></w:p></w:footnote>'
        '<w:footnote w:type="continuationSeparator" w:id="0"><w:p>'
        "<w:r><w:continuationSeparator/></w:r></w:p></w:footnote>"
    )
    return f"{_XML}<w:footnotes {_NS}>{separators}{entries}</w:footnotes>"


def _comment_part(entries: list[tuple[int, str, str, str]]) -> str:
    """``comments.xml`` from ``(id, author, text, w14:paraId)`` tuples."""
    body = "".join(
        f'<w:comment w:id="{cid}" w:author={quoteattr(author)} w:date="{_DATE}" w:initials="X">'
        f'<w:p w14:paraId="{para_id}"><w:r><w:annotationRef/></w:r>{_run(text)}</w:p></w:comment>'
        for cid, author, text, para_id in entries
    )
    return f"{_XML}<w:comments {_NS}>{body}</w:comments>"


def _comments_extended_part(entries: list[tuple[str, str | None]]) -> str:
    """``commentsExtended.xml`` from ``(paraId, paraIdParent)`` tuples."""
    body = "".join(
        f'<w15:commentEx w15:paraId="{pid}"'
        + (f' w15:paraIdParent="{parent}"' if parent else "")
        + ' w15:done="0"/>'
        for pid, parent in entries
    )
    return f"{_XML}<w15:commentsEx {_NS}>{body}</w15:commentsEx>"


def _comments_ids_part(entries: list[tuple[str, str]]) -> str:
    body = "".join(
        f'<w16cid:commentId w16cid:paraId="{pid}" w16cid:durableId="{did}"/>'
        for pid, did in entries
    )
    return f"{_XML}<w16cid:commentsIds {_NS}>{body}</w16cid:commentsIds>"


def _table(rows: tuple[tuple[str, ...], ...]) -> str:
    grid = '<w:tblGrid><w:gridCol w:w="2400"/><w:gridCol w:w="2400"/></w:tblGrid>'
    trs = "".join(
        "<w:tr>"
        + "".join(
            '<w:tc><w:tcPr><w:tcW w:w="2400" w:type="dxa"/></w:tcPr>'
            + _para(_run(cell))
            + "</w:tc>"
            for cell in row
        )
        + "</w:tr>"
        for row in rows
    )
    return (
        f'<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/></w:tblPr>{grid}{trs}</w:tbl>'
    )


_NUMBERING = (
    f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<w:numbering xmlns:w="{W_NS}">'
    '<w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0"><w:start w:val="1"/>'
    '<w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/><w:lvlJc w:val="left"/></w:lvl>'
    '</w:abstractNum><w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num></w:numbering>'
)


# ------------------------------------------------------------------ read-back


def document_xml(path: Path) -> str:
    """The raw ``word/document.xml`` of a ``.docx`` file."""
    with zipfile.ZipFile(path) as archive:
        return archive.read("word/document.xml").decode("utf-8")


def accepted_text(path: Path) -> str:
    """Text with every tracked change accepted: ``w:t`` runs only, one line per paragraph.

    ``w:delText`` (deleted text) is never collected, so a ``w:del`` disappears
    and a ``w:ins`` stays, which is what accepting all changes gives.
    """
    root = ET.fromstring(document_xml(path))
    lines = []
    for paragraph in root.iter(f"{_W}p"):
        lines.append("".join(t.text or "" for t in paragraph.iter(f"{_W}t")))
    return "\n".join(lines)
