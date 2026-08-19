"""The review report: built as markdown, rendered into the document.

One concern that used to live in two files, split at a string. The markdown was
assembled in ``finalization.py`` — 279 lines of it, inside a module about
applying patches — and turned into Word XML by sixteen methods on
``DocxEditor``, 444 lines inside a module about reading and writing a .docx.
Neither half had anything to do with its host: the report is prepended to the
document, it does not annotate it, and every annotation path was constructing
an editor that carried a markdown table parser it never used.

Fifteen of those sixteen methods touched no editor state at all. They are
module-level functions here, and ``prepend_review_section`` takes the editor it
needs rather than being a method on it.

The seam between the halves is unchanged: data goes in one end as
``_format_review_report``, markdown comes out, and ``prepend_review_section``
turns that markdown into paragraphs at the top of the file. It is prepended
AFTER annotations are placed, so the paragraphs it adds cannot shift an offset
out from under one.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from docx.oxml.ns import qn
from lxml import etree  # type: ignore

from .constants import (
    CRITICAL_ISSUES_HEADER,
    DEFAULT_DESCRIPTION,
    DEFAULT_DISCIPLINE,
    DEFAULT_EXPERT_NAME,
    DEFAULT_ISSUE_TITLE,
    DEFAULT_SEVERITY,
    EXECUTIVE_SUMMARY_HEADER,
    EXPERT_REVIEWS_HEADER,
    FIELD_LABEL_ADDITIONAL,
    FIELD_LABEL_ASSESSMENTS,
    FIELD_LABEL_DISCIPLINE,
    FIELD_LABEL_EDUCATION,
    FIELD_LABEL_LOCATION,
    FIELD_LABEL_POSITION,
    FIELD_LABEL_SCORES,
    DECLINED_FINDINGS_HEADER,
    NOVELTY_FINDINGS_HEADER,
    REPORT_TITLE,
    SECTION_SEPARATOR,
)
from .low_level import NS, ParagraphData, logger
from .reading import REVIEW_REPORT_BOOKMARK_PREFIX
from .model_utils import categorize_review_fields, extract_fields, normalize_to_dict


# ── data → markdown ─────────────────────────────────────────────────────


def _build_report_header(summary: str, issues_count: int) -> List[str]:
    """Build the report header with executive summary."""
    return [
        REPORT_TITLE,
        "",
        SECTION_SEPARATOR,
        "",
        EXECUTIVE_SUMMARY_HEADER,
        "",
        summary,
        "",
        SECTION_SEPARATOR,
        "",
        f"{CRITICAL_ISSUES_HEADER}: {issues_count}",
        "",
    ]


def _format_critical_issue(issue: Any, index: int) -> List[str]:
    """Format a single critical issue."""
    issue_data = extract_fields(
        issue,
        {
            "title": ["title", "category"],
            "description": "description",
            "severity": ["severity", "issue_type"],
            "location": "location",
            "recommendation": "recommendation",
            "source_reviewers": "source_reviewers",
        },
    )

    title = issue_data["title"] or DEFAULT_ISSUE_TITLE
    description = issue_data["description"] or DEFAULT_DESCRIPTION
    severity = issue_data["severity"] or DEFAULT_SEVERITY
    location = issue_data["location"] or ""
    recommendation = issue_data.get("recommendation") or ""
    source_reviewers = issue_data.get("source_reviewers") or []

    lines = []

    if severity:
        lines.append(f"### {index}. {title} ({severity.upper()})")
    else:
        lines.append(f"### {index}. {title}")

    if source_reviewers:
        reviewer_names = ", ".join(str(r) for r in source_reviewers)
        lines.append(f"**Identified by:** {reviewer_names}")

    if location:
        lines.append(f"**{FIELD_LABEL_LOCATION}:** {location}")

    lines.append("")
    lines.append(description)

    if recommendation:
        lines.append("")
        lines.append(f"**Recommendation:** {recommendation}")

    lines.append("")

    return lines


def _format_critical_issues(critical_issues: Optional[list]) -> List[str]:
    """Format all critical issues section."""
    if not critical_issues:
        return ["See detailed reviews below.", ""]

    lines = []
    for i, issue in enumerate(critical_issues, 1):
        lines.extend(_format_critical_issue(issue, i))

    return lines


def _format_expert_header(
    expert_name: str,
    discipline: str,
    position: Optional[str],
    education: Optional[str],
    index: int,
) -> List[str]:
    """Format expert header with metadata."""
    lines = [
        f"### Expert {index}: {expert_name}",
        f"**{FIELD_LABEL_DISCIPLINE}:** {discipline}",
    ]

    if position:
        lines.append(f"**{FIELD_LABEL_POSITION}:** {position}")
    if education:
        lines.append(f"**{FIELD_LABEL_EDUCATION}:** {education}")

    lines.append("")
    return lines


def _format_scores(scores: Dict[str, float]) -> List[str]:
    """Format scores section."""
    if not scores:
        return []

    lines = [f"**{FIELD_LABEL_SCORES}:**"]
    for label, value in sorted(scores.items()):
        lines.append(f"- {label}: {value}/10")
    lines.append("")
    return lines


def _format_justifications(justifications: Dict[str, str]) -> List[str]:
    """Format justifications section."""
    if not justifications:
        return []

    lines = [f"**{FIELD_LABEL_ASSESSMENTS}:**"]
    for label, text in sorted(justifications.items()):
        lines.append(f"**{label}:** {text}")
        lines.append("")
    return lines


def _format_structured_field(field_name: str, field_value: Any) -> List[str]:
    """Format a structured field (strengths, weaknesses, etc.)."""
    if not field_value:
        return []

    lines = []
    field_label = field_name.replace("_", " ").title()

    if isinstance(field_value, list):
        lines.append(f"**{field_label}:**")
        for item in field_value:
            item_text = item if isinstance(item, str) else str(item)
            lines.append(f"- {item_text}")
        lines.append("")
    elif isinstance(field_value, str):
        lines.append(f"**{field_label}:** {field_value}")
        lines.append("")

    return lines


def _format_other_fields(other_fields: Dict[str, Any]) -> List[str]:
    """Format any remaining unrecognized fields."""
    if not other_fields:
        return []

    lines = [f"**{FIELD_LABEL_ADDITIONAL}:**"]
    for key, value in sorted(other_fields.items()):
        if not value or not str(value).strip():
            continue

        field_label = key.replace("_", " ").title()

        if isinstance(value, list):
            lines.append(f"**{field_label}:**")
            for item in value:
                lines.append(f"- {item}")
        else:
            lines.append(f"**{field_label}:** {value}")

    lines.append("")
    return lines


def _format_single_expert_review(
    review: Any, expert_metadata: Optional[Dict[str, Any]], index: int, total: int
) -> List[str]:
    """Format a single expert's review."""
    lines = []

    # Extract expert metadata
    if expert_metadata:
        expert_name = expert_metadata.get("name", f"{DEFAULT_EXPERT_NAME} {index}")
        discipline = expert_metadata.get("discipline", DEFAULT_DISCIPLINE)
        position = expert_metadata.get("position")
        education = expert_metadata.get("education")
    else:
        expert_name = f"{DEFAULT_EXPERT_NAME} {index}"
        discipline = DEFAULT_DISCIPLINE
        position = None
        education = None

    # Add expert header
    lines.extend(
        _format_expert_header(expert_name, discipline, position, education, index)
    )

    # Extract and categorize review data
    review_data = normalize_to_dict(review)
    categories = categorize_review_fields(review_data)

    # Add sections in order
    lines.extend(_format_scores(categories["scores"]))
    lines.extend(_format_justifications(categories["justifications"]))

    # Add structured fields
    other_fields = {**categories["structured"], **categories["other"]}
    for field_name in (
        "strengths",
        "weaknesses",
        "recommendation",
        "overall_assessment",
    ):
        if field_name in other_fields:
            lines.extend(_format_structured_field(field_name, other_fields[field_name]))

    # Add any remaining fields (only if no justifications were found)
    if not categories["justifications"]:
        remaining = {
            k: v
            for k, v in other_fields.items()
            if k
            not in ("strengths", "weaknesses", "recommendation", "overall_assessment")
            and v
            and str(v).strip()
        }
        if remaining:
            lines.extend(_format_other_fields(remaining))

    # Add separator between experts
    if index < total:
        lines.extend(["---", ""])

    return lines


def _format_expert_reviews(
    expert_reviews: Optional[list], generated_experts: Optional[list]
) -> List[str]:
    """Format all expert reviews section."""
    if not expert_reviews or len(expert_reviews) == 0:
        return []

    lines = [
        SECTION_SEPARATOR,
        "",
        f"{EXPERT_REVIEWS_HEADER} ({len(expert_reviews)} experts)",
        "",
    ]

    for i, review in enumerate(expert_reviews, 1):
        # Get corresponding expert metadata
        expert_metadata = None
        if generated_experts and i <= len(generated_experts):
            expert_metadata = normalize_to_dict(generated_experts[i - 1])

        lines.extend(
            _format_single_expert_review(
                review, expert_metadata, i, len(expert_reviews)
            )
        )

    return lines


def _format_review_report(
    summary: str,
    issues_count: int,
    critical_issues: Optional[list] = None,
    expert_reviews: Optional[list] = None,
    generated_experts: Optional[list] = None,
    novelty_findings: str = "",
    declined_findings: str = "",
) -> str:
    """
    Format review report for prepending to document.

    This is the main entry point that orchestrates all formatting functions.
    """
    lines = []

    # Header with executive summary
    lines.extend(_build_report_header(summary, issues_count))

    # Critical issues
    lines.extend(_format_critical_issues(critical_issues))

    # Novelty assessment (if novelty checking was performed)
    if novelty_findings and novelty_findings.strip():
        lines.extend(
            [
                SECTION_SEPARATOR,
                "",
                NOVELTY_FINDINGS_HEADER,
                "",
                novelty_findings.strip(),
                "",
            ]
        )

    # Findings the editor handed back. Placed after the issues and before the
    # expert reviews: they are part of the answer to "what was done about the
    # review", not a separate opinion about the document.
    if declined_findings and declined_findings.strip():
        lines.extend(
            [
                SECTION_SEPARATOR,
                "",
                DECLINED_FINDINGS_HEADER,
                "",
                declined_findings.strip(),
                "",
            ]
        )

    # Individual expert reviews
    lines.extend(_format_expert_reviews(expert_reviews, generated_experts))

    # Footer
    lines.extend([SECTION_SEPARATOR, ""])

    return "\n".join(lines)


# ── markdown → Word ─────────────────────────────────────────────────────


def prepend_review_section(editor: Any, review_markdown: str) -> None:
    """
    Prepend review report as formatted content at document beginning.

    Parses the markdown into Word paragraphs with appropriate styles:

      • ``# / ## / ### / #### / ##### / ######`` → Heading 1–6
      • ``- item`` or ``* item``                  → ``List Bullet`` style
      • ``1. item`` (any leading digits + dot)    → ``List Number`` style
      • ``> text``                                → ``Quote`` style
      • ``---``                                   → horizontal rule
      • ``| a | b |`` (+ a ``| --- |`` row)       → a real Word table
      • ``**whole line**`` / ``*whole line*``     → bold / italic paragraph
      • everything else                           → regular paragraph

    Inline ``**bold**`` and ``*italic*`` are applied inside any
    block (including headings, list items, and table cells) via
    ``_parse_inline_markdown``.

    Tables are the one multi-line construct, so the markdown is
    converted block-at-a-time by
    ``_convert_markdown_to_block_elements`` rather than line-at-a-time.
    The resulting ``<w:tbl>`` elements are inserted into the body but
    are deliberately NOT registered in ``editor.paragraphs``: that list
    drives the run-level diff/track-changes machinery in
    :meth:`write` and the patch matching in ``PatchDocxEditor``, and
    a table is neither an editable manuscript paragraph nor a valid
    input to either (it has no direct-child ``<w:r>`` runs).

    After the parsed content, the method appends one final paragraph
    containing a hard page break so the original manuscript body
    starts on its own page — visually separating the review report
    from the document being reviewed.

    The whole block is bracketed by a review-report bookmark, which is
    what stops a later round reading its predecessor's findings as
    manuscript (see :func:`.reading.body_paragraphs`). Reports STACK: a
    second round prepends its own above the first rather than replacing
    it, because the earlier report is the earlier reviewer's record and
    deleting it would throw away findings the author may not have worked
    through yet.

    Args:
        review_markdown: Markdown-formatted review report
    """
    doc_path = editor._doc_path
    tree = editor.trees[doc_path]
    root = tree.getroot()
    body = root.find(".//w:body", namespaces=NS)

    if body is None:
        return  # No body to prepend to

    elements_to_prepend = _convert_markdown_to_block_elements(review_markdown)
    # Trailing page break separates the review report from the
    # manuscript that follows. Always added — review reports want
    # a clean visual boundary.
    elements_to_prepend.append(_build_page_break_paragraph())

    start, end = _build_review_report_bookmark(root)
    block = [start, *elements_to_prepend, end]

    # Create ParagraphData objects WITHOUT re-extracting from XML so
    # any changes made by apply_patches() are preserved.
    prepended_paragraphs = []
    for position, elem in enumerate(block):
        body.insert(position, elem)

        if elem.tag != f"{{{NS['w']}}}p":
            # A <w:tbl> is in the body but out of the paragraph model
            # — see this method's docstring for why. So are the two
            # bookmark markers.
            continue

        prepended_paragraphs.append(ParagraphData(doc_path, elem))

    editor.paragraphs = prepended_paragraphs + editor.paragraphs


def _build_review_report_bookmark(root) -> tuple[Any, Any]:
    """A ``bookmarkStart`` / ``bookmarkEnd`` pair naming one review report.

    The id and the number in the name are both taken above whatever the
    document already carries, so a second round's report gets its own
    bookmark instead of colliding with the first's — Word renames a
    duplicate bookmark on save, and a renamed marker is an unmarked report.
    """
    used_ids = {
        element.get(qn("w:id"))
        for element in root.iter(qn("w:bookmarkStart"))
        if element.get(qn("w:id")) is not None
    }
    used_names = {
        element.get(qn("w:name")) for element in root.iter(qn("w:bookmarkStart"))
    }
    numeric = [int(value) for value in used_ids if str(value).isdigit()]
    bookmark_id = str(max(numeric) + 1) if numeric else "1"

    ordinal = 1
    while f"{REVIEW_REPORT_BOOKMARK_PREFIX}{ordinal}" in used_names:
        ordinal += 1
    name = f"{REVIEW_REPORT_BOOKMARK_PREFIX}{ordinal}"

    start = etree.Element(qn("w:bookmarkStart"))
    start.set(qn("w:id"), bookmark_id)
    start.set(qn("w:name"), name)
    end = etree.Element(qn("w:bookmarkEnd"))
    end.set(qn("w:id"), bookmark_id)
    return start, end


def _convert_markdown_to_block_elements(review_markdown: str) -> list:
    """Convert review markdown into a flat list of Word body elements.

    Almost every line maps 1:1 onto a ``<w:p>`` via
    :meth:`_build_review_paragraph`. Tables are the exception: they
    are the only construct in the report vocabulary that spans
    several lines, so consecutive pipe-delimited lines are buffered
    and handed to :meth:`_build_markdown_table_elements` as a block.
    A line-at-a-time cascade structurally cannot build a table, which
    is why this wrapper exists rather than another branch in
    ``_build_review_paragraph``.
    """
    elements: list = []
    table_buffer: list[str] = []

    for line in review_markdown.split("\n"):
        if _is_markdown_table_row(line):
            table_buffer.append(line)
            continue
        if table_buffer:
            elements.extend(_build_markdown_table_elements(table_buffer))
            table_buffer = []
        elements.append(_build_review_paragraph(line))

    if table_buffer:
        elements.extend(_build_markdown_table_elements(table_buffer))

    return elements


def _build_review_paragraph(line: str):
    """Convert one markdown line into a Word `<w:p>` element.

    Returns the constructed XML element. The line classification
    cascade is order-sensitive: more specific patterns first.
    """
    # Match heading prefixes (# through ######)
    for level in range(6, 0, -1):
        prefix = "#" * level + " "
        if line.startswith(prefix):
            return _make_styled_paragraph(
                style_name=f"Heading{level}",
                body=line[len(prefix) :],
            )

    stripped = line.strip()

    # Horizontal rule
    if stripped == "---":
        return _make_horizontal_rule_paragraph()

    # Bullet list:  "- item" or "* item"  (single space; not "**")
    if (
        len(line) >= 2
        and line[0] in ("-", "*")
        and line[1] == " "
        and not line.startswith("**")
    ):
        return _make_styled_paragraph(
            style_name="ListBullet",
            body=line[2:],
        )

    # Numbered list:  "1. item" / "12. item"
    if stripped and stripped[0].isdigit():
        digits_end = 0
        while digits_end < len(stripped) and stripped[digits_end].isdigit():
            digits_end += 1
        if (
            digits_end > 0
            and digits_end + 1 < len(stripped)
            and stripped[digits_end] == "."
            and stripped[digits_end + 1] == " "
        ):
            return _make_styled_paragraph(
                style_name="ListNumber",
                body=stripped[digits_end + 2 :],
            )

    # Blockquote
    if line.startswith("> "):
        return _make_styled_paragraph(
            style_name="Quote",
            body=line[2:],
        )
    if stripped == ">":
        return _make_styled_paragraph(style_name="Quote", body="")

    # Whole-line bold / italic (no surrounding text)
    if line.startswith("**") and line.endswith("**") and len(line) > 4:
        return _make_inline_marked_paragraph(line[2:-2], bold=True)
    if (
        line.startswith("*")
        and line.endswith("*")
        and not line.startswith("**")
        and len(line) > 2
    ):
        return _make_inline_marked_paragraph(line[1:-1], italic=True)

    # Regular paragraph — empty lines preserve spacing
    return _make_styled_paragraph(style_name=None, body=line)


def _parse_inline_markdown(text: str, parent_elem):
    """
    Parse inline markdown formatting and create Word XML run elements.

    Handles **bold** and *italic* within text, creating properly formatted runs.

    Args:
        text: Text with inline markdown formatting
        parent_elem: Parent XML element to append runs to
    """
    if not text:
        return

    # Pattern to match **bold** or *italic* (non-greedy)
    pattern = r"(\*\*.*?\*\*|\*.*?\*)"

    parts = re.split(pattern, text)

    for part in parts:
        if not part:
            continue

        r_elem = etree.SubElement(parent_elem, f"{{{NS['w']}}}r")

        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            # Bold text
            rPr = etree.SubElement(r_elem, f"{{{NS['w']}}}rPr")
            etree.SubElement(rPr, f"{{{NS['w']}}}b")
            t_elem = etree.SubElement(r_elem, f"{{{NS['w']}}}t")
            t_elem.text = part[2:-2]
        elif (
            part.startswith("*")
            and part.endswith("*")
            and len(part) > 2
            and not part.startswith("**")
        ):
            # Italic text
            rPr = etree.SubElement(r_elem, f"{{{NS['w']}}}rPr")
            etree.SubElement(rPr, f"{{{NS['w']}}}i")
            t_elem = etree.SubElement(r_elem, f"{{{NS['w']}}}t")
            t_elem.text = part[1:-1]
        else:
            # Regular text
            t_elem = etree.SubElement(r_elem, f"{{{NS['w']}}}t")
            t_elem.text = part
            # Preserve space if text has leading/trailing spaces
            if part and (part[0].isspace() or part[-1].isspace()):
                t_elem.set(
                    "{http://www.w3.org/XML/1998/namespace}space", "preserve"
                )


def _make_styled_paragraph(*, style_name, body: str):
    """Build a paragraph with the given style; body uses inline markdown."""
    p_elem = etree.Element(f"{{{NS['w']}}}p")
    if style_name:
        pPr = etree.SubElement(p_elem, f"{{{NS['w']}}}pPr")
        pStyle = etree.SubElement(pPr, f"{{{NS['w']}}}pStyle")
        pStyle.set(f"{{{NS['w']}}}val", style_name)

    if body.strip():
        _parse_inline_markdown(body, p_elem)
    else:
        # Empty paragraph — preserve vertical spacing in the docx
        r_elem = etree.SubElement(p_elem, f"{{{NS['w']}}}r")
        t_elem = etree.SubElement(r_elem, f"{{{NS['w']}}}t")
        t_elem.text = ""
        t_elem.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return p_elem


def _make_horizontal_rule_paragraph():
    """Empty paragraph with a bottom border = a markdown horizontal rule."""
    p_elem = etree.Element(f"{{{NS['w']}}}p")
    pPr = etree.SubElement(p_elem, f"{{{NS['w']}}}pPr")
    pBdr = etree.SubElement(pPr, f"{{{NS['w']}}}pBdr")
    bottom = etree.SubElement(pBdr, f"{{{NS['w']}}}bottom")
    bottom.set(f"{{{NS['w']}}}val", "single")
    bottom.set(f"{{{NS['w']}}}sz", "6")
    bottom.set(f"{{{NS['w']}}}space", "1")
    bottom.set(f"{{{NS['w']}}}color", "auto")
    return p_elem


def _make_inline_marked_paragraph(
    text: str, *, bold: bool = False, italic: bool = False
):
    """Single-run paragraph with bold/italic applied to its only run."""
    p_elem = etree.Element(f"{{{NS['w']}}}p")
    r_elem = etree.SubElement(p_elem, f"{{{NS['w']}}}r")
    rPr = etree.SubElement(r_elem, f"{{{NS['w']}}}rPr")
    if bold:
        etree.SubElement(rPr, f"{{{NS['w']}}}b")
    if italic:
        etree.SubElement(rPr, f"{{{NS['w']}}}i")
    t_elem = etree.SubElement(r_elem, f"{{{NS['w']}}}t")
    t_elem.text = text
    return p_elem


def _is_markdown_table_row(line: str) -> bool:
    """True for a line shaped like a table row: ``| a | b |``.

    Shape only. Whether a buffered run of such lines is *actually* a
    table is decided by :meth:`_build_markdown_table_elements`, which
    additionally requires the ``| --- |`` separator row — a lone
    pipe-wrapped line is ordinary prose.
    """
    stripped = line.strip()
    return (
        len(stripped) >= 2 and stripped.startswith("|") and stripped.endswith("|")
    )


def _split_markdown_table_row(line: str) -> list[str]:
    r"""Split ``| a | b |`` into ``["a", "b"]``.

    Splits on unescaped pipes only and unescapes ``\|`` afterwards,
    so a finding whose text contains a literal pipe does not silently
    gain a column and make the whole table look ragged.
    """
    inner = line.strip()[1:-1]
    return [
        cell.replace("\\|", "|").strip() for cell in re.split(r"(?<!\\)\|", inner)
    ]


def _is_markdown_table_separator_row(line: str) -> bool:
    """True for the ``| --- | :---: |`` row that follows a table header."""
    cells = _split_markdown_table_row(line)
    return bool(cells) and all(
        re.fullmatch(r":?-+:?", cell) is not None for cell in cells
    )


def _build_markdown_table_elements(rows: list[str]) -> list:
    """Convert a buffered block of pipe-delimited lines into Word XML.

    Returns a single-element list holding a ``<w:tbl>`` when the block
    is a well-formed markdown table, and otherwise one literal
    paragraph per line — the pre-table behaviour.

    Falling back instead of raising is deliberate: this markdown is
    prepended to the author's manuscript, and a malformed table in
    generated report prose must not cost them the entire render. The
    ragged case is logged at warning level so the malformation is
    still visible rather than silently absorbed.
    """
    if len(rows) < 2 or not _is_markdown_table_separator_row(rows[1]):
        # Pipe-wrapped lines with no separator row are not a table.
        return [_build_review_paragraph(line) for line in rows]

    header_cells = _split_markdown_table_row(rows[0])
    body_rows = [_split_markdown_table_row(line) for line in rows[2:]]

    ragged_count = sum(1 for cells in body_rows if len(cells) != len(header_cells))
    if ragged_count:
        logger.warning(
            "[markdown-table] ragged table: header has %d column(s) but "
            "%d of %d body row(s) differ — rendering the %d line(s) as "
            "literal paragraphs instead of a Word table",
            len(header_cells),
            ragged_count,
            len(body_rows),
            len(rows),
        )
        return [_build_review_paragraph(line) for line in rows]

    return [
        _make_table_element(header_cells=header_cells, body_rows=body_rows)
    ]


def _make_table_element(
    *, header_cells: list[str], body_rows: list[list[str]]
):
    """Build a ``<w:tbl>`` with a bolded, page-repeating header row."""
    tbl = etree.Element(f"{{{NS['w']}}}tbl")

    tblPr = etree.SubElement(tbl, f"{{{NS['w']}}}tblPr")
    tblW = etree.SubElement(tblPr, f"{{{NS['w']}}}tblW")
    tblW.set(f"{{{NS['w']}}}w", "5000")  # 100% in fiftieths of a percent
    tblW.set(f"{{{NS['w']}}}type", "pct")
    # Borders are written out explicitly rather than referenced via
    # <w:tblStyle w:val="TableGrid"/>: the manuscript being reviewed
    # owns styles.xml and need not define any table style, and a
    # dangling style reference renders as an invisible grid.
    tblBorders = etree.SubElement(tblPr, f"{{{NS['w']}}}tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = etree.SubElement(tblBorders, f"{{{NS['w']}}}{edge}")
        border.set(f"{{{NS['w']}}}val", "single")
        border.set(f"{{{NS['w']}}}sz", "4")
        border.set(f"{{{NS['w']}}}space", "0")
        border.set(f"{{{NS['w']}}}color", "auto")

    # Equal columns across a 6.5in text block (9360 twips). Word
    # re-flows these to fit content; they only need to be present and
    # sane, since tblW above asks for the full text width.
    column_width = 9360 // max(len(header_cells), 1)
    tblGrid = etree.SubElement(tbl, f"{{{NS['w']}}}tblGrid")
    for _ in header_cells:
        gridCol = etree.SubElement(tblGrid, f"{{{NS['w']}}}gridCol")
        gridCol.set(f"{{{NS['w']}}}w", str(column_width))

    tbl.append(_make_table_row(cells=header_cells, is_header=True))
    for cells in body_rows:
        tbl.append(_make_table_row(cells=cells, is_header=False))

    return tbl


def _make_table_row(*, cells: list[str], is_header: bool):
    """Build one ``<w:tr>``; each cell gets the inline-markdown treatment."""
    tr = etree.Element(f"{{{NS['w']}}}tr")
    if is_header:
        trPr = etree.SubElement(tr, f"{{{NS['w']}}}trPr")
        # Repeat the header when the table spans a page break.
        etree.SubElement(trPr, f"{{{NS['w']}}}tblHeader")

    for cell in cells:
        tc = etree.SubElement(tr, f"{{{NS['w']}}}tc")
        tcPr = etree.SubElement(tc, f"{{{NS['w']}}}tcPr")
        tcW = etree.SubElement(tcPr, f"{{{NS['w']}}}tcW")
        tcW.set(f"{{{NS['w']}}}w", "0")
        tcW.set(f"{{{NS['w']}}}type", "auto")
        # Same builder as a normal paragraph, so **bold** / *italic*
        # inside a cell renders as formatting, not as literal asterisks.
        cell_paragraph = _make_styled_paragraph(style_name=None, body=cell)
        if is_header:
            _apply_bold_to_runs(cell_paragraph)
        tc.append(cell_paragraph)

    return tr


def _apply_bold_to_runs(p_elem) -> None:
    """Mark every run in a paragraph bold.

    Applied *after* ``_parse_inline_markdown`` rather than by wrapping
    the cell text in ``**``: pre-wrapping would collide with any
    markers the header cell already contains. ``<w:b/>`` is inserted
    at the front of ``<w:rPr>`` because ECMA-376 orders ``b`` before
    the ``i`` that inline parsing may already have emitted.
    """
    for r_elem in p_elem.findall(f"{{{NS['w']}}}r"):
        rPr = r_elem.find(f"{{{NS['w']}}}rPr")
        if rPr is None:
            rPr = etree.Element(f"{{{NS['w']}}}rPr")
            r_elem.insert(0, rPr)
        if rPr.find(f"{{{NS['w']}}}b") is None:
            rPr.insert(0, etree.Element(f"{{{NS['w']}}}b"))


def _build_page_break_paragraph():
    """A paragraph containing a hard page break — separates review from body."""
    p_elem = etree.Element(f"{{{NS['w']}}}p")
    r_elem = etree.SubElement(p_elem, f"{{{NS['w']}}}r")
    br_elem = etree.SubElement(r_elem, f"{{{NS['w']}}}br")
    br_elem.set(f"{{{NS['w']}}}type", "page")
    return p_elem
