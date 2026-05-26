"""Word document renderer for ``ReviewResult``.

Reuses the in-tree ``whetstone.docx.finalization.finalize_reviewed_document``
(the existing 5k-line track-changes machinery) by adapting v2's flat
``Edit`` and ``Finding`` types into v1's ``DocumentPatch`` shape:

  • Edit       → DocumentPatch(patch_type="text_edit",
                                text_pattern=original_text,
                                new_text=new_text,
                                explanation=rationale)
  • Finding    → DocumentPatch(patch_type="comment",
                                text_pattern=first_quote_text,
                                comment_text="<title>\\n\\n<rationale>",
                                explanation=rationale)

Findings without quotes are skipped — there's nothing to anchor the
comment to in the .docx structure. Edits whose ``original_text`` doesn't
appear in the .docx are also dropped silently (``apply_patches`` already
handles that).

Panel mode (``mode="panel"``) is now ALSO rendered: expert biosketches,
scored expert reviews, and the panel synthesis are folded into the
prepended review report. ``finalize_reviewed_document`` already accepts
``expert_reviews`` and ``generated_experts`` keywords; the v2 adapter
just passes them through. v2's ExpertProfile / ExpertReview pydantic
schemas are field-compatible with the v1 reader (which uses
``model_dump()`` then categorises by ``_score`` / ``_justification``
suffix).

Novelty findings (category="novelty") are also pulled out and passed as
``novelty_findings`` so they appear in their own report subsection
rather than mixed with the lens findings.

Requires an existing .docx file as input — Word's track-changes work
against a pre-existing structure. For PDF/HTML sources, render to
markdown or HTML instead, OR convert the source to .docx first via your
preferred tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .._watermark import (
    BANNER_TITLE,
    DISCLAIMER_SHORT,
    provenance_line,
    stamp_docx_core_properties,
)
from ..schemas import Edit, Finding, ReviewResult
from ._panel_layout import body_markdown


# The default tracked-change author. Deliberately includes "(AI)" so the
# Word File→Info pane and the tracked-changes attribution make AI
# authorship unmissable. Override via --allow-author-override on the CLI
# only; misrepresenting AI-generated edits as a human reviewer's may
# constitute research misconduct under most institutional codes.
DEFAULT_AI_AUTHOR = "andamentum-whetstone (AI)"


def render_docx(
    result: ReviewResult,
    *,
    source_docx_path: str | Path,
    output_path: str | Path,
    author: str = DEFAULT_AI_AUTHOR,
    model: str | None = None,
    visible_watermark: bool = True,
) -> Any:
    """Render a ReviewResult as a .docx with track changes + comments.

    Edits become tracked changes (deletion of the original span,
    insertion of the new span). Findings become Word comments anchored
    to the first verbatim quote of each finding. The executive summary
    becomes a prepended review report.

    For panel-mode results, the prepended report includes the panel
    synthesis prose, scored per-expert reviews, and expert biosketches.

    Parameters
    ----------
    result:
        The ReviewResult to render.
    source_docx_path:
        Path to the original .docx file. Track changes are applied
        against this document's structure.
    output_path:
        Where to write the reviewed .docx.
    author:
        Author name attributed to the tracked changes.

    Returns
    -------
    The ``PatchApplicationResult`` from the finalisation machinery
    (counts of applied/failed patches).
    """
    # Defer the heavy docx import: the track-changes machinery is large,
    # and we want v2 to be importable without dragging it into the import
    # graph by default.
    from andamentum.whetstone.docx.finalization import finalize_reviewed_document
    from andamentum.whetstone.models import DocumentPatch

    patches = _to_document_patches(result, DocumentPatch)
    findings = list(result.findings) + list(result.deterministic_findings)

    review_summary = _strip_leading_exec_heading(
        result.summary or _fallback_summary(result)
    )
    if result.panel_synthesis is not None:
        # Panel mode: the synthesis is the authoritative summary. ``result.summary``
        # is intentionally empty (see panel_synthesise.py); render the synthesis
        # body (no ``## Panel synthesis`` heading — content goes inside the
        # ``## Executive Summary`` already added by _build_report_header).
        review_summary = body_markdown(
            result.panel_synthesis, list(result.expert_reviews)
        )

    if visible_watermark:
        banner = (
            f"> **⚠ {BANNER_TITLE}.** {DISCLAIMER_SHORT}\n"
            f">\n"
            f"> *Produced by {provenance_line(model=model)}.*\n\n"
        )
        review_summary = banner + (review_summary or "")

    novelty_findings_text = _collect_novelty_findings(result)

    _, patch_result = finalize_reviewed_document(
        original_file_path=Path(source_docx_path),
        patches=patches,
        review_summary=review_summary,
        issues_count=len(findings),
        output_path=Path(output_path),
        author=author,
        # Panel-mode payload (None / empty when not in panel mode)
        expert_reviews=list(result.expert_reviews) or None,
        generated_experts=list(result.expert_profiles) or None,
        # Novelty-check payload (empty string when not used)
        novelty_findings=novelty_findings_text,
    )
    # Invisible metadata: always on, regardless of visible_watermark.
    stamp_docx_core_properties(Path(output_path), model=model)
    return patch_result


# ── Adapter: v2 → v1 DocumentPatch ──────────────────────────────────────


def _to_document_patches(result: ReviewResult, DocumentPatch) -> list:
    """Convert v2's Edit + Finding into v1's DocumentPatch list."""
    patches: list = []

    for e in result.edits:
        patches.append(_edit_to_patch(e, DocumentPatch))

    findings = list(result.findings) + list(result.deterministic_findings)
    for f in findings:
        # Novelty findings are surfaced separately via novelty_findings;
        # don't ALSO emit them as anchored comments because they have
        # no anchor in the manuscript text.
        if f.category == "novelty":
            continue
        patch = _finding_to_patch(f, DocumentPatch)
        if patch is not None:
            patches.append(patch)

    return patches


def _edit_to_patch(edit: Edit, DocumentPatch):
    return DocumentPatch(
        patch_type="text_edit",
        text_pattern=edit.original_text,
        new_text=edit.new_text,
        explanation=edit.rationale or edit.title or "Suggested edit",
        confidence=_confidence_to_float(edit.confidence),
    )


def _finding_to_patch(finding: Finding, DocumentPatch):
    if not finding.quotes:
        return None
    anchor = finding.quotes[0].text
    if not anchor or not anchor.strip():
        return None
    body = finding.title
    if finding.rationale:
        body = f"{finding.title}\n\n{finding.rationale}"
    # Cross-perspective corroboration recorded by Consolidate: surface it so
    # the author sees the comment is not a lone opinion.
    if len(finding.corroborated_by) >= 2:
        body = f"{body}\n\nRaised by {len(finding.corroborated_by)} perspectives: {', '.join(finding.corroborated_by)}."
    # explanation == comment_text on purpose: the comment body is already the
    # complete title+rationale, so the patch editor must NOT re-append it as a
    # "Note:" (which produced duplicated comment text). explanation is required
    # on DocumentPatch; setting it equal suppresses the duplication.
    return DocumentPatch(
        patch_type="comment",
        text_pattern=anchor,
        comment_text=body,
        explanation=body,
        confidence=_confidence_to_float(finding.confidence),
    )


def _strip_leading_exec_heading(summary: str) -> str:
    """Drop a leading ``## Executive Summary`` heading from the summary.

    The docx report header (``_build_report_header``) supplies its own
    ``## Executive Summary`` heading; v2's synthesised summary leads with the
    same heading, which produced "Executive Summary" twice in the rendered
    .docx. Stripped only here — the markdown/HTML renderers have no report
    header and keep the heading.
    """
    from andamentum.whetstone.docx.constants import EXECUTIVE_SUMMARY_HEADER

    stripped = summary.lstrip()
    if stripped.startswith(EXECUTIVE_SUMMARY_HEADER):
        return stripped[len(EXECUTIVE_SUMMARY_HEADER) :].lstrip("\n")
    return summary


def _confidence_to_float(level: str) -> float:
    """v2 uses low/medium/high enums; v1's DocumentPatch uses a 0..1 float."""
    return {"low": 0.4, "medium": 0.7, "high": 0.95}.get(level, 0.7)


def _fallback_summary(result: ReviewResult) -> str:
    """Generate a plain summary for when Synthesise didn't run."""
    n_findings = len(result.findings) + len(result.deterministic_findings)
    n_edits = len(result.edits)
    n_questions = len(result.author_questions)
    parts = [
        f"Whetstone review: {n_findings} finding(s), {n_edits} edit(s), "
        f"{n_questions} author question(s)."
    ]
    if result.expert_reviews:
        parts.append(f"Panel: {len(result.expert_reviews)} expert review(s).")
    if not (result.summary or "").strip():
        parts.append("Run with model= for synthesis prose.")
    return " ".join(parts)


def _collect_novelty_findings(result: ReviewResult) -> str:
    """Concatenate novelty-category findings into a markdown block.

    The v1 finalize_reviewed_document accepts a ``novelty_findings``
    string and renders it as its own report subsection. Pulling the
    novelty findings out of the lens-finding pool prevents them from
    becoming anchored comments against fictional text spans (they are
    document-level claims, not section-anchored observations).
    """
    novelty = [
        f
        for f in (list(result.findings) + list(result.deterministic_findings))
        if f.category == "novelty"
    ]
    if not novelty:
        return ""

    lines = [
        f"{len(novelty)} novelty claim(s) flagged by literature search:",
        "",
    ]
    for f in novelty:
        lines.append(f"- **{f.title}** ({f.severity}, {f.confidence} confidence)")
        if f.rationale:
            # Indent rationale under its bullet
            for ln in f.rationale.splitlines():
                lines.append(f"    {ln}")
        lines.append("")
    return "\n".join(lines)
