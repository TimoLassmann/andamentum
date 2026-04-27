"""Word document renderer for ``ReviewResult``.

Reuses v1's `whetstone.docx.finalization.finalize_reviewed_document`
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
appear in the .docx are also dropped silently (v1's ``apply_patches``
already handles that).

Requires an existing .docx file as input — Word's track-changes work
against a pre-existing structure. For PDF/HTML sources, render to
markdown or HTML instead, OR convert the source to .docx first via your
preferred tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schemas import Edit, Finding, ReviewResult


def render_docx(
    result: ReviewResult,
    *,
    source_docx_path: str | Path,
    output_path: str | Path,
    author: str = "Whetstone Review",
) -> Any:
    """Render a ReviewResult as a .docx with track changes + comments.

    Edits become tracked changes (deletion of the original span,
    insertion of the new span). Findings become Word comments anchored
    to the first verbatim quote of each finding. The executive summary
    becomes a prepended review report.

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
    The ``PatchApplicationResult`` from v1's finalisation machinery
    (counts of applied/failed patches).
    """
    # Defer the v1 import: v1 owns the heavy docx machinery, but we want
    # v2 to be importable without dragging it into the import graph by
    # default.
    from andamentum.whetstone.docx.finalization import finalize_reviewed_document
    from andamentum.whetstone.models import DocumentPatch

    patches = _to_document_patches(result, DocumentPatch)
    findings = list(result.findings) + list(result.deterministic_findings)

    _, patch_result = finalize_reviewed_document(
        original_file_path=Path(source_docx_path),
        patches=patches,
        review_summary=result.summary or _fallback_summary(result),
        issues_count=len(findings),
        output_path=Path(output_path),
        author=author,
    )
    return patch_result


# ── Adapter: v2 → v1 DocumentPatch ──────────────────────────────────────


def _to_document_patches(result: ReviewResult, DocumentPatch) -> list:
    """Convert v2's Edit + Finding into v1's DocumentPatch list."""
    patches: list = []

    for e in result.edits:
        patches.append(_edit_to_patch(e, DocumentPatch))

    findings = list(result.findings) + list(result.deterministic_findings)
    for f in findings:
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
    return DocumentPatch(
        patch_type="comment",
        text_pattern=anchor,
        comment_text=body,
        explanation=finding.rationale or finding.title,
        confidence=_confidence_to_float(finding.confidence),
    )


def _confidence_to_float(level: str) -> float:
    """v2 uses low/medium/high enums; v1's DocumentPatch uses a 0..1 float."""
    return {"low": 0.4, "medium": 0.7, "high": 0.95}.get(level, 0.7)


def _fallback_summary(result: ReviewResult) -> str:
    """Generate a plain summary for when Synthesise didn't run."""
    n_findings = len(result.findings) + len(result.deterministic_findings)
    n_edits = len(result.edits)
    n_questions = len(result.author_questions)
    return (
        f"Whetstone review: {n_findings} finding(s), {n_edits} edit(s), "
        f"{n_questions} author question(s). Run with model= for synthesis prose."
    )
