"""Word document renderer with track changes.

Applies DocumentPatch objects as Word track changes and prepends a review
report. Wraps the internal finalization module.
"""

from pathlib import Path
from typing import Optional

from ..models import DocumentPatch, PatchApplicationResult


def render_docx(
    *,
    input_path: Path,
    output_path: Path,
    patches: list[DocumentPatch],
    review_summary: str = "",
    critical_issues: Optional[list] = None,
    expert_reviews: Optional[list] = None,
    generated_experts: Optional[list] = None,
    novelty_findings: str = "",
    author: str = "Whetstone Review",
    checklist_items: Optional[list] = None,
) -> PatchApplicationResult:
    """Render review results as a Word document with track changes.

    Applies patches to the original .docx file as tracked changes,
    then prepends a formatted review report.

    Args:
        input_path: Path to original .docx file.
        output_path: Where to save the reviewed document.
        patches: DocumentPatch objects to apply as track changes.
        review_summary: Executive summary text for the report header.
        critical_issues: List of critical issue objects for the report.
        expert_reviews: List of expert review objects (panel task).
        generated_experts: List of expert profile objects (panel task).
        novelty_findings: External novelty findings text (optional).
        author: Default author name for track changes.
        checklist_items: Optional list of ChecklistItem objects to prepend
            to the review report (checklist task).

    Returns:
        PatchApplicationResult with applied/failed patch counts.

    Raises:
        FileNotFoundError: If input_path doesn't exist.
        RuntimeError: If document operations fail.
    """
    from ..docx.finalization import finalize_reviewed_document

    _, patch_result = finalize_reviewed_document(
        original_file_path=input_path,
        patches=patches,
        review_summary=review_summary,
        issues_count=len(critical_issues) if critical_issues else 0,
        output_path=output_path,
        author=author,
        critical_issues=critical_issues,
        expert_reviews=expert_reviews,
        generated_experts=generated_experts,
        novelty_findings=novelty_findings,
        checklist_items=checklist_items,
    )
    return patch_result
