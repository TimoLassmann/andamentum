"""
Document Review Finalization Utility

Framework-agnostic utility for applying editing patches and creating final reviewed documents.
This is Layer 1 (utilities) - no framework dependencies.

Can be used by:
- Tools (Layer 3)
- Post-processors (Layer 2/3 bridge)
- Applications (Layer 4)
"""

from pathlib import Path
from typing import List, Optional, Any
import tempfile

from .low_level import DocxEditor
from .patch_editor import PatchDocxEditor
from ..models import DocumentPatch, PatchApplicationResult
from .exceptions import FileNotFoundError as DocNotFoundError, FinalizationError
from .review_report import _format_review_report, prepend_review_section


def finalize_reviewed_document(
    original_file_path: str | Path,
    patches: List[DocumentPatch],
    review_summary: str = "",
    issues_count: int = 0,
    output_path: Optional[str | Path] = None,
    author: str = "Mosaic Review",
    critical_issues: Optional[list] = None,
    expert_reviews: Optional[list] = None,
    generated_experts: Optional[list] = None,
    novelty_findings: str = "",
) -> tuple[str, PatchApplicationResult]:
    """
    Apply editing patches to create final reviewed Word document with track changes.

    This is a pure utility function with no framework dependencies. It performs
    the mechanical work of:
    1. Loading the original document
    2. Applying all patches with track changes
    3. Prepending review report (if provided)
    4. Saving to output file

    Args:
        original_file_path: Path to original .docx file
        patches: List of DocumentPatch objects to apply
        review_summary: Executive summary text to prepend
        issues_count: Number of issues identified
        output_path: Where to save result (if None, creates temp file)
        author: Author recorded on every tracked change

    Returns:
        tuple of (output_file_path, PatchApplicationResult)

    Raises:
        FileNotFoundError: If original file doesn't exist
        Exception: If patch application or file operations fail
    """
    original_path = Path(original_file_path)

    if not original_path.exists():
        raise DocNotFoundError(f"Original file not found: {original_file_path}")

    try:
        # Step 1: Initialize editor
        editor = PatchDocxEditor(str(original_path), author=author)

        # Step 2: Apply all patches with track changes
        patch_result = editor.apply_patches(patches)

        # Step 3: Prepend review report if provided
        if review_summary:
            review_text = _format_review_report(
                review_summary,
                issues_count,
                critical_issues,
                expert_reviews,
                generated_experts,
                novelty_findings,
            )
            prepend_review_section(editor, review_text)

        # Step 4: Save to output file
        if output_path:
            output_file = str(output_path)
        else:
            # Create temp file with proper suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                output_file = tmp.name

        editor.write(output_file)

        return output_file, patch_result

    except (OSError, IOError) as e:
        # File errors
        raise FinalizationError(f"File operation failed: {str(e)}") from e
    except Exception as e:
        # Unexpected errors
        raise FinalizationError(f"Document finalization failed: {str(e)}") from e


def finalize_annotated_document(
    original_file_path: str | Path,
    *,
    rewrites: List[Any],
    notes: List[Any],
    reviewed_text: str,
    review_summary: str = "",
    issues_count: int = 0,
    output_path: Optional[str | Path] = None,
    author: str = "Whetstone Review",
    critical_issues: Optional[list] = None,
    expert_reviews: Optional[list] = None,
    generated_experts: Optional[list] = None,
    novelty_findings: str = "",
    declined_findings: str = "",
    novelty_claims: str = "",
) -> tuple[str, Any]:
    """Apply resolved annotations, prepend the report, and save.

    The counterpart to :func:`finalize_reviewed_document` for a caller that
    already knows where its annotations go. Same report, same output shape; the
    difference is entirely in how the annotations were located, which is: they
    were not, because the reviewer read this document's own text and the
    offsets it returned are addresses in it. See :mod:`.annotate`.

    ``finalize_reviewed_document`` stays for ``andamentum-whetstone
    apply-patches``, whose input is a list of text patterns from an outside
    agent that never saw the document's coordinates. Searching is the right
    answer there, and only there.

    The report is prepended AFTER the annotations are placed, so the paragraphs
    it adds cannot shift an offset out from under an annotation.
    """
    from .annotate import CoordinateError, apply_annotations

    original_path = Path(original_file_path)
    if not original_path.exists():
        raise DocNotFoundError(f"Original file not found: {original_file_path}")

    try:
        editor = DocxEditor(str(original_path), author=author)
        report = apply_annotations(
            editor, rewrites=rewrites, notes=notes, reviewed_text=reviewed_text
        )

        if review_summary:
            prepend_review_section(
                editor,
                _format_review_report(
                    review_summary,
                    issues_count,
                    critical_issues,
                    expert_reviews,
                    generated_experts,
                    novelty_findings,
                    declined_findings,
                    novelty_claims,
                )
            )

        if output_path:
            output_file = str(output_path)
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                output_file = tmp.name

        editor.write(output_file)
        return output_file, report

    except CoordinateError:
        # The one failure that must not be re-dressed as a generic
        # finalisation error: it names a specific, actionable mismatch and its
        # message is the whole diagnosis.
        raise
    except (OSError, IOError) as e:
        raise FinalizationError(f"File operation failed: {str(e)}") from e
    except Exception as e:
        raise FinalizationError(f"Document finalization failed: {str(e)}") from e


