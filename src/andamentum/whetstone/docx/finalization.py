"""
Document Review Finalization Utility

Framework-agnostic utility for creating final reviewed documents.
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
from .exceptions import FileNotFoundError as DocNotFoundError, FinalizationError
from .review_report import _format_review_report, prepend_review_section


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
    configuration: str = "",
    other_edits: str = "",
) -> tuple[str, Any]:
    """Apply resolved annotations, prepend the report, and save.

    The annotations are not searched for: the reviewer read this document's
    own text and the offsets it returned are addresses in it. See
    :mod:`.annotate`.

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
                    configuration,
                    other_edits,
                ),
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
