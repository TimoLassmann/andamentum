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
from typing import List, Optional, Dict, Any
import tempfile

from .patch_editor import PatchDocxEditor
from ..models import DocumentPatch, PatchApplicationResult
from .exceptions import FileNotFoundError as DocNotFoundError, FinalizationError
from .constants import (
    REPORT_TITLE,
    SECTION_SEPARATOR,
    EXECUTIVE_SUMMARY_HEADER,
    CRITICAL_ISSUES_HEADER,
    EXPERT_REVIEWS_HEADER,
    NOVELTY_FINDINGS_HEADER,
    FIELD_LABEL_LOCATION,
    FIELD_LABEL_DISCIPLINE,
    FIELD_LABEL_POSITION,
    FIELD_LABEL_EDUCATION,
    FIELD_LABEL_SCORES,
    FIELD_LABEL_ASSESSMENTS,
    FIELD_LABEL_ADDITIONAL,
    DEFAULT_DESCRIPTION,
    DEFAULT_SEVERITY,
    DEFAULT_ISSUE_TITLE,
    DEFAULT_EXPERT_NAME,
    DEFAULT_DISCIPLINE,
)
from .model_utils import normalize_to_dict, extract_fields, categorize_review_fields


def finalize_reviewed_document(
    original_file_path: str | Path,
    patches: List[DocumentPatch],
    review_summary: str = "",
    issues_count: int = 0,
    output_path: Optional[str | Path] = None,
    author: str = "Mosaic Review",
    use_patch_authors: bool = True,
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
        author: Default author for changes (if not using patch authors)
        use_patch_authors: Whether to use author from each patch

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
        patch_result = editor.apply_patches(patches, use_patch_authors=use_patch_authors)

        # Step 3: Prepend review report if provided
        if review_summary:
            review_text = _format_review_report(
                review_summary, issues_count, critical_issues, expert_reviews, generated_experts, novelty_findings
            )
            editor.prepend_review_section(review_text)

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
    expert_name: str, discipline: str, position: Optional[str], education: Optional[str], index: int
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
    lines.extend(_format_expert_header(expert_name, discipline, position, education, index))

    # Extract and categorize review data
    review_data = normalize_to_dict(review)
    categories = categorize_review_fields(review_data)

    # Add sections in order
    lines.extend(_format_scores(categories["scores"]))
    lines.extend(_format_justifications(categories["justifications"]))

    # Add structured fields
    other_fields = {**categories["structured"], **categories["other"]}
    for field_name in ("strengths", "weaknesses", "recommendation", "overall_assessment"):
        if field_name in other_fields:
            lines.extend(_format_structured_field(field_name, other_fields[field_name]))

    # Add any remaining fields (only if no justifications were found)
    if not categories["justifications"]:
        remaining = {
            k: v
            for k, v in other_fields.items()
            if k not in ("strengths", "weaknesses", "recommendation", "overall_assessment") and v and str(v).strip()
        }
        if remaining:
            lines.extend(_format_other_fields(remaining))

    # Add separator between experts
    if index < total:
        lines.extend(["---", ""])

    return lines


def _format_expert_reviews(expert_reviews: Optional[list], generated_experts: Optional[list]) -> List[str]:
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

        lines.extend(_format_single_expert_review(review, expert_metadata, i, len(expert_reviews)))

    return lines


def _format_review_report(
    summary: str,
    issues_count: int,
    critical_issues: Optional[list] = None,
    expert_reviews: Optional[list] = None,
    generated_experts: Optional[list] = None,
    novelty_findings: str = "",
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

    # Individual expert reviews
    lines.extend(_format_expert_reviews(expert_reviews, generated_experts))

    # Footer
    lines.extend([SECTION_SEPARATOR, ""])

    return "\n".join(lines)
