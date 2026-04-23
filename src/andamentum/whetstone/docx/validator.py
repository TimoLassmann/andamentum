#!/usr/bin/env python3
"""
Centralized patch validation for consistent validation logic.

This module eliminates the duplication of validation patterns found
across patch_docx_editor.py, sequential_processor.py, and other components.
"""

from typing import List, Tuple, Dict, Any
from difflib import SequenceMatcher

from ..models import DocumentPatch
from .result_types import ValidationResult, LocationResult


class PatchValidator:
    """
    Centralized validation for DocumentPatch objects.

    This class consolidates validation logic scattered across multiple
    components, providing consistent validation behavior.
    """

    @staticmethod
    def validate_patch_content(patch: DocumentPatch) -> ValidationResult:
        """
        Validate that a patch has all required content fields.

        Consolidates validation logic from simple_patch_models.py and other locations.

        Args:
            patch: DocumentPatch to validate

        Returns:
            ValidationResult indicating validity
        """
        # Check basic required fields
        if not patch.patch_type:
            return ValidationResult.invalid("Missing patch_type")

        if patch.patch_type not in ["text_edit", "comment", "document_analysis"]:
            return ValidationResult.invalid(f"Invalid patch_type: {patch.patch_type}")

        # text_pattern is required for text_edit and comment patches, but not for document_analysis
        if patch.patch_type in ["text_edit", "comment"]:
            if not patch.text_pattern or not patch.text_pattern.strip():
                return ValidationResult.invalid("Missing or empty text_pattern")

        if not patch.explanation or not patch.explanation.strip():
            return ValidationResult.invalid("Missing or empty explanation")

        # Validate confidence range
        if not (0.0 <= patch.confidence <= 1.0):
            return ValidationResult.invalid(f"Confidence {patch.confidence} not in range [0.0, 1.0]")

        # Check for suspicious patterns
        warnings = []

        # Type-specific validation
        if patch.patch_type == "text_edit":
            if not patch.new_text or not patch.new_text.strip():
                return ValidationResult.invalid("Text edit patch missing new_text")
        elif patch.patch_type == "comment":
            if not patch.comment_text or not patch.comment_text.strip():
                return ValidationResult.invalid("Comment patch missing comment_text")
        elif patch.patch_type == "document_analysis":
            if not patch.analysis_text or not patch.analysis_text.strip():
                return ValidationResult.invalid("Document analysis patch missing analysis_text")
            # Validate that it looks like structured analysis, not just a simple summary
            if len(patch.analysis_text.strip()) < 100:
                return ValidationResult.invalid("Document analysis text too short - should contain detailed analysis")
            # Check for markdown structure (## headers indicate structured format)
            if "##" not in patch.analysis_text:
                warnings.append(
                    "Document analysis appears unstructured - consider using markdown headers for better organization"
                )

        # Check if text_pattern is too short (might be ambiguous) - only for patches that use text_pattern
        if patch.text_pattern and len(patch.text_pattern.strip()) < 5:
            warnings.append("Text pattern is very short and might be ambiguous")

        # Check if confidence is suspiciously high for complex changes
        if patch.patch_type == "text_edit" and patch.confidence > 0.95:
            old_len = len(patch.text_pattern) if patch.text_pattern else 0
            new_len = len(patch.new_text) if patch.new_text else 0
            if abs(old_len - new_len) > old_len * 0.5:  # >50% length change
                warnings.append("High confidence for significant text change")

        if warnings:
            return ValidationResult.warning(
                f"Validation passed with warnings: {'; '.join(warnings)}", {"warnings": warnings}
            )

        return ValidationResult.valid("Patch content validation passed")

    @staticmethod
    def validate_patch_location(patch: DocumentPatch, paragraph_texts: List[str]) -> LocationResult:
        """
        Validate that a patch can be located in the document.

        Consolidates location validation from patch_docx_editor.py.

        Args:
            patch: DocumentPatch to validate location for
            paragraph_texts: List of paragraph text content

        Returns:
            LocationResult with location information
        """
        if not paragraph_texts:
            return LocationResult.not_found("Document has no paragraphs")

        # Check if text_pattern is provided
        if not patch.text_pattern:
            return LocationResult.not_found("No text pattern provided")

        # Try paragraph index first if specified
        if patch.paragraph_index is not None:
            if 0 <= patch.paragraph_index < len(paragraph_texts):
                para_text = paragraph_texts[patch.paragraph_index]
                if patch.text_pattern.lower().strip() in para_text.lower():
                    return LocationResult.found_at(
                        patch.paragraph_index, confidence=1.0, message="Found by paragraph index"
                    )
                else:
                    # Index is valid but pattern not found
                    return LocationResult.not_found(f"Pattern not found in paragraph {patch.paragraph_index}")
            else:
                return LocationResult.not_found(
                    f"Paragraph index {patch.paragraph_index} out of range [0, {len(paragraph_texts) - 1}]"
                )

        # Fall back to text pattern matching
        return PatchValidator._find_pattern_in_paragraphs(patch.text_pattern, paragraph_texts)

    @staticmethod
    def _find_pattern_in_paragraphs(pattern: str, paragraph_texts: List[str]) -> LocationResult:
        """
        Find a text pattern in paragraph list.

        Args:
            pattern: Text pattern to find
            paragraph_texts: List of paragraph texts to search

        Returns:
            LocationResult with search results
        """
        pattern_lower = pattern.lower().strip()
        exact_matches = []
        fuzzy_matches = []

        # First pass: exact matching
        for i, para_text in enumerate(paragraph_texts):
            if pattern_lower in para_text.lower():
                exact_matches.append(i)

        if len(exact_matches) == 1:
            return LocationResult.found_at(exact_matches[0], confidence=1.0, message="Exact pattern match found")
        elif len(exact_matches) > 1:
            return LocationResult.ambiguous(exact_matches, f"Pattern found in {len(exact_matches)} paragraphs")

        # Second pass: fuzzy matching
        for i, para_text in enumerate(paragraph_texts):
            similarity = PatchValidator._calculate_similarity(pattern_lower, para_text.lower())
            if similarity > 0.6:  # Threshold for fuzzy matching
                fuzzy_matches.append((i, similarity))

        if fuzzy_matches:
            # Sort by similarity, highest first
            fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
            best_match = fuzzy_matches[0]

            if best_match[1] > 0.8:  # High confidence fuzzy match
                return LocationResult.found_at(
                    best_match[0],
                    confidence=best_match[1],
                    message=f"Fuzzy match found (similarity: {best_match[1]:.2f})",
                )
            else:
                # Multiple fuzzy matches or low confidence
                alternatives = [match[0] for match in fuzzy_matches[:5]]  # Top 5
                return LocationResult.ambiguous(
                    alternatives, f"Multiple fuzzy matches found (best similarity: {best_match[1]:.2f})"
                )

        return LocationResult.not_found("Pattern not found in any paragraph")

    @staticmethod
    def _calculate_similarity(text1: str, text2: str) -> float:
        """
        Calculate text similarity using sequence matching.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        matcher = SequenceMatcher(None, text1, text2)
        return matcher.ratio()

    @staticmethod
    def validate_patch_application(patch: DocumentPatch, paragraph_index: int, paragraph_text: str) -> ValidationResult:
        """
        Validate that a patch can be safely applied to a specific paragraph.

        Args:
            patch: DocumentPatch to validate
            paragraph_index: Target paragraph index
            paragraph_text: Current paragraph text

        Returns:
            ValidationResult for application safety
        """
        # Basic content validation first
        content_result = PatchValidator.validate_patch_content(patch)
        if not content_result.is_valid:
            return content_result

        # Check that target text still exists
        if not patch.text_pattern:
            return ValidationResult.invalid("No text pattern provided")

        if patch.text_pattern.lower().strip() not in paragraph_text.lower():
            return ValidationResult.invalid("Target text pattern no longer exists in paragraph")

        # For text edits, check for potential conflicts
        if patch.patch_type == "text_edit":
            # Check if the replacement would create obvious problems
            if patch.new_text and patch.text_pattern:
                # Warn if replacement is dramatically different in length
                old_len = len(patch.text_pattern)
                new_len = len(patch.new_text)

                if new_len > old_len * 3:  # >3x length increase
                    return ValidationResult.warning(
                        "Text replacement significantly increases length",
                        {"old_length": old_len, "new_length": new_len},
                    )
                elif new_len < old_len * 0.3:  # <30% of original length
                    return ValidationResult.warning(
                        "Text replacement significantly decreases length",
                        {"old_length": old_len, "new_length": new_len},
                    )

        return ValidationResult.valid("Patch can be safely applied")

    @staticmethod
    def validate_patch_batch(patches: List[DocumentPatch], paragraph_texts: List[str]) -> Dict[str, Any]:
        """
        Validate a batch of patches for consistency and applicability.

        Consolidates batch validation logic from patch_docx_editor.py.

        Args:
            patches: List of patches to validate
            paragraph_texts: Document paragraph texts

        Returns:
            Dictionary with validation results
        """
        valid_patches = []
        invalid_patches = []
        warnings = []
        location_failures = 0
        content_failures = 0

        # Track patches by paragraph to detect conflicts
        patches_by_paragraph = {}

        for patch in patches:
            # Content validation
            content_result = PatchValidator.validate_patch_content(patch)
            if not content_result.is_valid:
                invalid_patches.append((patch, content_result.message))
                content_failures += 1
                continue

            if content_result.has_warning:
                warnings.append(f"Patch {patch.patch_id}: {content_result.message}")

            # Location validation
            location_result = PatchValidator.validate_patch_location(patch, paragraph_texts)
            if not location_result.found:
                invalid_patches.append((patch, location_result.message))
                location_failures += 1
                continue

            if location_result.is_ambiguous:
                warnings.append(f"Patch {patch.patch_id}: {location_result.message}")

            # Track for conflict detection
            para_idx = location_result.index
            if para_idx is not None:
                if para_idx not in patches_by_paragraph:
                    patches_by_paragraph[para_idx] = []
                patches_by_paragraph[para_idx].append(patch)

            valid_patches.append(patch)

        # Check for potential conflicts (multiple patches on same paragraph)
        conflicts = []
        for para_idx, para_patches in patches_by_paragraph.items():
            if len(para_patches) > 1:
                # Check if patches target overlapping text
                text_edit_patches = [p for p in para_patches if p.patch_type == "text_edit"]
                if len(text_edit_patches) > 1:
                    conflicts.append(
                        {
                            "paragraph": para_idx,
                            "patches": [p.patch_id for p in text_edit_patches],
                            "reason": "Multiple text edits on same paragraph",
                        }
                    )

        if conflicts:
            warnings.extend([f"Potential conflict in paragraph {c['paragraph']}: {c['reason']}" for c in conflicts])

        success_rate = len(valid_patches) / len(patches) * 100 if patches else 100

        return {
            "valid_patches": valid_patches,
            "invalid_patches": invalid_patches,
            "warnings": warnings,
            "conflicts": conflicts,
            "total_patches": len(patches),
            "valid_count": len(valid_patches),
            "invalid_count": len(invalid_patches),
            "success_rate": success_rate,
            "location_failures": location_failures,
            "content_failures": content_failures,
        }

    @staticmethod
    def get_validation_summary(validation_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of validation results.

        Args:
            validation_results: Results from validate_patch_batch

        Returns:
            Formatted summary string
        """
        results = validation_results

        lines = [
            "Patch Validation Summary",
            "=" * 30,
            f"Total patches: {results['total_patches']}",
            f"Valid patches: {results['valid_count']}",
            f"Invalid patches: {results['invalid_count']}",
            f"Success rate: {results['success_rate']:.1f}%",
            "",
        ]

        if results["invalid_count"] > 0:
            lines.extend(
                [
                    "Failure breakdown:",
                    f"  Content failures: {results['content_failures']}",
                    f"  Location failures: {results['location_failures']}",
                    "",
                ]
            )

        if results["warnings"]:
            lines.extend(
                [
                    f"Warnings ({len(results['warnings'])}):",
                ]
            )
            for warning in results["warnings"][:5]:  # Show first 5
                lines.append(f"  - {warning}")

            if len(results["warnings"]) > 5:
                lines.append(f"  ... and {len(results['warnings']) - 5} more")
            lines.append("")

        if results["conflicts"]:
            lines.extend(
                [
                    f"Conflicts detected ({len(results['conflicts'])}):",
                ]
            )
            for conflict in results["conflicts"]:
                lines.append(f"  - Paragraph {conflict['paragraph']}: {conflict['reason']}")
            lines.append("")

        return "\n".join(lines)


# Convenience functions


def validate_patch_content(patch: DocumentPatch) -> Tuple[bool, str]:
    """
    Legacy compatibility function for boolean validation.

    Args:
        patch: DocumentPatch to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    result = PatchValidator.validate_patch_content(patch)
    return result.is_valid, result.message


def validate_patch_location(patch: DocumentPatch, total_paragraphs: int) -> bool:
    """
    Legacy compatibility function for simple location validation.

    Args:
        patch: DocumentPatch to validate
        total_paragraphs: Total number of paragraphs in document

    Returns:
        True if location is valid
    """
    if patch.paragraph_index is not None:
        return 0 <= patch.paragraph_index < total_paragraphs

    # If no paragraph index, assume valid (will be checked during application)
    return bool(patch.text_pattern and patch.text_pattern.strip())
