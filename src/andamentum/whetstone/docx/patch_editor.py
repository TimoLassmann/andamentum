#!/usr/bin/env python3
"""
Enhanced DocxEditor with patch application capabilities.

This module extends the existing DocxEditor to support applying structured patches
while preserving all the sophisticated formatting and track changes functionality.
"""

import logging
import time
from typing import List, Optional, Tuple

from .low_level import DocxEditor
from ..models import DocumentPatch, PatchApplicationResult
from .text_processor import TextProcessor
from .validator import PatchValidator

logger = logging.getLogger("andamentum.whetstone")


class PatchDocxEditor(DocxEditor):
    """
    Enhanced DocxEditor that can apply structured patches while preserving formatting.

    Extends the existing DocxEditor with patch application capabilities, leveraging
    all the existing sophisticated formatting preservation and track changes logic.
    """

    def __init__(self, input_path: str, author: str = "Document Editor", context_size: int = 1):
        """
        Initialize the enhanced DocxEditor.

        Args:
            input_path: Path to input DOCX file
            author: Author name for track changes
            context_size: Context size for paragraph operations
        """
        super().__init__(input_path, author, context_size)
        self.applied_patches: List[DocumentPatch] = []
        self.failed_patches: List[DocumentPatch] = []
        self._doc_index = None  # lazy DocIndex over the baseline paragraphs

    def _get_doc_index(self):
        """Lazily build a document-level normalised anchor index over the
        baseline paragraphs (one segment per paragraph). Used to resolve a
        comment's target text — including targets that span a heading→body
        paragraph boundary, which paragraph-substring matching can't find."""
        if self._doc_index is None:
            from .anchor import DocIndex

            self._doc_index = DocIndex(
                [[(i, p.modified)] for i, p in enumerate(self.paragraphs)]
            )
        return self._doc_index

    def apply_patches(self, patches, use_patch_authors: bool = False) -> PatchApplicationResult:
        """
        Apply a list of patches to the document.

        Args:
            patches: List of DocumentPatch or AttributedPatch objects to apply
            use_patch_authors: If True, use individual patch authors for track changes

        Returns:
            PatchApplicationResult with detailed results
        """
        start_time = time.time()

        applied_count = 0
        failed_patches = []
        applied_edits = 0
        applied_comments = 0
        location_failures = 0
        validation_failures = 0

        # Extract patches and authors from the input
        patch_author_pairs = []
        for item in patches:
            if hasattr(item, "patch") and hasattr(item, "attribution"):
                # AttributedPatch object
                patch = item.patch
                author = item.attribution.author_name if use_patch_authors else self.author
                patch_author_pairs.append((patch, author))
            else:
                # Regular DocumentPatch object
                # Note: In sequential mode, this fallback should rarely be used
                # as all patches should be AttributedPatch objects
                author = self.author
                patch_author_pairs.append((item, author))

        # Sort patches by confidence (apply high-confidence patches first)
        sorted_patch_pairs = sorted(patch_author_pairs, key=lambda x: x[0].confidence, reverse=True)

        for patch, patch_author in sorted_patch_pairs:
            try:
                success, failure_reason = self._apply_single_patch_with_author(patch, patch_author)

                if success:
                    applied_count += 1
                    self.applied_patches.append(patch)

                    if patch.patch_type == "text_edit":
                        applied_edits += 1
                    elif patch.patch_type == "comment":
                        applied_comments += 1
                else:
                    failed_patches.append(patch)
                    self.failed_patches.append(patch)

                    # Categorize failure reasons
                    if "location" in failure_reason.lower():
                        location_failures += 1
                    elif "validation" in failure_reason.lower():
                        validation_failures += 1

            except Exception:
                failed_patches.append(patch)
                self.failed_patches.append(patch)
                validation_failures += 1

        processing_time = time.time() - start_time

        logger.info(
            "[anchor] %d/%d patch(es) anchored (%d comment(s), %d edit(s)); "
            "%d could not be anchored",
            applied_count,
            len(patches),
            applied_comments,
            applied_edits,
            len(failed_patches),
        )

        return PatchApplicationResult(
            total_patches=len(patches),
            applied_patches=applied_count,
            failed_patches=failed_patches,
            processing_time=processing_time,
            applied_edits=applied_edits,
            applied_comments=applied_comments,
            location_failures=location_failures,
            validation_failures=validation_failures,
        )

    def _apply_single_patch(self, patch: DocumentPatch) -> Tuple[bool, str]:
        """
        Apply a single patch to the document.

        Args:
            patch: DocumentPatch to apply

        Returns:
            Tuple of (success: bool, failure_reason: str)
        """
        # Comments are anchored at the document level (the target text may
        # span a heading→body paragraph boundary) — handled separately.
        if patch.patch_type == "comment":
            return self._apply_comment_patch(patch)

        # Locate the target paragraph
        paragraph_index = self._locate_patch_target(patch)

        if paragraph_index is None:
            return False, "Could not locate target text in document"

        # Validate that the paragraph still matches expectations
        if not self._validate_patch_location(patch, paragraph_index):
            return False, "Paragraph content has changed since patch generation"

        try:
            if patch.patch_type == "text_edit":
                return self._apply_text_edit_patch(patch, paragraph_index)
            else:
                return False, f"Unknown patch type: {patch.patch_type}"

        except Exception as e:
            return False, f"Error applying patch: {str(e)}"

    def _apply_single_patch_with_author(self, patch: DocumentPatch, author: str) -> Tuple[bool, str]:
        """
        Apply a single patch to the document with a specific author.

        Args:
            patch: DocumentPatch to apply
            author: Author name for track changes

        Returns:
            Tuple of (success: bool, failure_reason: str)
        """
        # Save current author
        original_author = self.author

        try:
            # Temporarily set the author for this patch
            self.author = author

            # Apply the patch using the existing logic
            return self._apply_single_patch(patch)

        finally:
            # Restore original author
            self.author = original_author

    def _locate_patch_target(self, patch: DocumentPatch) -> Optional[int]:
        """
        Find the paragraph index for a patch.

        Uses paragraph_index if available, otherwise fuzzy text matching.

        Args:
            patch: DocumentPatch to locate

        Returns:
            Paragraph index if found, None otherwise
        """
        # If paragraph index is specified and valid, use it first
        if patch.paragraph_index is not None and 0 <= patch.paragraph_index < len(self.paragraphs):
            return patch.paragraph_index

        # Fall back to text pattern matching
        if patch.text_pattern:
            return self._find_text_pattern(patch.text_pattern)

        return None

    def _find_text_pattern(self, pattern: str) -> Optional[int]:
        """
        Find paragraph containing the specified text pattern.

        Args:
            pattern: Text pattern to search for

        Returns:
            Paragraph index if found, None otherwise
        """
        pattern_lower = pattern.lower().strip()

        # First try exact matching
        for i, para in enumerate(self.paragraphs):
            if pattern_lower in para.modified.lower():
                return i

        # If exact match fails, try fuzzy matching
        best_match_idx = None
        best_similarity = 0.6  # Minimum similarity threshold

        for i, para in enumerate(self.paragraphs):
            similarity = self._calculate_text_similarity(pattern_lower, para.modified.lower())
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = i

        return best_match_idx

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        result = TextProcessor.calculate_similarity(text1, text2)
        return result.similarity

    def _validate_patch_location(self, patch: DocumentPatch, paragraph_index: int) -> bool:
        """
        Validate that a patch can still be applied to the specified paragraph.

        Args:
            patch: DocumentPatch to validate
            paragraph_index: Target paragraph index

        Returns:
            True if patch can be applied, False otherwise
        """
        if paragraph_index >= len(self.paragraphs):
            return False

        paragraph = self.paragraphs[paragraph_index]

        # For text edits, check that original text is still present
        if patch.patch_type == "text_edit" and patch.original_text:
            if patch.original_text.strip() not in paragraph.modified:
                return False

        # For any patch with text_pattern, verify it's still there
        if patch.text_pattern:
            if patch.text_pattern.lower().strip() not in paragraph.modified.lower():
                return False

        return True

    def _apply_text_edit_patch(self, patch: DocumentPatch, paragraph_index: int) -> Tuple[bool, str]:
        """
        Apply a text edit patch to a specific paragraph.

        Args:
            patch: Text edit patch to apply
            paragraph_index: Target paragraph index

        Returns:
            Tuple of (success: bool, failure_reason: str)
        """
        if not patch.new_text:
            return False, "Text edit patch missing new_text"

        paragraph = self.paragraphs[paragraph_index]
        current_text = paragraph.modified

        # If we have original_text, do precise replacement
        if patch.original_text and patch.original_text.strip() in current_text:
            new_text = current_text.replace(patch.original_text.strip(), patch.new_text.strip())
        elif patch.text_pattern and patch.text_pattern.strip() in current_text:
            # Use text_pattern for replacement
            new_text = current_text.replace(patch.text_pattern.strip(), patch.new_text.strip())
        else:
            return False, "Could not find target text for replacement"

        # Apply the change using existing DocxEditor logic
        original_text = paragraph.modified
        paragraph.modified = new_text

        # Track who made this change (for track changes attribution)
        paragraph.change_author = self.author  # Legacy single author - kept for compatibility

        # Track the specific token-level changes made by this agent
        paragraph.attribution_tracker.track_change(original_text, new_text, self.author)

        # Note: patch.explanation is preserved for console display and internal validation
        # but no longer automatically converted to Word comments for text edits

        return True, ""

    def _apply_comment_patch(self, patch: DocumentPatch) -> Tuple[bool, str]:
        """Resolve a comment's target text at the document level and attach it.

        Uses the normalised :class:`DocIndex` so a target that spans a
        heading→body paragraph boundary still resolves. The comment is
        attached to its START paragraph (so that paragraph is rebuilt into
        precise runs); the actual range is placed at write time against
        the resolved text span.

        Fail-loud: when the target cannot be located, the comment is NOT
        placed — it's reported as a failure, and the closest matching
        region is logged so the mismatch is visible.
        """
        if not patch.comment_text:
            return False, "Comment patch missing comment_text"
        target = (patch.text_pattern or "").strip()
        if not target:
            return False, "Comment patch missing text_pattern"

        index = self._get_doc_index()
        span = index.find(target)
        if span is None:
            score, snippet = index.closest(target)
            logger.info(
                "[anchor] comment NOT anchored — target text not found in document.\n"
                "         target  : %r\n"
                "         closest (similarity=%.2f): %r",
                target[:160],
                score,
                snippet[:160],
            )
            return False, "Could not anchor comment: target text not found in document"

        # span.start_key is the paragraph index where the target begins.
        paragraph = self.paragraphs[span.start_key]

        # Format the comment. (When explanation == comment_text — as the
        # finding→patch adapter now arranges — no "Note:" is appended.)
        comment_content = patch.comment_text
        if patch.explanation and patch.explanation != patch.comment_text:
            comment_content = f"{patch.comment_text}\n\nNote: {patch.explanation}"

        # Carry the target text so the write pass can place the range
        # precisely (3-tuple: author, content, target).
        paragraph.comments.append((self.author, comment_content, target))
        paragraph.change_author = self.author
        return True, ""

    def get_patch_summary(self) -> dict:
        """
        Get summary of patches applied to this document.

        Returns:
            Dictionary with patch application summary
        """
        applied_edits = sum(1 for p in self.applied_patches if p.patch_type == "text_edit")
        applied_comments = sum(1 for p in self.applied_patches if p.patch_type == "comment")
        failed_edits = sum(1 for p in self.failed_patches if p.patch_type == "text_edit")
        failed_comments = sum(1 for p in self.failed_patches if p.patch_type == "comment")

        avg_confidence = 0.0
        if self.applied_patches:
            avg_confidence = sum(p.confidence for p in self.applied_patches) / len(self.applied_patches)

        return {
            "total_applied": len(self.applied_patches),
            "total_failed": len(self.failed_patches),
            "applied_edits": applied_edits,
            "applied_comments": applied_comments,
            "failed_edits": failed_edits,
            "failed_comments": failed_comments,
            "average_confidence": avg_confidence,
            "success_rate": len(self.applied_patches) / (len(self.applied_patches) + len(self.failed_patches)) * 100
            if (self.applied_patches or self.failed_patches)
            else 0,
        }

    def generate_patch_report(self) -> str:
        """
        Generate a human-readable report of patch application.

        Returns:
            Formatted report string
        """
        summary = self.get_patch_summary()

        lines = [
            "Patch Application Report",
            "=" * 30,
            f"Successfully applied: {summary['total_applied']} patches",
            f"Failed to apply: {summary['total_failed']} patches",
            f"Success rate: {summary['success_rate']:.1f}%",
            "",
            "Applied patches:",
            f"  Text edits: {summary['applied_edits']}",
            f"  Comments: {summary['applied_comments']}",
            "",
            "Failed patches:",
            f"  Text edits: {summary['failed_edits']}",
            f"  Comments: {summary['failed_comments']}",
            "",
            f"Average confidence: {summary['average_confidence']:.2f}",
        ]

        # Add details about failed patches if any
        if self.failed_patches:
            lines.extend(
                [
                    "",
                    "Failed patch details:",
                ]
            )
            for i, patch in enumerate(self.failed_patches[:5]):  # Show first 5
                lines.append(f"  {i + 1}. {patch.patch_type}: {patch.explanation[:60]}...")

            if len(self.failed_patches) > 5:
                lines.append(f"  ... and {len(self.failed_patches) - 5} more")

        return "\n".join(lines)

    def preview_patches(self, patches: List[DocumentPatch]) -> str:
        """
        Generate a preview of patches before applying them.

        Args:
            patches: List of patches to preview

        Returns:
            Human-readable preview of patches
        """
        lines = [
            f"Patch Preview ({len(patches)} patches)",
            "=" * 30,
        ]

        for i, patch in enumerate(patches, 1):
            lines.append(f"{i}. {patch.patch_type.upper()}")
            if patch.text_pattern:
                lines.append(f"   Pattern: {patch.text_pattern[:60]}...")

            if patch.patch_type == "text_edit" and patch.new_text:
                lines.append(f"   New text: {patch.new_text[:60]}...")
            elif patch.patch_type == "comment" and patch.comment_text:
                lines.append(f"   Comment: {patch.comment_text[:60]}...")

            lines.append(f"   Confidence: {patch.confidence:.2f}")
            lines.append(f"   Explanation: {patch.explanation}")
            lines.append("")

        return "\n".join(lines)

    def validate_patches_before_application(self, patches: List[DocumentPatch]) -> dict:
        """
        Validate patches before attempting to apply them.

        Args:
            patches: List of patches to validate

        Returns:
            Dictionary with validation results
        """
        # Use centralized validation
        paragraph_texts = [para.modified for para in self.paragraphs]
        return PatchValidator.validate_patch_batch(patches, paragraph_texts)

    def get_document_text(self) -> str:
        """
        Get the complete current document text as a single string.

        This method concatenates all paragraph text for document state tracking
        in the context refresh system.

        Returns:
            Complete document text with paragraphs separated by newlines
        """
        if not self.paragraphs:
            return ""

        # Use modified text (current state) rather than original
        paragraph_texts = []
        for paragraph in self.paragraphs:
            # Use modified text which reflects any changes made
            text = paragraph.modified.strip()
            if text:  # Only include non-empty paragraphs
                paragraph_texts.append(text)

        return "\n\n".join(paragraph_texts)
