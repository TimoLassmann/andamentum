#!/usr/bin/env python3
"""
Enhanced DocxEditor with patch application capabilities.

This module extends the existing DocxEditor to support applying structured patches
while preserving all the sophisticated formatting and track changes functionality.
"""

import difflib
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .low_level import DocxEditor
from ..models import DocumentPatch, PatchApplicationResult

logger = logging.getLogger("andamentum.whetstone")


@dataclass(frozen=True)
class EditPlacement:
    """One text edit resolved to a concrete span, before anything mutates.

    Resolving every edit up front is what makes conflicts decidable. Applied
    one at a time against the running text, two edits on overlapping spans
    could BOTH succeed — the first rewrite can leave the second's pattern
    intact — and compose into a sentence neither patch proposed, with both
    reported as applied. Offsets index ``ParagraphData.modified`` as it stood
    before any edit in this batch was applied.
    """

    patch: DocumentPatch
    author: str
    paragraph_index: int
    start: int
    end: int
    target: str
    replacement: str


def find_occurrences(needle: str, haystack: str) -> list[int]:
    """Every start offset of *needle* in *haystack*, non-overlapping."""
    if not needle:
        return []
    found: list[int] = []
    cursor = haystack.find(needle)
    while cursor != -1:
        found.append(cursor)
        cursor = haystack.find(needle, cursor + len(needle))
    return found


def resolve_edit_conflicts(
    placements: List[EditPlacement],
) -> Tuple[List[EditPlacement], List[Tuple[EditPlacement, EditPlacement]]]:
    """Split *placements* into the ones to apply and the ones that clash.

    Greedy over the incoming order, which the caller sorts by confidence — the
    only priority signal a ``DocumentPatch`` carries. A placement is accepted
    when its span touches no already-accepted span in the same paragraph;
    otherwise it is returned paired with the placement that displaced it, so
    the caller can say which edit won rather than reporting a bare count.

    Pure, so the arbitration rule is testable without a Word document.
    """
    accepted: List[EditPlacement] = []
    conflicted: List[Tuple[EditPlacement, EditPlacement]] = []
    for candidate in placements:
        clash = next(
            (
                won
                for won in accepted
                if won.paragraph_index == candidate.paragraph_index
                and won.start < candidate.end
                and candidate.start < won.end
            ),
            None,
        )
        if clash is None:
            accepted.append(candidate)
        else:
            conflicted.append((candidate, clash))
    return accepted, conflicted


class PatchDocxEditor(DocxEditor):
    """
    Enhanced DocxEditor that can apply structured patches while preserving formatting.

    Extends the existing DocxEditor with patch application capabilities, leveraging
    all the existing sophisticated formatting preservation and track changes logic.
    """

    def __init__(
        self, input_path: str, author: str = "Document Editor", context_size: int = 1
    ):
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
                [[(i, p.anchor_text)] for i, p in enumerate(self.paragraphs)]
            )
        return self._doc_index

    def apply_patches(
        self, patches, use_patch_authors: bool = False
    ) -> PatchApplicationResult:
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
        failed_patches: List[DocumentPatch] = []
        applied_edits = 0
        applied_comments = 0
        location_failures = 0
        validation_failures = 0
        conflict_failures = 0

        # Extract patches and authors from the input
        patch_author_pairs = []
        for item in patches:
            if hasattr(item, "patch") and hasattr(item, "attribution"):
                # AttributedPatch object
                patch = item.patch
                author = (
                    item.attribution.author_name if use_patch_authors else self.author
                )
                patch_author_pairs.append((patch, author))
            else:
                # Regular DocumentPatch object
                # Note: In sequential mode, this fallback should rarely be used
                # as all patches should be AttributedPatch objects
                author = self.author
                patch_author_pairs.append((item, author))

        # Confidence still orders the work — it is the only priority signal a
        # DocumentPatch carries. What changed is that it no longer decides
        # conflicts by accident of application order: conflicts are resolved
        # explicitly below, before any text is touched.
        ordered = sorted(
            patch_author_pairs, key=lambda pair: pair[0].confidence, reverse=True
        )
        edit_pairs = [(p, a) for p, a in ordered if p.patch_type == "text_edit"]
        other_pairs = [(p, a) for p, a in ordered if p.patch_type != "text_edit"]

        # ── Phase 1: resolve every text edit against the untouched text ─────
        placements: List[EditPlacement] = []
        for patch, patch_author in edit_pairs:
            placement, reason = self._resolve_edit_placement(patch, patch_author)
            if placement is None:
                failed_patches.append(patch)
                self.failed_patches.append(patch)
                location_failures += 1
                logger.info(
                    "[patch] text edit not placed (%s): %r",
                    reason,
                    (patch.text_pattern or "")[:80],
                )
                continue
            placements.append(placement)

        # ── Phase 2: drop clashing edits before any of them is applied ──────
        accepted, conflicted = resolve_edit_conflicts(placements)
        for loser, winner in conflicted:
            failed_patches.append(loser.patch)
            self.failed_patches.append(loser.patch)
            conflict_failures += 1
            logger.warning(
                "[patch] text edit DROPPED — its span overlaps an edit already "
                "accepted in the same paragraph. Only one rewrite per span can "
                "be tracked.\n"
                "         dropped : %r (%s)\n"
                "         kept    : %r (%s)",
                loser.target[:80],
                loser.patch.explanation[:80],
                winner.target[:80],
                winner.patch.explanation[:80],
            )

        # ── Phase 3: apply survivors right to left within each paragraph ────
        # Right to left keeps every not-yet-applied offset valid: a splice only
        # moves text at or after its own start, and the accepted spans do not
        # overlap, so each remaining edit sits entirely to the left.
        for placement in sorted(
            accepted, key=lambda pl: (pl.paragraph_index, -pl.start)
        ):
            try:
                self._apply_placement(placement)
            except Exception as exc:
                failed_patches.append(placement.patch)
                self.failed_patches.append(placement.patch)
                validation_failures += 1
                logger.warning(
                    "[patch] text edit raised while applying (%s): %r",
                    exc,
                    placement.target[:80],
                )
                continue
            applied_count += 1
            applied_edits += 1
            self.applied_patches.append(placement.patch)

        # Comments resolve against the document as it reads AFTER the edits,
        # which is the same text the write pass indexes. Resolving them first
        # would validate a target here and then miss at write time, where the
        # rebuilt runs no longer say that — a comment that reports success and
        # is invisible in Word. A comment whose anchor an edit rewrote now
        # fails here instead, loudly and with a reason.
        if accepted:
            self._doc_index = None

        for patch, patch_author in other_pairs:
            try:
                success, failure_reason = self._apply_single_patch_with_author(
                    patch, patch_author
                )
            except Exception:
                failed_patches.append(patch)
                self.failed_patches.append(patch)
                validation_failures += 1
                continue

            if success:
                applied_count += 1
                self.applied_patches.append(patch)
                if patch.patch_type == "comment":
                    applied_comments += 1
            else:
                failed_patches.append(patch)
                self.failed_patches.append(patch)
                if "location" in failure_reason.lower():
                    location_failures += 1
                elif "validation" in failure_reason.lower():
                    validation_failures += 1
                else:
                    location_failures += 1

        processing_time = time.time() - start_time

        logger.info(
            "[anchor] %d/%d patch(es) anchored (%d comment(s), %d edit(s)); "
            "%d could not be anchored (%d of them dropped as conflicting)",
            applied_count,
            len(patches),
            applied_comments,
            applied_edits,
            len(failed_patches),
            conflict_failures,
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
            conflict_failures=conflict_failures,
        )

    def _resolve_edit_placement(
        self, patch: DocumentPatch, author: str
    ) -> Tuple[Optional[EditPlacement], str]:
        """Resolve one text edit to an unambiguous span, or explain why not.

        A pattern that occurs more than once in its paragraph is refused rather
        than guessed at. The producer knows which occurrence it read; this layer
        does not, and picking one silently rewrites a sentence the author never
        had reviewed. ``renderers/docx.py`` widens whetstone's own patterns with
        surrounding words so they arrive unique.
        """
        if not patch.new_text:
            return None, "text edit has no new_text"

        target = (patch.original_text or patch.text_pattern or "").strip()
        if not target:
            return None, "text edit has no target text"

        paragraph_index = self._locate_patch_target(patch)
        if paragraph_index is None:
            return None, "target text not found in document"
        if paragraph_index >= len(self.paragraphs):
            return None, "target paragraph index out of range"

        text = self.paragraphs[paragraph_index].modified
        offsets = find_occurrences(target, text)
        if not offsets:
            return None, "target text not found in the located paragraph"
        if len(offsets) > 1:
            return None, (
                f"target text occurs {len(offsets)} times in the paragraph and is "
                "ambiguous — extend the pattern with surrounding words so it "
                "identifies one span"
            )

        start = offsets[0]
        return (
            EditPlacement(
                patch=patch,
                author=author,
                paragraph_index=paragraph_index,
                start=start,
                end=start + len(target),
                target=target,
                replacement=patch.new_text.strip(),
            ),
            "",
        )

    def _apply_placement(self, placement: EditPlacement) -> None:
        """Splice one resolved edit into its paragraph.

        Splices the exact resolved span. The previous implementation called
        ``str.replace`` with no count, so a pattern appearing three times in a
        paragraph was rewritten three times — including in sentences no
        reviewer had looked at.
        """
        paragraph = self.paragraphs[placement.paragraph_index]
        before = paragraph.modified
        after = (
            before[: placement.start] + placement.replacement + before[placement.end :]
        )
        paragraph.modified = after
        paragraph.change_author = placement.author
        paragraph.attribution_tracker.track_change(before, after, placement.author)

    def _apply_single_patch(self, patch: DocumentPatch) -> Tuple[bool, str]:
        """
        Apply a single non-edit patch to the document.

        Text edits do not come through here. They are resolved to spans and
        arbitrated as a batch in :meth:`apply_patches` before any of them runs,
        because a conflict between two of them cannot be seen one at a time.

        Args:
            patch: DocumentPatch to apply

        Returns:
            Tuple of (success: bool, failure_reason: str)
        """
        # Comments are anchored at the document level (the target text may
        # span a heading→body paragraph boundary) — handled separately.
        if patch.patch_type == "comment":
            return self._apply_comment_patch(patch)

        if patch.patch_type == "text_edit":
            return False, "text edits are applied as a batch, not one at a time"

        return False, f"Unknown patch type: {patch.patch_type}"

    def _apply_single_patch_with_author(
        self, patch: DocumentPatch, author: str
    ) -> Tuple[bool, str]:
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
        if patch.paragraph_index is not None and 0 <= patch.paragraph_index < len(
            self.paragraphs
        ):
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
            similarity = self._calculate_text_similarity(
                pattern_lower, para.modified.lower()
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = i

        return best_match_idx

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Whole-string similarity score (0.0-1.0) used to find the best
        paragraph match when an exact lookup fails. One-line wrap over
        difflib — replaces the ~700-LOC TextProcessor module that was
        only ever used here."""
        return difflib.SequenceMatcher(None, text1, text2, autojunk=False).ratio()

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
        applied_edits = sum(
            1 for p in self.applied_patches if p.patch_type == "text_edit"
        )
        applied_comments = sum(
            1 for p in self.applied_patches if p.patch_type == "comment"
        )
        failed_edits = sum(
            1 for p in self.failed_patches if p.patch_type == "text_edit"
        )
        failed_comments = sum(
            1 for p in self.failed_patches if p.patch_type == "comment"
        )

        avg_confidence = 0.0
        if self.applied_patches:
            avg_confidence = sum(p.confidence for p in self.applied_patches) / len(
                self.applied_patches
            )

        return {
            "total_applied": len(self.applied_patches),
            "total_failed": len(self.failed_patches),
            "applied_edits": applied_edits,
            "applied_comments": applied_comments,
            "failed_edits": failed_edits,
            "failed_comments": failed_comments,
            "average_confidence": avg_confidence,
            "success_rate": len(self.applied_patches)
            / (len(self.applied_patches) + len(self.failed_patches))
            * 100
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
                lines.append(
                    f"  {i + 1}. {patch.patch_type}: {patch.explanation[:60]}..."
                )

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
