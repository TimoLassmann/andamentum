#!/usr/bin/env python3
"""Apply a batch of patches to a Word document.

Every patch — a rewrite or a comment — is located the same way: once, through
:class:`~andamentum.whetstone.docx.textmap.TextMap`, against the one atom
stream. Rewrites are then applied by
:mod:`~andamentum.whetstone.docx.surgery`, which cuts only the runs their span
covers.

That uniformity is the point. This module previously used two matchers over two
representations — a raw case-insensitive substring test on a lossy paragraph
string for edits, a normalised index over runs for comments — so a target could
resolve for one and miss for the other. Measured on a real review: 27 of 27
targets resolved through the index, 11 through the substring test.

Order is deliberate. Edits are resolved and applied first, then the stream is
rebuilt and comments are resolved against it. So a comment sees the document as
the reader will, and a comment whose anchor a rewrite consumed fails here,
loudly, instead of being written into ``comments.xml`` and silently never
anchored.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, List, Tuple

from ..models import DocumentPatch, PatchApplicationResult
from .atoms import document_atoms
from .low_level import DocxEditor, PlacedComment
from .surgery import Edit, apply_edits, isolate_span
from .textmap import TextMap

logger = logging.getLogger("andamentum.whetstone")


@dataclass(frozen=True)
class _Resolution:
    patch: DocumentPatch
    author: str
    edit: Edit


def _target_of(patch: DocumentPatch) -> str:
    return (patch.original_text or patch.text_pattern or "").strip()


class PatchDocxEditor(DocxEditor):
    """A :class:`DocxEditor` that applies :class:`DocumentPatch` batches."""

    def __init__(
        self, input_path: str, author: str = "Document Editor", context_size: int = 1
    ):
        super().__init__(input_path, author, context_size)
        self.applied_patches: List[DocumentPatch] = []
        self.failed_patches: List[DocumentPatch] = []

    def apply_patches(
        self, patches, use_patch_authors: bool = False
    ) -> PatchApplicationResult:
        """Apply *patches*, rewrites before comments.

        Returns a :class:`PatchApplicationResult` in which every count is a
        count of what reached the document. A comment that resolved but could
        not be bracketed is a failure here, not a success with nothing to show
        for it.
        """
        started = time.time()

        pairs: List[Tuple[DocumentPatch, str]] = []
        for item in patches:
            if hasattr(item, "patch") and hasattr(item, "attribution"):
                author = (
                    item.attribution.author_name if use_patch_authors else self.author
                )
                pairs.append((item.patch, author))
            else:
                pairs.append((item, self.author))

        # Confidence orders the work; it no longer decides conflicts by accident
        # of application order, because conflicts are arbitrated explicitly.
        ordered = sorted(pairs, key=lambda pair: pair[0].confidence, reverse=True)
        edit_pairs = [(p, a) for p, a in ordered if p.patch_type == "text_edit"]
        comment_pairs = [(p, a) for p, a in ordered if p.patch_type == "comment"]

        failed: List[DocumentPatch] = []
        location_failures = 0
        validation_failures = 0

        # ── rewrites ────────────────────────────────────────────────────
        textmap = TextMap(document_atoms(self.paragraph_elements))
        resolutions: List[_Resolution] = []
        for patch, author in edit_pairs:
            reason = self._unusable(patch)
            if reason:
                failed.append(patch)
                validation_failures += 1
                logger.info("[patch] rewrite rejected (%s)", reason)
                continue
            span = textmap.find(_target_of(patch))
            if span is None:
                failed.append(patch)
                location_failures += 1
                score, nearest = textmap.closest(_target_of(patch))
                logger.info(
                    "[patch] rewrite not placed — target not found exactly once.\n"
                    "        target : %r\n"
                    "        nearest (similarity %.2f): %r",
                    _target_of(patch)[:120],
                    score,
                    nearest[:120],
                )
                continue
            resolutions.append(
                _Resolution(
                    patch=patch,
                    author=author,
                    edit=Edit(
                        span=span,
                        new_text=(patch.new_text or "").strip(),
                        author=author,
                    ),
                )
            )

        by_edit = {id(r.edit): r.patch for r in resolutions}
        report = apply_edits(textmap, [r.edit for r in resolutions])
        for loser, _winner in report.conflicted:
            patch = by_edit.get(id(loser))
            if patch is not None:
                failed.append(patch)
        applied_edits = report.applied
        conflict_failures = len(report.conflicted)

        # ── comments, against the document as it now reads ──────────────
        applied_comments = 0
        if comment_pairs:
            after = TextMap(document_atoms(self.paragraph_elements))
            located: List[Tuple[DocumentPatch, str, Any]] = []
            for patch, author in comment_pairs:
                if not (patch.comment_text or "").strip():
                    failed.append(patch)
                    validation_failures += 1
                    continue
                span = after.find(_target_of(patch))
                if span is None:
                    failed.append(patch)
                    location_failures += 1
                    score, nearest = after.closest(_target_of(patch))
                    logger.info(
                        "[patch] comment not anchored — target not found exactly once.\n"
                        "        target : %r\n"
                        "        nearest (similarity %.2f): %r",
                        _target_of(patch)[:120],
                        score,
                        nearest[:120],
                    )
                    continue
                located.append((patch, author, span))

            # Right to left: isolating a span splits its runs, and a split
            # keeps the original element as the left-hand half. Working
            # backwards therefore leaves every not-yet-placed comment's address
            # untouched. Three comments in one run used to be placed against a
            # stream that the first split had already invalidated.
            for patch, author, span in sorted(
                located, key=lambda item: item[2].start, reverse=True
            ):
                bounds = isolate_span(after.atoms, span)
                if bounds is None:
                    failed.append(patch)
                    location_failures += 1
                    continue
                body = patch.comment_text or ""
                if patch.explanation and patch.explanation != patch.comment_text:
                    body = f"{body}\n\nNote: {patch.explanation}"
                start_run, end_run = bounds
                self.placed_comments.append(
                    PlacedComment(
                        author=author,
                        text=body,
                        start_run=start_run,
                        end_run=end_run,
                    )
                )
                applied_comments += 1

        applied_total = applied_edits + applied_comments
        self.applied_patches.extend(
            p for p in (r.patch for r in resolutions) if p not in failed
        )
        self.failed_patches.extend(failed)

        logger.info(
            "[patch] %d/%d applied (%d rewrite(s), %d comment(s)); "
            "%d not placed, %d dropped as conflicting",
            applied_total,
            len(pairs),
            applied_edits,
            applied_comments,
            location_failures,
            conflict_failures,
        )
        if report.paragraphs_merged or report.paragraphs_created:
            logger.info(
                "[patch] %d paragraph(s) merged, %d created",
                report.paragraphs_merged,
                report.paragraphs_created,
            )

        return PatchApplicationResult(
            total_patches=len(pairs),
            applied_patches=applied_total,
            failed_patches=failed,
            processing_time=time.time() - started,
            applied_edits=applied_edits,
            applied_comments=applied_comments,
            location_failures=location_failures,
            validation_failures=validation_failures,
            conflict_failures=conflict_failures,
        )

    @staticmethod
    def _unusable(patch: DocumentPatch) -> str:
        if not (patch.new_text or "").strip():
            return "no new_text"
        if not _target_of(patch):
            return "no target text"
        return ""
