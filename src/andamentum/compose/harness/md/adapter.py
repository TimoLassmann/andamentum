"""MarkdownAdapter: a ``compose.bundle`` folder as a DocumentView, ops as section edits.

Paragraph ids: a section's heading is ``<section id>:h`` (text = the section
title, heading level 1); its body blocks, split at blank lines, are
``<section id>:<n>`` counting from 0. Text is the raw markdown of the block.
Ids are stable for one bundle across re-opens; an edit that inserts a block
renumbers the blocks after it in the output, as the ids follow position.

Supported ops: Insert, Replace, Delete, InsertParagraph. Markdown has no
comments or content controls, so CommentOp, Reply, Bind, Move and Unbind raise
NotImplementedError. Each op reads the text as left by the ops before it, so
two ops on one paragraph must use offsets that are still valid; a stale
``expect`` refuses the op. Limits: a block never contains a blank line (a
fenced code block with one is split), and a section whose blocks are edited
is rewritten with exactly one blank line between blocks.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..kernel.adapter import ApplyResult, UntouchedReport
from ..kernel.ops import (
    Bind,
    CommentOp,
    Delete,
    Insert,
    InsertParagraph,
    Move,
    Replace,
    Reply,
    Transaction,
    Unbind,
)
from ...bundle import Bundle
from ..kernel.view import DocumentView, Paragraph

_BLANK_LINE = re.compile(r"\n[ \t]*\n")
_UNSUPPORTED = (CommentOp, Reply, Bind, Move, Unbind)


def _blocks(body: str) -> list[str]:
    return [b.strip("\n") for b in _BLANK_LINE.split(body) if b.strip()]


def _paragraphs_of(bundle: Bundle) -> list[Paragraph]:
    paragraphs: list[Paragraph] = []
    for section in bundle.sections():
        paragraphs.append(
            Paragraph(
                id=f"{section.id}:h",
                index=len(paragraphs),
                style="Heading 1",
                heading_level=1,
                text=section.title,
            )
        )
        for n, block in enumerate(_blocks(section.body)):
            paragraphs.append(
                Paragraph(
                    id=f"{section.id}:{n}",
                    index=len(paragraphs),
                    style="Normal",
                    heading_level=None,
                    text=block,
                )
            )
    return paragraphs


@dataclass
class _Entry:
    """One paragraph during a transaction: original id (None if new) and current text."""

    section: str
    source_id: str | None
    heading: bool
    text: str
    targeted: bool = False


class MarkdownAdapter:
    """DocumentAdapter over a bundle folder."""

    def open(self, path: Path) -> DocumentView:
        bundle = Bundle.open(path)
        project_id = bundle.metadata.get("project_id")
        return DocumentView(
            path=Path(path),
            project_id=str(project_id) if project_id is not None else None,
            paragraphs=_paragraphs_of(bundle),
            anchors=[],
            fields=[],
            comments=[],
        )

    def apply(
        self, view: DocumentView, txn: Transaction, *, output: Path
    ) -> ApplyResult:
        """Copy the bundle to ``output`` and apply the transaction there.

        ``output`` must not exist or be an empty directory. The input bundle is
        never modified. Raises if the bundle no longer matches ``view``, if an
        op type is unsupported, or if the written result does not read back as
        intended.
        """
        for op in txn.ops:
            if isinstance(op, _UNSUPPORTED):
                raise NotImplementedError(
                    f"the markdown adapter cannot apply a {op.op!r} op"
                )
        source = Bundle.open(view.path)
        if _paragraphs_of(source) != view.paragraphs:
            raise ValueError(
                f"{view.path} changed since the view was opened; open it again"
            )
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError(f"{output} already exists and is not empty")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(view.path, output, dirs_exist_ok=True)

        entries = [
            _Entry(
                section=p.id.split(":")[0],
                source_id=p.id,
                heading=p.heading_level is not None,
                text=p.text,
            )
            for p in view.paragraphs
        ]
        refused: list[str] = []
        applied = 0
        for op in txn.ops:
            reason = self._apply_op(entries, op)
            if reason is None:
                applied += 1
            else:
                refused.append(f"{op.op}: {reason}")

        self._write(output, entries)
        return ApplyResult(
            output=output,
            applied=applied,
            refused=refused,
            untouched=self._verify(output, entries),
        )

    # -- ops on the in-memory paragraph list ------------------------------

    @staticmethod
    def _find(entries: list[_Entry], paragraph_id: str) -> int | None:
        for i, e in enumerate(entries):
            if e.source_id == paragraph_id:
                return i
        return None

    def _apply_op(self, entries: list[_Entry], op: object) -> str | None:
        """Apply one op; return a refusal reason, or None when applied."""
        if isinstance(op, (Insert, Replace, Delete)):
            i = self._find(entries, op.paragraph_id)
            if i is None:
                return f"no paragraph {op.paragraph_id!r}"
            entry = entries[i]
            if isinstance(op, Insert):
                if op.at > len(entry.text):
                    return f"position {op.at} is beyond the paragraph"
                new = entry.text[: op.at] + op.text + entry.text[op.at :]
            else:
                if op.end > len(entry.text):
                    return f"range ends at {op.end}, paragraph has {len(entry.text)}"
                if entry.text[op.start : op.end] != op.expect:
                    return (
                        f"text at [{op.start}, {op.end}) is "
                        f"{entry.text[op.start : op.end]!r}, expected {op.expect!r}"
                    )
                replacement = op.text if isinstance(op, Replace) else ""
                new = entry.text[: op.start] + replacement + entry.text[op.end :]
            reason = self._check_text(new, heading=entry.heading)
            if reason is not None:
                return reason
            entry.text = new
            entry.targeted = True
            return None
        if isinstance(op, InsertParagraph):
            i = self._find(entries, op.after_paragraph_id)
            if i is None:
                return f"no paragraph {op.after_paragraph_id!r}"
            if op.style not in (None, "Normal"):
                return f"style {op.style!r} is not supported in markdown"
            reason = self._check_text(op.text, heading=False)
            if reason is not None:
                return reason
            entries.insert(
                i + 1,
                _Entry(
                    section=entries[i].section,
                    source_id=None,
                    heading=False,
                    text=op.text,
                    targeted=True,
                ),
            )
            return None
        raise NotImplementedError(f"unsupported op {op!r}")  # pragma: no cover

    @staticmethod
    def _check_text(text: str, *, heading: bool) -> str | None:
        if not text.strip():
            return "the paragraph would be empty"
        if heading and "\n" in text:
            return "a heading cannot contain a line break"
        if not heading and _BLANK_LINE.search(text):
            return "a paragraph cannot contain a blank line"
        return None

    # -- writing and verifying -------------------------------------------

    @staticmethod
    def _write(output: Path, entries: list[_Entry]) -> None:
        bundle = Bundle.open(output)
        for section in bundle.sections():
            mine = [e for e in entries if e.section == section.id]
            title = next(e.text for e in mine if e.heading)
            body = "\n\n".join(e.text for e in mine if not e.heading)
            if not any(e.targeted for e in mine):
                continue
            current = bundle.section(section.id)
            if title != current.title:
                current = bundle.rename_section(
                    section.id, title=title, expected_revision=current.revision
                )
            if body != current.body:
                bundle.write_section(
                    section.id, body=body, expected_revision=current.revision
                )

    @staticmethod
    def _verify(output: Path, entries: list[_Entry]) -> UntouchedReport:
        """Re-open the output: targeted paragraphs read as intended, others unchanged."""
        reread = _paragraphs_of(Bundle.open(output))
        if len(reread) != len(entries):
            raise RuntimeError(
                f"output has {len(reread)} paragraphs, expected {len(entries)}"
            )
        changed_outside: list[str] = []
        for entry, got in zip(entries, reread):
            if entry.text == got.text:
                continue
            if entry.targeted:
                raise RuntimeError(
                    f"{got.id} reads {got.text!r} after the edit, expected {entry.text!r}"
                )
            changed_outside.append(entry.source_id or got.id)
        return UntouchedReport(ok=not changed_outside, changed_outside=changed_outside)


__all__ = ["MarkdownAdapter"]
