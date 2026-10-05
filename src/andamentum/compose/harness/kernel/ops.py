"""The operation set. Ranges are ``(paragraph_id, start, end)`` on the view text."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

BindKind = Literal["claim", "number", "figure", "span"]


class _Op(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _Ranged(_Op):
    paragraph_id: str
    start: int
    end: int

    @model_validator(mode="after")
    def _check_range(self) -> _Ranged:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid range [{self.start}, {self.end})")
        return self


class Insert(_Op):
    op: Literal["insert"] = "insert"
    paragraph_id: str
    at: int
    text: str

    @model_validator(mode="after")
    def _check(self) -> Insert:
        if self.at < 0:
            raise ValueError(f"insert position {self.at} is negative")
        if not self.text:
            raise ValueError("insert text is empty")
        return self


class Replace(_Ranged):
    op: Literal["replace"] = "replace"
    text: str
    expect: str  # current text of the range; a mismatch refuses the op

    @model_validator(mode="after")
    def _check_expect(self) -> Replace:
        if len(self.expect) != self.end - self.start:
            raise ValueError("expect length does not match the range")
        return self


class Delete(_Ranged):
    op: Literal["delete"] = "delete"
    expect: str

    @model_validator(mode="after")
    def _check_expect(self) -> Delete:
        if self.end == self.start:
            raise ValueError("delete range is empty")
        if len(self.expect) != self.end - self.start:
            raise ValueError("expect length does not match the range")
        return self


class InsertParagraph(_Op):
    op: Literal["insert_paragraph"] = "insert_paragraph"
    after_paragraph_id: str
    text: str
    style: str | None = None


class CommentOp(_Ranged):
    op: Literal["comment"] = "comment"
    text: str

    @model_validator(mode="after")
    def _check_text(self) -> CommentOp:
        if not self.text.strip():
            raise ValueError("comment text is empty")
        return self


class Reply(_Op):
    op: Literal["reply"] = "reply"
    parent_comment_id: str
    text: str

    @model_validator(mode="after")
    def _check_text(self) -> Reply:
        if not self.text.strip():
            raise ValueError("reply text is empty")
        return self


class Bind(_Ranged):
    op: Literal["bind"] = "bind"
    anchor_id: str
    kind: BindKind

    @model_validator(mode="after")
    def _check_bind(self) -> Bind:
        if self.end == self.start:
            raise ValueError("bind range is empty")
        return self


class CitationItem(BaseModel):
    """One work in a citation: its CSL JSON and how this citation points at it.

    ``uri`` is set only when the work came from Zotero (its item URI); the
    writer puts it in the field's ``uris`` and never invents one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    csl: dict[str, object]
    locator: str | None = None
    prefix: str | None = None
    suffix: str | None = None
    uri: str | None = None


class InsertCitation(_Op):
    """A citation of one or more works at ``at`` in the paragraph text.

    The docx writer writes a Zotero field when the document already uses
    Zotero fields (or an item has a ``uri``) and plain formatted text
    otherwise; either way as a tracked insertion in the document's own style.
    """

    op: Literal["insert_citation"] = "insert_citation"
    paragraph_id: str
    at: int
    items: list[CitationItem]

    @model_validator(mode="after")
    def _check(self) -> InsertCitation:
        if self.at < 0:
            raise ValueError(f"citation position {self.at} is negative")
        if not self.items:
            raise ValueError("a citation needs at least one item")
        return self


class UpdateBibliography(_Op):
    """Document-level: list every cited work the reference list is missing.

    Works are those of the ``InsertCitation`` ops in the same transaction and
    of the Zotero fields already in the document. Entries are added as tracked
    changes; existing entries are never rewritten.
    """

    op: Literal["update_bibliography"] = "update_bibliography"


class Move(_Op):
    """Declared in the kernel; the docx writer raises NotImplementedError for it."""

    op: Literal["move"] = "move"
    paragraph_id: str
    after_paragraph_id: str


class Unbind(_Op):
    """Declared in the kernel; the docx writer raises NotImplementedError for it."""

    op: Literal["unbind"] = "unbind"
    anchor_id: str


Operation = Annotated[
    Insert
    | Replace
    | Delete
    | InsertParagraph
    | CommentOp
    | Reply
    | Bind
    | InsertCitation
    | UpdateBibliography
    | Move
    | Unbind,
    Field(discriminator="op"),
]


class Transaction(BaseModel):
    """An ordered batch of operations applied together, with who and why."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ops: list[Operation]
    author: str
    reason: str
    source: str  # check id or verb name
