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
