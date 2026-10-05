"""Works found in a library or a search, and the outcome of verifying one."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from andamentum.references import UpdateNotice

from ..kernel.csl import doi_of, title_of, year_of
from ..kernel.ops import CitationItem

WorkSource = Literal["library", "zotero", "crossref"]


class LiteratureError(RuntimeError):
    """A source that was asked for could not answer; never turned into "no results"."""


class Work(BaseModel):
    """One work as CSL JSON, with where it came from.

    ``uri`` is the Zotero item URI and is set only for a work read from Zotero.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    csl: dict[str, object]
    source: WorkSource
    uri: str | None = None

    @property
    def doi(self) -> str:
        return doi_of(self.csl)

    @property
    def title(self) -> str:
        return title_of(self.csl)

    @property
    def year(self) -> int | None:
        return year_of(self.csl)

    @property
    def authors(self) -> list[str]:
        """Family names (or literal names), first author first."""
        names: list[str] = []
        entries = self.csl.get("author")
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, dict):
                name = entry.get("family") or entry.get("literal") or ""
                if name:
                    names.append(str(name))
        return names

    def citation_item(
        self,
        *,
        locator: str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
    ) -> CitationItem:
        return CitationItem(
            csl=self.csl, locator=locator, prefix=prefix, suffix=suffix, uri=self.uri
        )


class VerifiedWork(BaseModel):
    """A work that exists, whose details agree, and that is not retracted.

    ``notices`` holds the non-retraction notices (a correction, an expression
    of concern) Crossref records; the work may be cited, the notice should be
    read first.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    work: Work
    notices: list[UpdateNotice] = Field(default_factory=list)


UnverifiedKind = Literal["not_found", "details_differ", "retracted", "unchecked"]


class Unverified(BaseModel):
    """A work that must not be cited; ``reason`` is the question for the author.

    ``unchecked`` means the lookup could not be done (service down, nothing to
    look up), which says nothing about the work.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    work: Work
    kind: UnverifiedKind
    reason: str
    notices: list[UpdateNotice] = Field(default_factory=list)
