"""Literature for a draft: verified works become sources whose facts can be cited.

``draft`` does not search or verify anything itself. A caller passes a
:class:`LiteratureSearch`, an async callable that answers a query with works
that already passed verification (DOI exists, metadata match, not retracted)
and descriptions of those that did not. Only verified works are ever cited; the
unverified ones come back as author questions.

A verified work with an abstract becomes a compose ``Source`` whose text is the
title and abstract, so the existing evidence worker extracts quote-checked
facts from it and a ``[[fN]]`` marker on such a fact is a citation of that
work. A work with no abstract has nothing a fact could be quoted from; it is
not used and the report says so.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from andamentum.compose.evidence import stored_source_path
from andamentum.compose.genres import Genre
from andamentum.compose.schemas import Brief, Source

from ..kernel.csl import doi_of, title_of, work_key

if TYPE_CHECKING:
    from andamentum.deep_research import ResearchResult

QUERIES_PER_DRAFT = 6  # one per key message, never more
WORKS_PER_QUERY = 4
MAX_WORKS = 8  # one fact-extraction model call each


class CitableWork(BaseModel):
    """A verified work: its CSL JSON, its abstract and, from Zotero, its item URI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    csl: dict[str, object]
    abstract: str
    uri: str | None = None

    @property
    def key(self) -> str:
        return work_key(self.csl)

    @property
    def title(self) -> str:
        return title_of(self.csl)


class LiteratureFinding(BaseModel):
    """What one query found: verified works and what could not be verified."""

    model_config = ConfigDict(extra="forbid")

    works: list[CitableWork] = Field(default_factory=list)
    unverified: list[str] = Field(default_factory=list)  # one line each: what and why


class LiteratureSearch(Protocol):
    def __call__(self, query: str, *, limit: int) -> Awaitable[LiteratureFinding]: ...


@dataclass
class ResearchFinding:
    """What a research run gives the draft: the result and which of its works
    may be cited.

    ``works`` maps a research-store work id to its verified :class:`CitableWork`;
    a work missing from it is never cited. ``unverified`` is one author
    question per work that failed verification, ``notes`` the works left out
    for another reason.
    """

    result: ResearchResult
    works: dict[str, CitableWork] = field(default_factory=dict)
    unverified: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class ResearchRun(Protocol):
    """Research the task in the genre's mode over the store at *database*, and
    verify the works with findings. Like :class:`LiteratureSearch`, the draft
    holds no search or verification of its own."""

    def __call__(
        self, task: str, *, genre: Genre, database: Path
    ) -> Awaitable[ResearchFinding]: ...


class WorkSet(BaseModel):
    """The works a draft may cite, and the notes about the rest."""

    model_config = ConfigDict(extra="forbid")

    works: list[CitableWork] = Field(default_factory=list)
    unverified: list[str] = Field(default_factory=list)
    unused: list[str] = Field(default_factory=list)  # verified but no abstract


async def gather_works(
    brief: Brief,
    *,
    search: LiteratureSearch,
    progress: Callable[[str], None],
) -> WorkSet:
    """Run one query per key message (sequentially) and pool the results.

    Duplicates (same work key) are kept once; at most ``MAX_WORKS`` works with an
    abstract are kept, in the order they were found.
    """
    found: dict[str, CitableWork] = {}
    unverified: list[str] = []
    unused: list[str] = []
    for query in brief.key_messages[:QUERIES_PER_DRAFT]:
        progress(f"Literature: {query[:60]}")
        finding = await search(query, limit=WORKS_PER_QUERY)
        for line in finding.unverified:
            if line not in unverified:
                unverified.append(line)
        for work in finding.works:
            if work.key in found:
                continue
            if not work.abstract.strip():
                note = f"{work.title or doi_of(work.csl) or work.key}: no abstract to quote"
                if note not in unused:
                    unused.append(note)
                continue
            found[work.key] = work
    return WorkSet(
        works=list(found.values())[:MAX_WORKS], unverified=unverified, unused=unused
    )


def work_source(
    work: CitableWork, *, number: int, bundle_dir: str | Path
) -> tuple[Source, str]:
    """The compose ``Source`` for *work* (id ``src<number>``) and its text.

    The text is written to the place the evidence worker reads sources from.
    """
    text = f"# {work.title}\n\n{work.abstract.strip()}\n"
    doi = doi_of(work.csl)
    source = Source(
        id=f"src{number}",
        path=f"https://doi.org/{doi}" if doi else work.key,
        title=work.title,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        chars=len(text),
        kind="url",
        label=work.title,
    )
    stored = stored_source_path(bundle_dir, source)
    stored.parent.mkdir(parents=True, exist_ok=True)
    stored.write_text(text, encoding="utf-8")
    return source, text
