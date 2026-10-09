"""Works a draft may cite: the verified work type and the source made from one.

``draft`` does not search or verify anything itself. The cite loop
(``cite.py``) and the research run (``ResearchRun``) are passed in; both cite
only works that passed verification (DOI exists, metadata match, not
retracted). A cited work becomes a compose ``Source`` (``work_source``) whose
text is the paper the cite loop read, and a ``[[fN]]`` marker on one of its
facts is a citation of that work.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict

from andamentum.compose.evidence import stored_source_path
from andamentum.compose.genres import Genre
from andamentum.compose.schemas import Source

from ..kernel.csl import doi_of, title_of, work_key

if TYPE_CHECKING:
    from andamentum.deep_research import ResearchResult


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
    verify the works with findings. The draft holds no search or verification
    of its own."""

    def __call__(
        self, task: str, *, genre: Genre, database: Path
    ) -> Awaitable[ResearchFinding]: ...


def work_source(
    work: CitableWork, *, number: int, bundle_dir: str | Path, text: str | None = None
) -> tuple[Source, str]:
    """The compose ``Source`` for *work* (id ``src<number>``) and its text.

    The text is the work's title and abstract, or *text* when given (the paper
    the cite loop read). It is written to the place the evidence worker reads
    sources from.
    """
    if text is None:
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
