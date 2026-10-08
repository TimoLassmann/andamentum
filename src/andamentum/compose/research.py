"""Research for a document: findings and their passages, never report prose.

``--research`` runs ``deep_research.research`` once, in the mode the genre
names, over the project's own store (``<project>/research.db``). What compose
takes from the result is the evidence table: every finding is a claim with a
verbatim quote that the engine checked against a stored passage of a real work.
Each work that has a finding becomes a compose ``Source`` (kind ``research``)
whose text is the passages the engine read, and each finding becomes a ``Fact``
whose quote is located in that text, so a ``[[fN]]`` marker in a draft traces
to words in a source with no model call in between. The engine's own report is
not used.

The work's citation is the source's label; a draft's Sources list prints it.
A fact is stated as another work's finding ("Another published work (...)
reports, not this study: ...") because the drafter sees only the statement and
would otherwise write a published result up as the author's own.

``deep_research`` is imported inside the functions that run it (compose depends
on it only behind ``--research``); the types it supplies appear here as strings.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .bundle import ComposeError
from .evidence import find_quote, stored_source_path
from .genres import Genre, ResearchMode
from .schemas import Fact, Source

if TYPE_CHECKING:
    from andamentum.deep_research import Budget, ResearchResult, SourceConfig
    from andamentum.references import Work

RESEARCH_DB = "research.db"
MAX_LABEL_AUTHORS = 3
ATTRIBUTION = "Another published work ({title}) reports, not this study: {claim}"


class ResearchSetupError(ComposeError, ValueError):
    """``--research`` was asked for without what its mode needs."""


@dataclass(frozen=True)
class ResearchSetup:
    """Everything ``--research`` needs besides the language model, explicit.

    ``sources`` carries the web search address (a SearxNG instance or a Chrome
    over CDP, for web and both) and the contact mail or keys of the scholarly
    sources. ``mode`` overrides the genre's mode. ``budget`` bounds the run
    (``None``: the engine's default).
    """

    sources: SourceConfig
    mode: ResearchMode | None = None
    budget: Budget | None = None


def research_mode(genre: Genre, setup: ResearchSetup) -> ResearchMode:
    """The mode to search in: the setup's override, else the genre's."""
    return setup.mode or genre.research


async def run_research(
    task: str,
    *,
    genre: Genre,
    setup: ResearchSetup,
    model: str,
    database: Path,
    progress: Callable[[str], None],
) -> ResearchResult:
    """Research *task* in the genre's mode over the store at *database*.

    The question is the user's own task text, as it was when research was
    plain web search: nothing derived from a source file is ever sent out.
    """
    from andamentum.deep_research import Budget, research

    mode = research_mode(genre, setup)
    if mode != "scholarly" and not (setup.sources.searxng_url or setup.sources.browser):
        raise ResearchSetupError(
            f"research mode {mode!r} (genre {genre.name!r}) searches the web and "
            "needs a SearxNG address (--searxng-url) or the isolated browser "
            "(--browser); pass one"
        )
    progress(f"Research: {mode} search")
    return await research(
        task,
        mode=mode,
        model=model,
        store=database,
        budget=setup.budget or Budget(),
        sources=setup.sources,
        progress=progress,
    )


def citation_label(work: Work) -> str:
    """How a reader of the finished document sees the work: authors, year,
    title, venue and where to find it."""
    names = work.authors
    authors = ", ".join(names[:MAX_LABEL_AUTHORS])
    if len(names) > MAX_LABEL_AUTHORS:
        authors += " et al."
    year = f"({work.year})" if work.year else ""
    head = " ".join(part for part in (authors, year) if part)
    where = f"https://doi.org/{work.doi}" if work.doi else (work.url or "")
    parts = [head, work.title.rstrip("."), work.venue, where]
    return ". ".join(part for part in parts if part)


def _work_path(work: Work) -> str:
    """Where the work is: a link when it has one (never a file on this machine)."""
    if work.doi:
        return f"https://doi.org/{work.doi}"
    return work.url or work.oa_url or work.work_id


def _source_text(work: Work, passages: list[str]) -> str:
    return f"# {work.title or '(untitled)'}\n\n" + "\n\n".join(passages) + "\n"


def research_sources(
    result: ResearchResult,
    *,
    bundle_dir: str | Path,
    first_number: int,
    only: Collection[str] | None = None,
) -> list[Source]:
    """One source per work that has a finding (in order of first citation).

    The source's text, written where the evidence worker reads sources from,
    is the work's title and the passages the findings point into, in the order
    they stand in the work. ``only`` limits the works (by store id) to those a
    caller found usable. Ids are ``src<first_number>`` onwards.
    """
    finding_works = {f.work_id for f in result.findings}
    passages_of: dict[str, list[tuple[int, str]]] = {}
    wanted = {f.passage_id for f in result.findings}
    for passage in result.passages:
        if passage.passage_id in wanted:
            passages_of.setdefault(passage.work_id, []).append(
                (passage.start, passage.text)
            )
    sources: list[Source] = []
    for work in result.works:
        if work.work_id not in finding_works or work.work_id not in passages_of:
            continue
        if only is not None and work.work_id not in only:
            continue
        ordered = [text for _start, text in sorted(passages_of[work.work_id])]
        text = _source_text(work, ordered)
        source = Source(
            id=f"src{first_number + len(sources)}",
            path=_work_path(work),
            title=work.title or "(untitled)",
            sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            chars=len(text),
            kind="research",
            label=citation_label(work),
            work_id=work.work_id,
        )
        stored = stored_source_path(bundle_dir, source)
        stored.parent.mkdir(parents=True, exist_ok=True)
        stored.write_text(text, encoding="utf-8")
        sources.append(source)
    return sources


def attribute(claim: str, *, title: str) -> str:
    """A finding's claim stated as another work's, for the drafter."""
    return ATTRIBUTION.format(title=title, claim=claim)


def research_facts(
    result: ResearchResult,
    source: Source,
    text: str,
    *,
    first_number: int,
) -> tuple[list[Fact], list[str]]:
    """The facts of one research source: its findings, quotes located in *text*.

    Returns the facts (ids ``f<first_number>`` onwards) and the notes: one for
    every finding whose quote could not be placed in the source text (the engine
    checked it against the passage, so this signals a bug, and the fact is
    dropped rather than kept without its words), and one line counting the
    findings the engine could not verify (their quote was not found in the page),
    which are not used.
    """
    facts: list[Fact] = []
    dropped: list[str] = []
    unverified = 0
    for finding in result.findings:
        if finding.work_id != source.work_id:
            continue
        if not finding.quote_verified:
            unverified += 1
            continue
        span = find_quote(text, finding.quote)
        if span is None:
            dropped.append(
                f"finding {finding.finding_id} of {source.title!r}: quote not found "
                "in the source text; not used"
            )
            continue
        facts.append(
            Fact(
                id=f"f{first_number + len(facts)}",
                source_id=source.id,
                statement=attribute(finding.claim, title=source.title),
                quote=text[span[0] : span[1]],
                start=span[0],
                end=span[1],
            )
        )
    if unverified:
        dropped.append(f"{unverified} unverified quotes not used")
    return facts, dropped


def degraded_notes(result: ResearchResult) -> list[str]:
    """What the reader of the report should know about how the research went."""
    notes: list[str] = []
    if result.degraded:
        shown = "; ".join(result.degraded[:3])
        more = (
            f" (and {len(result.degraded) - 3} more)"
            if len(result.degraded) > 3
            else ""
        )
        notes.append(f"Research was degraded: {shown}{more}.")
    if not result.findings:
        notes.append(
            f"Research found no usable passages (stopped: {result.stop_reason or 'unknown'})."
        )
    return notes


def research_setup(
    *,
    mailto: str | None,
    searxng_url: str | None,
    browser: bool,
    mode: Literal["web", "scholarly", "both"] | None,
) -> ResearchSetup:
    """The research settings from CLI flags; credentials come from the
    environment (``.env`` is loaded by the CLI), never from flags."""
    from andamentum.deep_research import SourceConfig

    return ResearchSetup(
        sources=SourceConfig(
            mailto=mailto or os.environ.get("OPENALEX_EMAIL") or None,
            openalex_api_key=os.environ.get("OPENALEX_API_KEY") or None,
            semantic_scholar_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None,
            searxng_url=searxng_url,
            browser=browser,
        ),
        mode=mode,
    )
