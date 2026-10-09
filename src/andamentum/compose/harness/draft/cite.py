"""The cite loop: after the draft, find, read and cite works for its sentences.

The draft is written first; then, sentence by sentence (the way AI Scientist
adds its references):

1. **find the sentence**: a background or interpretation sentence that cites
   nothing (``SectionAnalysis`` already labels every sentence; no model call),
   at most ``MAX_SENTENCES`` per draft, the rest named in the notes;
2. **find papers**: the user's library first, then the scholarly finder
   (``deep_research.find_works`` with one step of citation following);
3. **read**: the paper's text from the document store when it holds it, else
   fetched (``deep_research.read_work``) and put into the store under its
   identifier; only papers read are stored, candidates never;
4. **decide**: one model call per paper read, on the part of the text most
   relevant to the sentence: it supports the sentence, supports it once the
   sentence is changed (the model writes the changed sentence), or does not.
   The supporting quote must be found in the text and the work must pass the
   same verification as any cited work (DOI exists, details agree, not
   retracted); then the paper becomes a source, the quote a fact, and the
   sentence gets its ``[[fN]]`` marker. At most ``READS_PER_SENTENCE`` papers
   are read for one sentence.

The steps that touch the network and the store come in as :class:`CiteTools`
(the draft verb builds them, the tests pass stand-ins). A failed search or read
is a note for that sentence, never a silent skip; the sentence stays as drafted.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, Field

from andamentum.compose._call import call_model
from andamentum.compose.log import ActionLog
from andamentum.compose.research import attribute
from andamentum.compose.schemas import Fact, Source
from andamentum.references import Work

from .assemble import SectionDraft
from ..literature import LiteratureError
from .body import TO_FILL, sentence_spans
from .claims import Sentence, SourceInfo
from .works import CitableWork, work_source

if TYPE_CHECKING:
    from andamentum.deep_research import WorkText
    from andamentum.document_store import Store

MAX_SENTENCES = 8  # sentences one draft sends through the loop
CANDIDATES_PER_SENTENCE = 6  # works considered for one sentence, library first
READS_PER_SENTENCE = 2  # papers read for one sentence before it stays uncited
CITE_KINDS = ("background", "interpretation")
WINDOW_CHARS = 12_000  # the part of a paper the decision reads
MIN_QUOTE_WORDS = 8  # a shorter quote shows nothing a title does not

_TAG = re.compile(r"<[^>]+>")


class Finder(Protocol):
    async def __call__(self, sentence: str, *, paragraph: str) -> list[Work]: ...


class LibrarySearch(Protocol):
    async def __call__(self, sentence: str) -> list[Work]: ...


class Reader(Protocol):
    async def __call__(self, work: Work, *, query: str) -> WorkText: ...


class Verifier(Protocol):
    async def __call__(self, work: Work) -> CitableWork | str: ...


@dataclass(kw_only=True)
class CiteTools:
    """What the loop needs from outside: the searches, the reader, the
    verification (a verified work, or the reason it cannot be cited) and the
    document store the papers read go into."""

    library: LibrarySearch | None
    finder: Finder
    read: Reader
    verify: Verifier
    papers: Store
    store_path: Path
    source_notes: list[str] = field(default_factory=list)
    """Lines the tools record about search or read outages; the loop moves
    them into the outcome's notes."""


@dataclass(frozen=True)
class Rewrite:
    """A sentence changed to match the paper that supports it."""

    section_key: str
    old: str  # the sentence as drafted (cleaned text)
    new: str  # the sentence now in the body, without its marker


class CiteVerdict(BaseModel):
    supports: Literal["yes", "after_change", "no"]
    quote: str = Field(
        default="",
        description="the words of the excerpt that support the sentence, copied exactly",
    )
    sentence: str = Field(
        default="", description="with after_change: the sentence as it should read"
    )


@dataclass
class CiteOutcome:
    """What the loop did; ``notes`` go into the draft report."""

    checked: int = 0
    cited: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    uncited: list[str] = field(default_factory=list)
    read: int = 0
    stored: int = 0
    reused: int = 0
    changed_sections: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)
    rewrites: list[Rewrite] = field(default_factory=list)

    def summary(self, store_path: Path) -> str:
        return (
            f"cite loop: {self.checked} sentence(s) checked, {len(self.cited)} cited "
            f"({len(self.changed)} changed to match the paper), {len(self.uncited)} "
            f"left without a citation; {self.read} paper(s) read, {self.stored} added "
            f"to {store_path}, {self.reused} taken from it"
        )


INSTRUCTIONS = """\
You check whether one paper supports one sentence of a draft.

You get the sentence, its paragraph, the paper's title and an excerpt of the
paper. Decide from the excerpt only:

- supports "yes": the excerpt states what the sentence says.
- supports "after_change": the excerpt supports a narrower or corrected
  version of the sentence. Write that version in `sentence`: keep the author's
  words where you can, change only what the paper does not support, one
  sentence, no citation marker.
- supports "no": the excerpt does not support the sentence, or is about
  something else.

With "yes" or "after_change", copy into `quote` the words of the excerpt that
support it: one continuous passage, character for character, no ellipsis.
"""


def candidate_sentences(
    drafts: Sequence[SectionDraft],
) -> list[tuple[SectionDraft, Sentence]]:
    """The sentences that need a citation, in document order."""
    return [
        (section, sentence)
        for section in drafts
        for sentence in section.analysis.sentences
        if sentence.kind in CITE_KINDS
        and not sentence.fact_ids
        and not TO_FILL.search(sentence.text)
        and not _in_placeholder(section, sentence)
    ]


def _in_placeholder(section: SectionDraft, sentence: Sentence) -> bool:
    """Whether *sentence* overlaps a ``[TO FILL: ...]`` placeholder (one with a
    full stop inside is split into two sentences, neither matching TO_FILL)."""
    ranges = section.paragraphs[sentence.paragraph].to_fill()
    return any(start < sentence.end and sentence.start < end for start, end in ranges)


def _usable_rewrite(text: str) -> bool:
    """A changed sentence the loop may put into the draft: one sentence, one
    line, no fact marker and no placeholder."""
    return (
        bool(text)
        and "\n" not in text
        and "[[" not in text
        and not TO_FILL.search(text)
        and len(sentence_spans(text)) == 1
    )


def paper_key(work: Work) -> str | None:
    """The document-store key of a paper: its strongest identifier."""
    for prefix, value in (
        ("doi", work.doi),
        ("arxiv", work.arxiv_id),
        ("pmcid", work.pmcid),
        ("pmid", work.pmid),
        ("openalex", work.openalex_id),
        ("url", work.url),
    ):
        if value:
            return f"{prefix}:{value}"
    return None


def work_from_csl(csl: dict[str, object]) -> Work:
    """A library item (CSL JSON) as a ``Work`` the reader can look up."""
    authors = []
    for person in csl.get("author") or []:  # type: ignore[union-attr]
        if isinstance(person, dict):
            name = " ".join(
                str(person.get(k) or "") for k in ("given", "family")
            ).strip() or str(person.get("literal") or "")
            if name:
                authors.append(name)
    year = None
    issued = csl.get("issued")
    if isinstance(issued, dict):
        parts = issued.get("date-parts")
        if (
            isinstance(parts, list)
            and parts
            and isinstance(parts[0], list)
            and parts[0]
        ):
            first = parts[0][0]
            year = int(first) if str(first).isdigit() else None
    return Work(
        doi=str(csl.get("DOI") or "") or None,
        title=str(csl.get("title") or ""),
        authors=authors,
        year=year,
        abstract=" ".join(_TAG.sub(" ", str(csl.get("abstract") or "")).split()),
        url=str(csl.get("URL") or "") or None,
    )


def _same(a: Work, b: Work) -> bool:
    keys = set(a.identity_keys())
    return any(k in keys for k in b.identity_keys())


async def _candidates(
    tools: CiteTools, sentence: str, paragraph: str, outcome: CiteOutcome
) -> list[Work]:
    from andamentum.deep_research import ResearchError

    found: list[Work] = []
    if tools.library is not None:
        try:
            found += await tools.library(sentence)
        except LiteratureError as exc:
            note = f"library search failed: {exc}"
            if note not in outcome.notes:
                outcome.notes.append(note)
    try:
        for work in await tools.finder(sentence, paragraph=paragraph):
            if not any(_same(work, seen) for seen in found):
                found.append(work)
    except ResearchError as exc:
        outcome.notes.append(f"search failed for {sentence[:80]!r}: {exc}")
    return found[:CANDIDATES_PER_SENTENCE]


async def _paper_text(
    tools: CiteTools, work: Work, *, query: str, outcome: CiteOutcome
) -> tuple[str, str]:
    """The paper's whole text and its window for *query* ("" when unreadable)."""
    from andamentum.deep_research import ResearchError, text_window

    key = paper_key(work)
    if key is not None:
        doc_id = await tools.papers.resolve(key)
        if doc_id is not None:
            doc = await tools.papers.get(doc_id)
            if doc is not None and doc.text.strip():
                outcome.reused += 1
                return doc.text, text_window(
                    doc.text, query=query, max_chars=WINDOW_CHARS
                )
    try:
        found = await tools.read(work, query=query)
    except ResearchError as exc:
        outcome.notes.append(f"could not read {work.title!r}: {exc}")
        return "", ""
    if not found.text:
        return "", ""
    await tools.papers.put(
        found.text,
        key=key,
        metadata={
            "title": work.title,
            "doi": work.doi,
            "authors": work.authors,
            "year": work.year,
            "venue": work.venue,
            "url": found.url or work.url,
            "read_as": found.via,
            "licence": work.licence,
        },
    )
    outcome.stored += 1
    window = text_window(found.text, query=query, max_chars=WINDOW_CHARS)
    return found.text, window


async def _decide(
    *,
    sentence: str,
    paragraph: str,
    work: Work,
    window: str,
    model: str,
    log: ActionLog | None,
) -> CiteVerdict:
    from andamentum.deep_research import UNTRUSTED, fence

    message = (
        f"SENTENCE: {sentence}\n\nPARAGRAPH: {paragraph}\n\n"
        f"PAPER: {work.title} ({work.year or 'n.d.'})\n\nEXCERPT:\n{fence(window)}"
    )
    return await call_model(
        model=model,
        instructions=f"{INSTRUCTIONS}\n{UNTRUSTED}",
        output_type=CiteVerdict,
        user_message=message,
        node="cite",
        log=log,
        role="reader",
    )


_AT_START = re.compile(
    r"(?:\A\s*\Z"
    r"|\n[ \t]*(?:[-*\u2022]|\d+[.)])?[ \t]*\Z"
    r"|[.!?][\"')\]\u201d\u2019]*(?:\s*\[\[f\d+\]\])*\s+\Z)"
)


def _locate_sentence(body: str, text: str) -> tuple[int, int] | None:
    """Where *text* (a sentence's cleaned text) stands in the raw *body* as a
    whole sentence, any whitespace run (a hard-wrapped line) standing for a
    space; None unless there is exactly one such place."""
    words = text.split()
    if not words:
        return None
    pattern = re.compile(r"\s+".join(re.escape(w) for w in words))
    spans = [
        m.span()
        for m in pattern.finditer(body)
        if _AT_START.search(body[: m.start()])
        and (m.end() == len(body) or body[m.end()].isspace())
    ]
    return spans[0] if len(spans) == 1 else None


def _replace_sentence(body: str, old: str, new: str, marker: str) -> str | None:
    """*body* with the sentence *old* replaced by *new* plus *marker*; None when
    *old* is not found exactly once as a whole sentence."""
    span = _locate_sentence(body, old)
    if span is None:
        return None
    return f"{body[: span[0]]}{new.rstrip()} {marker}{body[span[1] :]}"


async def cite_sentences(
    drafts: Sequence[SectionDraft],
    *,
    sources: list[Source],
    facts: list[Fact],
    infos: dict[str, SourceInfo],
    tools: CiteTools,
    bundle_dir: Path,
    model: str,
    progress: Callable[[str], None],
    log: ActionLog | None,
) -> CiteOutcome:
    """Run the loop over *drafts* (see the module docstring).

    ``sources``, ``facts`` and ``infos`` are the run's evidence and are added to
    in place; a cited sentence's section body is edited in place and its key is
    in ``changed_sections`` (the caller analyses those again).
    """
    from andamentum.deep_research import locate_quote

    outcome = CiteOutcome()
    made: set[str] = set()  # ids of the sources this loop made, with full text
    wanted = candidate_sentences(drafts)
    if len(wanted) > MAX_SENTENCES:
        outcome.notes.append(
            f"cite loop: {len(wanted) - MAX_SENTENCES} more sentence(s) without a "
            f"citation were not checked (at most {MAX_SENTENCES} per draft)"
        )
    for section, sentence in wanted[:MAX_SENTENCES]:
        paragraph = section.paragraphs[sentence.paragraph].text
        if _locate_sentence(section.body, sentence.text) is None:
            outcome.notes.append(
                f"{sentence.text[:80]!r}: the sentence could not be placed in the "
                "draft text (found zero or several times); not checked"
            )
            continue
        outcome.checked += 1
        progress(f"Cite: {sentence.text[:60]}")
        cited = False
        reads = 0
        for work in await _candidates(tools, sentence.text, paragraph, outcome):
            if reads >= READS_PER_SENTENCE:
                break
            text, window = await _paper_text(
                tools, work, query=sentence.text, outcome=outcome
            )
            if not text:
                continue
            reads += 1
            outcome.read += 1
            verdict = await _decide(
                sentence=sentence.text,
                paragraph=paragraph,
                work=work,
                window=window,
                model=model,
                log=log,
            )
            if verdict.supports == "no":
                continue
            if len(verdict.quote.split()) < MIN_QUOTE_WORDS:
                outcome.notes.append(
                    f"{work.title!r}: quote too short to show support; not cited"
                )
                continue
            span = locate_quote(verdict.quote, text)
            if span is None:
                outcome.notes.append(
                    f"{work.title!r}: the supporting quote is not in the paper; not cited"
                )
                continue
            checked = await tools.verify(work)
            if isinstance(checked, str):
                outcome.notes.append(f"not cited, could not be verified: {checked}")
                continue
            new_text = sentence.text
            if verdict.supports == "after_change":
                new_text = verdict.sentence.strip()
                if not _usable_rewrite(new_text):
                    outcome.notes.append(
                        f"{work.title!r} supports a changed version of "
                        f"{sentence.text[:80]!r}, but the changed sentence is not "
                        "one plain sentence; not cited"
                    )
                    continue
            fact_id = f"f{len(facts) + 1}"
            edited = _replace_sentence(
                section.body, sentence.text, new_text, f"[[{fact_id}]]"
            )
            if edited is None:
                outcome.notes.append(
                    f"{sentence.text[:80]!r}: supported by {work.title!r}, but the "
                    "sentence could not be placed in the draft text; not cited"
                )
                break
            source = _source_of(checked, infos, made)
            if source is None:  # a paper cited twice is one source
                source, _ = work_source(
                    checked, number=len(sources) + 1, bundle_dir=bundle_dir, text=text
                )
                sources.append(source)
                infos[source.id] = SourceInfo(source=source, work=checked)
                made.add(source.id)
            facts.append(
                Fact(
                    id=fact_id,
                    source_id=source.id,
                    statement=attribute(new_text, title=source.title),
                    quote=text[span[0] : span[1]],
                    start=span[0],
                    end=span[1],
                )
            )
            section.body = edited
            outcome.changed_sections.add(section.key)
            outcome.cited.append(new_text)
            if new_text != sentence.text:
                outcome.changed.append(f"{sentence.text} -> {new_text}")
                outcome.rewrites.append(
                    Rewrite(section_key=section.key, old=sentence.text, new=new_text)
                )
            cited = True
            break
        if not cited:
            outcome.uncited.append(sentence.text)
    outcome.notes.insert(0, outcome.summary(tools.store_path))
    outcome.notes[1:1] = tools.source_notes
    tools.source_notes.clear()
    outcome.notes += [f"no citation found for: {text}" for text in outcome.uncited]
    outcome.notes += [f"changed to match the paper: {line}" for line in outcome.changed]
    return outcome


def _source_of(
    work: CitableWork, infos: dict[str, SourceInfo], made: set[str]
) -> Source | None:
    """The source this loop made for *work*, if any. Sources made elsewhere
    (the literature step) hold only a title and abstract, so a fact quoting the
    full paper cannot point into them."""
    for source_id in made:
        info = infos.get(source_id)
        if info is not None and info.work is not None and info.work.key == work.key:
            return info.source
    return None
