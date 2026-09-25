"""The data a reference check produces. Flat, so any consumer can read it.

Every located thing carries ``start`` / ``end`` offsets into the text that was
checked. A caller that holds a document in the same coordinates (whetstone's
reviewed text, for one) can therefore attach a problem to the exact citation or
entry it concerns without searching for it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ProblemKind = Literal[
    "cited_not_listed",
    "listed_not_cited",
    "duplicate_number",
    "duplicate_entry",
    "out_of_order",
    "year_mismatch",
    "ambiguous_citation",
    "ibid_unclear",
    "short_before_full",
    "not_found",
    "doi_not_found",
    "details_differ",
]

# How a document cites. One value per document; see ``decide.decide_style``.
Style = Literal[
    "numeric_brackets",  # [3], [4, 5], [7-9]
    "numeric_superscript",  # a raised 3 after the word
    "numeric_parentheses",  # (3), (4, 5)
    "author_year",  # (Smith et al., 2020), Smith and Lee (2019)
    "notes",  # footnotes or endnotes that carry the citations (Chicago notes)
    "undecided",
]

# Where a candidate citation was seen.
CandidateSource = Literal[
    "brackets",
    "parentheses",
    "superscript",  # formatted as superscript in the source file
    "glued",  # digits stuck to a word, in a text that lost its formatting
    "parenthetical",  # (Smith et al., 2020; Lee 2019)
    "narrative",  # Smith et al. (2020)
    "field",  # a reference manager's citation field (Zotero, EndNote, ...)
    "note",  # a footnote or endnote; its text is in ``note_text``
]

# What a note says, by its form. "comment" is prose that cites nothing.
NoteForm = Literal["full", "short", "ibid", "comment", "unclear"]


class Hint(BaseModel):
    """Evidence the text alone does not carry, from the source file's format.

    Supplied by the caller that read the file — for a Word file, whetstone's
    reader, which sees run formatting and field codes. Offsets are in the same
    text that is checked.
    """

    kind: Literal["superscript", "field_citation", "note"]
    start: int
    end: int  # equal to start for a note: its mark is not a character
    # A field citation's instruction ("ADDIN ZOTERO_ITEM CSL_CITATION {...}"),
    # or a note's own text. Empty for a superscript run.
    payload: str = ""


class CitedWork(BaseModel):
    """One work inside an author-year citation."""

    surname: str  # first author's family name, as written
    year: int | None = None  # None in a note's short form
    suffix: str = ""  # the "a" of 2020a
    title: str = ""  # a note's title words, which tell one author's works apart


class Candidate(BaseModel):
    """Something in the text that may be a citation."""

    id: int
    source: CandidateSource
    text: str
    start: int
    end: int
    numbers: list[int] = Field(default_factory=list)  # numeric sources
    works: list[CitedWork] = Field(default_factory=list)  # author-year and note sources
    note_text: str = ""  # a note's own text
    note_form: NoteForm | None = None


class Judged(BaseModel):
    """A candidate after the decision: a citation or not, and why."""

    candidate: Candidate
    is_citation: bool
    # "rule: ...", "model", or "not judged: ..." — the report shows these, so
    # a reader can see why a number was or was not read as a citation.
    reason: str


class StyleDecision(BaseModel):
    style: Style
    decided_by: Literal["evidence", "model", "none"]
    reason: str
    # Candidates that fit each style and point at a real entry.
    support: dict[Style, int] = Field(default_factory=dict)


class ReferenceEntry(BaseModel):
    """One entry of the reference list, as written."""

    number: int | None  # the entry's own number; None in an unnumbered list
    text: str  # the entry without its number
    start: int
    end: int
    doi: str = ""  # a DOI written in the entry, if any
    lead: str = ""  # the author part: the text before the first year
    years: list[int] = Field(default_factory=list)
    year: int | None = None  # the first year written — the publication year
    year_suffix: str = ""  # the "a" of "2020a", when the entry has one


class Match(BaseModel):
    """The published work a lookup matched an entry to."""

    title: str
    year: int | None = None  # the service's main ("issued") year
    # Every year the record carries — issued, online, print. A citation may
    # give any of them: a paper online in November 2019 is in the 2020 issue,
    # and "2020" is what most reference lists print.
    years: list[int] = Field(default_factory=list)
    first_author: str = ""
    doi: str = ""
    kind: str = ""  # the service's own type, e.g. "journal-article"
    volume: str = ""
    first_page: str = ""


class LookupResult(BaseModel):
    """What the lookup made of one entry.

    ``service_error`` means the entry was NOT checked — the service failed or
    refused — and says nothing about the reference. It is counted, never
    reported as a problem with the entry.
    """

    entry_number: int | None
    status: Literal["found", "not_found", "details_differ", "service_error"]
    match: Match | None = None
    detail: str = ""  # what differs, or what went wrong
    entry_start: int = 0  # locates the entry when it has no number


class Problem(BaseModel):
    """One thing wrong with the references, located and with a fix."""

    kind: ProblemKind
    severity: Literal["minor", "moderate", "major"]
    headline: str
    issue: str
    fix: str
    start: int
    end: int
    entry_number: int | None = None


class ReferenceReport(BaseModel):
    """Everything one check found. ``found_list`` False means no reference
    list was recognised, and then nothing below it was checked."""

    found_list: bool
    list_start: int = 0
    list_end: int = 0
    entries: list[ReferenceEntry] = Field(default_factory=list)
    style: StyleDecision | None = None
    judged: list[Judged] = Field(default_factory=list)
    problems: list[Problem] = Field(default_factory=list)
    lookups: list[LookupResult] = Field(default_factory=list)
    looked_up: bool = False  # whether the online lookup ran at all

    @property
    def citations(self) -> list[Candidate]:
        return [j.candidate for j in self.judged if j.is_citation]
