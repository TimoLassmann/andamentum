"""Every possible citation in the text, of every kind, with no judgement yet.

Deterministic. The scan is deliberately generous — a superscript "2" after
"cm" and a bracketed "[3]" are both candidates — because deciding what is a
citation is the next stage's job (:mod:`.decide`), with the whole document's
evidence in view.

Seven sources: numbers in square brackets or parentheses; numbers the source
file formats as superscript (from a :class:`Hint`); digits glued to a word,
for a text that lost its formatting; author-year citations in parentheses or
in running text; and reference-manager fields.
"""

from __future__ import annotations

import re

from .models import Candidate, CitedWork, Hint
from .notes import read_note

_NUMBER_LIST = r"\d{1,4}(?:[ \t]*[,;–—-][ \t]*\d{1,4})*"
_BRACKETS = re.compile(rf"\[({_NUMBER_LIST})\]")
_PARENTHESES = re.compile(rf"\(({_NUMBER_LIST})\)")
# Digits stuck to the end of a word or after closing punctuation:
# "disease12", "disease.12", "disease,3-5". Only lowercase words of three or
# more letters, so gene and cell names (CD34, IL6), chemistry (H2O) and units
# (m2) are not even candidates.
_GLUED = re.compile(rf"(?:(?<=[a-z]{{3}})|(?<=[.,;:)\]]))({_NUMBER_LIST})(?=[\s.,;:)]|$)")
_MAX_RANGE = 50
# No reference list runs to a thousand entries; a four-digit number in
# brackets or parentheses is a year, "(2019)", not citation 2019.
_MAX_NUMBER = 999

# A surname: a capitalised word, allowing particles and hyphens
# (van der Berg, O'Neil, García-López).
# A surname may be several capitalised words ("Nguengang Wakap", "Le Cam");
# matching tries the whole run first and then drops leading words, so a
# sentence-initial "As Turro et al." still finds Turro (see resolve).
_NAME = (
    r"(?:(?:van|von|de|der|den|del|da|di|le|la|du|dos|el)\s+)*"
    r"[A-Z][\w'’\-]+(?:\s+[A-Z][\w'’\-]+)*"
)
# One or more names: "Smith", "Smith et al.", "Smith and Lee",
# "LeCun, Bengio and Hinton" (Harvard names up to three).
_AUTHORS = rf"{_NAME}(?:\s*,\s*{_NAME})*(?:,?\s+(?:et\s+al\.?|(?:and|&)\s+{_NAME}))?"
_YEARS = r"(?:1[89]\d\d|20\d\d)[a-z]?(?:\s*,\s*(?:1[89]\d\d|20\d\d)[a-z]?)*"
_PAREN_GROUP = re.compile(r"\(([^()]{4,400}?)\)")
_CITE_IN_GROUP = re.compile(rf"({_AUTHORS})(?:,|\s)\s*({_YEARS})")
_NARRATIVE = re.compile(rf"\b({_AUTHORS})\s+\(({_YEARS})\)")
_YEAR_ITEM = re.compile(r"(1[89]\d\d|20\d\d)([a-z]?)")
# Capitalised words that precede a year and are not authors.
_NOT_NAMES = frozenset(
    """
    table tables figure figures fig fig. supplementary appendix section chapter
    january february march april may june july august september october november
    december spring summer autumn fall winter version phase wave cohort round
    trial study since from until in on by the a an of for and or between during
    before after up to published accessed retrieved updated edition vol volume
    covid sars mers who cdc nih nhs usa uk eu
    """.split()
)


_PANDOC_NOTE_MARK = re.compile(r"\[\^([^\]\s]+)\](?!:)")
_PANDOC_NOTE_DEF = re.compile(r"^\[\^([^\]\s]+)\]:[ \t]*(.+)$", re.MULTILINE)


def note_anchor(text: str, offset: int) -> tuple[int, int]:
    """The word just before a note mark — where a comment about the note
    goes, since the mark itself is not a character of the text. The word
    after, when the mark opens a paragraph."""
    line_start = text.rfind("\n", 0, offset) + 1
    before = text[line_start:offset].rstrip()
    if before:
        word = re.search(r"\S+$", before)
        assert word is not None
        return line_start + word.start(), line_start + len(before)
    after = re.match(r"\s*(\S+)", text[offset:])
    if after:
        return offset + after.start(1), offset + after.end(1)
    return max(0, offset - 1), max(1, offset)


def pandoc_notes(text: str) -> tuple[list[Hint], list[tuple[int, int]]]:
    """Markdown footnotes: ``[^1]`` marks with ``[^1]: text`` definitions.

    Returns note hints at the marks and the definitions' spans, which belong
    to the notes and are not scanned as body text.
    """
    definitions = {m.group(1): m for m in _PANDOC_NOTE_DEF.finditer(text)}
    if not definitions:
        return [], []
    hints = [
        Hint(kind="note", start=m.start(), end=m.start(), payload=definitions[m.group(1)].group(2))
        for m in _PANDOC_NOTE_MARK.finditer(text)
        if m.group(1) in definitions
    ]
    return hints, [(m.start(), m.end()) for m in definitions.values()]


def scan(
    text: str,
    *,
    start: int,
    end: int,
    hints: list[Hint] | None = None,
    first_id: int = 0,
) -> list[Candidate]:
    """All candidates in ``text[start:end]``, in document order."""
    found: list[Candidate] = []
    hints = list(hints or [])
    covered: list[tuple[int, int]] = []  # spans a field, superscript or note owns
    markdown_notes, definitions = pandoc_notes(text)
    hints += markdown_notes
    covered += definitions

    def add(source, span_start, span_end, *, numbers=(), works=(), note_text="", note_form=None) -> None:
        found.append(
            Candidate(
                id=0,
                source=source,
                text=text[span_start:span_end],
                start=span_start,
                end=span_end,
                numbers=list(numbers),
                works=list(works),
                note_text=note_text,
                note_form=note_form,
            )
        )

    for hint in hints:
        if hint.kind != "note" or not (start <= hint.start <= end):
            continue
        form, works = read_note(hint.payload)
        anchor_start, anchor_end = note_anchor(text, hint.start)
        add("note", anchor_start, anchor_end, works=works, note_text=hint.payload, note_form=form)

    for hint in hints:
        if not (start <= hint.start < end):
            continue
        if hint.kind == "field_citation":
            snippet = text[hint.start : hint.end]
            add(
                "field",
                hint.start,
                hint.end,
                numbers=_expand(re.sub(r"[^\d,;–—-]", "", snippet)) if re.fullmatch(r"[\s\[\]()\d,;–—-]+", snippet) else (),
                works=_works_in(snippet),
            )
            covered.append((hint.start, hint.end))
    has_superscript = any(h.kind == "superscript" for h in hints)
    for hint in hints:
        if hint.kind != "superscript" or not (start <= hint.start < end):
            continue
        if _inside(hint.start, covered):
            continue
        body = text[hint.start : hint.end].strip()
        if re.fullmatch(_NUMBER_LIST, body):
            numbers = _expand(body)
            if numbers:
                add("superscript", hint.start, hint.end, numbers=numbers)
                covered.append((hint.start, hint.end))

    for pattern, source in ((_BRACKETS, "brackets"), (_PARENTHESES, "parentheses")):
        for m in pattern.finditer(text, start, end):
            if _inside(m.start(), covered):
                continue
            numbers = _expand(m.group(1))
            if numbers:
                add(source, m.start(), m.end(), numbers=numbers)

    # Glued digits only when the source had no formatting to go on: where the
    # file marks superscript, a digit it did NOT raise is not a superscript
    # citation, and scanning for glued ones would only add noise.
    if not has_superscript:
        for m in _GLUED.finditer(text, start, end):
            if _inside(m.start(), covered):
                continue
            numbers = _expand(m.group(1))
            if numbers:
                add("glued", m.start(1), m.end(1), numbers=numbers)

    for group in _PAREN_GROUP.finditer(text, start, end):
        if _inside(group.start(), covered):
            continue
        works = _works_in(group.group(1))
        if works:
            add("parenthetical", group.start(), group.end(), works=works)
    for m in _NARRATIVE.finditer(text, start, end):
        if _inside(m.start(), covered):
            continue
        works = _narrative_works(m.group(1), m.group(2))
        if works:
            add("narrative", m.start(), m.end(), works=works)

    found.sort(key=lambda c: (c.start, c.end))
    return [c.model_copy(update={"id": first_id + i}) for i, c in enumerate(found)]


def _inside(offset: int, spans: list[tuple[int, int]]) -> bool:
    return any(a <= offset < b for a, b in spans)


def _expand(body: str) -> list[int]:
    numbers: list[int] = []
    for part in re.split(r"[ \t]*[,;][ \t]*", body.strip()):
        if not part:
            continue
        bounds = re.split(r"[ \t]*[–—-][ \t]*", part)
        try:
            low, high = int(bounds[0]), int(bounds[-1])
        except ValueError:
            return []
        if max(low, high) > _MAX_NUMBER:
            return []
        if len(bounds) == 1:
            numbers.append(low)
            continue
        if high < low or high - low > _MAX_RANGE:
            return []
        numbers.extend(range(low, high + 1))
    return numbers


def _surname(authors: str) -> str:
    """The first author's family name from "Smith et al." / "van der Berg and Lee"."""
    first = re.split(r"\s+(?:et\s+al\.?|and\b|&)|,", authors)[0].strip()
    return first


def _plausible(surname: str) -> bool:
    words = surname.lower().split()
    return bool(words) and words[-1].strip(".") not in _NOT_NAMES


def _works_in(group: str) -> list[CitedWork]:
    """The author-year works inside one parenthesis: "(Smith et al., 2020a;
    Lee and Park 2019, 2021)"."""
    works: list[CitedWork] = []
    for m in _CITE_IN_GROUP.finditer(group):
        surname = _surname(m.group(1))
        if not _plausible(surname):
            continue
        for year in _YEAR_ITEM.finditer(m.group(2)):
            works.append(
                CitedWork(surname=surname, year=int(year.group(1)), suffix=year.group(2))
            )
    return works


def _narrative_works(authors: str, years: str) -> list[CitedWork]:
    surname = _surname(authors)
    if not _plausible(surname):
        return []
    return [
        CitedWork(surname=surname, year=int(y.group(1)), suffix=y.group(2))
        for y in _YEAR_ITEM.finditer(years)
    ]
