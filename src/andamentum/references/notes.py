"""What a footnote or endnote says: a full citation, a short form, "Ibid.",
or commentary. Deterministic.

Notes styles (Chicago notes and bibliography, MHRA, OSCOLA and their kin) put
the citation in the note. A first citation is full — "Sophie Gudmundsson et
al., “Variant Interpretation,” Human Mutation 43 (2021): 1012." — and later
ones short — "Gudmundsson et al., “Variant Interpretation,” 1015." — or
"Ibid." for the note just before. Names in a note run first name first, so the
surname is the LAST word of the author part; in the bibliography it comes
first. Signal words ("See also", "Cf.", "For a contrary view, see") open a
note without being part of what it cites, and one note may cite several works
separated by semicolons.

A note whose form the rules cannot read — no recognisable author, but a year
or a quoted title — is "unclear", and goes to the model if there is one.
"""

from __future__ import annotations

import re

from .models import CitedWork, NoteForm

_IBID = re.compile(r"^\s*(?:ibid|id)\b\.?", re.IGNORECASE)
_SIGNAL = re.compile(
    r"^\s*(?:(?:[A-Z][\w ,]{0,40}?,\s*)?(?:see also|see|cf\.|compare|e\.g\.,?|for example,?)\s+)",
    re.IGNORECASE,
)
_YEAR = re.compile(r"(?<!\d)(1[5-9]\d\d|20\d\d)(?!\d)")
_QUOTED = re.compile(r"[“\"‘']([^”\"’']{3,200})[”\"’']")
_NAME_WORD = re.compile(r"^[A-Z][\w'’\-]*\.?$")
_MAX_AUTHOR_WORDS = 8


def read_note(text: str) -> tuple[NoteForm, list[CitedWork]]:
    """The note's form and the works it cites (empty for ibid and commentary)."""
    if _IBID.match(text):
        return "ibid", []
    works: list[CitedWork] = []
    full = False
    for segment in re.split(r";\s+", text):
        work, has_year = _read_segment(segment)
        if work is not None:
            works.append(work)
            full = full or has_year
    if works:
        return ("full" if full else "short"), works
    if _YEAR.search(text) or _QUOTED.search(text):
        return "unclear", []
    return "comment", []


def _read_segment(segment: str) -> tuple[CitedWork | None, bool]:
    body = _SIGNAL.sub("", segment.strip(), count=1).strip()
    if "," not in body:
        return None, False
    author_part, rest = body.split(",", 1)
    author_part = re.sub(r"\s+et\s+al\.?", "", author_part).strip()
    first_author = re.split(r"\s+(?:and|&)\s+", author_part)[0].strip()
    words = first_author.split()
    if not words or len(words) > _MAX_AUTHOR_WORDS:
        return None, False
    if not all(_NAME_WORD.match(w) or w.lower() in _PARTICLES for w in words):
        return None, False
    surname = words[-1].rstrip(".")
    if len(surname) < 2 or not surname[0].isupper():
        return None, False
    title = _title_of(rest)
    if not title:
        return None, False
    year = _year_of(rest, title=title)
    return CitedWork(surname=surname, year=year, title=title), year is not None


def _year_of(rest: str, *, title: str) -> int | None:
    """The publication year, or None for a short form.

    Chicago puts it in parentheses — "43, no. 8 (2021): 1012", "(New York:
    Penguin, 2006), 99" — so a year in parentheses wins. A short form carries
    only a page after its title ("…,” 1755."), which must not be read as the
    year 1755: when nothing but a number follows the title, there is no year.
    """
    in_parentheses = re.search(r"\([^()]*?(?<!\d)(1[5-9]\d\d|20\d\d)(?!\d)[^()]*\)", rest)
    if in_parentheses:
        return int(in_parentheses.group(1))
    after = rest.split(title, 1)[-1]
    if re.fullmatch(r"[\s\W]*\d+(?:\s*[–-]\s*\d+)?[\s.,;]*", after):
        return None
    found = _YEAR.search(after)
    return int(found.group(1)) if found else None


_PARTICLES = frozenset({"van", "von", "de", "der", "den", "del", "da", "di", "le", "la", "du", "dos", "el"})


def _title_of(rest: str) -> str:
    quoted = _QUOTED.search(rest)
    if quoted:
        return quoted.group(1).strip(" ,.")
    # An unquoted title (a book, set in italics the text has lost): up to the
    # next comma or parenthesis.
    candidate = re.split(r"[,(]", rest.strip(), maxsplit=1)[0].strip(" .")
    if len(candidate.split()) >= 1 and candidate[:1].isupper() and not candidate[:1].isdigit():
        return candidate
    return ""
