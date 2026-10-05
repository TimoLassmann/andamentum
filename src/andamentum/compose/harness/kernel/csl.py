"""A small CSL formatter: citation text and reference entries from CSL JSON.

Not a CSL engine. It writes the two shapes the harness meets, author-year
(``(Smith 2020; Jones et al. 2019)``) and numeric (``[1, 2]``), in the
punctuation the document already uses (:func:`detect_style`), and a plain-text
reference entry for each. Zotero reformats its own fields on the next refresh,
so for a Zotero document this text is a placeholder that reads right; for a
plain-text document it is the text that stays. Known limits: no year suffixes
(``2020a``) for two works with one first author and year, no italics, no
sort order beyond the order works are given.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .ops import CitationItem

# Citation-shaped text a person or a model typed: author-year in parentheses, a
# numeric bracket, and a narrative "Smith (2019)". One definition for every place
# that must tell typed text from a citation the harness inserted.
TYPED_CITATION = re.compile(
    r"\((?=[^()]*[A-Z][a-z])[^()]*\b(?:1[5-9]|20)\d{2}[a-z]?\b[^()]*\)"
    r"|\[\d+(?:\s*[,;\u2013-]\s*\d+)*\]"
    r"|[A-Z][\w'-]+(?: et al\.| (?:and|&) [A-Z][\w'-]+)?,? \((?:1[5-9]|20)\d{2}[a-z]?\)"
)

_NUMERIC = re.compile(r"^[\[(]?\s*\d+(\s*[,;–-]\s*\d+)*\s*[\])]?$")
_PAGES = re.compile(r"^\d")


class CitationStyle(BaseModel):
    """How a document writes its citations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["author-year", "numeric"] = "author-year"
    open: str = "("
    close: str = ")"
    comma: bool = False  # "Smith, 2020" rather than "Smith 2020"
    ampersand: bool = False  # "Smith & Jones" rather than "Smith and Jones"


def typed_citations(text: str) -> list[tuple[int, int]]:
    """Ranges of the citation-shaped strings in *text*."""
    return [m.span() for m in TYPED_CITATION.finditer(text)]


def detect_style(displays: Sequence[str]) -> CitationStyle:
    """The style the existing citation texts are written in; author-year by default.

    Numeric when most displays are digits (in brackets, parentheses or bare,
    as a superscript's text is); the bracket kind follows the first one that
    has any. Author-year takes its comma and ampersand habits from the displays.
    """
    shown = [d.strip() for d in displays if d.strip()]
    numeric = [d for d in shown if _NUMERIC.match(d)]
    if shown and len(numeric) * 2 > len(shown):
        for display in numeric:
            if display[0] in "[(":
                return CitationStyle(
                    kind="numeric",
                    open=display[0],
                    close=")" if display[0] == "(" else "]",
                )
        return CitationStyle(kind="numeric", open="[", close="]")
    return CitationStyle(
        comma=any(re.search(r"[A-Za-z.],\s*\d{4}", d) for d in shown),
        ampersand=any(" & " in d for d in shown),
    )


# ------------------------------------------------------------------ CSL access


def _names(csl: Mapping[str, object], field: str = "author") -> list[dict[str, str]]:
    value = csl.get(field)
    if not isinstance(value, list):
        return []
    names: list[dict[str, str]] = []
    for entry in value:
        if isinstance(entry, dict):
            names.append({str(k): str(v) for k, v in entry.items()})
    return names


def _family(name: Mapping[str, str]) -> str:
    return name.get("family") or name.get("literal") or name.get("name") or ""


def year_of(csl: Mapping[str, object]) -> int | None:
    issued = csl.get("issued")
    if not isinstance(issued, dict):
        return None
    parts = issued.get("date-parts")
    if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
        try:
            return int(str(parts[0][0]))
        except ValueError:
            return None
    found = re.search(r"\b(1[5-9]\d\d|20\d\d)\b", str(issued.get("raw", "")))
    return int(found.group(1)) if found else None


def title_of(csl: Mapping[str, object]) -> str:
    return " ".join(str(csl.get("title", "")).split())


def doi_of(csl: Mapping[str, object]) -> str:
    doi = str(csl.get("DOI", "")).strip()
    return doi.removeprefix("https://doi.org/").removeprefix("doi:").lower()


def work_key(csl: Mapping[str, object]) -> str:
    """Identity of a work across citations: its DOI, else its normalised title."""
    doi = doi_of(csl)
    if doi:
        return f"doi:{doi}"
    title = re.sub(r"[^a-z0-9]+", " ", title_of(csl).lower()).strip()
    return f"title:{title}" if title else f"id:{csl.get('id', '')}"


def _lead_names(csl: Mapping[str, object]) -> list[str]:
    authors = [_family(a) for a in _names(csl) if _family(a)]
    if authors:
        return authors
    editors = [_family(a) for a in _names(csl, "editor") if _family(a)]
    if editors:
        return editors
    fallback = str(csl.get("container-title") or title_of(csl)).split()
    return [" ".join(fallback[:3])] if fallback else ["Anon."]


# ------------------------------------------------------------------ in-text


def _locator(value: str) -> str:
    value = value.strip()
    if not _PAGES.match(value):
        return value
    return ("pp. " if re.search(r"[-–,]", value) else "p. ") + value


def _author_year(item: CitationItem, style: CitationStyle) -> str:
    names = _lead_names(item.csl)
    joiner = " & " if style.ampersand else " and "
    if len(names) == 1:
        lead = names[0]
    elif len(names) == 2:
        lead = joiner.join(names)
    else:
        lead = f"{names[0]} et al."
    year = year_of(item.csl)
    text = f"{lead}{',' if style.comma else ''} {year if year else 'n.d.'}"
    return _decorate(text, item)


def _decorate(text: str, item: CitationItem) -> str:
    if item.locator:
        text = f"{text}, {_locator(item.locator)}"
    if item.prefix:
        text = f"{item.prefix.strip()} {text}"
    if item.suffix:
        text = f"{text} {item.suffix.strip()}"
    return text


def citation_text(
    items: Sequence[CitationItem],
    style: CitationStyle,
    *,
    numbers: Mapping[str, int] | None = None,
) -> str:
    """The text a reader sees for a citation of *items*.

    Numeric styles need *numbers* (work key to reference number); a work
    missing from it is a caller error and raises ``KeyError``.
    """
    if style.kind == "numeric":
        if numbers is None:
            raise ValueError("a numeric citation needs reference numbers")
        shown = [_decorate(str(numbers[work_key(i.csl)]), i) for i in items]
        return f"{style.open}{', '.join(shown)}{style.close}"
    shown = [_author_year(i, style) for i in items]
    return f"{style.open}{'; '.join(shown)}{style.close}"


# ------------------------------------------------------------- reference list


def _initials(given: str) -> list[str]:
    parts = re.split(r"[\s.]+", given.strip())
    return [p[0].upper() for p in parts if p and p[0].isalpha()]


def _sentence(text: str) -> str:
    text = text.strip()
    return text if not text or text[-1] in ".?!" else text + "."


def bibliography_entry(
    csl: Mapping[str, object], style: CitationStyle, *, number: int | None = None
) -> str:
    """A plain-text reference entry; numeric entries start with their number."""
    authors = [a for a in _names(csl) if _family(a)]
    title = _sentence(title_of(csl))
    container = str(csl.get("container-title", "")).strip()
    volume = str(csl.get("volume", "")).strip()
    issue = str(csl.get("issue", "")).strip()
    pages = re.sub(r"\s*[-‐–]+\s*", "–", str(csl.get("page", "")).strip())
    year = year_of(csl)
    doi = doi_of(csl)
    if style.kind == "numeric":
        if number is None:
            raise ValueError("a numeric reference entry needs its number")
        who = ", ".join(
            f"{_family(a)} {''.join(_initials(a.get('given', '')))}".strip()
            for a in authors[:6]
        )
        if len(authors) > 6:
            who += ", et al"
        locus = f"{year or ''}"
        if volume:
            locus += f";{volume}"
            if issue:
                locus += f"({issue})"
        if pages:
            locus += f":{pages}"
        parts = [
            _sentence(who),
            title,
            _sentence(container),
            _sentence(locus) if locus else "",
            f"doi:{doi}" if doi else "",
        ]
        lead = (
            f"{style.open}{number}{style.close}" if style.open == "[" else f"{number}."
        )
        return f"{lead} " + " ".join(p for p in parts if p)
    names = [
        f"{_family(a)}, {' '.join(i + '.' for i in _initials(a.get('given', '')))}"
        if _initials(a.get("given", ""))
        else _family(a)
        for a in authors[:6]
    ]
    if len(authors) > 6:
        who = ", ".join(names) + ", et al."
    elif len(names) > 1:
        who = ", ".join(names[:-1]) + ", & " + names[-1]
    else:
        who = names[0] if names else ""
    locus = container
    if volume:
        locus += f", {volume}" + (f"({issue})" if issue else "")
    if pages:
        locus += f", {pages}"
    parts = [
        _sentence(who),
        f"({year if year else 'n.d.'}).",
        title,
        _sentence(locus) if locus else "",
        f"https://doi.org/{doi}" if doi else "",
    ]
    return " ".join(p for p in parts if p)
