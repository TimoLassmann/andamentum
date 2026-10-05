"""Readers for a user's own library: CSL JSON and BibTeX.

``read_library(path)`` returns :class:`Work` objects with ``source="library"``.
The format follows the extension: ``.json`` (a CSL JSON list, or an object with
an ``items`` list) or ``.bib``. A file that does not parse raises
``LiteratureError`` naming the problem; entries without a title and a DOI are
reported the same way rather than dropped, since a library entry the user wrote
down is something they expect to find.

The BibTeX reader is small: entries ``@type{key, field = {..} | ".." | n}``,
nested braces, ``and``-separated authors in ``Family, Given`` or ``Given
Family`` order, ``--`` page ranges, and the common LaTeX accents and escapes.
``@string`` macros, ``@preamble`` and ``@comment`` are skipped; ``#``
concatenation is not supported.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from .models import LiteratureError, Work

_TYPES = {
    "article": "article-journal",
    "book": "book",
    "inbook": "chapter",
    "incollection": "chapter",
    "inproceedings": "paper-conference",
    "conference": "paper-conference",
    "phdthesis": "thesis",
    "mastersthesis": "thesis",
    "techreport": "report",
    "unpublished": "manuscript",
    "online": "webpage",
    "misc": "document",
}
_ACCENTS = {
    "`": "\u0300",
    "'": "\u0301",
    "^": "\u0302",
    "~": "\u0303",
    '"': "\u0308",
    "c": "\u0327",
    "v": "\u030c",
    "u": "\u0306",
    "H": "\u030b",
    "=": "\u0304",
    ".": "\u0307",
}
_LETTERS = {
    "\\ss": "ß",
    "\\o": "ø",
    "\\O": "Ø",
    "\\ae": "æ",
    "\\AE": "Æ",
    "\\aa": "å",
    "\\AA": "Å",
    "\\l": "ł",
    "\\L": "Ł",
    "\\i": "ı",
}


def read_library(path: Path) -> list[Work]:
    """The works in the CSL JSON or BibTeX file at *path*."""
    suffix = path.suffix.lower()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise LiteratureError(f"cannot read the library {path}: {exc}") from exc
    if suffix == ".json":
        items = _csl_items(text, path)
    elif suffix == ".bib":
        items = _bibtex_items(text, path)
    else:
        raise LiteratureError(
            f"unknown library format {suffix!r} for {path}; use .json (CSL JSON) or .bib"
        )
    works: list[Work] = []
    for position, item in enumerate(items):
        if not (item.get("title") or item.get("DOI")):
            raise LiteratureError(
                f"entry {position + 1} of {path} (id {item.get('id', '?')!r}) has "
                "neither a title nor a DOI"
            )
        works.append(Work(csl=item, source="library"))
    return works


# --------------------------------------------------------------------- CSL JSON


def _csl_items(text: str, path: Path) -> list[dict[str, object]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LiteratureError(f"{path} is not valid JSON: {exc}") from exc
    if isinstance(data, dict):
        data = data.get("items")
    if not isinstance(data, list) or not all(isinstance(i, dict) for i in data):
        raise LiteratureError(f"{path} is not a CSL JSON list of items")
    items: list[dict[str, object]] = []
    for position, item in enumerate(data):
        entry = {str(k): v for k, v in item.items()}
        entry.setdefault("id", f"item{position + 1}")
        items.append(entry)
    return items


# ----------------------------------------------------------------------- BibTeX


def _bibtex_items(text: str, path: Path) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    position = 0
    while True:
        at = text.find("@", position)
        if at < 0:
            return items
        found = re.compile(r"@\s*(\w+)\s*([{(])").match(text, at)
        if found is None:
            position = at + 1
            continue
        kind = found.group(1).lower()
        opener = found.group(2)
        end = _matching(text, found.end() - 1, opener, path)
        body = text[found.end() : end]
        position = end + 1
        if kind in {"string", "preamble", "comment"}:
            continue
        items.append(_bibtex_entry(kind, body, path))


def _matching(text: str, start: int, opener: str, path: Path) -> int:
    """Index of the bracket closing the one at *start*."""
    closer = "}" if opener == "{" else ")"
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index
    raise LiteratureError(
        f"{path}: an entry opened at character {start} is never closed"
    )


def _bibtex_entry(kind: str, body: str, path: Path) -> dict[str, object]:
    key, _, rest = body.partition(",")
    fields = _bibtex_fields(rest, path)
    entry: dict[str, object] = {
        "id": key.strip(),
        "type": _TYPES.get(kind, "document"),
    }
    if "title" in fields:
        entry["title"] = fields["title"]
    authors = _names(fields.get("author", ""))
    if authors:
        entry["author"] = authors
    editors = _names(fields.get("editor", ""))
    if editors:
        entry["editor"] = editors
    year = re.search(r"\d{4}", fields.get("year", fields.get("date", "")))
    if year:
        entry["issued"] = {"date-parts": [[int(year.group(0))]]}
    container = fields.get("journal") or fields.get("booktitle")
    if container:
        entry["container-title"] = container
    for source, target in (
        ("volume", "volume"),
        ("number", "issue"),
        ("publisher", "publisher"),
        ("url", "URL"),
        ("abstract", "abstract"),
        ("keywords", "keyword"),
    ):
        if source in fields:
            entry[target] = fields[source]
    if "pages" in fields:
        entry["page"] = re.sub(r"\s*-{1,3}\s*", "–", fields["pages"])
    if "doi" in fields:
        entry["DOI"] = fields["doi"].removeprefix("https://doi.org/")
    return entry


def _bibtex_fields(text: str, path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    index = 0
    while True:
        found = re.compile(r"\s*,?\s*(\w[\w-]*)\s*=\s*").match(text, index)
        if found is None:
            return fields
        name = found.group(1).lower()
        value, index = _bibtex_value(text, found.end(), path)
        # Braces mean "keep together" in a name list (a corporate author).
        fields[name] = _clean(value, keep_braces=name in {"author", "editor"})


def _bibtex_value(text: str, start: int, path: Path) -> tuple[str, int]:
    if start >= len(text):
        raise LiteratureError(f"{path}: a field has no value")
    if text[start] == "{":
        end = _matching(text, start, "{", path)
        return text[start + 1 : end], end + 1
    if text[start] == '"':
        depth = 0
        for index in range(start + 1, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            elif text[index] == '"' and depth == 0:
                return text[start + 1 : index], index + 1
        raise LiteratureError(f"{path}: a quoted field is never closed")
    found = re.compile(r"[^,}\s]+").match(text, start)
    if found is None:
        raise LiteratureError(f"{path}: a field has no value")
    return found.group(0), found.end()


def _clean(value: str, *, keep_braces: bool = False) -> str:
    """Accents and escapes out, braces too unless *keep_braces*; whitespace collapsed."""
    for command, char in _LETTERS.items():
        value = re.sub(re.escape(command) + r"(?![A-Za-z])\s*", char, value)

    def accent(match: re.Match[str]) -> str:
        mark = _ACCENTS[match.group(1)]
        letter = match.group(2)
        return unicodedata.normalize("NFC", letter + mark)

    value = re.sub(r"\\([`'^~\"=.])\s*\{?\\?([A-Za-z])\}?", accent, value)
    value = re.sub(r"\\([cvuH])\s*\{?([A-Za-z])\}?", accent, value)
    value = re.sub(r"\\([&%$#_])", r"\1", value).replace("~", " ")
    if not keep_braces:
        value = value.replace("{", "").replace("}", "")
    return " ".join(value.split())


def _names(value: str) -> list[dict[str, str]]:
    names: list[dict[str, str]] = []
    for person in _split_and(value):
        person = person.strip()
        if not person or person == "others":
            continue
        if person.startswith("{") and person.endswith("}") and "," not in person[1:-1]:
            names.append({"literal": _unbrace(person)})
            continue
        if "," in person:
            family, _, given = person.partition(",")
        else:
            *given_parts, family = _split_words(person)
            given = " ".join(given_parts)
        entry = {"family": _unbrace(family)}
        if given.strip():
            entry["given"] = _unbrace(given)
        names.append(entry)
    return names


def _split_and(value: str) -> list[str]:
    """Split a name list at top-level ``and``."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for word in re.split(r"(\s+)", value):
        depth += word.count("{") - word.count("}")
        if word == "and" and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(word)
    parts.append("".join(current))
    return parts


def _split_words(text: str) -> list[str]:
    """Whitespace-separated words, a braced group counting as one."""
    words: list[str] = []
    depth = 0
    current = ""
    for char in text:
        if char.isspace() and depth == 0:
            if current:
                words.append(current)
            current = ""
            continue
        depth += (char == "{") - (char == "}")
        current += char
    if current:
        words.append(current)
    return words


def _unbrace(text: str) -> str:
    return " ".join(text.replace("{", "").replace("}", "").split())
