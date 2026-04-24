"""Deterministic scanners for baseline checklist items.

Each function takes document text and returns a `(status, notes)` tuple
that the orchestrator wraps into a `ChecklistItem`.

Constitution Rule 4: items that can be verified by regex/string-search
go here. Items requiring reading comprehension go to the LLM path.
"""

from __future__ import annotations

import re
from typing import Literal

Status = Literal["pass", "fail", "unclear"]


# ---------------------------------------------------------------------------
# Figures & tables
# ---------------------------------------------------------------------------

_FIGURE_CAPTION = re.compile(r"^\s*(?:Figure|Fig\.?)\s+(\d+)[\.:]", re.MULTILINE)
_FIGURE_REF = re.compile(r"\b(?:Figure|Fig\.?)\s+(\d+)\b")
_TABLE_CAPTION = re.compile(r"^\s*Table\s+(\d+)[\.:]", re.MULTILINE)
_TABLE_REF = re.compile(r"\bTable\s+(\d+)\b")


def check_all_figures_referenced(text: str) -> tuple[Status, str]:
    captions = {int(m.group(1)) for m in _FIGURE_CAPTION.finditer(text)}
    if not captions:
        return ("unclear", "No figure captions found in document.")
    # Remove caption lines so they don't self-count as in-text references.
    body = _FIGURE_CAPTION.sub("", text)
    refs = {int(m.group(1)) for m in _FIGURE_REF.finditer(body)}
    missing = captions - refs
    if missing:
        return ("fail", f"Figures without in-text references: {sorted(missing)}")
    return ("pass", f"All {len(captions)} figure captions referenced in text.")


def check_figure_numbering_sequential(text: str) -> tuple[Status, str]:
    nums = [int(m.group(1)) for m in _FIGURE_CAPTION.finditer(text)]
    if not nums:
        return ("unclear", "No figure captions found.")
    expected = list(range(1, len(nums) + 1))
    if sorted(nums) != expected:
        return (
            "fail",
            f"Figure captions numbered {sorted(nums)}; expected 1..{len(nums)}.",
        )
    return ("pass", f"Figure captions numbered sequentially 1..{len(nums)}.")


def check_all_tables_referenced(text: str) -> tuple[Status, str]:
    captions = {int(m.group(1)) for m in _TABLE_CAPTION.finditer(text)}
    if not captions:
        return ("unclear", "No table captions found in document.")
    # Remove caption lines so they don't self-count as in-text references.
    body = _TABLE_CAPTION.sub("", text)
    refs = {int(m.group(1)) for m in _TABLE_REF.finditer(body)}
    missing = captions - refs
    if missing:
        return ("fail", f"Tables without in-text references: {sorted(missing)}")
    return ("pass", f"All {len(captions)} table captions referenced in text.")


def check_table_numbering_sequential(text: str) -> tuple[Status, str]:
    nums = [int(m.group(1)) for m in _TABLE_CAPTION.finditer(text)]
    if not nums:
        return ("unclear", "No table captions found.")
    expected = list(range(1, len(nums) + 1))
    if sorted(nums) != expected:
        return (
            "fail",
            f"Table captions numbered {sorted(nums)}; expected 1..{len(nums)}.",
        )
    return ("pass", f"Table captions numbered sequentially 1..{len(nums)}.")


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

_REFERENCES_HEADER = re.compile(
    r"^\s*(?:References|Bibliography)\s*$", re.MULTILINE | re.IGNORECASE
)


def check_citations_resolve(text: str) -> tuple[Status, str]:
    ref_match = _REFERENCES_HEADER.search(text)
    if not ref_match:
        return ("unclear", "No References section found.")
    body = text[: ref_match.start()]
    refs_text = text[ref_match.end() :]
    ref_nums: set[int] = set()
    for m in re.finditer(r"^\s*(?:\[(\d+)\]|(\d+)\.)\s", refs_text, re.MULTILINE):
        ref_nums.add(int(m.group(1) or m.group(2)))
    if not ref_nums:
        return ("unclear", "References section found but no numbered entries detected.")
    cit_nums: set[int] = set()
    for m in re.finditer(r"\[(\d+(?:\s*[,-]\s*\d+)*)\]", body):
        for part in re.split(r"\s*,\s*", m.group(1)):
            if "-" in part:
                halves = [h.strip() for h in part.split("-", 1)]
                if len(halves) == 2 and halves[0].isdigit() and halves[1].isdigit():
                    cit_nums.update(range(int(halves[0]), int(halves[1]) + 1))
                # silently skip malformed ranges
            elif part.strip().isdigit():
                cit_nums.add(int(part.strip()))
    unresolved = cit_nums - ref_nums
    if unresolved:
        return ("fail", f"Citations with no matching reference: {sorted(unresolved)}")
    return ("pass", f"All {len(cit_nums)} citations resolve to reference entries.")


# ---------------------------------------------------------------------------
# Required statements
# ---------------------------------------------------------------------------


def check_coi_statement(text: str) -> tuple[Status, str]:
    pattern = re.compile(
        r"(?:conflicts?\s+of\s+interests?|competing\s+(?:financial\s+)?interests?|declarations?\s+of\s+interest)",
        re.IGNORECASE,
    )
    if pattern.search(text):
        return ("pass", "Conflict-of-interest / competing-interests statement found.")
    return ("fail", "No conflict-of-interest or competing-interests statement found.")


def check_data_availability_statement(text: str) -> tuple[Status, str]:
    pattern = re.compile(
        r"data\s+(?:availability|accessibility|sharing)", re.IGNORECASE
    )
    if pattern.search(text):
        return ("pass", "Data availability statement found.")
    return ("fail", "No data availability statement found.")


def check_ethics_statement(text: str) -> tuple[Status, str]:
    subjects = bool(
        re.search(
            r"\b(?:human\s+subjects?|participants?|patients?|volunteers?|animals?|mice|rats|murine|primates?)\b",
            text,
            re.IGNORECASE,
        )
    )
    if not subjects:
        return (
            "unclear",
            "No human/animal subjects mentioned — ethics statement may not apply.",
        )
    has_ethics = bool(
        re.search(
            r"(?:ethics\s+(?:approval|committee|statement|review\s+board)|IRB|IACUC|institutional\s+review)",
            text,
            re.IGNORECASE,
        )
    )
    if has_ethics:
        return ("pass", "Ethics / IRB / IACUC statement found.")
    return ("fail", "Human/animal subjects mentioned but no ethics statement found.")


def check_funding_statement(text: str) -> tuple[Status, str]:
    pattern = re.compile(
        r"(?:funding|supported\s+by|grant\s+(?:number|no\.?)|acknowledg(?:e)?ments?)",
        re.IGNORECASE,
    )
    if pattern.search(text):
        return ("pass", "Funding / acknowledgements statement found.")
    return ("fail", "No funding or acknowledgements statement found.")


# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------


def check_keywords_section(text: str) -> tuple[Status, str]:
    pattern = re.compile(r"^\s*Key\s*words?\s*[:\s]", re.MULTILINE | re.IGNORECASE)
    if pattern.search(text):
        return ("pass", "Keywords section found.")
    return ("fail", "No keywords section found.")


def check_authors_listed(text: str) -> tuple[Status, str]:
    """Scan the document head for affiliation markers.

    Returns "pass" if institutional keywords appear in the first 2000
    characters, "unclear" otherwise. Never returns "fail": absence of
    these keywords does not prove authors are unlisted — the document
    may start with an abstract, use non-standard affiliations, or have
    been passed in as a body-only excerpt.
    """
    head = text[:2000]
    pattern = re.compile(
        r"\b(?:Department|Institute|School|University|Laboratory|Center|Centre|Faculty|Hospital)\b"
    )
    if pattern.search(head):
        return ("pass", "Affiliation markers found near document head.")
    return (
        "unclear",
        "No standard affiliation keywords found in the first 2000 characters.",
    )
