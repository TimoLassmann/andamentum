"""Deterministic scanners for internal consistency.

Pure functions. No LLM, no async, no IO. Each scanner examines
document text and emits DocumentIssue objects for problems it can
verify without reading comprehension.

Constitution Rule 4: countable work lives here; reading-comprehension
work lives in the consistency_reviewer agent.
"""

from __future__ import annotations

import re

from .issues import DocumentIssue

_FIGURE_PATTERN = re.compile(r"\b(?:Figure\s+|Fig\.\s*|Fig\s+)(\d+)\b")
_ACRONYM_PATTERN = re.compile(r"\b([A-Z]{2,6})\b")
_CITATION_NUMERIC = re.compile(r"\[(\d+(?:\s*[,-]\s*\d+)*)\]")
_REFERENCES_HEADER = re.compile(
    r"^\s*(?:References|Bibliography)\s*$", re.MULTILINE | re.IGNORECASE
)

# Common acronyms skipped by check_acronym_first_use — universally recognised,
# not worth flagging. Only pure uppercase ASCII entries matter: mixed-case or
# digit-containing acronyms (mRNA, CO2, MHz, etc.) would never reach this
# allowlist because _ACRONYM_PATTERN only extracts [A-Z]{2,6} runs.
_COMMON_ACRONYMS: frozenset[str] = frozenset(
    {
        "DNA",
        "RNA",
        "PCR",
        "HIV",
        "USA",
        "UK",
        "EU",
        "CI",
        "SD",
        "SEM",
        "FDA",
        "NIH",
        "NASA",
        "NSF",
        "PDF",
        "HTML",
        "URL",
        "API",
        "CPU",
        "GPU",
        "RAM",
        "USB",
        "AI",
        "ML",
        "ATP",
        "GDP",
        "OECD",
        "PCA",
        "SVM",
        "MB",
        "GB",
    }
)


def check_figure_order(text: str) -> list[DocumentIssue]:
    """Find figures first-referenced out of ascending order.

    Emits one issue per out-of-order first-reference.
    """
    seen: set[int] = set()
    expected_next = 1
    issues: list[DocumentIssue] = []

    for match in _FIGURE_PATTERN.finditer(text):
        n = int(match.group(1))
        if n in seen:
            continue
        seen.add(n)
        if n > expected_next:
            # Skipped ahead — first reference to this figure appeared before
            # the expected next figure in sequence.
            issues.append(
                DocumentIssue(
                    issue_type="minor",
                    category="structure",
                    title=f"Figure {n} introduced out of order",
                    description=(
                        f"Figure {n} is first referenced before Figure {expected_next}. "
                        f"Figures should be introduced in ascending numerical order."
                    ),
                    recommendation=f"Move the first mention of Figure {n} after Figure {expected_next}.",
                    location=f"Offset {match.start()}",
                    agent_type="scanner:figure_order",
                    confidence=1.0,
                    priority="medium",
                )
            )
        expected_next = max(expected_next, n + 1)

    return issues


def check_acronym_first_use(text: str) -> list[DocumentIssue]:
    """Find acronyms whose first use is not accompanied by a parenthesised definition.

    Definition pattern recognised: capitalised words immediately followed
    by ' (ACRONYM)'. Acronyms in _COMMON_ACRONYMS are skipped.
    """
    issues: list[DocumentIssue] = []
    seen: set[str] = set()

    for match in _ACRONYM_PATTERN.finditer(text):
        acronym = match.group(1)
        if acronym in seen or acronym in _COMMON_ACRONYMS:
            continue
        seen.add(acronym)

        # Look backward for '(ACRONYM)' definition pattern within 200 chars.
        start = max(0, match.start() - 200)
        window = text[start : match.end() + 1]
        if re.search(rf"\(\s*{re.escape(acronym)}\s*\)", window):
            continue  # defined earlier in the window

        # Or it's defined *at* this occurrence: "Phrase (ACR)" — look backward
        # from the opening paren.
        before = text[max(0, match.start() - 100) : match.start()]
        if re.search(r"[A-Za-z][^.\n]*\s+\(\s*$", before):
            continue  # preceded by a phrase and opening paren — first-use definition

        issues.append(
            DocumentIssue(
                issue_type="minor",
                category="structure",
                title=f"Acronym '{acronym}' used before being defined",
                description=(
                    f"The acronym '{acronym}' appears without a parenthesised "
                    f"definition before or at its first use."
                ),
                recommendation=f"Expand on first use: 'Full Phrase ({acronym})'.",
                location=f"Offset {match.start()}",
                agent_type="scanner:acronym_first_use",
                confidence=0.75,
                priority="low",
            )
        )

    return issues


def check_citation_resolution(text: str) -> list[DocumentIssue]:
    """Verify every [N]-style in-text citation has a matching reference entry.

    Returns [] if no References section is found (out of scope for this
    scanner) or if no numbered entries are detected in the references.
    """
    ref_match = _REFERENCES_HEADER.search(text)
    if not ref_match:
        return []

    body = text[: ref_match.start()]
    refs_text = text[ref_match.end() :]

    ref_nums: set[int] = set()
    for m in re.finditer(r"^\s*(?:\[(\d+)\]|(\d+)\.)\s", refs_text, re.MULTILINE):
        ref_nums.add(int(m.group(1) or m.group(2)))
    if not ref_nums:
        return []

    issues: list[DocumentIssue] = []
    seen: set[int] = set()
    for m in _CITATION_NUMERIC.finditer(body):
        raw = m.group(1)
        nums: list[int] = []
        for part in re.split(r"\s*,\s*", raw):
            if "-" in part:
                halves = part.split("-", 1)
                if len(halves) == 2 and halves[0].isdigit() and halves[1].isdigit():
                    nums.extend(range(int(halves[0]), int(halves[1]) + 1))
                # silently skip malformed ranges like [1-3-5]
            elif part.isdigit():
                nums.append(int(part))
        for n in nums:
            if n in seen:
                continue
            seen.add(n)
            if n not in ref_nums:
                issues.append(
                    DocumentIssue(
                        issue_type="major",
                        category="references",
                        title=f"Citation [{n}] has no matching reference entry",
                        description=(
                            f"In-text citation [{n}] found, but no reference [{n}] in References section."
                        ),
                        recommendation=f"Add a reference entry [{n}] or renumber the citation.",
                        location=f"Offset {m.start()}",
                        agent_type="scanner:citation_resolution",
                        confidence=1.0,
                        priority="high",
                    )
                )
    return issues


def run_all(text: str) -> list[DocumentIssue]:
    """Run all consistency scanners and return the merged issue list."""
    return (
        check_figure_order(text)
        + check_acronym_first_use(text)
        + check_citation_resolution(text)
    )
