"""Deterministic pre-submission checklist as v2 Findings.

Ported from v1 ``whetstone/checklist_scanners.py`` and the deterministic
items in v1 ``whetstone/agents/checklist.py``. v1 returned
``ChecklistItem`` objects with pass/fail/unclear status; v2 emits Findings
only for failures (passing checks produce no output — silence is success).

Scope is deliberately limited to checks that are not already covered by
``deterministic_findings.py``: required-statement presence, author block,
keywords, title, and abstract structure. Figure/table referencing,
sequential numbering, and unused references already live in
``deterministic_findings.py`` and are not duplicated here.

All checks are pure regex/string-search over the harvested markdown
(plus the section list when section-level scoping helps). No LLM, no
network, no embeddings — runs in milliseconds.
"""

from __future__ import annotations

import re

from ..schemas import Finding, Quote
from .types import SectionRef


# ── Public entry point ─────────────────────────────────────────────────


def run_checklist_checks(
    *,
    markdown: str,
    sections: list[SectionRef],
    abstract_min_words: int = 150,
    abstract_max_words: int = 300,
    title_min_words: int = 5,
    title_max_words: int = 25,
    keywords_min: int = 3,
    keywords_max: int = 8,
) -> list[Finding]:  # noqa: ARG001 - sections kept for future per-section context
    """Run all checklist checks against the harvested document.

    Parameters control the narrow numerical thresholds that vary by
    journal — defaults track common biomedical norms. None of these
    parameters are exposed via the CLI yet; if you find yourself wanting
    a non-default value, that's a signal to lift them into ReviewState.
    """
    out: list[Finding] = []
    out.extend(_check_required_statements(markdown))
    out.extend(_check_authors_listed(markdown))
    out.extend(
        _check_keywords(
            markdown,
            min_count=keywords_min,
            max_count=keywords_max,
        )
    )
    out.extend(
        _check_title(
            markdown,
            min_words=title_min_words,
            max_words=title_max_words,
        )
    )
    out.extend(
        _check_abstract(
            markdown=markdown,
            sections=sections,
            min_words=abstract_min_words,
            max_words=abstract_max_words,
        )
    )
    return out


# ── Required statements ────────────────────────────────────────────────


_COI_PATTERN = re.compile(
    r"(?:conflicts?\s+of\s+interests?"
    r"|competing\s+(?:financial\s+)?interests?"
    r"|declarations?\s+of\s+interest)",
    re.IGNORECASE,
)

_DATA_AVAIL_PATTERN = re.compile(
    r"data\s+(?:availability|accessibility|sharing)",
    re.IGNORECASE,
)

_ETHICS_PATTERN = re.compile(
    r"(?:ethics\s+(?:approval|committee|statement|review\s+board)"
    r"|IRB|IACUC|institutional\s+review)",
    re.IGNORECASE,
)

_SUBJECTS_PATTERN = re.compile(
    r"\b(?:human\s+subjects?|participants?|patients?|volunteers?"
    r"|animals?|mice|rats|murine|primates?)\b",
    re.IGNORECASE,
)

_FUNDING_PATTERN = re.compile(
    r"(?:funding|supported\s+by|grant\s+(?:number|no\.?)|acknowledg(?:e)?ments?)",
    re.IGNORECASE,
)


def _check_required_statements(markdown: str) -> list[Finding]:
    out: list[Finding] = []

    if not _COI_PATTERN.search(markdown):
        out.append(
            Finding(
                title="Conflict-of-interest statement missing",
                severity="major",
                confidence="high",
                rationale=(
                    "No conflict-of-interest, competing-interests, or declaration-of-interest "
                    "statement was found anywhere in the document. Most journals require an "
                    "explicit declaration before a manuscript can enter peer review."
                ),
                category="compliance",
            )
        )

    if not _DATA_AVAIL_PATTERN.search(markdown):
        out.append(
            Finding(
                title="Data availability statement missing",
                severity="major",
                confidence="high",
                rationale=(
                    "No data availability, data accessibility, or data sharing statement "
                    "was found. A data availability statement is required by most journals "
                    "and many funders, even if the answer is 'data are available on request'."
                ),
                category="compliance",
            )
        )

    if _SUBJECTS_PATTERN.search(markdown) and not _ETHICS_PATTERN.search(markdown):
        out.append(
            Finding(
                title="Ethics statement missing despite human/animal work",
                severity="major",
                confidence="high",
                rationale=(
                    "The document mentions human subjects, animals, or related study "
                    "populations but contains no ethics-approval, IRB, IACUC, or "
                    "institutional-review statement. Add a statement naming the approving "
                    "body and approval number."
                ),
                category="compliance",
            )
        )

    if not _FUNDING_PATTERN.search(markdown):
        out.append(
            Finding(
                title="Funding / acknowledgements statement missing",
                severity="moderate",
                confidence="high",
                rationale=(
                    "No funding, grant-support, or acknowledgements statement was found. "
                    "Even unfunded work usually warrants an explicit 'No specific funding' "
                    "declaration."
                ),
                category="compliance",
            )
        )

    return out


# ── Author affiliations ────────────────────────────────────────────────


_AFFILIATION_PATTERN = re.compile(
    r"\b(?:Department|Institute|School|University|Laboratory|Center"
    r"|Centre|Faculty|Hospital)\b"
)


def _check_authors_listed(markdown: str) -> list[Finding]:
    head = markdown[:2000]
    if _AFFILIATION_PATTERN.search(head):
        return []
    return [
        Finding(
            title="Author affiliations not detected at document head",
            severity="moderate",
            confidence="medium",
            rationale=(
                "No standard institutional keywords (Department, Institute, School, "
                "University, Laboratory, Center/Centre, Faculty, Hospital) appear in the "
                "first 2000 characters. Verify that the author list and affiliation block "
                "is present — the document may be a body-only excerpt, in which case "
                "ignore this finding."
            ),
            category="metadata",
        )
    ]


# ── Keywords ───────────────────────────────────────────────────────────


_KEYWORDS_HEADER = re.compile(
    r"^[ \t]*(?:#{1,6}\s*)?Key\s*words?\s*[:\s]\s*(?P<body>.+?)$",
    re.MULTILINE | re.IGNORECASE,
)


def _check_keywords(
    markdown: str,
    *,
    min_count: int,
    max_count: int,
) -> list[Finding]:
    match = _KEYWORDS_HEADER.search(markdown)
    if match is None:
        return [
            Finding(
                title="Keywords section missing",
                severity="minor",
                confidence="high",
                rationale=(
                    "No 'Keywords:' line or section header was found. Many journals use "
                    "the keywords list for indexing and discoverability."
                ),
                category="metadata",
            )
        ]
    body = match.group("body")
    items = [k.strip() for k in re.split(r"[,;]", body) if k.strip()]
    n = len(items)
    if n < min_count:
        return [
            Finding(
                title=f"Keywords list has {n} item(s); expected at least {min_count}",
                severity="minor",
                confidence="high",
                rationale=(
                    f"The keywords line was found but contains only {n} comma-separated "
                    f"item(s). Most journals expect {min_count}–{max_count}."
                ),
                category="metadata",
            )
        ]
    if n > max_count:
        return [
            Finding(
                title=f"Keywords list has {n} items; expected at most {max_count}",
                severity="minor",
                confidence="high",
                rationale=(
                    f"The keywords line contains {n} items. Most journals expect "
                    f"{min_count}–{max_count}; trim to the most discoverable terms."
                ),
                category="metadata",
            )
        ]
    return []


# ── Title ──────────────────────────────────────────────────────────────


_H1_PATTERN = re.compile(r"^[ \t]*#\s+(?P<title>.+?)\s*$", re.MULTILINE)


def _check_title(
    markdown: str,
    *,
    min_words: int,
    max_words: int,
) -> list[Finding]:
    head = markdown[:1000]
    match = _H1_PATTERN.search(head)
    if match is None:
        return [
            Finding(
                title="Document title not identified",
                severity="moderate",
                confidence="medium",
                rationale=(
                    "No H1 heading was found in the first 1000 characters. The harvested "
                    "markdown should begin with the manuscript's title — verify the source "
                    "document was harvested correctly, or that this is not a body-only "
                    "excerpt."
                ),
                category="metadata",
            )
        ]
    title = match.group("title").strip()
    n_words = len(title.split())
    if n_words < min_words:
        return [
            Finding(
                title=f"Title is only {n_words} word(s)",
                severity="minor",
                confidence="high",
                rationale=(
                    f'Title "{title}" is {n_words} word(s); typical research-article '
                    f"titles are {min_words}–{max_words} words. A short title may be "
                    "too generic to convey the contribution."
                ),
                category="metadata",
            )
        ]
    if n_words > max_words:
        return [
            Finding(
                title=f"Title is {n_words} words; consider trimming",
                severity="minor",
                confidence="high",
                rationale=(
                    f'Title "{title}" is {n_words} words; typical research-article '
                    f"titles are {min_words}–{max_words} words. Long titles often hide "
                    "the contribution — see if the methods or qualifiers can move into "
                    "the abstract."
                ),
                category="metadata",
            )
        ]
    return []


# ── Abstract ───────────────────────────────────────────────────────────


_IMRAD_CUES = {
    "background": (
        re.compile(r"\b(?:background|aim|purpose|introduction|objective)\b", re.IGNORECASE),
        "background/aim",
    ),
    "methods": (
        re.compile(r"\b(?:method|approach|design|setting|procedure|experiment)", re.IGNORECASE),
        "methods",
    ),
    "results": (
        re.compile(r"\b(?:result|finding|observed|showed|demonstrated)", re.IGNORECASE),
        "results",
    ),
    "conclusion": (
        re.compile(r"\b(?:conclude|conclusion|implication|in summary)", re.IGNORECASE),
        "conclusion",
    ),
}

_ABSTRACT_HEADING = re.compile(
    r"^[ \t]*(?P<hashes>#{1,6})\s+abstract\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_NEXT_HEADING = re.compile(r"^[ \t]*#{1,6}\s+", re.MULTILINE)


def _extract_abstract_body(markdown: str) -> tuple[str, int] | None:
    """Find the abstract heading in the document and return (body, start_pos).

    ``start_pos`` is the global character offset of the abstract body
    (excluding the heading line) so callers can locate which section
    chunker assigned the abstract to.

    Falls back across two strategies:
      1. Look for an explicit ``## Abstract`` (or any-level) heading.
      2. (Not implemented) — could fall back to "first paragraph of the
         doc" but that is too unreliable; better to flag absence.
    """
    m = _ABSTRACT_HEADING.search(markdown)
    if m is None:
        return None
    body_start = m.end()
    after = markdown[body_start:]
    next_match = _NEXT_HEADING.search(after)
    body = (after[: next_match.start()] if next_match else after).strip()
    return body, body_start


def _section_id_for_offset(
    sections: list[SectionRef], global_offset: int
) -> str | None:
    """Return the section_id whose [char_start, char_end] contains the offset."""
    for s in sections:
        if s.char_start <= global_offset < s.char_end:
            return s.id
    return None


def _check_abstract(
    *,
    markdown: str,
    sections: list[SectionRef],
    min_words: int,
    max_words: int,
) -> list[Finding]:
    extracted = _extract_abstract_body(markdown)
    if extracted is None:
        return [
            Finding(
                title="Abstract section not identified",
                severity="major",
                confidence="medium",
                rationale=(
                    "No '## Abstract' (or similar-level) heading was found in the "
                    "harvested document. Either the abstract is missing, or the "
                    "section heading differs from the standard label and was not "
                    "detected."
                ),
                category="abstract",
            )
        ]

    body, body_start = extracted
    section_id = _section_id_for_offset(sections, body_start)
    n_words = len(body.split())

    quote_kwargs = (
        {
            "quotes": [
                Quote(
                    section_id=section_id,
                    char_start=0,
                    char_end=min(len(body), 200),
                    text=body[:200],
                )
            ],
            "sections_involved": [section_id],
        }
        if section_id is not None
        else {}
    )

    out: list[Finding] = []

    if n_words < min_words:
        out.append(
            Finding(
                title=f"Abstract is only {n_words} words",
                severity="moderate",
                confidence="high",
                rationale=(
                    f"The abstract contains {n_words} words; most journals require "
                    f"{min_words}–{max_words}. Short abstracts often skip the methods "
                    "or results, leaving readers to guess at the contribution."
                ),
                category="abstract",
                **quote_kwargs,  # type: ignore[arg-type]
            )
        )
    elif n_words > max_words:
        out.append(
            Finding(
                title=f"Abstract is {n_words} words; consider trimming",
                severity="moderate",
                confidence="high",
                rationale=(
                    f"The abstract contains {n_words} words; most journals require "
                    f"{min_words}–{max_words}. Trim background and tighten methods "
                    "language to fit."
                ),
                category="abstract",
                **quote_kwargs,  # type: ignore[arg-type]
            )
        )

    matched = [
        label
        for _, (pattern, label) in _IMRAD_CUES.items()
        if pattern.search(body)
    ]
    if len(matched) < 3:
        missing = [
            label
            for _, (pattern, label) in _IMRAD_CUES.items()
            if not pattern.search(body)
        ]
        out.append(
            Finding(
                title="Abstract may lack IMRAD structure",
                severity="minor",
                confidence="medium",
                rationale=(
                    f"The abstract contains cues for only {len(matched)} of the four "
                    f"IMRAD elements (background, methods, results, conclusion). Missing "
                    f"cues for: {', '.join(missing)}. Consider whether the abstract "
                    "covers all four or restructuring would help readers locate them."
                ),
                category="abstract",
                **quote_kwargs,  # type: ignore[arg-type]
            )
        )

    return out
