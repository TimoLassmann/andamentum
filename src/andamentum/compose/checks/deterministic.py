"""Deterministic checks: no model, same input gives the same results.

Each check yields one or more ``CheckResult``. A check named in
``WARNING_CHECKS`` is advisory: it never counts against the goal and never
drives revision. Everything else is an error when it fails.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from andamentum.proofread import analyze
from andamentum.scribe import Section

from ..genres import Genre
from ..render import MARKER_RE, plain_text, strip_markers
from ..schemas import Brief, CheckResult, Fact, Outline

WARNING_CHECKS = frozenset({"section_words", "to_fill", "style"})
SECTION_WORD_TOLERANCE = 0.5  # plan target +/- 50%
MIN_WORDS_FOR_GRADE = 30  # readability scores are noise on a handful of words

_HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s")
_TO_FILL = re.compile(r"\[TO FILL[^\]]*\]?")


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def is_error(result: CheckResult) -> bool:
    """A failed check that counts against the goal (warnings do not)."""
    return not result.passed and result.name not in WARNING_CHECKS


def errors(results: Sequence[CheckResult]) -> list[CheckResult]:
    return [r for r in results if is_error(r)]


def to_fill_items(results: Sequence[CheckResult]) -> list[str]:
    """The ``[TO FILL: ...]`` markers found, for the open questions list."""
    return [d for r in results if r.name == "to_fill" for d in r.details]


def _check_total_words(sections: Sequence[Section], *, brief: Brief) -> CheckResult:
    total = sum(word_count(strip_markers(s.body)) for s in sections)
    lo, hi = brief.word_range
    if lo <= total <= hi:
        return CheckResult(
            name="total_words", passed=True, details=[f"{total} words in {lo}-{hi}"]
        )
    side = "short" if total < lo else "long"
    return CheckResult(
        name="total_words",
        passed=False,
        details=[f"{total} words, too {side} for the range {lo}-{hi}"],
    )


def _check_section_words(
    sections: Sequence[Section], *, outline: Outline | None
) -> list[CheckResult]:
    if outline is None:
        return []
    targets = {p.title.casefold(): p.target_words for p in outline.sections}
    out: list[CheckResult] = []
    for section in sections:
        target = targets.get(section.title.casefold())
        if not target:
            continue
        n = word_count(strip_markers(section.body))
        lo = int(target * (1 - SECTION_WORD_TOLERANCE))
        hi = int(target * (1 + SECTION_WORD_TOLERANCE))
        if not lo <= n <= hi:
            out.append(
                CheckResult(
                    name="section_words",
                    passed=False,
                    details=[f"{section.title}: {n} words, target {target}"],
                    section_id=section.id,
                )
            )
    return out or [CheckResult(name="section_words", passed=True, details=[])]


def _check_required_sections(
    sections: Sequence[Section],
    *,
    brief: Brief,
    genre: Genre,
    outline: Outline | None,
) -> CheckResult:
    if genre.flexible_outline and brief.guidelines.strip():
        return CheckResult(
            name="required_sections",
            passed=True,
            details=["skipped: the call text sets the sections"],
        )
    if genre.flexible_outline and outline is not None:
        expected = [p.title for p in outline.sections]
    else:
        expected = [s.title for s in genre.sections if s.required]
    present = {s.title.casefold() for s in sections}
    missing = [t for t in expected if t.casefold() not in present]
    return CheckResult(
        name="required_sections",
        passed=not missing,
        details=[f"missing section: {t}" for t in missing],
    )


def _check_must_include(sections: Sequence[Section], *, brief: Brief) -> CheckResult:
    text = plain_text(sections).casefold()
    missing = [item for item in brief.must_include if item.casefold() not in text]
    return CheckResult(
        name="must_include",
        passed=not missing,
        details=[f"missing: {item}" for item in missing],
    )


def _check_headings(sections: Sequence[Section]) -> list[CheckResult]:
    out = [
        CheckResult(
            name="headings_in_bodies",
            passed=False,
            details=[f"{s.title}: heading line {line.strip()!r} inside the body"],
            section_id=s.id,
        )
        for s in sections
        for line in s.body.splitlines()
        if _HEADING_LINE.match(line)
    ]
    return out or [CheckResult(name="headings_in_bodies", passed=True, details=[])]


def _check_fact_markers(
    sections: Sequence[Section], *, facts: Sequence[Fact]
) -> list[CheckResult]:
    known = {f.id for f in facts}
    out: list[CheckResult] = []
    for s in sections:
        unknown = sorted({m for m in MARKER_RE.findall(s.body) if m not in known})
        if unknown:
            out.append(
                CheckResult(
                    name="fact_markers",
                    passed=False,
                    details=[f"{s.title}: unknown fact id {m}" for m in unknown],
                    section_id=s.id,
                )
            )
    return out or [CheckResult(name="fact_markers", passed=True, details=[])]


def _check_to_fill(sections: Sequence[Section]) -> CheckResult:
    found = [
        f"{s.title}: {m.group(0).strip()}"
        for s in sections
        for m in _TO_FILL.finditer(s.body)
    ]
    return CheckResult(name="to_fill", passed=not found, details=found)


def _check_grade_level(
    sections: Sequence[Section], *, genre: Genre
) -> list[CheckResult]:
    ceiling = genre.max_grade_level
    if ceiling is None:
        return []
    out: list[CheckResult] = []
    for s in sections:
        text = strip_markers(s.body)
        if word_count(text) < MIN_WORDS_FOR_GRADE:
            continue
        grade = analyze(text).readability.flesch_kincaid_grade
        if grade > ceiling:
            out.append(
                CheckResult(
                    name="grade_level",
                    passed=False,
                    details=[
                        f"{s.title}: reading grade {grade:.1f}, ceiling {ceiling:g}; "
                        "use shorter sentences and plainer words"
                    ],
                    section_id=s.id,
                )
            )
    return out or [CheckResult(name="grade_level", passed=True, details=[])]


def _style_info(sections: Sequence[Section]) -> CheckResult:
    """Style counts for the report only. Style never drives idea revision."""
    text = "\n\n".join(strip_markers(s.body) for s in sections)
    if not text.strip():
        return CheckResult(name="style", passed=True, details=[])
    result = analyze(text)
    return CheckResult(
        name="style",
        passed=True,
        details=[
            f"weasel words: {len(result.weasel_words)}",
            f"passive voice: {len(result.passive_voice)}",
        ],
    )


def run_deterministic_checks(
    sections: Sequence[Section],
    *,
    brief: Brief,
    genre: Genre,
    outline: Outline | None,
    facts: Sequence[Fact],
) -> list[CheckResult]:
    """Run every deterministic check over the drafted sections.

    Per-section failures carry ``section_id``; document-level ones do not.
    """
    ordered = sorted(sections, key=lambda s: s.order)
    results: list[CheckResult] = [
        _check_total_words(ordered, brief=brief),
        _check_required_sections(ordered, brief=brief, genre=genre, outline=outline),
        _check_must_include(ordered, brief=brief),
    ]
    results += _check_headings(ordered)
    results += _check_fact_markers(ordered, facts=facts)
    results += _check_grade_level(ordered, genre=genre)
    results += _check_section_words(ordered, outline=outline)
    results.append(_check_to_fill(ordered))
    results.append(_style_info(ordered))
    return results
