"""Numbers and result phrases in the abstract must occur in the body (spec section 4, Tier 0).

The abstract is the paragraphs under a heading whose text is "abstract" (any
case), or paragraphs whose style is "Abstract". Everything else that is not a
heading is the body. No abstract found: no diagnostics.

Numbers: each number in the abstract (optional sign, decimals, thousands
commas) must occur with the same value in the body. A percent sign or a unit
from a short fixed list (mg, kg, ml, mm, cm, years, months, weeks, days, hours,
min, ms, Hz, ...) counts: the body must give that value with that unit
(``45 percent`` equals ``45%``). One error per missing number. Skipped: years
(1900-2099 as bare integers) and digits glued to letters ("COVID-19", "H2O",
"CI95").

Result phrases: only the word "significant". An abstract that claims a
significant result needs "significant" un-negated somewhere in the body; one
that says "no/not/non-significant" needs a negated form there. Warning.

Coverage limits: table and figure contents are not in the paragraphs, so a
number that appears only in a table is reported missing. Spelled-out numbers
("forty") and number words are not compared. Rounding differences (45.2 in the
abstract, 45.17 in the body) are reported missing.
"""

from __future__ import annotations

import re

from ..kernel import (
    CheckContext,
    CheckSpec,
    Diagnostic,
    DocumentView,
    register,
)

_UNITS = (
    "mg|kg|g|ml|mL|l|mm|cm|km|m|ms|s|sec|min|h|hr|hours?|days?|weeks?|months?|years?|"
    "Hz|kHz|mmHg|mol|mM|uM|nM|kb|bp|GB|MB"
)
_NUMBER = re.compile(
    rf"(?P<neg>(?<=[\s(\[])[-−])?(?<![\w.])(?<![A-Za-z]-)"
    rf"(?P<num>\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    rf"(?![\w]*\d)"
    rf"(?:\s?(?P<unit>%|percent\b|(?:{_UNITS})\b))?"
)
_SIGNIFICANT = re.compile(
    r"\b(?P<neg>(?:no|not|non)[- ]?(?:statistically[- ])?)?significant(?:ly)?\b",
    re.IGNORECASE,
)


def _norm(match: re.Match[str]) -> tuple[str, str]:
    value = match["num"].replace(",", "")
    if match["neg"]:
        value = "-" + value
    unit = (match["unit"] or "").lower()
    if unit == "percent":
        unit = "%"
    return value, unit


def _numbers(text: str) -> list[tuple[str, str, int, int]]:
    found = []
    for match in _NUMBER.finditer(text):
        value, unit = _norm(match)
        if re.fullmatch(r"(19|20)\d\d", value) and not unit:
            continue
        found.append((value, unit, match.start(), match.end()))
    return found


class AbstractBodyCheck:
    spec = CheckSpec(id="abstract_body", tier=0, scope="document", needs=[])
    version = "1"

    def run(self, view: DocumentView, ctx: CheckContext) -> list[Diagnostic]:
        abstract_ids: set[str] = set()
        for heading, ids in view.sections():
            if heading.strip().lower() == "abstract":
                abstract_ids.update(ids)
        for paragraph in view.paragraphs:
            if paragraph.style.strip().lower() == "abstract":
                abstract_ids.add(paragraph.id)
        if not abstract_ids:
            return []
        body = [
            p
            for p in view.paragraphs
            if p.id not in abstract_ids and p.heading_level is None
        ]
        body_numbers = {
            (value, unit) for p in body for value, unit, _s, _e in _numbers(p.text)
        }
        body_values = {value for value, _unit in body_numbers}
        body_text = "\n".join(p.text for p in body)
        out: list[Diagnostic] = []
        for paragraph in view.paragraphs:
            if paragraph.id not in abstract_ids:
                continue
            for value, unit, start, end in _numbers(paragraph.text):
                present = (
                    (value, unit) in body_numbers if unit else value in body_values
                )
                if present:
                    continue
                shown = paragraph.text[start:end]
                out.append(
                    self._diag(
                        "error",
                        f"Abstract number {shown!r} does not occur in the body.",
                        paragraph.id,
                        start,
                        end,
                        shown,
                    )
                )
            out.extend(self._phrases(paragraph.id, paragraph.text, body_text))
        return out

    def _phrases(
        self, paragraph_id: str, text: str, body_text: str
    ) -> list[Diagnostic]:
        body_hits = list(_SIGNIFICANT.finditer(body_text))
        body_positive = any(not m["neg"] for m in body_hits)
        body_negative = any(m["neg"] for m in body_hits)
        out: list[Diagnostic] = []
        for match in _SIGNIFICANT.finditer(text):
            negated = bool(match["neg"])
            if body_negative if negated else body_positive:
                continue
            kind = "negated 'significant'" if negated else "'significant'"
            out.append(
                self._diag(
                    "warning",
                    f"Abstract uses {kind} but the body has no matching statement.",
                    paragraph_id,
                    match.start(),
                    match.end(),
                    match.group(0),
                )
            )
        return out

    def _diag(
        self,
        severity: str,
        message: str,
        paragraph_id: str,
        start: int,
        end: int,
        evidence: str,
    ) -> Diagnostic:
        return Diagnostic(
            check=self.spec.id,
            tier=self.spec.tier,
            severity=severity,  # type: ignore[arg-type]
            paragraph_id=paragraph_id,
            start=start,
            end=end,
            message=message,
            evidence=[evidence],
        )


register(AbstractBodyCheck())
