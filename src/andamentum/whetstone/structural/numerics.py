"""Extract numeric claims from sections.

Phase 1 covers four high-value categories of numeric claim:
  • sample size:   ``N=50``, ``n = 250``  (note: case-sensitive `N` vs `n`
                    is significant — both appear and we keep the case)
  • percentage:   ``42%``, ``42.5 %``, ``+12%``
  • p-value:      ``p<0.05``, ``p = 0.001``, ``p < .05``
  • count:        ``5 participants``, ``three subjects`` — *not* yet;
                    requires NLP. Reserved as a kind for later expansion.

The synthesiser uses the (kind, value) tuples to flag inconsistencies
(e.g. one section says ``N=50``, another ``N=48``).
"""

from __future__ import annotations

import re

from .types import NumericClaim, SectionRef

# N=50 / n = 250 / N = 1,000
_SAMPLE_SIZE_RE = re.compile(r"\b([Nn])\s*=\s*([\d,]+)\b")
# 42% / 42.5% / +12 %
_PERCENT_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*%")
# p < 0.05 / p = 0.001 / p < .05
_P_VALUE_RE = re.compile(r"\bp\s*([<>=≤≥])\s*(0?\.\d+|\d+\.\d+|\d+)\b", re.IGNORECASE)


def extract_numeric_claims(sections: list[SectionRef]) -> list[NumericClaim]:
    """Walk every section, extract every recognised numeric claim."""
    out: list[NumericClaim] = []
    for section in sections:
        out.extend(_extract_one_section(section))
    return out


def _extract_one_section(section: SectionRef) -> list[NumericClaim]:
    out: list[NumericClaim] = []
    text = section.text

    for m in _SAMPLE_SIZE_RE.finditer(text):
        # Preserve case of N/n — they're often distinct (subjects vs
        # within-subject observations).
        prefix = m.group(1)
        value = m.group(2).replace(",", "")
        out.append(
            NumericClaim(
                raw=m.group(0),
                kind="sample_size",
                value=f"{prefix}={value}",
                section_id=section.id,
                char_start=m.start(),
                char_end=m.end(),
            )
        )

    for m in _PERCENT_RE.finditer(text):
        out.append(
            NumericClaim(
                raw=m.group(0),
                kind="percentage",
                value=m.group(1),
                section_id=section.id,
                char_start=m.start(),
                char_end=m.end(),
            )
        )

    for m in _P_VALUE_RE.finditer(text):
        op = m.group(1)
        val = m.group(2)
        # Normalise ".05" → "0.05" for comparison.
        if val.startswith("."):
            val = "0" + val
        out.append(
            NumericClaim(
                raw=m.group(0),
                kind="p_value",
                value=f"{op}{val}",
                section_id=section.id,
                char_start=m.start(),
                char_end=m.end(),
            )
        )

    return out
