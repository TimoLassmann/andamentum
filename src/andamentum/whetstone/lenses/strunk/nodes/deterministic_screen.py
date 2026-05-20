"""DeterministicScreen — regex / dictionary rule checks. No LLM.

Kind: deterministic
Reads: state.section
Writes: state.findings
Successor: R11ActiveVoice

Phase A implements one rule:

* **R2** — Series comma (Oxford comma). In a series of three or more
  terms with a single conjunction (and/or/nor/but), use a comma after
  each term except the last.

Future phases extend this node with:

* R1 — possessive singular ``'s``
* R5 — comma splice (joining independent clauses with a comma)
* Chapter V — ``Words and Expressions Commonly Misused`` (~180 entries)
* Chapter VI — common misspellings
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from pydantic_graph import BaseNode, GraphRunContext

from ....schemas import Finding
from ..kinds import NodeKind
from ..models import StrunkFinding
from ..state import StrunkLensDeps, StrunkLensState

if TYPE_CHECKING:
    from .r11_active_voice import R11ActiveVoice


# R2 — Series comma detector.
#
# Matches a series of 3+ items followed by a coordinating conjunction
# *without* an Oxford comma. The pre-conjunction repetition consumes
# greedily, so when the Oxford comma is present (",  and X"), the
# trailing `\s+(conjunction)` fails to match and the regex returns no
# hit. False-positive risk on prose where comma usage is structural
# (lists, parentheticals); kept loose for Phase A and tightened as
# fixtures accumulate.
_SERIES_NO_OXFORD = re.compile(
    r"\b(\w+(?:,\s+\w+){1,})\s+(and|or|nor|but)\s+\w+",
    re.IGNORECASE,
)


def _insert_oxford(span_text: str) -> str:
    """Insert a comma before the conjunction in a matched series."""
    return re.sub(
        r"(\s+)(and|or|nor|but)(\s+)",
        r",\1\2\3",
        span_text,
        count=1,
    )


def _check_series_comma(text: str) -> list[StrunkFinding]:
    """Return one StrunkFinding per Oxford-comma violation in ``text``.

    Char offsets in the returned findings are relative to ``text`` —
    typically the whole section's ``section.text``."""
    findings: list[StrunkFinding] = []
    for m in _SERIES_NO_OXFORD.finditer(text):
        span_text = m.group()
        findings.append(
            StrunkFinding(
                rule_number=2,
                rule_name="series-comma",
                char_start=m.start(),
                char_end=m.end(),
                title="R2: Missing Oxford comma in series",
                rationale=(
                    "Strunk Ch II §2 — in a series of three or more terms "
                    "with a single conjunction, use a comma after each "
                    "term except the last."
                ),
                severity="minor",
                confidence="high",
                category="r2-series-comma",
                span_text=span_text,
                suggested_replacement=_insert_oxford(span_text),
            )
        )
    return findings


@dataclass
class DeterministicScreen(
    BaseNode[StrunkLensState, StrunkLensDeps, list[Finding]]
):
    """Run every deterministic Strunk rule across the section's text.

    Each rule is implemented as a pure helper that takes the section
    text and returns ``list[StrunkFinding]``. The node iterates the
    rules, extends ``state.findings``, and hands off to the first
    LLM-backed rule node.
    """

    kind: ClassVar[NodeKind] = NodeKind.DETERMINISTIC
    reads: ClassVar[frozenset[str]] = frozenset({"section"})
    writes: ClassVar[frozenset[str]] = frozenset({"findings"})

    async def run(
        self,
        ctx: GraphRunContext[StrunkLensState, StrunkLensDeps],
    ) -> "R11ActiveVoice":
        ctx.state.findings.extend(_check_series_comma(ctx.state.section.text))
        from .r11_active_voice import R11ActiveVoice

        return R11ActiveVoice()
