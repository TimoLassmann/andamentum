"""R13 — Omit needless words. Agent node, one call per section.

Kind: agent
Reads: state.section
Writes: state.findings, state.demands
Model: ``deps.model_for_rule[13]`` or ``deps.model_default``
Output: ``OmitNeedlessWordsReport`` (list of ``OmitNeedlessWordsViolation``)
Successor: ResolveDemands
Source: Strunk, *Elements of Style*, Ch III §13

The agent receives the whole section as one prompt and returns every
needless-words violation it finds, each classified into a closed-set
``category``. The empty list is the "no violations" answer.

Each violation's ``span`` is matched back against ``section.text`` via
``andamentum.chunker.validation.find_anchor`` so the finding carries
verbatim char offsets even when the model paraphrases slightly —
unanchorable spans are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from pydantic_graph import BaseNode, GraphRunContext

from andamentum.chunker.validation import find_anchor
from andamentum.core.agents import AgentDefinition

from ....schemas import Finding
from ..kinds import NodeKind
from ..models import (
    OmitNeedlessWordsReport,
    OmitNeedlessWordsViolation,
    StrunkDemand,
    StrunkFinding,
)
from ..state import StrunkLensDeps, StrunkLensState

if TYPE_CHECKING:
    from .resolve_demands import ResolveDemands


_R13_PROMPT = """\
You apply ONE rule from Strunk's *Elements of Style*:

**Rule 13. Omit needless words.** Vigorous writing is concise. A
sentence should contain no unnecessary words, a paragraph no
unnecessary sentences.

You receive ONE SECTION of prose. Find every clear-cut needless-words
violation. Pick ONE category per violation from the closed list. For
each issue, return:

* `span`: the EXACT verbatim substring from the section text — copy
  it character-for-character; do not paraphrase, do not summarise.
* `category`: one of the closed-set values below.
* `suggested_deletion`: the FULL rewritten sentence with the
  needless words removed (not a diff).
* `confidence`: how sure you are (low / medium / high).

Return an empty list if the section reads cleanly.

Categories (closed set):

* `throat-clearing` — empty preamble that delays the point.
  Examples: "the reason that ... is that ...", "it is the case that
  ...", "the fact that ...".
* `redundancy` — repeats the meaning. Examples: "advance planning",
  "consensus of opinion", "general consensus", "ATM machine".
* `weak-qualifier` — "rather", "very", "little", "pretty" used as
  intensifiers that add no information.
* `filler-prepositional` — "of a X nature" used in place of an
  adjective ("of a fragile nature" → "fragile").
* `other` — anything else clearly violating Rule 13.

Examples:

Section: "The reason that I came is that I wanted to see you. She
moved quickly."
→ violations=[
    OmitNeedlessWordsViolation(
      span="The reason that I came is that",
      category="throat-clearing",
      suggested_deletion="I came because I wanted to see you.",
      confidence="high"),
  ]

Section: "The reaction was rather slow. The result is interesting."
→ violations=[
    OmitNeedlessWordsViolation(
      span="rather",
      category="weak-qualifier",
      suggested_deletion="The reaction was slow.",
      confidence="high"),
  ]

Section: "She moved quickly. He worked hard. The cat sat on the mat."
→ violations=[]

Guidance:

* Be conservative. Short, direct sentences usually have NO violation.
* DO NOT include a violation whose ``span`` does not appear verbatim
  in the input section text.
* `suggested_deletion` is the full rewritten sentence — not a diff,
  not a fragment.
"""


OMIT_NEEDLESS_WORDS_AGENT = AgentDefinition(
    name="strunk.r13_omit_needless_words",
    prompt=_R13_PROMPT,
    output_model=OmitNeedlessWordsReport,
    retries=2,
    output_retries=3,
)


def _violation_to_finding(
    v: OmitNeedlessWordsViolation,
    section_text: str,
) -> StrunkFinding | None:
    """Anchor a violation back to the section text and shape a StrunkFinding.

    Returns ``None`` if the LLM's span cannot be located in
    ``section_text``."""
    if not v.span.strip():
        return None
    match = find_anchor(v.span, section_text, search_from=0)
    if match is None:
        return None
    verbatim = section_text[match.start : match.end]
    rationale_parts = [
        "Strunk Ch III §13 — Omit needless words.",
        f"Category: {v.category}.",
        f"Offending span: {verbatim!r}.",
    ]
    return StrunkFinding(
        rule_number=13,
        rule_name="omit-needless-words",
        char_start=match.start,
        char_end=match.end,
        title=f"R13: Needless words ({v.category})",
        rationale=" ".join(rationale_parts),
        severity="minor",
        confidence=v.confidence,
        category=f"r13-{v.category}",
        span_text=verbatim,
        suggested_replacement=v.suggested_deletion,
    )


@dataclass
class R13OmitNeedlessWords(
    BaseNode[StrunkLensState, StrunkLensDeps, list[Finding]]
):
    """R13 — One LLM call per section. Returns a list of violations."""

    kind: ClassVar[NodeKind] = NodeKind.AGENT
    reads: ClassVar[frozenset[str]] = frozenset({"section"})
    writes: ClassVar[frozenset[str]] = frozenset({"findings", "demands"})
    model: ClassVar[str] = "ollama:gemma3:4b-it-q4_K_M"
    output_model: ClassVar[type] = OmitNeedlessWordsReport
    rule_number: ClassVar[int] = 13
    rule_source: ClassVar[str] = "Strunk Elements of Style, Ch III §13"

    async def run(
        self,
        ctx: GraphRunContext[StrunkLensState, StrunkLensDeps],
    ) -> "ResolveDemands":
        from .resolve_demands import ResolveDemands

        if ctx.deps.executor is None:
            return ResolveDemands()
        section_text = ctx.state.section.text
        try:
            report = await ctx.deps.executor.run(
                OMIT_NEEDLESS_WORDS_AGENT,
                section_text=section_text,
            )
        except Exception:
            ctx.state.demands.append(
                StrunkDemand(rule="r13", reason="executor_exception")
            )
            return ResolveDemands()
        if not isinstance(report, OmitNeedlessWordsReport):
            ctx.state.demands.append(
                StrunkDemand(rule="r13", reason="schema_validation_failed")
            )
            return ResolveDemands()
        for violation in report.violations:
            finding = _violation_to_finding(violation, section_text)
            if finding is not None:
                ctx.state.findings.append(finding)
        return ResolveDemands()
