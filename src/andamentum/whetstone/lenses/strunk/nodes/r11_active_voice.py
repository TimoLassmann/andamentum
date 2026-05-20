"""R11 — Use the active voice. Agent node, one call per section.

Kind: agent
Reads: state.section
Writes: state.findings, state.demands
Model: ``deps.model_for_rule[11]`` or ``deps.model_default``
Output: ``ActiveVoiceReport`` (list of ``ActiveVoiceViolation``)
Successor: R13OmitNeedlessWords
Source: Strunk, *Elements of Style*, Ch III §11

The agent receives the whole section as one prompt and returns every
passive-voice issue it finds as a list of ``ActiveVoiceViolation``
objects. The empty list is the "no violations" answer. Each
violation's ``span`` is matched back against ``section.text`` via
``andamentum.chunker.validation.find_anchor`` so the finding carries
verbatim char offsets even when the model paraphrases slightly —
unanchorable spans are dropped.

A ``StrunkDemand`` is appended only on executor exceptions or shape
mismatches; per-section calls have no natural "abstained" state (an
empty list IS the no-op answer).
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
    ActiveVoiceReport,
    ActiveVoiceViolation,
    StrunkDemand,
    StrunkFinding,
)
from ..state import StrunkLensDeps, StrunkLensState

if TYPE_CHECKING:
    from .r13_omit_needless_words import R13OmitNeedlessWords


_R11_PROMPT = """\
You apply ONE rule from Strunk's *Elements of Style*:

**Rule 11. Use the active voice.** The active voice is usually more
direct and vigorous than the passive: "I shall always remember my
first visit to Boston" is preferable to "My first visit to Boston
will always be remembered by me."

You receive ONE SECTION of prose. Find every clear-cut passive-voice
construction that would read better in the active voice. For each
issue, return:

* `span`: the EXACT verbatim substring from the section text — copy
  it character-for-character; do not paraphrase, do not summarise.
* `suggested_active_rewrite`: the FULL sentence in active voice.
* `confidence`: how sure you are (low / medium / high).

Return an empty list if the section has no passive-voice issues.

Examples:

Section: "The reports were submitted by the committee. The cat sat
on the mat. Errors were made."
→ violations=[
    ActiveVoiceViolation(span="were submitted by the committee",
        suggested_active_rewrite="The committee submitted the reports.",
        confidence="high"),
    ActiveVoiceViolation(span="were made",
        suggested_active_rewrite="We made errors.",
        confidence="medium"),
  ]

Section: "The cat sat on the mat. The result is interesting."
→ violations=[]

Guidance:

* Adjectival ``be`` ("is interesting", "was happy") is NOT passive.
  Do NOT include it.
* Intransitive perfect ("has gone", "has been to Paris") is NOT
  passive.
* When the agent of the passive is elided ("Errors were made"), the
  rewrite has to introduce one — note this with lower confidence.
* Be conservative. If a sentence is borderline, leave it out.
* DO NOT include a violation whose ``span`` does not appear verbatim
  in the input section text.
"""


ACTIVE_VOICE_AGENT = AgentDefinition(
    name="strunk.r11_active_voice",
    prompt=_R11_PROMPT,
    output_model=ActiveVoiceReport,
    retries=2,
    output_retries=3,
)


def _violation_to_finding(
    v: ActiveVoiceViolation,
    section_text: str,
) -> StrunkFinding | None:
    """Anchor a violation back to the section text and shape a StrunkFinding.

    Returns ``None`` if the LLM's span cannot be located in
    ``section_text`` (a fabricated quote). The on-disk
    ``StrunkFinding.span_text`` is always the verbatim source slice,
    not the model's input span — so even when the chunker's fuzzy
    matcher kicks in, downstream readers get the canonical source
    text.
    """
    if not v.span.strip():
        return None
    match = find_anchor(v.span, section_text, search_from=0)
    if match is None:
        return None
    verbatim = section_text[match.start : match.end]
    rationale_parts = [
        "Strunk Ch III §11 — Use the active voice.",
        f"Passive phrase: {verbatim!r}.",
    ]
    return StrunkFinding(
        rule_number=11,
        rule_name="active-voice",
        char_start=match.start,
        char_end=match.end,
        title="R11: Prefer the active voice",
        rationale=" ".join(rationale_parts),
        severity="minor",
        confidence=v.confidence,
        category="r11-active-voice",
        span_text=verbatim,
        suggested_replacement=v.suggested_active_rewrite,
    )


@dataclass
class R11ActiveVoice(BaseNode[StrunkLensState, StrunkLensDeps, list[Finding]]):
    """R11 — One LLM call per section. Returns a list of violations."""

    kind: ClassVar[NodeKind] = NodeKind.AGENT
    reads: ClassVar[frozenset[str]] = frozenset({"section"})
    writes: ClassVar[frozenset[str]] = frozenset({"findings", "demands"})
    model: ClassVar[str] = "ollama:gemma3:4b-it-q4_K_M"
    output_model: ClassVar[type] = ActiveVoiceReport
    rule_number: ClassVar[int] = 11
    rule_source: ClassVar[str] = "Strunk Elements of Style, Ch III §11"

    async def run(
        self,
        ctx: GraphRunContext[StrunkLensState, StrunkLensDeps],
    ) -> "R13OmitNeedlessWords":
        from .r13_omit_needless_words import R13OmitNeedlessWords

        if ctx.deps.executor is None:
            return R13OmitNeedlessWords()
        section_text = ctx.state.section.text
        try:
            report = await ctx.deps.executor.run(
                ACTIVE_VOICE_AGENT,
                section_text=section_text,
            )
        except Exception:
            ctx.state.demands.append(
                StrunkDemand(rule="r11", reason="executor_exception")
            )
            return R13OmitNeedlessWords()
        if not isinstance(report, ActiveVoiceReport):
            ctx.state.demands.append(
                StrunkDemand(rule="r11", reason="schema_validation_failed")
            )
            return R13OmitNeedlessWords()
        for violation in report.violations:
            finding = _violation_to_finding(violation, section_text)
            if finding is not None:
                ctx.state.findings.append(finding)
        return R13OmitNeedlessWords()
