"""The default investigator: read sections of the document itself.

Calls ``investigate_agent`` with the hypothesis + 1–3 relevant sections
and converts the agent's flat ``InvestigationOutput`` into an
``InvestigationOutcome`` for the InvestigateLoop. Quote anchoring is
done here via the chunker's tiered ``find_anchor`` (exact → whitespace
→ fuzzy) so the agent's verbatim quote strings get located inside the
section text and turned into proper ``Quote`` objects with offsets.

If the agent emits a quote the section text doesn't contain at all
(LLM fabrication), the quote is dropped silently. The Challenge phase
catches findings that consequently end up unsupported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from andamentum.chunker.validation import find_anchor

from ..agents import build_pydantic_ai_agent
from ..agents import InvestigationOutput
from ..schemas import Finding, Hypothesis, Quote

if TYPE_CHECKING:
    from . import InvestigationOutcome
    from ..deps import ReviewDeps
    from ..state import ReviewState
    from ..structural.types import SectionRef


# Cap on context sent per investigate call. The skim_agent is supposed
# to limit relevant_section_ids to 1–5; this is a defence against runaway.
_MAX_SECTIONS_PER_INVESTIGATION = 4


async def investigate_internal(
    hypothesis: Hypothesis,
    state: "ReviewState",
    deps: "ReviewDeps",
) -> "InvestigationOutcome":
    """Resolve a hypothesis by reading the cited sections via investigate_agent."""
    from . import InvestigationOutcome

    sections_by_id = {s.id: s for s in state.sections}
    relevant: list["SectionRef"] = []
    for sid in hypothesis.relevant_section_ids[:_MAX_SECTIONS_PER_INVESTIGATION]:
        sec = sections_by_id.get(sid)
        if sec is not None:
            relevant.append(sec)

    if not relevant:
        # Nothing to read — treat as unfounded with a clear reason.
        return InvestigationOutcome.unfounded(
            "no relevant sections were resolvable from hypothesis.relevant_section_ids"
        )

    prompt = _build_prompt(hypothesis, state.document_map, relevant)
    agent = build_pydantic_ai_agent("investigate", deps.model)
    result = await agent.run(prompt)
    # Agent.run returns AgentRunResult[BaseModel] generically; narrow with cast.
    from typing import cast

    output = cast(InvestigationOutput, result.output)
    state.llm_calls += 1

    if output.decision == "unfounded":
        return InvestigationOutcome.unfounded(output.unfounded_reason or "no support")

    if output.decision == "needs_subhypotheses":
        sub_hypotheses = [
            Hypothesis(
                text=t,
                priority=hypothesis.priority,
                relevant_section_ids=hypothesis.relevant_section_ids,
                investigation_type=hypothesis.investigation_type,
            )
            for t in output.sub_hypothesis_texts
            if t and t.strip()
        ]
        return InvestigationOutcome.split(sub_hypotheses)

    # decision == "finding"
    quotes = _locate_quotes(output.finding_quotes, relevant)
    finding = Finding(
        title=output.finding_title or hypothesis.text,
        severity=output.finding_severity,
        confidence=output.finding_confidence,
        rationale=output.finding_rationale,
        quotes=quotes,
        sections_involved=output.finding_sections or [s.id for s in relevant],
        source="investigate",
    )
    return InvestigationOutcome.found(finding, raw_quotes=output.finding_quotes)


# ── Prompt construction ─────────────────────────────────────────────────


def _build_prompt(
    hypothesis: Hypothesis,
    document_map,
    relevant_sections: "list[SectionRef]",
) -> str:
    """Compose the user message sent to investigate_agent."""
    map_lines = "\n".join(
        f"  • {c.section_id} — {c.title}: {c.one_line_gist}"
        for c in document_map
    )
    sections_text = "\n\n".join(
        f"--- BEGIN {s.id} ({s.title}) ---\n{s.text}\n--- END {s.id} ---"
        for s in relevant_sections
    )
    return f"""HYPOTHESIS:
{hypothesis.text}
(priority: {hypothesis.priority}; relevant section ids: {", ".join(hypothesis.relevant_section_ids)})

DOCUMENT MAP (overview, not for direct quoting):
{map_lines}

RELEVANT SECTION TEXT (quote VERBATIM from these only):
{sections_text}

Decide: finding | unfounded | needs_subhypotheses. Fill the matching
fields of InvestigationOutput."""


# ── Quote anchoring ─────────────────────────────────────────────────────


def _locate_quotes(
    raw_quotes: list[str],
    relevant_sections: "list[SectionRef]",
) -> list[Quote]:
    """Find each verbatim quote in one of the cited sections.

    Uses ``chunker.validation.find_anchor`` (tiered exact → whitespace
    normalised → fuzzy) so minor formatting differences don't drop
    legitimate quotes. Quotes that don't match any section are silently
    omitted; the Challenge phase will flag findings whose evidence has
    evaporated.
    """
    out: list[Quote] = []
    for raw in raw_quotes:
        if not raw or not raw.strip():
            continue
        for section in relevant_sections:
            match = find_anchor(raw, section.text, search_from=0)
            if match is None:
                continue
            out.append(
                Quote(
                    section_id=section.id,
                    char_start=match.start,
                    char_end=match.end,
                    text=section.text[match.start : match.end],
                )
            )
            break  # first matching section wins
    return out
