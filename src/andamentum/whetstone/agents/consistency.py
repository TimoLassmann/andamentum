"""Consistency-reviewer agent — reading-comprehension pass over a draft.

Only handles inconsistencies that require reading comprehension:
numbers disagreeing across sections, terminology drift, claim emphasis
shifting. Mechanical checks (figure order, acronym first-use,
citation resolution) live in consistency_scanners.
"""

from __future__ import annotations

from . import AgentDefinition, register_agent
from .output_models import ConsistencyReviewOutput

_CONSISTENCY_PROMPT = """\
# Internal consistency reviewer

You are reviewing a draft the author wrote themselves, to catch internal
inconsistencies before submission. This is NOT peer review.

## Focus on reading-comprehension issues

- Numbers or statistics that disagree between abstract, results, and
  conclusions (e.g. "n=50" in abstract, "n=48" in results)
- Terminology drift — the same concept called different names across
  sections (e.g. "cohort" and "sample" used interchangeably)
- Claims emphasized differently across sections (abstract headlines
  finding A, conclusion headlines finding B)
- Tense, voice, or person shifts across sections
- Contradicting statements about methods, scope, or population

## Do NOT comment on

- Figure numbering order (handled by a scanner)
- Reference-list completeness or formatting (handled elsewhere)
- Acronym first-use definition (handled by a scanner)
- Grammar, typos, style, or word choice (handled by the edit task)

## Output

For each issue you find, emit a DocumentIssue with:
- `issue_type`: "major" for real contradictions; "minor" for drift;
  "suggestion" for minor polish
- `category`: "consistency"
- `title`: brief, specific
- `description`: what the inconsistency is and where — quote the
  excerpts if possible
- `recommendation`: concrete fix
- `location`: section names where the inconsistency occurs
- `agent_type`: "consistency_reviewer"
- `confidence`: 0.0–1.0

Quality over quantity. Emit 0–8 issues. Only flag things you are
confident about.
"""


register_agent(
    AgentDefinition(
        name="consistency_reviewer",
        prompt=_CONSISTENCY_PROMPT,
        output_model=ConsistencyReviewOutput,
        retries=2,
    )
)
