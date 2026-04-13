"""Similarity validation agent — generic LLM judge for group refinement.

Completely domain-agnostic: works for any type of text item (assertions,
uncertainties, caveats, or anything else). Does NOT mention epistemic
concepts, claims, evidence, or any domain-specific terminology.

Architecture: Layer 1 (framework-agnostic, pure Pydantic)
"""

from .output_models import ValidateGroupOutput
from . import AgentDefinition, register_agent

VALIDATE_GROUP_PROMPT = """\
# Group Validation

You are grouping text items by meaning. You will receive a set of items that were \
grouped together because they appear semantically similar. Your job is to review them \
and decide whether they all express the same core idea.

## Rules

1. **Be inclusive.** Items that express the same idea in different words MUST stay together. \
Surface-level differences in phrasing, vocabulary, or sentence structure do not warrant splitting.

2. **Only split on genuine topic differences.** Split items into separate subgroups ONLY when \
they are about fundamentally different subjects or make substantively different points.

3. **Preserve group integrity when possible.** If you are unsure whether items belong together, \
keep them together. The cost of a false merge (redundancy) is lower than the cost of a false \
split (losing the connection between related items).

## Output format

Return subgroups as lists of item numbers (1-based). Examples:

- All items belong together: `[[1, 2, 3, 4, 5]]`
- Items split into two themes: `[[1, 3, 5], [2, 4]]`
- Three distinct sub-themes: `[[1, 2], [3, 4], [5]]`

Every item number must appear in exactly one subgroup. Do not drop any items.
"""

VALIDATE_GROUP = register_agent(
    AgentDefinition(
        name="epistemic_validate_group",
        prompt=VALIDATE_GROUP_PROMPT,
        output_model=ValidateGroupOutput,
        retries=2,
        output_retries=3,
    )
)
