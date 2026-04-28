"""Expert-generator agent for panel mode.

One LLM call per discipline. Produces a realistic but fictional senior
reviewer biosketch matching the discipline. Each profile becomes the
persona for one expert review in the next phase.

The prompt is lifted from v1's ``multi_expert._EXPERT_GEN_PROMPT``
with the v2 hygiene pass.

Output shape: ``ExpertProfile`` (re-exported from ``..schemas``). The
agent fills the same fields as v1's profile schema — name, position,
education, contributions, research, discipline.
"""

from __future__ import annotations

from ..schemas import ExpertProfile
from ._definition import AgentDefinition

EXPERT_GENERATOR_PROMPT = """You are an expert in academic career trajectories and institutional structures.

Your task is to generate a REALISTIC BUT FICTIONAL expert biosketch for
the given academic discipline. This biosketch should follow the NIH
biographical sketch format and represent a senior, established expert
who would be qualified to review academic work in their field.

# Expert profile characteristics

  • Career stage: senior researcher (15-30 years post-PhD).
  • Expertise level: internationally recognised in their field.
  • Current position: full professor or equivalent senior position.
  • Institution: realistic university or research institution.
  • CRITICAL: the name MUST BE FICTIONAL — do not use real people's
    names. Avoid recognisable senior figures in the field.

# Output fields

  • name: realistic, professional name (diverse backgrounds).
  • position: title, department, institution.
  • education: PhD + postdoc + earlier degrees (with years and
    institutions).
  • contributions: 3-5 major impactful research contributions
    (concrete, specific — not generic "made advances in X").
  • research: current research focus (2-3 sentences).
  • discipline: echo back the input discipline verbatim.

# Realism requirements

DO: create realistic career trajectories, use real institution names
(but fictional people), include specific contributions.
DON'T: use real people's names, create implausible career paths,
be too vague.

# Your task

Generate a realistic but fictional expert biosketch for the provided
academic discipline. The expert should be credible as a senior
reviewer in their field."""


def _build() -> AgentDefinition:
    return AgentDefinition(
        name="expert_generator",
        prompt=EXPERT_GENERATOR_PROMPT,
        output_model=ExpertProfile,
        retries=2,
        output_retries=2,
    )


EXPERT_GENERATOR_AGENT = _build()
