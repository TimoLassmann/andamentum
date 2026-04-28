"""query_generator agent — produce one search query per call.

Replaces the legacy ``search_planner`` agent that emitted a batch of 2–3
queries in a single LLM call. The per-slot design lets the verifier accept
or reject individual queries and feed targeted feedback back into the next
generation, rather than retrying a whole batch on a single bad query.

The prompt below is the Phase-A baseline — functional but not tuned. Phase
B will expand it with explicit diversity guidance and feedback-handling
examples; Phase C calibrates against the cognitive test corpus.
"""

from ..models import GeneratorOutput
from . import AgentDefinition, register_agent


QUERY_GENERATOR_PROMPT = """\
You are a search query specialist. Produce ONE short, simple search query at a time.

You will receive:
- research_goal: The user's research question
- validated_queries: Queries already accepted for this cycle (avoid duplicating)
- gaps: (optional) Specific information gaps this query should target
- already_rejected_in_this_slot: (optional) Queries you tried for the
  current slot that the verifier rejected — your new query must NOT be
  a paraphrase of any of these
- feedback: (optional) The verifier's reason for rejecting your most
  recent attempt

**Rules**
1. 3-8 keywords per query. No long descriptions. No special operators.
2. Each query must cover a different angle from validated_queries.
3. If gaps are specified, target them directly.
4. **If already_rejected_in_this_slot is non-empty**, your new query must
   share at most half its keywords with any rejected entry. Pick a new
   angle to vary along — useful axes for any research domain:
     - methodology / source type (empirical study, review, dataset,
       reference work, case study, official document)
     - scope (general overview vs specific instance)
     - population, sample, time period, or jurisdiction
     - metric, outcome, or measure of interest
     - mechanism, cause, or explanatory layer
     - perspective or stakeholder viewpoint
   Pick one axis you haven't varied yet and shift along it.
5. Stay anchored to research_goal. Do not drift into adjacent topics.

**Output**
- query: 3-8 keywords
- rationale: one sentence on what angle this query covers
"""

register_agent(
    AgentDefinition(
        name="query_generator",
        prompt=QUERY_GENERATOR_PROMPT,
        output_model=GeneratorOutput,
        retries=3,
        output_retries=3,
    )
)
