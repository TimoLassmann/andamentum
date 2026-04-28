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
- feedback: (optional) If your prior attempt was rejected, the reason why

**Rules**
1. 3-8 keywords per query. No long descriptions. No special operators.
2. Each query must cover a different angle from validated_queries.
3. If gaps are specified, target them directly.
4. If feedback is provided, your prior query was rejected for that reason —
   produce a substantively different query that addresses it.
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
