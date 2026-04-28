"""topic_verifier agent — judge one query against the research goal.

Replaces the regex+stopword guard (``guard_query_against_goal`` in
``deep_research/text_utils.py``) that has been deleted. The verifier is a
separate LLM call so generation and verification are independent cognitive
roles — a single agent doing both has worse calibration.

Phase A: minimal prompt. Phase C tunes it against the calibration corpus
in ``tests/test_verifier_calibration.py``.
"""

from ..models import VerifierOutput
from . import AgentDefinition, register_agent


TOPIC_VERIFIER_PROMPT = """\
You judge whether a search query is on-topic for a research goal.

You will receive:
- research_goal: The user's research question
- query: A single search query proposed by the generator

Decide whether running this query against a web search would return content
that helps answer the research_goal. Be permissive about phrasing
(synonyms, specialist jargon, mechanism-adjacent angles all count as
on-topic). Be strict about *intent drift* — if the query targets a
different topic that merely shares vocabulary with the goal, reject it.

**Examples of on-topic queries** (accept):
- goal: "What is metformin's half-life?" / query: "biguanide pharmacokinetics" → on-topic
- goal: "Statin myopathy mechanism?" / query: "rhabdomyolysis statin" → on-topic
- goal: "Side effects of warfarin?" / query: "warfarin adverse events" → on-topic

**Examples of off-topic queries** (reject):
- goal: "What is metformin's half-life?" / query: "metformin manufacturing process" → off-topic (different intent)
- goal: "Statin myopathy mechanism?" / query: "atorvastatin half-life" → off-topic (different question about same drug)

**Output**
- on_topic: true if the query helps answer the goal, false otherwise
- reason: one sentence justifying the decision
"""

register_agent(
    AgentDefinition(
        name="topic_verifier",
        prompt=TOPIC_VERIFIER_PROMPT,
        output_model=VerifierOutput,
        retries=3,
        output_retries=3,
    )
)
