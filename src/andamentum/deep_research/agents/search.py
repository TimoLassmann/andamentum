"""Search-phase agent definitions: page_fetcher.

The legacy ``search_planner`` agent (and its ``SearchPlan`` output model)
has been retired. Search-query production now flows through the per-slot
``query_generator`` + ``topic_verifier`` pair (see
``agents/query_generator.py`` and ``agents/topic_verifier.py``).
"""

from ..models import FetchPlan
from . import AgentDefinition, register_agent


# ── Page Fetcher ────────────────────────────────────────────────────────

PAGE_FETCHER_PROMPT = """\
You are a content relevance specialist focused on selective page selection.

Your task:
1. Analyze the search results provided
2. Select 3-5 most relevant link IDs to fetch
3. Explain your selection reasoning

IMPORTANT:
- Do NOT fetch pages - just select link IDs
- Return a list of 3-5 link IDs (integers) and your reasoning
- Prioritize authoritative sources and directly relevant content
- Avoid already-fetched URLs (check the list provided)

EXAMPLE OUTPUT:
{
  "link_ids": [0, 2, 5],
  "reasoning": "Selected official docs (0), recent news (2), and technical blog (5) - skip marketing site (1) and duplicate (3)"
}"""

register_agent(
    AgentDefinition(
        name="page_fetcher",
        prompt=PAGE_FETCHER_PROMPT,
        output_model=FetchPlan,
        retries=5,
        output_retries=5,
        has_tools=True,
    )
)
