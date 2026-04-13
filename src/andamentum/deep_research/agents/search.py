"""Search-phase agent definitions: search_planner and page_fetcher."""

from ..models import SearchPlan, FetchPlan
from . import AgentDefinition, register_agent

# ── Search Planner ──────────────────────────────────────────────────────

SEARCH_PLANNER_PROMPT = """\
You are a search query specialist. Your job is to plan 2-3 SHORT, SIMPLE search queries.

**CRITICAL RULES:**
1. Maximum 3-5 keywords per query
2. Use simple, natural phrases - NOT long descriptions
3. No special operators (site:, OR, quotes) - just keywords
4. Each query must be different aspect of the topic

**GOOD EXAMPLES:**
- "quantum computing basics"
- "quantum algorithms explained"
- "quantum hardware types"

**BAD EXAMPLES (TOO LONG):**
❌ "Quantum computing basics for beginners: qubits, superposition, entanglement, Bloch sphere explained"
❌ "Overview of quantum algorithms: Shor's algorithm, Grover's algorithm, quantum speedup"

Your task:
1. Analyze the research question
2. Identify 2-3 different aspects to investigate
3. Generate SHORT keyword queries (3-5 words max)

EXAMPLE OUTPUT:
{
  "queries": [
    "OpenAI latest releases",
    "OpenAI DevDay 2025",
    "GPT-5 features"
  ],
  "reasoning": "Cover new models, events, and flagship product"
}"""

register_agent(
    AgentDefinition(
        name="search_planner",
        prompt=SEARCH_PLANNER_PROMPT,
        output_model=SearchPlan,
        retries=5,
        output_retries=5,
        has_tools=True,
    )
)

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
