"""LLM-powered query planner for natural language search.

Parses natural language queries into a search plan: one semantic query
plus one optional metadata filter on a closed-set field.

Filterable fields (closed-set only — LLM knows all valid values):
  source: manual, slack, claude_code, zotero, voice
  created_at: date filtering (after/before)

Open-ended fields (people, projects, topics, methods) are NOT filterable —
semantic search handles them. The LLM can't know what values exist in the database.

``has_decision`` / ``has_action_item`` used to be filterable. They were removed
along with the per-chunk LLM tagging that populated them: nothing read the tags,
and a filter over a field nothing writes can only ever match zero rows.
Consumers that want such flags write their own metadata and query it with
``find_by_metadata`` — deterministically, no LLM.

Requires: pydantic-ai (installed as part of andamentum).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Only closed-set fields and booleans. The LLM knows all valid values.
FilterField = Literal[
    "source",  # 5 values: manual, slack, claude_code, zotero, voice
    "created_at",  # date filtering (after/before)
]

FilterOperator = Literal["equals", "after", "before"]

_OUTPUT_RETRIES = 5
_RETRIES = 3


class MetadataFilter(BaseModel):
    """A single metadata filter to narrow search results.

    Only closed-set fields are filterable. Open-ended fields
    (people, projects, topics) are handled by semantic search.
    """

    field: FilterField = Field(
        description=(
            "Metadata field to filter on. "
            "source: manual, slack, claude_code, zotero, voice. "
            "created_at: document creation date. "
            "Do NOT filter on people, projects, or topics — semantic search handles those."
        ),
    )
    operator: FilterOperator = Field(
        description=(
            "How to compare. "
            "equals: exact match (for source). "
            "after: date >= value (for created_at). "
            "before: date <= value (for created_at)."
        ),
    )
    value: str = Field(
        default="",
        description="Value to match. For equals: the exact string. For dates: YYYY-MM-DD.",
    )


class SearchPlan(BaseModel):
    """LLM-generated plan for a natural language search query.

    One semantic query + one optional filter. Simple by design —
    semantic search handles most of the work, the filter just narrows scope.
    """

    semantic_query: str = Field(
        description=(
            "What to search for in document content. Keep the full meaning "
            "of the query. Only remove purely structural terms like 'show me all' or 'list'. "
            "If the query is purely a filter (e.g. 'show me all decisions'), "
            "set this to empty string."
        ),
    )
    filter: MetadataFilter | None = Field(
        default=None,
        description=(
            "Optional: the single most useful filter for this query. "
            "None if the query is pure content search with no metadata constraint."
        ),
    )
    needs_semantic_search: bool = Field(
        default=True,
        description=(
            "Whether to run content search (FTS5 + embeddings). "
            "False only for pure metadata queries like 'show me all decisions'."
        ),
    )


_PLANNER_SYSTEM_PROMPT = """\
You are a search query planner for a personal knowledge base.
Given a natural language query, produce a search plan with:
- A semantic query (what to search for in content)
- Optionally ONE metadata filter on a closed-set field

Filterable fields: source, created_at.
Do NOT filter on people, projects, methods, or topics — semantic search handles those.

Today's date is {today}.

Examples:
- "What have I captured about MAP-Elites?" → semantic_query="MAP-Elites", filter=None
- "What decisions did I make about GROVE?" → semantic_query="decisions about GROVE", filter=None
- "What has Sarah said about the grant?" → semantic_query="Sarah grant", filter=None
- "Meeting notes from last month" → semantic_query="meeting notes", filter={{field: "created_at", operator: "after", value: "{last_month}"}}
- "What did I capture from Slack this week?" → semantic_query="", filter={{field: "source", operator: "equals", value: "slack"}}, needs_semantic_search=False
"""


def _build_planner_agent(model: str):  # type: ignore[no-untyped-def]
    """Build a PydanticAI agent for query planning with output validation."""
    import re

    from pydantic_ai import Agent, ModelRetry, RunContext

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    last_month = datetime.now(timezone.utc).replace(day=1).strftime("%Y-%m-%d")

    agent = Agent(
        model,
        system_prompt=_PLANNER_SYSTEM_PROMPT.format(today=today, last_month=last_month),
        output_type=SearchPlan,
        retries={"tools": _RETRIES, "output": _OUTPUT_RETRIES},
    )

    @agent.output_validator
    async def validate_plan(ctx: RunContext[None], output: SearchPlan) -> SearchPlan:
        issues: list[str] = []

        if output.filter is not None:
            f = output.filter
            date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")

            if (
                f.operator in ("after", "before")
                and f.value
                and not date_re.match(f.value)
            ):
                issues.append(f"Date filter needs YYYY-MM-DD format, got: '{f.value}'")

            if f.operator == "equals" and not f.value:
                issues.append("'equals' operator requires a non-empty value")

            if f.field == "created_at" and f.operator not in ("after", "before"):
                issues.append(
                    f"created_at only supports 'after' or 'before', got: '{f.operator}'"
                )

            if f.field == "source" and f.operator != "equals":
                issues.append(
                    f"'{f.field}' only supports 'equals' operator, got: '{f.operator}'"
                )

        if not output.needs_semantic_search and output.filter is None:
            issues.append("If needs_semantic_search is False, a filter is required")

        if issues:
            raise ModelRetry("\n".join(issues))

        return output

    return agent


async def plan_search(query: str, model: str) -> SearchPlan:
    """Parse a natural language query into a structured search plan.

    Args:
        query: Natural language search query
        model: PydanticAI model string

    Returns:
        SearchPlan with semantic_query and optional filter

    Raises:
        RuntimeError: If the LLM is unreachable or fails after all retries
    """
    try:
        agent = _build_planner_agent(model)
        result = await agent.run(query)
        plan = result.output

        # Safety: if LLM emptied the semantic query but we need search, use original
        if plan.needs_semantic_search and not plan.semantic_query.strip():
            plan.semantic_query = query

        return plan
    except ImportError:
        raise RuntimeError(
            "pydantic-ai not installed. Install with: pip install andamentum"
        )
    except Exception as e:
        raise RuntimeError(f"Query planning failed: {e}") from e
