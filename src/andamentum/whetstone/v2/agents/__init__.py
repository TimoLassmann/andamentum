"""Agent definitions for whetstone v2.

Mirrors the deep_research/agents/ pattern: each agent is a tiny
``AgentDefinition`` of (prompt, output_model, retry budget). Nodes look
agents up by name via ``get_agent`` and instantiate a pydantic-ai
``Agent`` from the definition. This keeps the agent surface small,
testable, and audit-able.
"""

from __future__ import annotations

from typing import Any

from ._definition import AgentDefinition
from .author_question import AUTHOR_QUESTION_AGENT, AuthorQuestionOutput
from .challenge import CHALLENGE_AGENT, ChallengeVerdict
from .custom_reviewer import CUSTOM_REVIEWER_AGENT
from .editor import EDITOR_AGENT, EditorOutput, EditProposal
from .expert_generator import EXPERT_GENERATOR_AGENT
from .expert_reviewer import EXPERT_REVIEWER_AGENT
from .extract_checkable_items import (
    EXTRACT_CHECKABLE_ITEMS_AGENT,
    ExtractedItemsList,
)
from .extract_keywords import EXTRACT_KEYWORDS_AGENT, KeywordExtractionOutput
from .guideline_item_evaluator import GUIDELINE_ITEM_EVALUATOR_AGENT
from .investigator import (
    INVESTIGATOR_AGENT,
    InvestigatorOutput,
    NewNote,
    NoteUpdate,
)
from .lens import (
    LensIssueProposal,
    LensReadOutput,
    build_lens_agent_definition,
    list_available_lenses,
)
from .panel_synthesise import PANEL_SYNTHESISE_AGENT
from .reflection import REFLECTION_AGENT, ReflectionOutput, ReflectionTask
from .synthesise import SYNTHESISE_AGENT, ReviewSummary


_REGISTRY: dict[str, AgentDefinition] = {
    EDITOR_AGENT.name: EDITOR_AGENT,
    CHALLENGE_AGENT.name: CHALLENGE_AGENT,
    SYNTHESISE_AGENT.name: SYNTHESISE_AGENT,
    AUTHOR_QUESTION_AGENT.name: AUTHOR_QUESTION_AGENT,
    REFLECTION_AGENT.name: REFLECTION_AGENT,
    INVESTIGATOR_AGENT.name: INVESTIGATOR_AGENT,
    # Panel mode
    EXTRACT_KEYWORDS_AGENT.name: EXTRACT_KEYWORDS_AGENT,
    EXPERT_GENERATOR_AGENT.name: EXPERT_GENERATOR_AGENT,
    EXPERT_REVIEWER_AGENT.name: EXPERT_REVIEWER_AGENT,
    PANEL_SYNTHESISE_AGENT.name: PANEL_SYNTHESISE_AGENT,
    # Guidelines mode
    EXTRACT_CHECKABLE_ITEMS_AGENT.name: EXTRACT_CHECKABLE_ITEMS_AGENT,
    GUIDELINE_ITEM_EVALUATOR_AGENT.name: GUIDELINE_ITEM_EVALUATOR_AGENT,
    # Custom mode (output_model is built at call time)
    CUSTOM_REVIEWER_AGENT.name: CUSTOM_REVIEWER_AGENT,
}

# Register every available lens under its lens.<name> key.
for _lens_name in list_available_lenses():
    _defn = build_lens_agent_definition(_lens_name)
    _REGISTRY[_defn.name] = _defn


def get_agent(name: str) -> AgentDefinition:
    """Look up an agent by name. Raises KeyError on unknown agent."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown agent {name!r}. Registered: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def build_pydantic_ai_agent(name: str, model: Any):
    """Build a pydantic-ai ``Agent`` from a registry definition.

    Thin wrapper around ``andamentum.core.agents.build_pydantic_ai_agent``
    that adds the registry-name lookup. Kept as a thin shim so
    ``mock.patch("...build_pydantic_ai_agent", ...)`` in node tests
    continues to work.
    """
    from andamentum.core.agents import build_pydantic_ai_agent as _build

    return _build(get_agent(name), model)


def build_dynamic_output_agent(name: str, model: Any, output_type: Any):
    """Build a pydantic-ai ``Agent`` whose ``output_type`` is supplied at call time.

    For agents whose output schema is built at runtime (custom-criteria
    mode). The registry definition carries ``output_model=None``; the
    caller passes the concrete type here. Mirrors
    :func:`build_pydantic_ai_agent` so node tests can ``mock.patch`` it
    in the same way.
    """
    from pydantic_ai import Agent

    defn = get_agent(name)
    return Agent(
        model,
        instructions=defn.prompt,
        output_type=output_type,
        retries=defn.retries,
        output_retries=defn.output_retries,
    )


__all__ = [
    "AgentDefinition",
    "build_dynamic_output_agent",
    "build_pydantic_ai_agent",
    "get_agent",
    # Module-level definition constants
    "INVESTIGATOR_AGENT",
    "REFLECTION_AGENT",
    "EXTRACT_KEYWORDS_AGENT",
    "EXPERT_GENERATOR_AGENT",
    "EXPERT_REVIEWER_AGENT",
    "PANEL_SYNTHESISE_AGENT",
    "EXTRACT_CHECKABLE_ITEMS_AGENT",
    "GUIDELINE_ITEM_EVALUATOR_AGENT",
    "CUSTOM_REVIEWER_AGENT",
    # Lens helpers
    "build_lens_agent_definition",
    "list_available_lenses",
    # Output schemas re-exported so node code can import from one place
    "AuthorQuestionOutput",
    "ChallengeVerdict",
    "EditProposal",
    "EditorOutput",
    "ExtractedItemsList",
    "InvestigatorOutput",
    "KeywordExtractionOutput",
    "LensIssueProposal",
    "LensReadOutput",
    "NewNote",
    "NoteUpdate",
    "ReflectionOutput",
    "ReflectionTask",
    "ReviewSummary",
]
