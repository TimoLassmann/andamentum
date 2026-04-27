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
from .editor import EDITOR_AGENT, EditorOutput, EditProposal
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
from .reflection import REFLECTION_AGENT, ReflectionOutput, ReflectionTask
from .synthesise import SYNTHESISE_AGENT, ReviewSummary


_REGISTRY: dict[str, AgentDefinition] = {
    EDITOR_AGENT.name: EDITOR_AGENT,
    CHALLENGE_AGENT.name: CHALLENGE_AGENT,
    SYNTHESISE_AGENT.name: SYNTHESISE_AGENT,
    AUTHOR_QUESTION_AGENT.name: AUTHOR_QUESTION_AGENT,
    REFLECTION_AGENT.name: REFLECTION_AGENT,
    INVESTIGATOR_AGENT.name: INVESTIGATOR_AGENT,
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
    """Build a pydantic-ai Agent from a registry definition.

    Defers the pydantic-ai import so this module can be loaded without it
    (e.g. for type-checking or for tests that mock at the node level).
    """
    from pydantic_ai import Agent

    defn = get_agent(name)
    return Agent(
        model,
        output_type=defn.output_model,
        instructions=defn.prompt,
        retries=defn.retries,
        output_retries=defn.output_retries,
    )


__all__ = [
    "AgentDefinition",
    "build_pydantic_ai_agent",
    "get_agent",
    # Module-level definition constants
    "INVESTIGATOR_AGENT",
    "REFLECTION_AGENT",
    # Lens helpers
    "build_lens_agent_definition",
    "list_available_lenses",
    # Output schemas re-exported so node code can import from one place
    "AuthorQuestionOutput",
    "ChallengeVerdict",
    "EditProposal",
    "EditorOutput",
    "InvestigatorOutput",
    "LensIssueProposal",
    "LensReadOutput",
    "NewNote",
    "NoteUpdate",
    "ReflectionOutput",
    "ReflectionTask",
    "ReviewSummary",
]
