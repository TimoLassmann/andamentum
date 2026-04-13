"""Deep research agent definitions — prompts, output models, and registry.

Agent knowledge is Python code: prompts are string constants, output models
are BaseModel classes, and definitions are frozen dataclasses.  All are
importable, testable, and IDE-navigable.

Architecture: Layer 1 (framework-agnostic, pure Pydantic)
"""

from dataclasses import dataclass

from pydantic import BaseModel


@dataclass(frozen=True)
class AgentDefinition:
    """Immutable definition of a deep-research agent.

    Attributes:
        name: Agent identifier (e.g. "search_planner")
        prompt: System prompt
        output_model: Pydantic BaseModel class for structured output
        retries: Number of LLM retries on failure
        output_retries: Number of retries for output parsing
        has_tools: Whether this agent uses tools (search, fetch, etc.)
    """

    name: str
    prompt: str
    output_model: type[BaseModel]
    retries: int = 3
    output_retries: int = 5
    has_tools: bool = False


# Global registry populated by domain modules at import time
AGENT_REGISTRY: dict[str, AgentDefinition] = {}


def register_agent(defn: AgentDefinition) -> AgentDefinition:
    """Register an agent definition and return it for assignment."""
    if defn.name in AGENT_REGISTRY:
        raise ValueError(f"Duplicate agent registration: {defn.name}")
    AGENT_REGISTRY[defn.name] = defn
    return defn


def get_agent(name: str) -> AgentDefinition:
    """Get an agent definition by name.

    Raises:
        KeyError: If agent is not registered.
    """
    if name not in AGENT_REGISTRY:
        raise KeyError(f"Unknown deep-research agent: {name}. Available: {sorted(AGENT_REGISTRY)}")
    return AGENT_REGISTRY[name]


# Import domain modules to populate registry on first access.
from . import search as _search  # noqa: E402, F401
from . import analysis as _analysis  # noqa: E402, F401
from . import novelty as _novelty  # noqa: E402, F401

__all__ = [
    "AgentDefinition",
    "AGENT_REGISTRY",
    "register_agent",
    "get_agent",
]
