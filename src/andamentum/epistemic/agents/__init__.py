"""Epistemic agent definitions — prompts, output models, and registry.

Agent knowledge is Python code: prompts are string constants, output models
are BaseModel classes, and definitions are frozen dataclasses.  All are
importable, testable, and IDE-navigable.

AgentDefinition is imported from andamentum.core — shared across all modules.

Architecture: Layer 1 (framework-agnostic, pure Pydantic)
"""

from andamentum.core.agents import AgentDefinition


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
        raise KeyError(
            f"Unknown epistemic agent: {name}. Available: {sorted(AGENT_REGISTRY)}"
        )
    return AGENT_REGISTRY[name]


# Import domain modules to populate registry on first access.
# Order doesn't matter — each module calls register_agent() at import time.
from . import preplanning as _preplanning  # noqa: E402, F401
from . import evidence as _evidence  # noqa: E402, F401
from . import verification as _verification  # noqa: E402, F401
from . import uncertainty as _uncertainty  # noqa: E402, F401
from . import synthesis as _synthesis  # noqa: E402, F401
from . import similarity as _similarity  # noqa: E402, F401
from . import judge as _judge  # noqa: E402, F401
from . import integration as _integration  # noqa: E402, F401
from . import dispatch as _dispatch  # noqa: E402, F401

__all__ = [
    "AgentDefinition",
    "AGENT_REGISTRY",
    "register_agent",
    "get_agent",
]
