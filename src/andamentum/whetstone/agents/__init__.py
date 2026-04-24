"""Agent definitions for whetstone.

Prompts and output models are the single source of truth. All agents register
themselves into AGENT_REGISTRY at import time via register_agent().

AgentDefinition is imported from andamentum.core — shared across all modules.
"""

from __future__ import annotations

from andamentum.core.agents import AgentDefinition

AGENT_REGISTRY: dict[str, AgentDefinition] = {}


def register_agent(defn: AgentDefinition) -> AgentDefinition:
    """Register an agent definition. Raises on duplicate name."""
    if defn.name in AGENT_REGISTRY:
        raise ValueError(f"Duplicate agent registration: {defn.name}")
    AGENT_REGISTRY[defn.name] = defn
    return defn


def get_agent(name: str) -> AgentDefinition:
    """Look up an agent by name. Raises KeyError if not found."""
    if name not in AGENT_REGISTRY:
        raise KeyError(
            f"Unknown whetstone agent: {name}. Available: {sorted(AGENT_REGISTRY)}"
        )
    return AGENT_REGISTRY[name]


# Import domain modules to populate the registry on first access.
from . import editing as _editing  # noqa: E402, F401
from . import review as _review  # noqa: E402, F401
from . import synthesis as _synthesis  # noqa: E402, F401
from . import multi_expert as _multi_expert  # noqa: E402, F401
from . import custom as _custom  # noqa: E402, F401
from . import consistency as _consistency  # noqa: E402, F401
from . import checklist as _checklist  # noqa: E402, F401

__all__ = [
    "AgentDefinition",
    "AGENT_REGISTRY",
    "register_agent",
    "get_agent",
]
