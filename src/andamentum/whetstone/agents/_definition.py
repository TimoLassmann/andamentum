"""Re-export ``core.agents.AgentDefinition`` so whetstone v2 agent files share
one definition class with deep_research, epistemic, and any future module.

Kept as its own file (rather than importing core directly in
``__init__.py``) so individual agent modules can import the symbol without
triggering the v2 agent-registry init in ``__init__.py``.
"""

from __future__ import annotations

from andamentum.core.agents import AgentDefinition

__all__ = ["AgentDefinition"]
