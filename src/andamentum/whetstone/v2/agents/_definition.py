"""Tiny module holding ``AgentDefinition`` so individual agent files can
import it without triggering the agent-registry init in ``__init__.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class AgentDefinition:
    """A single (prompt, output_model, retries) tuple."""

    name: str
    prompt: str
    output_model: type[BaseModel]
    retries: int = 2
    output_retries: int = 2
