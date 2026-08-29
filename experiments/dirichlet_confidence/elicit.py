"""Model-call layer: ask one model, under one phrasing, for a 0-100 histogram.

Isolated here so both the GATHER step and any ad-hoc probing share exactly one
elicitation implementation. OpenAI models pass through pydantic-ai's inference
(chat completions, which honour ``temperature``); local ollama models go through
``core.resolve_model``. Tool-based structured output falls back to PromptedOutput
for models that don't support tools. A failed call returns ``None``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.output import PromptedOutput
from pydantic_ai.settings import ModelSettings

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from andamentum.core import resolve_model  # noqa: E402

from experiments.dirichlet_confidence.phrasings import (  # noqa: E402
    PHRASINGS,
    SYSTEM_INSTRUCTIONS,
    render,
)


class VerbalizedDistribution(BaseModel):
    """The model's 0-100 points across the three SciFact classes."""

    support: int = Field(ge=0, le=100)
    contradict: int = Field(ge=0, le=100)
    noinfo: int = Field(ge=0, le=100)

    def to_row(self) -> list[float]:
        return [float(self.support), float(self.contradict), float(self.noinfo)]


def _agent(model: str, *, prompted: bool) -> Agent:
    resolved = resolve_model(model) if model.startswith("ollama:") else model
    output_type = (
        PromptedOutput(VerbalizedDistribution) if prompted else VerbalizedDistribution
    )
    return Agent(resolved, instructions=SYSTEM_INSTRUCTIONS, output_type=output_type)


async def elicit_row(
    claim: str, evidence: str, model: str, phrasing_id: int, temperature: float
) -> list[float] | None:
    """Ask ``model`` the claim under ``PHRASINGS[phrasing_id]``. Returns the raw
    0-100 row, or ``None`` if the call ultimately fails."""
    prompt = render(PHRASINGS[phrasing_id], claim, evidence)
    settings = ModelSettings(temperature=temperature)
    try:
        result = await _agent(model, prompted=False).run(prompt, model_settings=settings)
        return result.output.to_row()
    except (UnexpectedModelBehavior, ModelHTTPError):
        try:
            result = await _agent(model, prompted=True).run(prompt, model_settings=settings)
            return result.output.to_row()
        except Exception:  # noqa: BLE001
            return None
    except Exception:  # noqa: BLE001
        return None
