"""Verbalized-distribution elicitation.

For one (model, phrasing) pair this asks the model to distribute 0-100 across
{SUPPORT, CONTRADICT, NOINFO} and returns the raw integer row. Parsing is via a
pydantic structured output (robust across very different models); the *phrasing*
variation lives in the user prompt, so the numbers still reflect prompt
sensitivity even though extraction is uniform.

We reuse ``andamentum.core.resolve_model`` for ollama/bedrock/passthrough
resolution, but build the pydantic-ai agent here so we control per-backend model
settings:

- **OpenAI** models go through the Responses API (``OpenAIResponsesModel``) — the
  unified endpoint that serves both chat-capable and responses-only reasoning /
  codex models. These are reasoning models: they reject ``temperature`` and
  require ``reasoning.effort`` to be one of low/medium/high (pydantic-ai's
  default ``'auto'`` is rejected with a 400), so we pass an explicit effort and
  no temperature. Determinism is therefore not available for these models — a
  real limitation of using reasoning models for verbalized confidence.
- **Local / other** models get ``temperature`` (0 by default) for reproducibility.

The tool-based structured output falls back to PromptedOutput (schema in the
prompt) for models that don't support tool output. Genuinely failed elicitations
return ``None`` so the caller can drop them from the mixture.
"""

from __future__ import annotations

import logging

import numpy as np
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.output import PromptedOutput
from pydantic_ai.settings import ModelSettings

from andamentum.core import resolve_model

from .phrasings import CLASSES, PHRASINGS, SYSTEM_INSTRUCTIONS, render

logger = logging.getLogger(__name__)

# Reasoning effort for OpenAI Responses models. low keeps latency/cost down while
# still letting the chain-of-thought phrasing reason a little.
_OPENAI_REASONING_EFFORT = "low"


class VerbalizedDistribution(BaseModel):
    """Structured target for the verbalized integers (0-100 per class)."""

    support: int = Field(ge=0, le=100, description="Points for SUPPORT")
    contradict: int = Field(ge=0, le=100, description="Points for CONTRADICT")
    noinfo: int = Field(ge=0, le=100, description="Points for NOINFO")

    def to_row(self) -> np.ndarray:
        return np.array([self.support, self.contradict, self.noinfo], dtype=float)


class Elicitor:
    """Resolves each model once and elicits verbalized rows from it.

    Args:
        models: model ids (e.g. ``"ollama:gemma4:31b-nvfp4"``, ``"openai:gpt-5.1"``).
        temperature: sampling temperature for non-OpenAI models (0.0 for
            reproducibility). OpenAI reasoning models ignore it.
        retries / output_retries: passed through to the pydantic-ai agent.
    """

    def __init__(
        self,
        models: list[str],
        temperature: float = 0.0,
        retries: int = 2,
        output_retries: int = 3,
    ) -> None:
        self.models = models
        self.temperature = temperature
        self.retries = retries
        self.output_retries = output_retries
        self._resolved = {m: self._resolve(m) for m in models}

    @staticmethod
    def _resolve(model: str):
        """Resolve a model id to a pydantic-ai model object."""
        if model.startswith("openai:"):
            from pydantic_ai.models.openai import OpenAIResponsesModel

            return OpenAIResponsesModel(model.split(":", 1)[1])
        return resolve_model(model)

    def _settings(self, model: str) -> ModelSettings:
        """Per-backend model settings.

        OpenAI Responses (reasoning) models: explicit ``reasoning.effort`` and no
        temperature. Everything else: ``temperature``.
        """
        if model.startswith("openai:"):
            from pydantic_ai.models.openai import OpenAIResponsesModelSettings

            return OpenAIResponsesModelSettings(
                openai_reasoning_effort=_OPENAI_REASONING_EFFORT
            )
        return ModelSettings(temperature=self.temperature)

    def _agent(self, model: str, *, prompted: bool) -> Agent:
        resolved = self._resolved[model]
        output_type = (
            PromptedOutput(VerbalizedDistribution) if prompted else VerbalizedDistribution
        )
        return Agent(
            resolved,
            instructions=SYSTEM_INSTRUCTIONS,
            output_type=output_type,
            retries=self.retries,
            output_retries=self.output_retries,
        )

    async def _run_once(
        self, model: str, prompt: str, settings: ModelSettings | None
    ) -> np.ndarray:
        """One attempt: tool-based output, falling back to PromptedOutput."""
        try:
            result = await self._agent(model, prompted=False).run(
                prompt, model_settings=settings
            )
            return result.output.to_row()
        except (UnexpectedModelBehavior, ModelHTTPError):
            logger.info("model=%s: tool output failed, trying PromptedOutput", model)
            result = await self._agent(model, prompted=True).run(
                prompt, model_settings=settings
            )
            return result.output.to_row()

    async def elicit(
        self, claim: str, evidence: str, model: str, phrasing_id: int
    ) -> np.ndarray | None:
        """Elicit one verbalized row, or ``None`` if the call ultimately fails."""
        prompt = render(PHRASINGS[phrasing_id], claim, evidence)
        try:
            return await self._run_once(model, prompt, self._settings(model))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "elicit retry (no settings) model=%s phrasing=%d: %s",
                model,
                phrasing_id,
                exc,
            )
        try:
            return await self._run_once(model, prompt, None)
        except Exception as exc:  # noqa: BLE001
            logger.error("elicit failed model=%s phrasing=%d: %s", model, phrasing_id, exc)
            return None


def n_phrasings() -> int:
    return len(PHRASINGS)


__all__ = ["Elicitor", "VerbalizedDistribution", "CLASSES", "n_phrasings"]
