"""Model registry: which model fills which role or decision, with its digest
Entries come from code or from ``checks.yaml``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict

from andamentum.core.decisions import DecisionModel


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str  # e.g. "numbers", "needs_citation"
    kind: Literal["llm", "decision"] = "llm"
    id: str  # "provider:name", as everywhere in andamentum
    digest: str | None = None  # content digest of local weights, when known


class ModelRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, ModelEntry] = {}

    def register(self, entry: ModelEntry) -> ModelEntry:
        if entry.role in self._entries:
            raise ValueError(f"model role {entry.role!r} is already registered")
        self._entries[entry.role] = entry
        return entry

    def get(self, role: str) -> ModelEntry:
        try:
            return self._entries[role]
        except KeyError:
            raise KeyError(f"no model registered for role {role!r}") from None

    def entries(self) -> list[ModelEntry]:
        return list(self._entries.values())

    def as_context(self) -> dict[str, str]:
        """Role to model id, the shape ``CheckContext.models`` takes."""
        return {role: entry.id for role, entry in self._entries.items()}

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> ModelRegistry:
        """Build from the ``models:`` mapping of ``checks.yaml``.

        Each key is a role; the value is a model id string or a mapping with
        ``id`` and the optional ``kind``, ``digest``.
        """
        registry = cls()
        raw = config.get("models", {})
        if not isinstance(raw, Mapping):
            raise ValueError("'models' in checks.yaml must be a mapping")
        for role, value in raw.items():
            if isinstance(value, str):
                registry.register(ModelEntry(role=str(role), id=value))
            elif isinstance(value, Mapping):
                registry.register(ModelEntry(role=str(role), **value))
            else:
                raise ValueError(
                    f"model entry for {role!r} must be a string or mapping"
                )
        return registry


async def open_decision_model(
    name: str | None, *, injected: DecisionModel | None = None
) -> tuple[DecisionModel | None, bool]:
    """The run's decision-model handle and whether the caller must close it.

    ``injected`` (tests, library callers) is used as given; otherwise ``name``
    (``--decision-model``) is opened and checked against the local Ollama
    server. Raises ``DecisionModelUnavailable`` when it cannot be.
    """
    if injected is not None:
        return injected, False
    if not name:
        return None, False
    return await DecisionModel.open(name), True
