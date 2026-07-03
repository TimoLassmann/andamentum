"""The fixed spine that generated systems import — written once, engine-free.

Four pieces, deliberately tiny:

- ``run_head`` — the single LLM seam. A generated head node calls this; it routes
  through ``andamentum.core.run_agent_with_fallback`` (tool-calling output with a
  PromptedOutput fallback for small models) and honours the ``agent_overrides`` test
  seam so a generated graph runs with no live model.
- ``EnvelopeTolerantModel`` — the base for every agent-*output* model (forge's own
  heads and the rendered heads of generated systems). A deterministic ``mode="before"``
  validator unwraps the one structured-output failure small models actually produce:
  answering with the JSON *schema envelope* (``{"properties": {...}, "required": ...,
  "type": "object", ...}``) instead of the instance, with the real answer intact inside
  ``properties``. Model proposes, code disposes — lossless normalisation at the
  transport seam, never a silent fallback (a payload that is still invalid after
  unwrapping fails validation exactly as loudly as before).
- ``loop_allowed`` — the termination guard. A checkpoint node calls this before it
  loops back; the bound is a counter in State checked against a Deps cap (recipe I6 /
  dialect L5). The model never decides when to stop.
- ``Store`` — the durable memory Port for a **stateful function** (dialect L1: cross-run
  memory lives in a Port-backed store, loaded at start, saved at end). A keyed
  ``(collection, key, value-as-JSON)`` CRUD store over stdlib ``sqlite3``; ``path=None``
  is an in-memory store (rung-1 behaviour / the offline test mode), ``path="..."`` a
  durable file (rung-2). Five operations, closed surface — no query/where (a sixth
  operation is a signal the brief left rung 2; see ``docs/plans/forge-functions/``).

Dialect note: this module imports **no graph engine** (``pydantic_graph``). It is a
leaf worker library — it may be imported by a generated package's ``nodes.py`` (which
*is* engine-aware) and its ``deps.py`` without dragging the engine into a worker file.
That is what lets a generated package stay dialect-conforming while reusing the fixed
spine.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Protocol, TypeVar

from pydantic import BaseModel, model_validator

_OutT = TypeVar("_OutT", bound=BaseModel)

# The JSON-Schema vocabulary. A payload whose top-level keys ALL come from this set —
# with a dict under "properties" — is a schema envelope, not an instance: no forge or
# generated agent-output model uses these words as its complete field set.
_SCHEMA_ENVELOPE_KEYS = frozenset(
    {
        "$defs",
        "$schema",
        "additionalProperties",
        "definitions",
        "description",
        "properties",
        "required",
        "title",
        "type",
    }
)


class EnvelopeTolerantModel(BaseModel):
    """Base for agent-output models: unwraps the schema-envelope confusion.

    Small models on the PromptedOutput path sometimes echo the JSON schema back with
    the actual field values placed under ``properties`` — e.g.
    ``{"properties": {"body": "<code>"}, "required": ["body"], "type": "object",
    "title": "PieceOut"}``. The instance is fully recoverable, so recovering it is a
    deterministic normalisation, not a guess. Two guards keep it precise: every
    top-level key must be JSON-Schema vocabulary, and at least one corroborating
    envelope marker must be present beyond ``properties`` itself. If the unwrapped
    payload is not a valid instance either (e.g. ``properties`` holds field *schemas*,
    not values), validation still fails loud and the runner's retry behaviour is
    unchanged.
    """

    @model_validator(mode="before")
    @classmethod
    def _unwrap_schema_envelope(cls, data: object) -> object:
        if (
            isinstance(data, dict)
            and isinstance(data.get("properties"), dict)
            and set(data) <= _SCHEMA_ENVELOPE_KEYS
            and (
                data.get("type") == "object"
                or "required" in data
                or "title" in data
                or "$schema" in data
            )
        ):
            return data["properties"]
        return data


class _HasOverrides(Protocol):
    """The slice of a generated ``Deps`` that ``run_head`` reads (read-only, so a
    frozen-dataclass Deps satisfies it as readily as a mutable one)."""

    @property
    def model(self) -> str: ...

    @property
    def agent_overrides(self) -> dict[str, object]: ...


async def run_head(
    deps: _HasOverrides,
    name: str,
    output_type: type[_OutT],
    instructions: str,
    user_message: str,
) -> _OutT:
    """Run one LLM head, honouring the ``agent_overrides`` test seam.

    If ``deps.agent_overrides`` carries a stub under ``name``, its ``.output`` is
    returned directly (no model call) — the dialect's "stub an agent by swapping a
    fake into Deps". Otherwise the call goes through the shared ``core`` runner with
    its small-model PromptedOutput fallback.
    """
    override = deps.agent_overrides.get(name)
    if override is not None:
        out = getattr(override, "output", override)
        if not isinstance(out, output_type):
            raise TypeError(
                f"agent_overrides[{name!r}] produced {type(out).__name__}, expected {output_type.__name__}"
            )
        return out

    from andamentum.core import run_agent_with_fallback

    return await run_agent_with_fallback(
        deps.model,
        instructions=instructions,
        output_type=output_type,
        user_message=user_message,
    )


def loop_allowed(count: int, cap: int) -> bool:
    """True while a bounded loop may run again (recipe I6 / dialect L5).

    ``count`` is the loop counter held in State; ``cap`` is the bound from Deps. The
    checkpoint node increments the counter and loops back only while this is True —
    so termination is structural, never the model's judgment.
    """
    return count < cap


class Store:
    """A durable keyed store for a stateful function's cross-run memory (dialect L1).

    One table — ``(collection, key, value-as-JSON)`` — and exactly five operations. It is
    schema-agnostic: ``value`` is the entity serialised to a JSON object; the per-system
    schema lives in the generated entity model, never here.

    - ``path=None`` → an in-memory database (``:memory:``) that forgets at process exit.
      This is rung-1 behaviour and the offline test mode: the smoke test injects
      ``Store(None)`` and exercises the real load/save paths against RAM.
    - ``path="..."`` → a durable file at exactly that path (rung-2: memory survives runs).

    A node never sees the path: the run entry resolves ``Store(path-or-None)`` once and the
    body uses ``ctx.deps.store`` — so the same body runs stateless or stateful with no
    branch. ``add`` is create-or-update (``INSERT OR REPLACE``), idempotent by key (L8), so
    there is no separate ``update``. There is deliberately no ``query``/``where``: a sixth
    operation signals the brief has left rung 2.
    """

    def __init__(self, path: str | None = None) -> None:
        self._db = sqlite3.connect(path or ":memory:")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS records ("
            "  collection TEXT NOT NULL,"
            "  key        TEXT NOT NULL,"
            "  value      TEXT NOT NULL,"  # JSON object
            "  PRIMARY KEY (collection, key))"
        )
        self._db.commit()

    def add(self, collection: str, key: str, value: dict[str, object]) -> None:
        """Create or overwrite the record at ``(collection, key)``. Idempotent (L8)."""
        self._db.execute(
            "INSERT OR REPLACE INTO records (collection, key, value) VALUES (?, ?, ?)",
            (collection, key, json.dumps(value)),
        )
        self._db.commit()

    def get(self, collection: str, key: str) -> dict[str, object] | None:
        """The record at ``(collection, key)``, or ``None`` if absent."""
        row = self._db.execute(
            "SELECT value FROM records WHERE collection = ? AND key = ?",
            (collection, key),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, collection: str) -> list[dict[str, object]]:
        """Every record in ``collection``, ordered by key (a stable, deterministic order)."""
        rows = self._db.execute(
            "SELECT value FROM records WHERE collection = ? ORDER BY key",
            (collection,),
        ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def remove(self, collection: str, key: str) -> None:
        """Delete the record at ``(collection, key)``. A no-op if it is absent."""
        self._db.execute(
            "DELETE FROM records WHERE collection = ? AND key = ?",
            (collection, key),
        )
        self._db.commit()
