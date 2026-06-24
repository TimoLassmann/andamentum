"""The fixed spine that generated systems import — written once, engine-free.

Two helpers, deliberately tiny:

- ``run_head`` — the single LLM seam. A generated head node calls this; it routes
  through ``andamentum.core.run_agent_with_fallback`` (tool-calling output with a
  PromptedOutput fallback for small models) and honours the ``agent_overrides`` test
  seam so a generated graph runs with no live model.
- ``loop_allowed`` — the termination guard. A checkpoint node calls this before it
  loops back; the bound is a counter in State checked against a Deps cap (recipe I6 /
  dialect L5). The model never decides when to stop.

Dialect note: this module imports **no graph engine** (``pydantic_graph``). It is a
leaf worker library — it may be imported by a generated package's ``nodes.py`` (which
*is* engine-aware) without dragging the engine into a worker file. That is what lets a
generated package stay dialect-conforming while reusing the fixed spine.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

_OutT = TypeVar("_OutT", bound=BaseModel)


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
