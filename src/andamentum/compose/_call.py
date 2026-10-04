"""The one place compose workers call a model, so every call is logged alike."""

from __future__ import annotations

from pydantic import BaseModel

from andamentum.core.agents import run_agent_with_usage

from .log import ActionLog
from .schemas import Role


async def call_model[T: BaseModel](
    *,
    model: str,
    instructions: str,
    output_type: type[T],
    user_message: str,
    node: str,
    log: ActionLog | None,
    role: Role = "drafter",
) -> T:
    """One structured model call; appends a log record when ``log`` is given.

    Token counts come from the model run's usage and are logged with the call.
    """
    output, usage = await run_agent_with_usage(
        model,
        instructions=instructions,
        output_type=output_type,
        user_message=user_message,
    )
    if log is not None:
        log.call(
            node=node,
            role=role,
            model=model,
            prompt=instructions,
            input_text=user_message,
            output_text=output.model_dump_json(),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
    return output
