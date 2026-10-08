"""Offline stand-ins for the models and the network, for the READ, COVERAGE,
VERIFY and WRITE tests.

``ScriptedLLM`` is a pydantic-ai ``FunctionModel`` whose answers a test writes
as a function of the instructions and the user message; ``http`` is a ``ReferenceHttp`` over a mock transport.
Nothing here touches the network or a real model.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping

import httpx
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from andamentum.references import ReferenceHttp

from ..ask import Asker
from ..store import ResearchStore
from ..types import Budget, RunStats

Answer = Mapping[str, object]
LLMRespond = Callable[[str, str], Answer]


class ScriptedLLM:
    """A model that answers by calling ``respond(instructions, user_message)``.

    ``respond`` returns the fields of the structured answer. ``calls`` records
    every ``(instructions, user_message)``; a repair round after a validator's
    ``ModelRetry`` is another call with the same message. The answer is given
    without a thread or a timer, so concurrent calls finish in the order they
    started and a test of parallel work is repeatable.
    """

    def __init__(self, respond: LLMRespond) -> None:
        self.respond = respond
        self.calls: list[tuple[str, str]] = []
        self.model = FunctionModel(self._answer)

    async def _answer(
        self, messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        # async on purpose: pydantic-ai runs a sync function in a worker thread,
        # and the order concurrent calls finish in would then depend on the threads
        instructions = ""
        prompt = ""
        for message in messages:
            if isinstance(message, ModelRequest):
                instructions = message.instructions or instructions
                for part in message.parts:
                    if isinstance(part, UserPromptPart) and not prompt:
                        prompt = str(part.content)
        self.calls.append((instructions, prompt))
        fields = self.respond(instructions, prompt)
        if (
            not info.output_tools
        ):  # a PromptedOutput retry: the answer is plain JSON text
            return ModelResponse(parts=[TextPart(content=json.dumps(fields))])
        tool = info.output_tools[0].name
        return ModelResponse(
            parts=[ToolCallPart(tool_name=tool, args=json.dumps(fields))]
        )


def http(handler: Callable[[httpx.Request], httpx.Response]) -> ReferenceHttp:
    """A ``ReferenceHttp`` over *handler*, with every URL treated as safe."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ReferenceHttp(client=client, url_check=lambda url: (True, ""))


def make_asker(
    *,
    model: ScriptedLLM | None = None,
    budget: Budget | None = None,
    store: ResearchStore | None = None,
) -> Asker:
    """An ``Asker`` over an in-memory store with a started run."""
    store = store or ResearchStore.in_memory()
    run_id = store.start_run(question="q", mode="scholarly", config={}, models={})
    return Asker(
        model=model.model if model else None,
        store=store,
        run_id=run_id,
        budget=budget or Budget(),
        stats=RunStats(),
    )
