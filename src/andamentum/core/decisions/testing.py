"""A stub Ollama decision server for offline tests (``httpx.MockTransport``).

``StubDecisions(respond)`` serves ``/api/version``, ``/api/show``, ``/api/tags``
and ``/v1/systemone``. ``respond(state, questions)`` receives the state and the
wire form of each question (``{"type", "instructions", "criteria"?}``) and
returns an ``Answer`` per question. ``.model()`` builds a ``DecisionModel`` on
it; ``await DecisionModel.open(name, transport=stub.transport())`` exercises the
real checks. ``.calls`` records every ``(state, questions)`` asked.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx

from .api import Answer
from .model import DecisionModel

STUB_DIGEST = "sha256:" + "ab" * 32
STUB_NAME = "nimble:9b"

Responder = Callable[[object, dict[str, dict[str, object]]], dict[str, Answer]]


def wire_answer(question: dict[str, object], answer: Answer) -> dict[str, object]:
    if question["type"] == "noul":
        return {"type": "noul", "noul": answer.p_true}
    if question["type"] == "choice":
        return {
            "type": "choice",
            "choice": answer.choice,
            "probabilities": answer.probabilities,
            "confidence": answer.confidence,
        }
    return {
        "type": "score",
        "score": answer.score,
        "legend": {},
        "probabilities": answer.probabilities,
        "confidence": answer.confidence,
    }


def noul_answer(p: float) -> Answer:
    return Answer(
        choice="yes" if p >= 0.5 else "no",
        probabilities={"yes": p, "no": 1.0 - p},
        confidence=max(p, 1.0 - p),
        p_true=p,
    )


def choice_answer(options: list[str], choice: str, p: float) -> Answer:
    rest = (1.0 - p) / max(len(options) - 1, 1)
    return Answer(
        choice=choice,
        probabilities={o: (p if o == choice else rest) for o in options},
        confidence=p,
    )


class StubDecisions:
    def __init__(
        self,
        respond: Responder,
        *,
        name: str = STUB_NAME,
        digest: str = STUB_DIGEST,
        version: str = "0.35.1",
        capabilities: list[str] | None = None,
    ) -> None:
        self.respond = respond
        self.name = name
        self.digest = digest
        self.version = version
        self.capabilities = (
            ["completion", "decision"] if capabilities is None else capabilities
        )
        self.calls: list[tuple[object, dict[str, dict[str, object]]]] = []

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": self.version})
        if path == "/api/show":
            wanted = json.loads(request.content)["model"]
            if wanted != self.name:
                return httpx.Response(404, json={"error": "model not found"})
            return httpx.Response(200, json={"capabilities": self.capabilities})
        if path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": self.name, "digest": self.digest}]}
            )
        if path == "/v1/systemone":
            body = json.loads(request.content)
            self.calls.append((body["state"], body["questions"]))
            answers = self.respond(body["state"], body["questions"])
            return httpx.Response(
                200,
                json={
                    "model": body["model"],
                    "answers": {
                        key: wire_answer(body["questions"][key], answer)
                        for key, answer in answers.items()
                    },
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                },
            )
        return httpx.Response(404, json={"error": path})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def model(self, *, digest: str | None = None) -> DecisionModel:
        """A handle on this stub, built without the open() checks."""
        return DecisionModel(
            self.name,
            digest=digest or self.digest,
            base_url="http://stub.invalid",
            transport=self.transport(),
        )
