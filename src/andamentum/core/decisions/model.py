"""``DecisionModel``: a long-lived handle on one local Ollama decision model.

Mirrors ``core.embeddings.make_ollama_embedder``: one client for the life of
the handle, plus one-shot helpers (``decide``, ``model_digest``) for code that
asks once and moves on.

``DecisionModel.open`` checks the server (Ollama 0.35 or later), that the
model is installed and declares the ``decision`` capability, and reads its
digest, which identifies the exact weights (a cache key for answers).

Every request to one server is serialised by a lock shared by all handles in
the event loop: local calls are never concurrent, whoever makes them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import warnings
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType

import httpx

from ..embeddings import DEFAULT_OLLAMA_URL
from .api import (
    Answer,
    Question,
    Usage,
    build_body,
    chunk_questions,
    model_name,
    parse_answers,
    parse_usage,
)
from .errors import (
    DecisionModelUnavailable,
    DecisionRequestRejected,
    DecisionResponseError,
)

logger = logging.getLogger(__name__)

MIN_OLLAMA_VERSION = (0, 35, 0)
DEFAULT_TIMEOUT = 30.0
DEFAULT_LOAD_TIMEOUT = 120.0
DECISION_CAPABILITY = "decision"
_QUESTION_KEY = "q"

LogFn = Callable[[dict[str, object]], None]

_LOCKS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, asyncio.Lock]
] = weakref.WeakKeyDictionary()


def _lock_for(base_url: str) -> asyncio.Lock:
    """The lock of one server in the running loop (a lock binds to its loop)."""
    per_url = _LOCKS.setdefault(asyncio.get_running_loop(), {})
    return per_url.setdefault(base_url.rstrip("/"), asyncio.Lock())


def server_lock(base_url: str = DEFAULT_OLLAMA_URL) -> asyncio.Lock:
    """The lock every decision request to the Ollama server at *base_url*
    holds. Other local model calls to the same server (an embedding request)
    take it too, so no two local calls run at once."""
    return _lock_for(base_url)


def _parse_version(text: str) -> tuple[int, ...]:
    found = re.match(r"(\d+)\.(\d+)\.(\d+)", text)
    if found is None:
        raise DecisionModelUnavailable(f"cannot read the Ollama version {text!r}")
    return tuple(int(part) for part in found.groups())


def find_digest(listed: object, name: str) -> str | None:
    """The digest of ``name`` in an ``/api/tags`` ``models`` listing."""
    if not isinstance(listed, list):
        return None
    wanted = {name, name if ":" in name else f"{name}:latest"}
    for entry in listed:
        if isinstance(entry, dict) and wanted & {entry.get("name"), entry.get("model")}:
            digest = entry.get("digest")
            if isinstance(digest, str) and digest:
                return digest
    return None


@dataclass(frozen=True)
class DecisionResult:
    answers: dict[str, Answer]
    usage: Usage
    ms: float


class DecisionModel:
    """A handle on one decision model of one Ollama server.

    ``log`` receives one record per request: model, digest, number of
    questions, token usage and milliseconds (and ``error`` on failure).
    ``usage`` and ``calls`` total the successful requests of this handle.
    """

    def __init__(
        self,
        name: str,
        *,
        digest: str | None = None,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = DEFAULT_TIMEOUT,
        load_timeout: float = DEFAULT_LOAD_TIMEOUT,
        log: LogFn | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = model_name(name)
        self.digest = digest
        self.capabilities: list[str] | None = None
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.load_timeout = load_timeout
        self.usage = Usage()
        self.calls = 0
        self._log = log
        self._loaded = False
        self._client = httpx.AsyncClient(base_url=self.base_url, transport=transport)

    @property
    def id(self) -> str:
        """The ``ollama:<name>`` model id."""
        return f"ollama:{self.name}"

    @classmethod
    async def open(
        cls,
        name: str,
        *,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = DEFAULT_TIMEOUT,
        load_timeout: float = DEFAULT_LOAD_TIMEOUT,
        require_capability: bool = True,
        log: LogFn | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> DecisionModel:
        """Check the server and the model, read the digest, return the handle.

        Raises ``DecisionModelUnavailable`` with the fix in the message when
        the server is unreachable or older than 0.35, the model is not
        installed, or it does not declare the ``decision`` capability. A
        server that reports no capabilities at all is accepted with a warning.
        """
        model = cls(
            name,
            base_url=base_url,
            timeout=timeout,
            load_timeout=load_timeout,
            log=log,
            transport=transport,
        )
        try:
            await model._check(require_capability=require_capability)
        except BaseException:
            await model.aclose()
            raise
        return model

    async def _check(self, *, require_capability: bool) -> None:
        version_text = (await self._json("GET", "/api/version")).get("version")
        if not isinstance(version_text, str):
            raise DecisionModelUnavailable(
                f"{self.base_url}/api/version returned no version"
            )
        if _parse_version(version_text) < MIN_OLLAMA_VERSION:
            raise DecisionModelUnavailable(
                f"Ollama {version_text} at {self.base_url} has no decision "
                "endpoint; Ollama 0.35 or later is needed"
            )
        shown = await self._json(
            "POST",
            "/api/show",
            body=json.dumps({"model": self.name}).encode(),
            missing=(
                f"the model {self.name!r} is not installed on {self.base_url}; "
                f"run: ollama pull {self.name}"
            ),
        )
        reported = shown.get("capabilities")
        capabilities = (
            [str(c) for c in reported] if isinstance(reported, list) else None
        )
        self.capabilities = capabilities
        if capabilities is None:
            warnings.warn(
                f"Ollama reports no capabilities for {self.name!r}; "
                "assuming it is a decision model.",
                stacklevel=3,
            )
        elif DECISION_CAPABILITY not in capabilities and require_capability:
            raise DecisionModelUnavailable(
                f"the model {self.name!r} does not declare the "
                f"{DECISION_CAPABILITY!r} capability (it declares "
                f"{capabilities}); it is not a decision model"
            )
        digest = find_digest(
            (await self._json("GET", "/api/tags")).get("models"), self.name
        )
        if digest is None:
            raise DecisionModelUnavailable(
                f"/api/tags lists no digest for {self.name!r}"
            )
        self.digest = digest

    async def _send(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None,
        timeout: float,
    ) -> httpx.Response:
        async with _lock_for(self.base_url):
            try:
                return await self._client.request(
                    method,
                    path,
                    content=body,
                    headers={"Content-Type": "application/json"} if body else None,
                    timeout=timeout,
                )
            except httpx.HTTPError as exc:
                raise DecisionModelUnavailable(
                    f"cannot reach Ollama at {self.base_url} ({exc!r}); "
                    "is `ollama serve` running?"
                ) from exc

    async def _json(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        missing: str | None = None,
    ) -> dict[str, object]:
        response = await self._send(method, path, body=body, timeout=self.timeout)
        if response.status_code == 404 and missing is not None:
            raise DecisionModelUnavailable(missing)
        if response.status_code >= 400:
            raise DecisionModelUnavailable(
                f"{method} {self.base_url}{path} answered {response.status_code}: "
                f"{response.text[:200]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise DecisionModelUnavailable(
                f"{self.base_url}{path} returned no JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise DecisionModelUnavailable(
                f"{self.base_url}{path} returned {payload!r}"
            )
        return payload

    def _emit(self, record: dict[str, object]) -> None:
        if self._log is not None:
            self._log({"model": self.name, "digest": self.digest, **record})

    async def decide_with_usage(
        self, state: dict[str, object] | str, questions: dict[str, Question]
    ) -> DecisionResult:
        """One request: ``questions`` about ``state``, answers and token usage.

        Raises ``DecisionRequestTooLarge`` before sending when a limit is
        exceeded, ``DecisionModelUnavailable`` when the server fails,
        ``DecisionRequestRejected`` on a client error and
        ``DecisionResponseError`` on a response of the wrong shape.
        """
        body = build_body(state, questions, name=self.name)
        started = time.monotonic()
        try:
            response = await self._send(
                "POST",
                "/v1/systemone",
                body=body,
                timeout=self.timeout if self._loaded else self.load_timeout,
            )
            if response.status_code == 404:
                raise DecisionModelUnavailable(
                    f"{self.base_url} has no /v1/systemone endpoint or no model "
                    f"{self.name!r}; run: ollama pull {self.name}"
                )
            if response.status_code >= 500:
                raise DecisionModelUnavailable(
                    f"Ollama answered {response.status_code}: {response.text[:200]}"
                )
            if response.status_code >= 400:
                raise DecisionRequestRejected(response.status_code, response.text[:300])
            try:
                payload = response.json()
            except ValueError as exc:
                raise DecisionResponseError("the response is not JSON") from exc
            answers = parse_answers(payload, questions)
            usage = parse_usage(payload)
        except Exception as exc:
            self._emit(
                {
                    "questions": len(questions),
                    "ms": round((time.monotonic() - started) * 1000, 1),
                    "error": str(exc),
                }
            )
            raise
        self._loaded = True
        ms = (time.monotonic() - started) * 1000
        self.usage += usage
        self.calls += 1
        self._emit(
            {
                "questions": len(questions),
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "ms": round(ms, 1),
            }
        )
        return DecisionResult(answers=answers, usage=usage, ms=ms)

    async def decide(
        self, state: dict[str, object] | str, questions: dict[str, Question]
    ) -> dict[str, Answer]:
        """``decide_with_usage`` without the usage."""
        return (await self.decide_with_usage(state, questions)).answers

    async def decide_many(
        self, states: list[dict[str, object] | str], question: Question
    ) -> list[Answer]:
        """The same question over many states: one request per state, one
        after the other, answers in the order of ``states``."""
        return [
            (await self.decide(state, {_QUESTION_KEY: question}))[_QUESTION_KEY]
            for state in states
        ]

    async def ask_all(
        self, state: dict[str, object] | str, questions: dict[str, Question]
    ) -> dict[str, Answer]:
        """Many questions about one state, in as few requests as the limits
        allow (at most 64 questions and 64 KiB each), sent one after the other."""
        answers: dict[str, Answer] = {}
        for chunk in chunk_questions(state, questions, name=self.name):
            answers.update(await self.decide(state, chunk))
        return answers

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> DecisionModel:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()


async def make_decision_model(
    name: str,
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = DEFAULT_TIMEOUT,
    load_timeout: float = DEFAULT_LOAD_TIMEOUT,
    require_capability: bool = True,
    log: LogFn | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> DecisionModel:
    """Open a checked ``DecisionModel`` (the counterpart of ``make_ollama_embedder``)."""
    return await DecisionModel.open(
        name,
        base_url=base_url,
        timeout=timeout,
        load_timeout=load_timeout,
        require_capability=require_capability,
        log=log,
        transport=transport,
    )


async def decide(
    state: dict[str, object] | str,
    questions: dict[str, Question],
    *,
    model: str,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = DEFAULT_TIMEOUT,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Answer]:
    """Put ``questions`` about ``state`` to a decision model, once.

    Builds a temporary, unchecked ``DecisionModel`` (no capability or digest
    check): for a loop, open one handle and reuse it. ``model`` is
    ``ollama:<name>`` or ``<name>``; ``transport`` exists for tests.
    """
    async with DecisionModel(
        model,
        base_url=base_url,
        timeout=timeout,
        load_timeout=timeout,
        transport=transport,
    ) as handle:
        return await handle.decide(state, questions)


async def model_digest(
    model: str,
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = DEFAULT_TIMEOUT,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    """The model's digest from Ollama's ``/api/tags`` listing (``/api/show``
    carries none), which identifies the exact weights. Raises if the model is not listed or has no digest."""
    name = model_name(model)
    async with httpx.AsyncClient(
        base_url=base_url, timeout=timeout, transport=transport
    ) as client:
        response = await client.get("/api/tags")
        response.raise_for_status()
        digest = find_digest(response.json().get("models", []), name)
    if digest is None:
        raise ValueError(f"/api/tags lists no digest for {model!r}")
    return digest
