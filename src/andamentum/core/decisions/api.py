"""The decision-model vocabulary: questions, answers, the wire shape, the limits.

Pure data and parsing, no I/O. Ollama's ``POST /v1/systemone`` (0.35 and
later, TypeSafe's Jev API) takes ``{model, state, questions: {name: {type,
instructions, criteria}}}`` and answers ``{model, answers: {name: {type,
...}}, usage}``:

- ``choice`` carries ``criteria`` as an object mapping each option to a
  description; the answer has ``choice``, ``probabilities``, ``confidence``.
- ``score`` carries ``criteria`` as an array of ordered levels; the answer has
  ``score``, ``legend``, ``probabilities``, ``confidence``. Weak (50-55%
  accuracy in the vendor numbers): never a gate.
- ``noul`` (a yes/no condition) carries optional ``criteria`` ``{true, false}``;
  the answer is ``{noul: p}``, the probability the condition holds.

The wire shape lives in ``Question.wire``, ``build_body`` and ``parse_answers``
only; if Ollama changes it, those are the whole change.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from .errors import DecisionRequestTooLarge, DecisionResponseError

MAX_QUESTIONS = 64
MAX_BODY_BYTES = 64 * 1024
MIN_OPTIONS = 2
MAX_CHOICE_OPTIONS = 26
MAX_SCORE_LEVELS = 10

_FOREIGN_PREFIXES = frozenset(
    {
        "openai",
        "anthropic",
        "bedrock",
        "google-gla",
        "google-vertex",
        "groq",
        "mistral",
        "cohere",
        "huggingface",
        "gateway",
        "azure",
        "deepseek",
        "cerebras",
        "xai",
    }
)


def model_name(model: str) -> str:
    """The Ollama model name of ``ollama:<name>`` or ``<name>``.

    Ollama names carry their own colon (``nimble:9b``), so a bare name is
    accepted; an id for another provider is refused.
    """
    prefix, _, rest = model.partition(":")
    name = rest if prefix == "ollama" else model
    if not name or prefix in _FOREIGN_PREFIXES:
        raise ValueError(
            f"decision models run on local Ollama and need an 'ollama:<name>' "
            f"or '<name>' model id, got {model!r}"
        )
    return name


class Question(BaseModel):
    """One question put to a decision model.

    ``options`` (choice, score): a ``{option: description}`` mapping, or a
    list (a choice list becomes ``{option: option}``; a score list is the
    ordered levels). ``criteria`` (noul only): optional ``{"true": ...,
    "false": ...}`` descriptions of the two outcomes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["choice", "noul", "score"]
    instructions: str
    options: dict[str, str] | list[str] | None = None
    criteria: dict[str, str] | None = None

    @model_validator(mode="after")
    def _shape_matches_type(self) -> Question:
        if self.type == "noul":
            if self.options is not None:
                raise ValueError("a noul question takes no options")
            if self.criteria is not None and not set(self.criteria) <= {
                "true",
                "false",
            }:
                raise ValueError("noul criteria keys must be 'true' and 'false'")
            return self
        if self.criteria is not None:
            raise ValueError(f"a {self.type} question takes options, not criteria")
        if not self.options:
            raise ValueError(f"a {self.type} question needs options")
        names = list(self.options)
        if len(set(names)) != len(names):
            raise ValueError("options must be distinct")
        if self.type == "score":
            if not isinstance(self.options, list):
                raise ValueError("score levels are a list, in order")
            if not MIN_OPTIONS <= len(names) <= MAX_SCORE_LEVELS:
                raise ValueError(
                    f"a score question takes {MIN_OPTIONS} to {MAX_SCORE_LEVELS} levels"
                )
        elif not MIN_OPTIONS <= len(names) <= MAX_CHOICE_OPTIONS:
            raise ValueError(
                f"a choice question takes {MIN_OPTIONS} to {MAX_CHOICE_OPTIONS} options"
            )
        return self

    @property
    def option_names(self) -> list[str]:
        return list(self.options or [])

    @property
    def option_descriptions(self) -> dict[str, str]:
        if isinstance(self.options, dict):
            return dict(self.options)
        return {o: o for o in self.options or []}

    def wire(self) -> dict[str, object]:
        entry: dict[str, object] = {
            "type": self.type,
            "instructions": self.instructions,
        }
        if self.type == "choice":
            entry["criteria"] = self.option_descriptions
        elif self.type == "score":
            entry["criteria"] = self.option_names
        elif self.criteria is not None:
            entry["criteria"] = dict(self.criteria)
        return entry


def question_hash(question: Question) -> str:
    """A hash of what the question asks (type, instructions, options with
    descriptions, criteria); changing any of these changes the hash (a cache
    key for answers)."""
    canonical = {
        "type": question.type,
        "instructions": question.instructions,
        "options": (
            question.option_names
            if question.type == "score"
            else question.option_descriptions
        ),
        "criteria": question.criteria,
    }
    text = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Answer(BaseModel):
    """A decision model's answer to one question.

    ``choice`` is the selected option (``choice`` questions; for ``noul`` it
    is "yes" or "no" by the 0.5 rule). ``probabilities`` maps each option to
    its probability; for ``noul`` it is ``{"yes": p, "no": 1 - p}``.
    ``confidence`` is the model's reported concentration, except for ``noul``,
    where the endpoint reports none and it is ``max(p, 1 - p)``; it is not
    correctness. ``p_true`` is the probability a ``noul`` condition holds.
    ``score`` is set for ``score`` questions only.
    """

    model_config = ConfigDict(extra="forbid")

    choice: str | None
    probabilities: dict[str, float]
    confidence: float
    score: float | None = None
    p_true: float | None = None


DecisionAnswer = Answer
DecisionQuestion = Question


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


def _encode(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def build_body(
    state: dict[str, object] | str, questions: dict[str, Question], *, name: str
) -> bytes:
    """The request body, checked against the endpoint's limits before anything
    is sent: 1 to 64 questions and at most 64 KiB (the endpoint answers an
    over-length input with HTTP 400, it never truncates)."""
    if not questions:
        raise ValueError("decide() needs at least one question")
    if len(questions) > MAX_QUESTIONS:
        raise DecisionRequestTooLarge(
            f"{len(questions)} questions in one request; the limit is "
            f"{MAX_QUESTIONS} (use ask_all to chunk)"
        )
    body = _encode(
        {
            "model": name,
            "state": state,
            "questions": {key: q.wire() for key, q in questions.items()},
        }
    )
    if len(body) > MAX_BODY_BYTES:
        raise DecisionRequestTooLarge(
            f"the request is {len(body)} bytes; the limit is {MAX_BODY_BYTES}"
        )
    return body


def chunk_questions(
    state: dict[str, object] | str, questions: dict[str, Question], *, name: str
) -> list[dict[str, Question]]:
    """Split ``questions`` into requests of at most 64 questions that each fit
    the body limit with ``state``. The size estimate errs high."""
    base = len(_encode({"model": name, "state": state, "questions": {}}))
    chunks: list[dict[str, Question]] = []
    current: dict[str, Question] = {}
    size = base
    for key, question in questions.items():
        added = len(_encode(key)) + len(_encode(question.wire())) + 2
        if base + added > MAX_BODY_BYTES:
            raise DecisionRequestTooLarge(
                f"question {key!r} with its state is {base + added} bytes; "
                f"the limit is {MAX_BODY_BYTES}"
            )
        if current and (len(current) >= MAX_QUESTIONS or size + added > MAX_BODY_BYTES):
            chunks.append(current)
            current, size = {}, base
        current[key] = question
        size += added
    if current:
        chunks.append(current)
    return chunks


def _probability(raw: object, *, where: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise DecisionResponseError(f"{where}: expected a number, got {raw!r}")
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise DecisionResponseError(f"{where}: probability {value} outside [0, 1]")
    return value


def _probabilities(raw: object, *, where: str) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise DecisionResponseError(
            f"{where}: probabilities must be an object, got {raw!r}"
        )
    return {str(k): _probability(v, where=f"{where}.{k}") for k, v in raw.items()}


def parse_answers(payload: object, questions: dict[str, Question]) -> dict[str, Answer]:
    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
        raise DecisionResponseError(
            f"systemone response has no 'answers' object: {payload!r}"
        )
    answers: dict[str, object] = payload["answers"]
    missing = set(questions) - set(answers)
    if missing:
        raise DecisionResponseError(
            f"systemone response lacks answers for {sorted(missing)}"
        )
    parsed: dict[str, Answer] = {}
    for key, q in questions.items():
        raw = answers[key]
        if not isinstance(raw, dict):
            raise DecisionResponseError(f"answer {key!r} is not an object: {raw!r}")
        where = f"answer {key!r}"
        if q.type == "noul":
            p = _probability(raw.get("noul"), where=f"{where}.noul")
            parsed[key] = Answer(
                choice="yes" if p >= 0.5 else "no",
                probabilities={"yes": p, "no": 1.0 - p},
                confidence=max(p, 1.0 - p),
                p_true=p,
            )
            continue
        probs = _probabilities(raw.get("probabilities"), where=where)
        confidence = _probability(raw.get("confidence"), where=f"{where}.confidence")
        if q.type == "choice":
            choice = raw.get("choice")
            if not isinstance(choice, str) or choice not in q.option_names:
                raise DecisionResponseError(
                    f"{where}: choice {choice!r} is not one of {q.option_names}"
                )
            parsed[key] = Answer(
                choice=choice, probabilities=probs, confidence=confidence
            )
        else:
            score = raw.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise DecisionResponseError(
                    f"{where}: score must be a number, got {score!r}"
                )
            parsed[key] = Answer(
                choice=None,
                probabilities=probs,
                confidence=confidence,
                score=float(score),
            )
    return parsed


def parse_usage(payload: object) -> Usage:
    """Token usage from the response; absent counts are zero."""
    raw = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        return Usage()

    def count(key: str) -> int:
        value = raw.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DecisionResponseError(f"usage.{key}: expected a count, got {value!r}")
        return value

    return Usage(count("input_tokens"), count("output_tokens"))
