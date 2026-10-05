"""Decision models: choices, probabilities and scores instead of text.

Talks to Ollama's ``POST /v1/systemone`` endpoint (Ollama 0.35 and later,
based on TypeSafe's Jev API). Shape confirmed against the Ollama blog post
"Ollama now supports Jev-style decision models" and the v0.35.0 release notes:

Request  ``{model, state, questions: {name: {type, instructions, criteria}}}``
  - ``choice`` carries ``criteria`` as an object mapping each option to a
    description; ``score`` carries ``criteria`` as an array of options;
    ``noul`` (a yes/no condition) carries no criteria.
Response ``{model, answers: {name: {type, ...}}, usage}``
  - ``choice``: ``choice``, ``probabilities``, ``confidence``
  - ``score``: ``score``, ``legend``, ``probabilities``, ``confidence``
  - ``noul``: ``noul`` (the probability the condition holds)

The wire shape lives in ``_request_body`` and ``_parse_answers`` only; if
Ollama changes it, those two functions are the whole change.

Decision models order and escalate work, they never certify. Use
``calibrate`` to measure how far a model's probabilities can be trusted on a
labelled set before acting on a threshold.

Calls are sequential: ``decide`` sends one request and awaits it, and callers
must not gather several local-model calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .embeddings import DEFAULT_OLLAMA_URL

__all__ = [
    "CalibrationRecord",
    "DecisionAnswer",
    "DecisionQuestion",
    "ReliabilityBin",
    "ThresholdStat",
    "calibrate",
    "decide",
    "model_digest",
]


class DecisionQuestion(BaseModel):
    """One question put to a decision model."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice", "noul", "score"]
    instructions: str
    options: list[str] | None = None  # required for choice and score

    @model_validator(mode="after")
    def _options_match_type(self) -> DecisionQuestion:
        if self.type == "noul":
            if self.options is not None:
                raise ValueError("a noul question takes no options")
        elif not self.options:
            raise ValueError(f"a {self.type} question needs options")
        elif len(set(self.options)) != len(self.options):
            raise ValueError("options must be distinct")
        return self


class DecisionAnswer(BaseModel):
    """A decision model's answer to one question.

    ``choice`` is the selected option (``choice`` questions; for ``noul`` it
    is "yes" or "no" by the 0.5 rule). ``probabilities`` maps each option to
    its probability; for ``noul`` it is ``{"yes": p, "no": 1 - p}``.
    ``confidence`` is the model's reported confidence, except for ``noul``,
    where the endpoint reports none and it is ``max(p, 1 - p)``. ``score`` is
    set for ``score`` questions only.
    """

    model_config = ConfigDict(extra="forbid")

    choice: str | None
    probabilities: dict[str, float]
    confidence: float
    score: float | None = None


def _model_name(model: str) -> str:
    prefix, _, name = model.partition(":")
    if prefix != "ollama" or not name:
        raise ValueError(
            f"decision models need an 'ollama:<name>' model id, got {model!r}"
        )
    return name


def _request_body(
    state: dict[str, object] | str,
    questions: dict[str, DecisionQuestion],
    *,
    name: str,
) -> dict[str, object]:
    wire: dict[str, dict[str, object]] = {}
    for key, q in questions.items():
        entry: dict[str, object] = {"type": q.type, "instructions": q.instructions}
        if q.type == "choice":
            assert q.options is not None
            entry["criteria"] = {option: option for option in q.options}
        elif q.type == "score":
            entry["criteria"] = list(q.options or [])
        wire[key] = entry
    return {"model": name, "state": state, "questions": wire}


def _probability(raw: object, *, where: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{where}: expected a number, got {raw!r}")
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{where}: probability {value} outside [0, 1]")
    return value


def _probabilities(raw: object, *, where: str) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: probabilities must be an object, got {raw!r}")
    return {str(k): _probability(v, where=f"{where}.{k}") for k, v in raw.items()}


def _parse_answers(
    payload: object, questions: dict[str, DecisionQuestion]
) -> dict[str, DecisionAnswer]:
    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
        raise ValueError(f"systemone response has no 'answers' object: {payload!r}")
    answers: dict[str, object] = payload["answers"]
    missing = set(questions) - set(answers)
    if missing:
        raise ValueError(f"systemone response lacks answers for {sorted(missing)}")
    parsed: dict[str, DecisionAnswer] = {}
    for key, q in questions.items():
        raw = answers[key]
        if not isinstance(raw, dict):
            raise ValueError(f"answer {key!r} is not an object: {raw!r}")
        where = f"answer {key!r}"
        if q.type == "noul":
            p = _probability(raw.get("noul"), where=f"{where}.noul")
            parsed[key] = DecisionAnswer(
                choice="yes" if p >= 0.5 else "no",
                probabilities={"yes": p, "no": 1.0 - p},
                confidence=max(p, 1.0 - p),
            )
            continue
        probs = _probabilities(raw.get("probabilities"), where=where)
        confidence = _probability(raw.get("confidence"), where=f"{where}.confidence")
        if q.type == "choice":
            choice = raw.get("choice")
            if not isinstance(choice, str) or choice not in (q.options or []):
                raise ValueError(
                    f"{where}: choice {choice!r} is not one of {q.options}"
                )
            parsed[key] = DecisionAnswer(
                choice=choice, probabilities=probs, confidence=confidence
            )
        else:
            score = raw.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"{where}: score must be a number, got {score!r}")
            parsed[key] = DecisionAnswer(
                choice=None,
                probabilities=probs,
                confidence=confidence,
                score=float(score),
            )
    return parsed


async def decide(
    state: dict[str, object] | str,
    questions: dict[str, DecisionQuestion],
    *,
    model: str,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 30.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, DecisionAnswer]:
    """Put ``questions`` about ``state`` to an Ollama decision model.

    ``model`` is ``ollama:<name>``. One request, awaited; raises on HTTP
    errors and on any response that does not match the documented shape.
    ``transport`` exists for tests.
    """
    if not questions:
        raise ValueError("decide() needs at least one question")
    body = _request_body(state, questions, name=_model_name(model))
    async with httpx.AsyncClient(
        base_url=base_url, timeout=timeout, transport=transport
    ) as client:
        response = await client.post("/v1/systemone", json=body)
        response.raise_for_status()
        return _parse_answers(response.json(), questions)


async def model_digest(
    model: str,
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 30.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    """The model's digest from Ollama's ``/api/tags`` listing (``/api/show``
    carries none), to pin a calibration to the exact weights it was measured
    on. Raises if the model is not listed or has no digest."""
    name = _model_name(model)
    async with httpx.AsyncClient(
        base_url=base_url, timeout=timeout, transport=transport
    ) as client:
        response = await client.get("/api/tags")
        response.raise_for_status()
        listed = response.json().get("models", [])
    for entry in listed:
        if name in (entry.get("name"), entry.get("model")):
            digest = entry.get("digest")
            if isinstance(digest, str) and digest:
                return digest
    raise ValueError(f"/api/tags lists no digest for {model!r}")


# --- calibration -----------------------------------------------------------


class ReliabilityBin(BaseModel):
    """Predictions whose probability falls in [lower, upper) (last bin closed)."""

    lower: float
    upper: float
    count: int
    mean_probability: float | None
    fraction_positive: float | None


class ThresholdStat(BaseModel):
    """Precision and recall of the rule "probability >= threshold"."""

    threshold: float
    predicted: int
    precision: float | None  # None when nothing is predicted
    recall: float | None  # None when the set has no positives


class CalibrationRecord(BaseModel):
    """How far a model's probabilities can be trusted on one labelled set."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    model: str
    digest: str
    task: str
    n: int
    base_rate: float
    ece: float
    bins: list[ReliabilityBin]
    thresholds: list[ThresholdStat]
    pairs: list[tuple[float, bool]] = Field(repr=False)

    def stats_at(self, threshold: float) -> ThresholdStat:
        return _threshold_stat(self.pairs, threshold)

    def threshold_for(self, *, precision: float, min_support: int = 5) -> float | None:
        """Lowest threshold whose precision is at least ``precision`` with at
        least ``min_support`` predictions (lowest keeps recall highest).
        None when no threshold qualifies, so callers stay advisory."""
        return threshold_for(self.pairs, precision=precision, min_support=min_support)

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CalibrationRecord:
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _threshold_stat(pairs: list[tuple[float, bool]], threshold: float) -> ThresholdStat:
    predicted = [label for p, label in pairs if p >= threshold]
    positives = sum(label for _, label in pairs)
    true_positives = sum(predicted)
    return ThresholdStat(
        threshold=threshold,
        predicted=len(predicted),
        precision=true_positives / len(predicted) if predicted else None,
        recall=true_positives / positives if positives else None,
    )


def threshold_for(
    pairs: list[tuple[float, bool]], *, precision: float, min_support: int = 5
) -> float | None:
    """See ``CalibrationRecord.threshold_for``."""
    if not 0.0 < precision <= 1.0:
        raise ValueError("precision must be in (0, 1]")
    for candidate in sorted({p for p, _ in pairs}):
        stat = _threshold_stat(pairs, candidate)
        if (
            stat.predicted >= min_support
            and stat.precision is not None
            and stat.precision >= precision
        ):
            return candidate
    return None


def calibrate(
    pairs: list[tuple[float, bool]],
    *,
    model: str,
    digest: str,
    task: str,
    n_bins: int = 10,
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
) -> CalibrationRecord:
    """Measure reliability of ``(probability, label)`` pairs.

    ECE is the count-weighted mean gap between each bin's mean probability and
    its observed positive fraction.
    """
    if not pairs:
        raise ValueError("calibrate() needs at least one pair")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    for p, _ in pairs:
        _probability(p, where="calibration pair")
    grouped: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for p, label in pairs:
        grouped[min(int(p * n_bins), n_bins - 1)].append((p, label))
    bins: list[ReliabilityBin] = []
    ece = 0.0
    for i, group in enumerate(grouped):
        mean_p = sum(p for p, _ in group) / len(group) if group else None
        frac = sum(label for _, label in group) / len(group) if group else None
        if mean_p is not None and frac is not None:
            ece += len(group) / len(pairs) * abs(mean_p - frac)
        bins.append(
            ReliabilityBin(
                lower=i / n_bins,
                upper=(i + 1) / n_bins,
                count=len(group),
                mean_probability=mean_p,
                fraction_positive=frac,
            )
        )
    return CalibrationRecord(
        model=model,
        digest=digest,
        task=task,
        n=len(pairs),
        base_rate=sum(label for _, label in pairs) / len(pairs),
        ece=ece,
        bins=bins,
        thresholds=[_threshold_stat(pairs, t) for t in thresholds],
        pairs=list(pairs),
    )
