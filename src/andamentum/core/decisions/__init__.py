"""Decision models: choices and probabilities instead of text.

Local Ollama only: ``POST /v1/systemone`` (Ollama 0.35 and later, TypeSafe's
Jev API). One request puts 1 to 64 named questions about one ``state``; a
question is ``noul`` (probability a condition holds), ``choice`` (probabilities
over described options) or ``score`` (weak, never a gate).

- ``DecisionModel`` / ``make_decision_model``: a long-lived, checked handle
  (server version, capability, digest); every request to one server is
  serialised by one lock, so local calls are never concurrent.
- ``Question`` / ``Answer``, ``decide`` (one-shot), ``DecisionModel.decide``,
  ``decide_many``, ``ask_all`` (chunked at 64 questions and 64 KiB).

A site that is given a decision model uses it instead of its LLM call: a
``choice`` question gives the top choice, a ``noul`` question is a yes when
``p_true >= 0.5``. Without one the site calls its LLM as before.

``python -m andamentum.core.decisions check --model NAME`` is the CLI.
``testing.StubDecisions`` stubs the server for offline tests.
"""

from .api import (
    MAX_BODY_BYTES,
    MAX_QUESTIONS,
    Answer,
    DecisionAnswer,
    DecisionQuestion,
    Question,
    Usage,
    model_name,
    question_hash,
)
from .errors import (
    DecisionError,
    DecisionModelUnavailable,
    DecisionRequestRejected,
    DecisionRequestTooLarge,
    DecisionResponseError,
)
from .model import (
    DecisionModel,
    DecisionResult,
    decide,
    find_digest,
    make_decision_model,
    model_digest,
    server_lock,
)

__all__ = [
    "Answer",
    "DecisionAnswer",
    "DecisionError",
    "DecisionModel",
    "DecisionModelUnavailable",
    "DecisionQuestion",
    "DecisionRequestRejected",
    "DecisionRequestTooLarge",
    "DecisionResponseError",
    "DecisionResult",
    "MAX_BODY_BYTES",
    "MAX_QUESTIONS",
    "Question",
    "Usage",
    "decide",
    "find_digest",
    "make_decision_model",
    "model_digest",
    "model_name",
    "question_hash",
    "server_lock",
]
