"""Classify enricher: user-defined labels chosen by a decision model (opt-in)."""

from __future__ import annotations

from andamentum.core.decisions import DecisionModel, Question

from ..store import Store
from ._base import EnrichResult, load, write_key

KEY = "labels"
_THRESHOLD = 0.5
_MIN_CHOICE_LABELS = 2


async def enrich_classify(
    store: Store,
    doc_id: str,
    *,
    decision_model: DecisionModel,
    labels: dict[str, str],
    max_chars: int,
    multi: bool = False,
) -> EnrichResult:
    """Write the chosen label(s) under ``metadata["labels"]``.

    ``labels`` maps each label to a description; there is no built-in ontology.
    Single-label (default): one ``choice`` question, the top option wins (needs
    at least two labels). ``multi=True``: one yes/no question per label, a label
    is chosen when its probability is at least 0.5.

    The decision server rejects over-size requests, so the text is cut to its
    first ``max_chars`` characters; the stored value records ``truncated``.
    The value holds ``labels`` (chosen), ``probabilities``, ``multi``,
    ``decision_model``, ``digest``, ``max_chars`` and ``truncated``.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if not labels:
        raise ValueError("labels must not be empty")
    if not multi and len(labels) < _MIN_CHOICE_LABELS:
        raise ValueError("single-label classification needs at least two labels")
    doc = await load(store, doc_id)
    state: dict[str, object] = {"document": doc.text[:max_chars]}

    chosen: list[str]
    probabilities: dict[str, float]
    if multi:
        questions = {
            label: Question(
                type="noul",
                instructions=f"Does the label '{label}' apply to the document? "
                f"'{label}' means: {description}",
            )
            for label, description in labels.items()
        }
        answers = await decision_model.ask_all(state, questions)
        probabilities = {}
        for label in labels:
            p = answers[label].p_true
            if p is None:
                raise ValueError(f"the decision model gave no probability for {label!r}")
            probabilities[label] = p
        chosen = [label for label, p in probabilities.items() if p >= _THRESHOLD]
    else:
        question = Question(
            type="choice",
            instructions="Which label best describes the document?",
            options=dict(labels),
        )
        answers = await decision_model.ask_all(state, {"label": question})
        answer = answers["label"]
        if answer.choice is None:
            raise ValueError("the decision model gave no choice")
        chosen = [answer.choice]
        probabilities = dict(answer.probabilities)

    value = {
        "labels": chosen,
        "probabilities": probabilities,
        "multi": multi,
        "decision_model": decision_model.name,
        "digest": decision_model.digest,
        "max_chars": max_chars,
        "truncated": len(doc.text) > max_chars,
    }
    return await write_key(store, doc, key=KEY, value=value)
