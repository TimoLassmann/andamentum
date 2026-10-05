"""Decision-model pre-gate: which uncited sentences look like they need a citation.

Optional (``--decision-model ollama:<name>``). For each sentence of the body the
decision model answers one yes/no question, "does this sentence state a factual
claim that needs a citation?", as a probability. A sentence at or above the
threshold that overlaps no citation field is an *info* diagnostic. It is a
pointer for the author, never a finding of fact.

A sentence counts when the decision model's probability of "yes" is at least
``THRESHOLD`` (0.5, the model's own yes). The check is model-judged, so the
kernel's promotion policy reports it at tier 2 (advisory) unless a measured
precision is on record in ``precision.json``. Sentences
are asked one at a time, sequentially. Headings, captions, reference-list
paragraphs and sentences under six words are not asked, and at most
``MAX_SENTENCES`` sentences are asked per run (an info diagnostic says how many
were left).

Coverage limits: a sentence is cited when a citation field or plain citation
the reader found overlaps it; a citation in the following sentence does not
count. Sentence splitting is a regular expression (full stop, question or
exclamation mark followed by a capital), so abbreviations can split one; a
citation that follows the full stop ("... day. [3]") stays with its sentence.
"""

from __future__ import annotations

import re

from andamentum.core.decisions import DecisionModel, Question, question_hash

from ..kernel import (
    CheckContext,
    CheckSpec,
    Diagnostic,
    DocumentView,
    Paragraph,
    register,
)
from ..kernel.checks import cache_key

THRESHOLD = 0.5
MAX_SENTENCES = 400
MIN_WORDS = 6
CHECK_ID = "needs_citation"
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])")
_REFERENCE_HEADINGS = {"references", "bibliography", "works cited", "reference list"}
QUESTION = Question(
    type="noul",
    instructions=(
        "Does this sentence state a factual claim that needs a citation? "
        "Answer yes for a statement about what is known, found or true in the "
        "world; no for the authors' own plans, methods, definitions or "
        "transitions."
    ),
)


def _sentences(paragraph: Paragraph) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    start = 0
    for piece in _SENTENCE.split(paragraph.text):
        at = paragraph.text.find(piece, start)
        if at < 0:
            continue
        start = at + len(piece)
        if len(piece.split()) >= MIN_WORDS:
            out.append((piece, at, at + len(piece)))
    return out


def _cited_spans(view: DocumentView) -> dict[str, list[tuple[int, int]]]:
    spans: dict[str, list[tuple[int, int]]] = {}
    for token in view.fields:
        if token.kind == "citation":
            spans.setdefault(token.paragraph_id, []).append((token.start, token.end))
    return spans


class NeedsCitationCheck:
    spec = CheckSpec(id=CHECK_ID, tier=1, scope="span", needs=["decision_model"])
    version = "1"

    async def run(self, view: DocumentView, ctx: CheckContext) -> list[Diagnostic]:
        model = ctx.decision_model
        if model is None:
            raise ValueError("the needs_citation check needs --decision-model")
        cited = _cited_spans(view)
        todo: list[tuple[Paragraph, str, int, int]] = []
        for paragraph in view.paragraphs:
            if paragraph.heading_level is not None:
                if paragraph.text.strip().strip(":").lower() in _REFERENCE_HEADINGS:
                    break
                continue
            if "caption" in paragraph.style.lower():
                continue
            for sentence, start, end in _sentences(paragraph):
                overlaps = any(
                    s < end and start < e for s, e in cited.get(paragraph.id, [])
                )
                if not overlaps:
                    todo.append((paragraph, sentence, start, end))
        asked, left = todo[:MAX_SENTENCES], todo[MAX_SENTENCES:]
        out: list[Diagnostic] = []
        for paragraph, sentence, start, end in asked:
            p_yes = await self._probability(sentence, model, ctx)
            if p_yes >= THRESHOLD:
                out.append(
                    Diagnostic(
                        check=self.spec.id,
                        tier=self.spec.tier,
                        severity="info",
                        paragraph_id=paragraph.id,
                        start=start,
                        end=end,
                        message=(
                            "This sentence may state a claim that needs a "
                            f"citation (probability {p_yes:.2f})."
                        ),
                        evidence=[sentence],
                        data={"probability": p_yes, "threshold": THRESHOLD},
                    )
                )
        if left:
            out.append(
                Diagnostic(
                    check=self.spec.id,
                    tier=self.spec.tier,
                    severity="info",
                    paragraph_id=None,
                    start=None,
                    end=None,
                    message=(
                        f"{len(left)} sentence(s) were not asked "
                        f"(limit {MAX_SENTENCES} per run)."
                    ),
                    evidence=[],
                )
            )
        return out

    async def _probability(
        self, sentence: str, model: DecisionModel, ctx: CheckContext
    ) -> float:
        # Keyed by the weights and the question, not the model's name.
        key = cache_key(
            self,
            version=f"{self.version}:{model.digest}:{question_hash(QUESTION)}",
            inputs=[sentence],
        )
        if ctx.store is not None:
            cached = ctx.store.cache_get(CHECK_ID, key)
            if isinstance(cached, float):
                return cached
        result = await model.decide_with_usage(sentence, {CHECK_ID: QUESTION})
        p_yes = result.answers[CHECK_ID].p_true
        assert p_yes is not None
        ctx.stats.add_usage(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
        )
        if ctx.store is not None:
            ctx.store.cache_put(CHECK_ID, key, p_yes)
        return p_yes


register(NeedsCitationCheck())
