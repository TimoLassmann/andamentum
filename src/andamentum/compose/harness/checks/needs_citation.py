"""Decision-model pre-gate: which uncited sentences look like they need a citation.

Optional (``--decision-model ollama:<name>``). For each sentence of the body the
decision model answers one yes/no question, "does this sentence state a factual
claim that needs a citation?", as a probability. A sentence at or above the
threshold that overlaps no citation field is an *info* diagnostic. It is a
pointer for the author, never a finding of fact.

The threshold comes from the model's calibration record
(``CheckContext.calibrations["needs_citation"]``): the lowest probability at
which the measured precision reaches ``PROMOTE_AT``. With no record, or when
no threshold reaches that precision, ``ADVISORY_THRESHOLD`` is used and the
check stays advisory (tier 2, through the kernel's promotion policy). Sentences
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

from andamentum.core.decisions import CalibrationRecord, DecisionQuestion, decide

from ..kernel import (
    PROMOTE_AT,
    CheckContext,
    CheckSpec,
    Diagnostic,
    DocumentView,
    Paragraph,
    register,
)
from ..kernel.checks import cache_key

ADVISORY_THRESHOLD = 0.8
MAX_SENTENCES = 400
MIN_WORDS = 6
CHECK_ID = "needs_citation"
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])")
_REFERENCE_HEADINGS = {"references", "bibliography", "works cited", "reference list"}
_QUESTION = DecisionQuestion(
    type="noul",
    instructions=(
        "Does this sentence state a factual claim that needs a citation? "
        "Answer yes for a statement about what is known, found or true in the "
        "world; no for the authors' own plans, methods, definitions or "
        "transitions."
    ),
)


def threshold_for_run(record: CalibrationRecord | None) -> tuple[float, bool]:
    """``(threshold, calibrated)``: the record's threshold at ``PROMOTE_AT``, or the advisory default."""
    if record is not None:
        found = record.threshold_for(precision=PROMOTE_AT)
        if found is not None:
            return found, True
    return ADVISORY_THRESHOLD, False


def measured_precision(record: CalibrationRecord | None) -> float | None:
    """The precision the record measured at the threshold the check will use."""
    if record is None:
        return None
    threshold, calibrated = threshold_for_run(record)
    return record.stats_at(threshold).precision if calibrated else None


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
        model = ctx.models.get("decision_model")
        if model is None:
            raise ValueError("the needs_citation check needs --decision-model")
        threshold, calibrated = threshold_for_run(ctx.calibrations.get(CHECK_ID))
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
            if p_yes >= threshold:
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
                        data={
                            "probability": p_yes,
                            "threshold": threshold,
                            "calibrated": calibrated,
                        },
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

    async def _probability(self, sentence: str, model: str, ctx: CheckContext) -> float:
        key = cache_key(self, version=f"{self.version}:{model}", inputs=[sentence])
        if ctx.store is not None:
            cached = ctx.store.cache_get(CHECK_ID, key)
            if isinstance(cached, float):
                return cached
        answers = await decide(sentence, {CHECK_ID: _QUESTION}, model=model)
        p_yes = answers[CHECK_ID].probabilities["yes"]
        ctx.stats.model_calls += 1
        if ctx.store is not None:
            ctx.store.cache_put(CHECK_ID, key, p_yes)
        return p_yes


register(NeedsCitationCheck())
