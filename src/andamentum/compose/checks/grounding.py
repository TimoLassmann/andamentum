"""Grounding check: does each cited fact support the sentence that cites it?

One judge-role call per section whose content changed. Results are cached by
section content hash, so an unchanged section costs nothing on a later pass.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel

from andamentum.core.agents import run_agent_with_usage
from andamentum.scribe import Section

from ..log import ActionLog
from ..render import MARKER_RE
from ..schemas import CheckResult, Fact

GROUNDING_INSTRUCTIONS = (
    "You check whether sentences are supported by the evidence they cite. "
    "You are given numbered sentences. Each lists the cited facts as an id, a "
    "statement and a verbatim quote from the source. A sentence is supported "
    "when the cited quotes state what the sentence claims, including any number, "
    "name, date or direction of change in it. A sentence is unsupported when a "
    "claim in it goes beyond the quotes, contradicts them, or the cited facts "
    "are about something else. Judge only the claims that rely on the cited "
    "facts. List only unsupported sentences; if all are supported, return an "
    "empty list. For each, give the marker (for example f3), the sentence, and "
    "a short reason."
)


class Unsupported(BaseModel):
    marker: str
    sentence: str
    reason: str


class GroundingVerdict(BaseModel):
    unsupported: list[Unsupported]


def cited_sentences(body: str) -> list[tuple[str, list[str]]]:
    """Sentences of *body* that carry markers, with the fact ids each cites.

    A marker belongs to the sentence before it: ``Sales rose. [[f3]]`` and
    ``Sales rose [[f3]].`` both attach f3 to "Sales rose".
    """
    pieces = [p for p in re.split(r"(?<=[.!?])\s+|\n{2,}", body) if p and p.strip()]
    merged: list[str] = []
    for piece in pieces:
        lead = re.match(r"\s*((?:\[\[f\d+\]\]\s*)+)", piece)
        if lead and merged:
            merged[-1] = f"{merged[-1]} {lead.group(1).strip()}"
            piece = piece[lead.end() :]
            if not piece.strip():
                continue
        merged.append(piece)
    out: list[tuple[str, list[str]]] = []
    for sentence in merged:
        ids = MARKER_RE.findall(sentence)
        if ids:
            clean = MARKER_RE.sub("", sentence)
            clean = re.sub(r"\s+", " ", clean).strip()
            clean = re.sub(r" ([.,;:!?])", r"\1", clean)
            out.append((clean, list(dict.fromkeys(ids))))
    return out


def _user_message(
    sentences: Sequence[tuple[str, list[str]]], *, by_id: dict[str, Fact]
) -> str:
    lines: list[str] = []
    for i, (sentence, ids) in enumerate(sentences, 1):
        lines.append(f"{i}. {sentence}")
        for fact_id in ids:
            fact = by_id[fact_id]
            lines.append(f"   [{fact.id}] {fact.statement} | quote: {fact.quote}")
    return "\n".join(lines)


async def check_grounding(
    sections: Sequence[Section],
    *,
    facts: Sequence[Fact],
    model: str,
    log: ActionLog,
    cache: dict[str, list[Unsupported]],
) -> list[CheckResult]:
    """One ``grounding`` CheckResult per section that cites facts.

    *cache* maps a section's content hash to its verdict and is owned by the
    caller (the run state); an unchanged section is never judged twice.
    Sentences citing an id that is not a known fact are left to the
    deterministic ``fact_markers`` check. Sequential: one call at a time.
    """
    by_id = {f.id: f for f in facts}
    results: list[CheckResult] = []
    for section in sorted(sections, key=lambda s: s.order):
        sentences = [
            (text, [i for i in ids if i in by_id])
            for text, ids in cited_sentences(section.body)
        ]
        sentences = [(text, ids) for text, ids in sentences if ids]
        if not sentences:
            continue
        key = section.content_hash
        if key not in cache:
            message = _user_message(sentences, by_id=by_id)
            verdict, usage = await run_agent_with_usage(
                model,
                instructions=GROUNDING_INSTRUCTIONS,
                output_type=GroundingVerdict,
                user_message=message,
            )
            log.call(
                node="Check",
                role="judge",
                model=model,
                prompt=GROUNDING_INSTRUCTIONS,
                input_text=message,
                output_text=verdict.model_dump_json(),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )
            cache[key] = list(verdict.unsupported)
        bad = cache[key]
        results.append(
            CheckResult(
                name="grounding",
                passed=not bad,
                details=[f"{u.marker}: {u.sentence} ({u.reason})".strip() for u in bad],
                section_id=section.id,
            )
        )
    return results
