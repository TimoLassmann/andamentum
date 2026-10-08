"""Claim extraction + verify (the only LLM in document-model building).

ExtractDigest (agent, per section) returns verbatim claim spans; VerifyDigest
(deterministic `locate` + an agent re-quote on a miss, ≤3 attempts) keeps only
the spans that are actually in the source. A claim's stored quote is the
*located source text*, so it is verbatim by construction.
"""

from __future__ import annotations

import asyncio
import logging
from typing import cast

from pydantic import BaseModel, Field

from andamentum.core.agents import AgentDefinition, build_pydantic_ai_agent
from andamentum.core.models import resolve_model

from .locate import locate
from .model import Claim, Section, Span

logger = logging.getLogger("andamentum.whetstone.v3")

# Dropped from 4 → 2: per-section extract fans out across many sections
# at once, and the resulting concurrent burst was saturating the OpenAI
# connection path (Connection-error waves in batches of exactly N).
_MAX_CONCURRENT = 2
_MAX_REQUOTE = 3


# ── Agents ────────────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """You are reading ONE section of a document. Return the \
verbatim spans that make a CLAIM — an assertion the author wants the reader to \
believe: a result, a contribution, a comparison, a capability, or a factual \
assertion.

Multi-sentence claims. A claim often spans 2-3 sentences: a claim sentence \
followed by a scope qualifier ("Our result is comparable to the best known \
bound for this general convex online learning problem.") or an equation/bound \
that is the load-bearing payload of the claim. When the second sentence is \
necessary to judge the claim's scope or correctness, INCLUDE it as part of the \
same verbatim span. Don't truncate at the first period if the qualifier \
follows immediately. Each entry in your output is one claim; multi-sentence \
spans count as ONE entry.

Algorithms, equations, and figures.
  - Algorithm pseudocode is EVIDENCE, not a claim. The claim is in the prose \
that introduces or interprets the algorithm ("the magnitudes of parameter \
updates are invariant to rescaling of the gradient"). Do not extract \
individual pseudocode lines.
  - HTML comment placeholders like ``<!-- formula-not-decoded -->`` mean an \
equation was stripped from the source. Extract the surrounding prose sentence \
that names the result ("Adam achieves O(√T) regret comparable to the best \
known bound"). The equation itself is unrecoverable from this text — accept \
that and use the prose claim.
  - Figure captions are valid claim sources but usually redundant with the \
paragraph that references them. Prefer the prose; only extract from a caption \
if it carries a claim that doesn't appear in the prose.

Copy each claim span EXACTLY from the section text, character for character. \
Do not paraphrase, summarise, shorten, or fix anything — an exact copy is \
essential. Do NOT rewrite math notation, glyph names, or Docling-style spaced \
symbols (e.g. "β 1", "glyph[circledot]", "θ t -1"); keep them exactly as the \
section text shows them. Skip pure background, method mechanics that assert \
nothing, and headings. If the section makes no claims, return an empty list."""


class _ClaimSpans(BaseModel):
    claims: list[str] = Field(
        default_factory=list, description="Verbatim claim sentences, copied exactly."
    )


_REQUOTE_PROMPT = """A claim was identified in a section, but the text quoted \
for it was not found verbatim in the section. Given the section text and the \
claim, copy the EXACT verbatim span (character for character) from the section \
that states this claim. Output only that span, exactly as it appears.

Punctuation must match byte-for-byte: copy the exact dash variant (- vs – vs —), \
exact quote marks (straight " ' vs curly “ ” ‘ ’), ellipses (… vs ...), and any \
accents or special characters as they appear in the section. Whitespace, case, \
and markdown emphasis (**, *, _) do not matter — but every other character must \
be identical. Do not rewrite math notation, glyph names, or Docling-style \
spaced symbols (e.g. "β 1", "glyph[circledot]", "θ t -1"); keep them exactly \
as the section text shows them."""


class _Requote(BaseModel):
    quote: str = Field(description="The exact verbatim span from the section.")


def _agent(name: str, prompt: str, output_model: type[BaseModel], model: str):
    defn = AgentDefinition(
        name=name, prompt=prompt, output_model=output_model, retries=2, output_retries=2
    )
    return build_pydantic_ai_agent(defn, resolve_model(model))


# ── Orchestration ───────────────────────────────────────────────────────────


async def _extract_section(section: Section, *, model: str) -> list[str]:
    from ._metrics import bump_from_result

    agent = _agent("v3_extract_claims", _EXTRACT_PROMPT, _ClaimSpans, model)
    result = await agent.run(
        f"SECTION: {section.title}\n\n--- BEGIN ---\n{section.text}\n--- END ---"
    )
    bump_from_result(result)
    return list(cast(_ClaimSpans, result.output).claims)


async def _verify(
    raw_quote: str, section: Section, source: str, *, model: str
) -> tuple[str, Span] | None:
    """Locate the claim in its section; on a miss, ask the agent to re-quote
    (≤3). Returns (verbatim source text, span) or None."""
    within = (section.start, section.end)
    span = locate(raw_quote, source, within=within)
    attempts = 0
    requote = raw_quote
    while span is None and attempts < _MAX_REQUOTE:
        attempts += 1
        try:
            from ._metrics import bump_from_result

            agent = _agent("v3_requote", _REQUOTE_PROMPT, _Requote, model)
            res = await agent.run(
                f"SECTION:\n{section.text}\n\nCLAIM (not found verbatim):\n{requote}"
            )
            bump_from_result(res)
            requote = cast(_Requote, res.output).quote
        except Exception as exc:
            logger.warning("[v3.verify] requote crashed: %s", exc)
            break
        span = locate(requote, source, within=within)
    if span is None:
        return None
    return source[span[0] : span[1]], Span(
        section_id=section.id, start=span[0], end=span[1]
    )


async def build_claims(
    sections: list[Section], source: str, *, model: str
) -> list[Claim]:
    """Extract claims per section (parallel), verify+locate each, assign ids."""
    sem = asyncio.Semaphore(_MAX_CONCURRENT)

    async def one(section: Section) -> list[tuple[str, Span]]:
        async with sem:
            try:
                raw = await _extract_section(section, model=model)
            except Exception as exc:
                logger.warning("[v3.extract] %s crashed: %s", section.id, exc)
                return []
        verified: list[tuple[str, Span]] = []
        for q in raw:
            located = await _verify(q, section, source, model=model)
            if located is not None:
                verified.append(located)
        return verified

    results = await asyncio.gather(*[one(s) for s in sections])
    claims: list[Claim] = []
    n = 0
    for verified in results:
        for quote, span in verified:
            n += 1
            claims.append(Claim(id=f"c{n}", quote=quote, span=span))
    logger.info("[v3] %d verified claim(s)", len(claims))
    return claims
