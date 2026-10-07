"""CHOOSE: which of the unread search results to read in full.

One model call (``step="Choose"``) sees the title, site, year and snippet of the
sub-question's unread results and returns the ones to read (``picks``, at most
K) and as many ``reserves``, read only when a pick cannot be opened. When the
model fails, or returns no usable pick while results exist, the top results by
reciprocal rank fusion of their source ranks are read instead (``fallback``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry

from . import fusion
from .ask import MODEL_ERRORS, UNTRUSTED, Asker, BudgetExhausted, fence
from .origin import origin_label, registrable_domain
from .search import work_url
from .types import Candidate, SubQuestion

# Results put to the model at once, best fused rank first.
MAX_SHOWN = 40
SNIPPET_CHARS = 400


class ChoiceDraft(BaseModel):
    picks: list[int] = Field(description="Result numbers to read, best first.")
    reserves: list[int] = Field(
        default_factory=list,
        description="More result numbers, best first, read only if a pick cannot be opened.",
    )


def choose_instructions(*, k: int) -> str:
    return f"""\
You choose which search results to read in full to answer a question. You see \
numbered results: the title, the site, the year when known, and the snippet the \
search engine showed.
Return `picks`: the results most likely to contain the specific answer, best \
first, at most {k}.
Return `reserves`: up to {k} more, best first, read only if a picked page \
cannot be opened.
Prefer primary, official and encyclopedic sources (the organisation, person, \
publisher, database or reference work the answer comes from) over aggregators, \
listings, forums, content farms and pages written to rank in search. Avoid a \
page whose snippet only repeats the question without stating anything. Prefer \
different sites over several pages of one site, unless that site is the \
authority.
A snippet that states the answer is a good sign; a page of an authoritative \
site that probably holds it is better. Return numbers only.
{UNTRUSTED}"""


@dataclass
class ChoiceOutcome:
    picks: list[Candidate]
    reserves: list[Candidate]
    # The results the model saw, in the order it saw them.
    shown: list[Candidate] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    spent: str = ""
    fallback: bool = False


def site_of(candidate: Candidate) -> str:
    work = candidate.work
    return work.venue or registrable_domain(work_url(work)) or origin_label(work)


def ordered_by_fusion(candidates: Sequence[Candidate]) -> list[Candidate]:
    """*candidates* best first by reciprocal rank fusion of their source ranks;
    ties keep the order the candidates came in (a work id is random per store,
    so it cannot break a tie the same way twice)."""
    scores = fusion.fused_scores({c.work.work_id: c for c in candidates})
    return sorted(candidates, key=lambda c: -scores.get(c.work.work_id, 0.0))


def _listing(shown: Sequence[Candidate]) -> str:
    lines: list[str] = []
    for number, candidate in enumerate(shown, start=1):
        work = candidate.work
        year = f", {work.year}" if work.year else ""
        lines.append(
            f"[{number}] {' '.join(work.title.split())} ({site_of(candidate)}{year})"
        )
        snippet = " ".join(candidate.snippet.split())[:SNIPPET_CHARS]
        if snippet:
            lines.append(f"    {snippet}")
    return "\n".join(lines)


async def choose(
    subquestion: SubQuestion,
    *,
    question: str,
    candidates: Sequence[Candidate],
    k: int,
    asker: Asker,
) -> ChoiceOutcome:
    """Up to *k* picks and *k* reserves among *candidates* (the sub-question's
    unread results). The model sees at most ``MAX_SHOWN`` of them, best fused
    rank first."""
    if not candidates:
        return ChoiceOutcome([], [])
    shown = ordered_by_fusion(candidates)[:MAX_SHOWN]

    # async on purpose: pydantic-ai runs a sync validator in a worker thread, and
    # the order concurrent calls finish in would then depend on the threads
    async def numbers_exist(draft: ChoiceDraft) -> ChoiceDraft:
        wrong = sorted(
            {n for n in (*draft.picks, *draft.reserves) if not 1 <= n <= len(shown)}
        )
        if wrong:
            raise ModelRetry(
                f"there are no results numbered {wrong}; use 1 to {len(shown)}"
            )
        return draft

    outcome = ChoiceOutcome([], [], shown=shown)
    message = "\n".join(
        [
            f"QUESTION: {question.strip()}",
            f"SUB-QUESTION: {subquestion.text.strip()}",
            "",
            "RESULTS:",
            fence(_listing(shown)),
        ]
    )
    try:
        draft = await asker.llm(
            step="Choose",
            instructions=choose_instructions(k=k),
            output_type=ChoiceDraft,
            user_message=message,
            validators=[numbers_exist],
        )
    except BudgetExhausted as spent:
        outcome.spent = spent.name
        draft = None
    except MODEL_ERRORS as error:
        outcome.degraded.append(
            f"choosing what to read failed ({type(error).__name__}: {error}); the "
            "top results by rank are read"
        )
        draft = None
    if draft is not None:
        picked = list(dict.fromkeys(draft.picks))[:k]
        spare = [n for n in dict.fromkeys(draft.reserves) if n not in picked][:k]
        outcome.picks = [shown[n - 1] for n in picked]
        outcome.reserves = [shown[n - 1] for n in spare]
    if not outcome.picks:
        outcome.fallback = True
        outcome.picks = shown[:k]
        outcome.reserves = shown[k : 2 * k]
    return outcome
