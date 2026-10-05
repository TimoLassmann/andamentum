"""The checks that need a model or the network: claim support and link health.

``ClaimJudge`` asks one question of each (claim, evidence) pair, through the
same ``Asker`` the engine uses: a decision model answers a ``noul`` question
(supported when ``p >= 0.5``), else the language model answers yes or no. The
judge is separate from the system under test; give it a model of its own, or
the engine grades itself.

``best_window`` picks the stretch of a long page that shares most with a claim,
so a judge reads the passage that could support the claim rather than the top
of the page. The same window rule applies to every arm.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from pydantic import BaseModel

from andamentum.core.decisions import DecisionModel
from andamentum.references import UrlHealth

from ..ask import Asker
from ..store import ResearchStore
from ..types import Budget, RunStats
from .metrics import UrlIntegrity, url_integrity

JUDGE_INSTRUCTION = (
    "Does the evidence state what the claim says? Answer yes only when the "
    "evidence itself supports the claim; a claim that goes beyond the evidence, "
    "or that the evidence does not mention, is a no."
)
JUDGE_BUDGET_CALLS = 1_000_000
WINDOW_CHARS = 2400
MAX_LINKS_PER_ANSWER = 25
_WORD = re.compile(r"[a-z0-9]{4,}")

UrlChecker = Callable[[str], Awaitable[UrlHealth]]


class ClaimEvidence(BaseModel):
    """One claim and the text that is supposed to support it."""

    claim: str
    evidence: str


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.casefold()))


def best_window(claim: str, texts: Sequence[str], *, width: int = WINDOW_CHARS) -> str:
    """The *width*-character stretch of *texts* with most of the claim's words.

    Windows start every half width; the earliest best one wins. Returns ``""``
    when there is no text.
    """
    wanted = _words(claim)
    best = ""
    best_score = -1
    for text in texts:
        starts = range(0, max(len(text) - width // 2, 1), max(width // 2, 1))
        for start in starts:
            window = text[start : start + width]
            score = len(wanted & _words(window))
            if score > best_score:
                best, best_score = window, score
    return best


class ClaimJudge:
    """Judges claims one pair at a time, sequentially, and counts its calls."""

    def __init__(
        self,
        *,
        model: Any | None = None,
        decision_model: DecisionModel | None = None,
    ) -> None:
        if model is None and decision_model is None:
            raise ValueError("a claim judge needs a model or a decision model")
        self.stats = RunStats()
        store = ResearchStore.in_memory()
        self._asker = Asker(
            model=model,
            store=store,
            run_id=store.start_run(
                question="claim support", mode="web", config={}, models={}
            ),
            budget=Budget(
                max_llm_calls=JUDGE_BUDGET_CALLS,
                max_decision_calls=JUDGE_BUDGET_CALLS,
            ),
            stats=self.stats,
            decision_model=decision_model,
        )

    async def judge(self, pairs: Sequence[ClaimEvidence]) -> list[bool]:
        """Is each claim supported by its evidence? One verdict per pair, in order."""
        if not pairs:
            return []
        probabilities = await self._asker.ask_each(
            "claim_support",
            instruction=JUDGE_INSTRUCTION,
            states=[{"claim": p.claim, "evidence": p.evidence} for p in pairs],
        )
        return [p >= 0.5 for p in probabilities]


async def check_links(
    urls: Sequence[str], checker: UrlChecker, *, limit: int = MAX_LINKS_PER_ANSWER
) -> UrlIntegrity:
    """Probe up to *limit* links, one after the other, and sort the results."""
    health = [await checker(url) for url in urls[:limit]]
    return url_integrity([h.status for h in health])
