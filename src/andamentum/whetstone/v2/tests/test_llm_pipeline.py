"""End-to-end tests for the LLM-driven path.

The full pipeline (CriticalRead → ReflectAndInvestigate → EditSections →
Challenge → AuthorQuestions → Synthesise) is exercised in
``test_pipeline_e2e.py`` with stubbed agents.

This file just confirms the no-LLM path still works (model=None on the
deterministic-only branch).
"""

from __future__ import annotations

from andamentum.whetstone.v2 import review_document


PAPER = """## 1 Introduction

This paper studies Reinforcement Learning (RL) applied to bipedal walking.
We had N = 50 participants in our user study, and cite prior work [1, 42].
As shown in Figure 1, the results are striking.

## 2 Methods

We compare two variants of RL on the same benchmark.
Across N=48 trials, the new method outperforms baselines significantly.
Figure 1: Comparison of accuracy across methods.

## References

[1] First Author. (2020). Title one.
[2] Second Author. (2021).
"""


async def test_phase_1_path_still_works_without_model() -> None:
    """No model → ChunkAndScan returns End[ReviewResult] directly, no LLM
    phases run. The deterministic-only path."""
    result = await review_document(PAPER)  # no model arg
    assert result.metrics.llm_calls == 0
    assert result.findings == []  # no LLM-driven findings
    assert result.summary == ""  # no synthesis
    # But deterministic findings still populated
    assert len(result.deterministic_findings) > 0
