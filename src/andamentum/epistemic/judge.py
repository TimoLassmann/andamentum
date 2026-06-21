"""Central judge module — focused evaluative LLM judgments.

Provides two functions that are the ONLY evaluative LLM calls feeding
into the confidence score. All other confidence-relevant computation
is deterministic counting of these judgments.

    judge_evidence()      — supports / contradicts / no_bearing
    judge_independence()  — independent / not independent

Architecture: Layer 1 (framework-agnostic, async)
"""

from __future__ import annotations

from .agents.output_models import EvidenceJudgmentOutput, IndependenceJudgmentOutput
from .entities.evidence import Evidence
from .operations.base import AgentRunner


def apply_judgment(evidence: Evidence, judgment: EvidenceJudgmentOutput) -> None:
    """Map an evidence-claim judgment onto an Evidence entity (Tier 0).

    The single place that writes a judgment's results to an Evidence: the
    load-bearing categorical ``support_judgment`` (the verdict — argmax of the
    belief distribution, unchanged in meaning) plus the *additive* verbalized
    distribution that the confidence/entropy/one-hot properties derive from.

    Keeping this in one function means the four LLM-judge call sites
    (``claims``, ``seed_claim``, ``multi_seed_claim``, graph ``nodes``) stay in
    lock-step — adding a future derived signal touches here only. The caller
    is responsible for persisting the evidence afterwards.
    """
    evidence.support_judgment = judgment.verdict
    evidence.judgment_reasoning = judgment.reasoning
    evidence.judgment_distribution = judgment.distribution


async def judge_evidence(
    claim_statement: str,
    claim_scope: str,
    evidence_content: str,
    evidence_source: str,
    runner: AgentRunner,
) -> EvidenceJudgmentOutput:
    """Judge whether evidence supports, contradicts, or has no bearing on a claim.

    One LLM call. Returns a three-way classification with reasoning.

    Args:
        claim_statement: What the claim asserts.
        claim_scope: Under what conditions the claim holds.
        evidence_content: The evidence text.
        evidence_source: Where the evidence comes from (URL, DOI, etc.).
        runner: Agent runner for LLM execution.

    Returns:
        EvidenceJudgmentOutput with verdict and reasoning.

    Raises:
        RuntimeError: If the LLM is unavailable.
    """
    result: EvidenceJudgmentOutput = await runner.run(
        "epistemic_judge_evidence",
        claim_statement=claim_statement,
        claim_scope=claim_scope,
        evidence_content=evidence_content,
        evidence_source=evidence_source,
    )
    return result


async def judge_independence(
    evidence_a_content: str,
    evidence_a_source: str,
    evidence_b_content: str,
    evidence_b_source: str,
    runner: AgentRunner,
) -> IndependenceJudgmentOutput:
    """Judge whether two evidence items are methodologically independent.

    One LLM call. Returns a binary judgment with reasoning.

    Args:
        evidence_a_content: Text of the first evidence item.
        evidence_a_source: Source of the first evidence item.
        evidence_b_content: Text of the second evidence item.
        evidence_b_source: Source of the second evidence item.
        runner: Agent runner for LLM execution.

    Returns:
        IndependenceJudgmentOutput with independent flag and reasoning.

    Raises:
        RuntimeError: If the LLM is unavailable.
    """
    result: IndependenceJudgmentOutput = await runner.run(
        "epistemic_judge_independence",
        evidence_a=f"[{evidence_a_source}]\n{evidence_a_content}",
        evidence_b=f"[{evidence_b_source}]\n{evidence_b_content}",
    )
    return result
