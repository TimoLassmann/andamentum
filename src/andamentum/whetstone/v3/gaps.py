"""The gap re-examination loop — re-grounds the review in the real source.

Reviewing the digest alone is unsafe (a missed claim is invisible from inside
the representation). This bounded loop re-reads the ORIGINAL source to (a)
re-check existing findings' veracity against the text and (b) surface issues
the forward pass missed. Demand-routed and capped (the termination guarantee),
in the spirit of deep_research / epistemic lazy escalation.

Functions here; the graph nodes wrap them in Phase 5. Deterministic pieces
(coverage summary, loop control, verify) are separated from the agent pieces
(gap analysis, satisfy).
"""

from __future__ import annotations

import logging
from typing import Literal, cast

from pydantic import BaseModel, Field

from pydantic_ai import RunContext

from andamentum.core.agents import AgentDefinition, build_pydantic_ai_agent
from andamentum.core.models import resolve_model

from .criteria import SPECS
from .model import DocumentModel
from .review import Finding, Severity, anchor_quotes_or_retry, verify_findings

logger = logging.getLogger("andamentum.whetstone.v3")

_DEFAULT_CAP = 2

# Per-round demand cap. analyze_gaps may emit any number of demands;
# we truncate to this many before satisfying them. Structural ceiling
# on LLM calls in the loop: _DEFAULT_CAP rounds * _DEFAULT_PER_ROUND_DEMANDS
# demands per round = 6 calls. With both reexamine (full agent) and
# recheck (single-call) demands costing seconds each, this keeps the
# loop's wall-clock contribution bounded.
_DEFAULT_PER_ROUND_DEMANDS = 3


class Demand(BaseModel):
    """A pull request for more work, routed to its minimal satisfier."""

    kind: Literal["reexamine", "recheck"]
    detail: str = ""  # what to look for (reexamine) / why (recheck)
    target_section_id: str | None = None  # for reexamine
    finding_index: int | None = None  # for recheck (index into the numbered list shown)

    def signature(self) -> str:
        return f"{self.kind}|{self.target_section_id}|{self.finding_index}|{self.detail[:40]}"


class _DemandList(BaseModel):
    demands: list[Demand] = Field(default_factory=list)


class _Holds(BaseModel):
    holds: bool = Field(
        description="True if the finding is genuinely supported by the text."
    )
    reason: str = ""


class _ReexamineFinding(BaseModel):
    issue: str = Field(description="Short description of the problem (1-2 sentences).")
    quote: str = Field(description="A VERBATIM span from the section it concerns.")
    severity: Severity = "moderate"
    criterion: str = Field(
        description="Which review criterion this problem falls under "
        "(use one of the names provided, verbatim)."
    )


class _ReexamineFindings(BaseModel):
    findings: list[_ReexamineFinding] = Field(default_factory=list)


def _snap_criterion(value: str, names: list[str]) -> str:
    """Map the agent's criterion onto the active set (case-insensitive) so a
    re-examined finding lands under a real criterion rather than a generic
    label. Falls back to the first criterion if nothing matches."""
    v = value.strip().lower()
    for n in names:
        if n.lower() == v:
            return n
    for n in names:
        if n.lower() in v or v in n.lower():
            return n
    return names[0] if names else "Finding"


# ── Deterministic: coverage summary + loop control ──────────────────────────


def coverage_summary(findings: list[Finding], model: DocumentModel) -> str:
    """Light reflection input: findings per criterion + which sections have any."""
    by_crit: dict[str, int] = {}
    for f in findings:
        by_crit[f.criterion] = by_crit.get(f.criterion, 0) + 1
    touched = {f.span.section_id for f in findings if f.span}
    crit_line = ", ".join(f"{k}:{v}" for k, v in sorted(by_crit.items())) or "(none)"
    untouched = [s.id for s in model.sections if s.id not in touched]
    return (
        f"findings by criterion: {crit_line}\n"
        f"sections with no finding yet: {', '.join(untouched) or '(none)'}"
    )


# ── Agents: gap analysis + satisfy ──────────────────────────────────────────

_GAP_PROMPT = """You are re-examining a review before it is finalised. Your job \
is NOT to re-list every section — it is to find what is worth a second look:
  • findings that should be re-checked against the real text (they may be wrong
    or overstated) → emit a "recheck" demand with the finding's index;
  • places the review likely MISSED something worth raising → emit a "reexamine"
    demand naming the section and what to look for.

Be sparing. Emit only high-value demands. If nothing needs a second look, emit
an empty list. Do NOT repeat demands already listed as previously requested."""


async def analyze_gaps(
    model: DocumentModel,
    findings: list[Finding],
    prior: set[str],
    *,
    agent_model: str,
) -> list[Demand]:
    defn = AgentDefinition(
        name="v3_gap_analysis",
        prompt=_GAP_PROMPT,
        output_model=_DemandList,
        retries=2,
        output_retries=2,
    )
    agent = build_pydantic_ai_agent(defn, resolve_model(agent_model))
    numbered = (
        "\n".join(
            f"  [{i}] ({f.criterion}/{f.severity}) {f.issue} — quote: {f.quote!r}"
            for i, f in enumerate(findings)
        )
        or "  (no findings yet)"
    )
    gists = "\n".join(f"  - {g.section_id} {g.title}: {g.gist}" for g in model.gists)
    prior_block = "\n".join(f"  - {p}" for p in sorted(prior)) or "  (none)"
    result = await agent.run(
        f"CURRENT FINDINGS:\n{numbered}\n\n"
        f"COVERAGE:\n{coverage_summary(findings, model)}\n\n"
        f"SECTIONS:\n{gists}\n\n"
        f"PREVIOUSLY REQUESTED (do not repeat):\n{prior_block}\n\n"
        f"What is worth a second look?"
    )
    demands = cast(_DemandList, result.output).demands
    return [d for d in demands if d.signature() not in prior]


async def _satisfy_reexamine(
    demand: Demand,
    model: DocumentModel,
    *,
    agent_model: str,
    criterion_names: list[str],
) -> list[Finding]:
    section = model.section_by_id(demand.target_section_id or "")
    if section is None:
        return []
    # output_retries=3 makes room for the anchor validator's two re-quote
    # attempts plus one reserve for pydantic-ai's own structured-output
    # coercion — matches the cascade's _build_agent.
    defn = AgentDefinition(
        name="v3_reexamine",
        prompt=(
            "Re-read this section of the document and report any real problems "
            "matching the request. Quote VERBATIM (non-verbatim quotes are "
            "dropped). Classify each problem under one of the review criteria "
            "listed. Report nothing if there is no real problem."
        ),
        output_model=_ReexamineFindings,
        retries=2,
        output_retries=3,
    )
    agent = build_pydantic_ai_agent(defn, resolve_model(agent_model))

    # Attach the same verbatim-quote anchor validator the cascade uses
    # (review.py). On unanchored quotes pydantic-ai sends the model back
    # to re-quote (up to two attempts); on exhaustion the validator
    # silently drops misses and verify_findings is the deterministic
    # floor below — same shape as the cascade.
    source = model.source

    @agent.output_validator
    async def _validate_quotes(
        ctx: RunContext[None], output: _ReexamineFindings
    ) -> _ReexamineFindings:
        if ctx.partial_output:
            return output
        anchored = anchor_quotes_or_retry(
            source, output.findings, ctx_retry=ctx.retry
        )
        return _ReexamineFindings(findings=anchored)

    res = await agent.run(
        f"REQUEST: {demand.detail}\n\n"
        f"CRITERIA (classify each problem as one of these): "
        f"{', '.join(criterion_names)}\n\n"
        f"SECTION ({section.title}):\n{section.text}"
    )
    raw = cast(_ReexamineFindings, res.output).findings
    return [
        Finding(
            criterion=_snap_criterion(r.criterion, criterion_names),
            issue=r.issue,
            quote=r.quote,
            severity=r.severity,
        )
        for r in raw
    ]


async def _satisfy_recheck(
    demand: Demand, findings: list[Finding], model: DocumentModel, *, agent_model: str
) -> bool:
    """Re-verify a finding against its real section text. Returns True to KEEP."""
    if demand.finding_index is None or not (0 <= demand.finding_index < len(findings)):
        return True
    f = findings[demand.finding_index]
    section = model.section_by_id(f.span.section_id) if f.span else None
    context = section.text if section else model.source
    defn = AgentDefinition(
        name="v3_recheck",
        prompt=(
            "Decide whether a review finding is genuinely supported by the "
            "document text shown. Answer holds=true only if the text really "
            "bears out the finding; otherwise holds=false."
        ),
        output_model=_Holds,
        retries=2,
        output_retries=2,
    )
    agent = build_pydantic_ai_agent(defn, resolve_model(agent_model))
    res = await agent.run(
        f"FINDING: {f.issue}\nQUOTE: {f.quote!r}\n\nDOCUMENT TEXT:\n{context}"
    )
    return bool(cast(_Holds, res.output).holds)


# ── Orchestration (the loop) ────────────────────────────────────────────────


async def gap_loop(
    model: DocumentModel,
    findings: list[Finding],
    *,
    agent_model: str,
    cap: int = _DEFAULT_CAP,
    per_round_demand_cap: int = _DEFAULT_PER_ROUND_DEMANDS,
    criterion_names: list[str] | None = None,
) -> list[Finding]:
    """Re-examine findings + surface misses against the source, bounded by *cap*
    rounds and *per_round_demand_cap* demands per round.

    Timing instrumentation: each round logs start, demands fired, findings
    added, and wall-clock duration. The audit found no timing data was
    being emitted; the logs here let future audits see if the cap is the
    right number.
    """
    import time
    names = criterion_names or [c.name for c in SPECS]
    current = list(findings)
    prior: set[str] = set()
    for round_idx in range(1, cap + 1):
        round_start = time.monotonic()
        try:
            demands = await analyze_gaps(model, current, prior, agent_model=agent_model)
        except Exception as exc:
            logger.warning("[v3.gaps] round %d — analysis crashed: %s", round_idx, exc)
            break
        if not demands:
            logger.info(
                "[v3.gaps] round %d — no demands, exiting (%.1fs)",
                round_idx, time.monotonic() - round_start,
            )
            break
        # Per-round cap. analyze_gaps' prompt says "Be sparing" but doesn't
        # enforce a number; truncate here so a chatty round can't dominate
        # the loop's wall-clock budget.
        if len(demands) > per_round_demand_cap:
            logger.info(
                "[v3.gaps] round %d — %d demand(s) emitted, truncating to %d",
                round_idx, len(demands), per_round_demand_cap,
            )
            demands = demands[:per_round_demand_cap]
        else:
            logger.info("[v3.gaps] round %d — %d demand(s)", round_idx, len(demands))
        for d in demands:
            prior.add(d.signature())
        drop_idxs: set[int] = set()
        new: list[Finding] = []
        for d in demands:
            try:
                if d.kind == "reexamine":
                    new += await _satisfy_reexamine(
                        d, model, agent_model=agent_model, criterion_names=names
                    )
                elif d.kind == "recheck":
                    keep = await _satisfy_recheck(
                        d, current, model, agent_model=agent_model
                    )
                    if not keep and d.finding_index is not None:
                        drop_idxs.add(d.finding_index)
            except Exception as exc:
                logger.warning("[v3.gaps] satisfy crashed: %s", exc)
        current = [f for i, f in enumerate(current) if i not in drop_idxs]
        added_anchored = verify_findings(new, model)
        current += added_anchored
        logger.info(
            "[v3.gaps] round %d — added %d finding(s), dropped %d, total %d (%.1fs)",
            round_idx, len(added_anchored), len(drop_idxs), len(current),
            time.monotonic() - round_start,
        )
    return current
