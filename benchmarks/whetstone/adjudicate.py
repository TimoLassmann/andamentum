"""Judge-model adjudication + blinded human worksheet.

A strong judge model aligns the two arms' findings, buckets each
(both / a_only / b_only) and tags it (critical|minor, cross_section|local) per
the pre-registered rubric. Because an LLM judging LLM output is circular, we
also emit a BLINDED worksheet: the subset that matters (every b_only-critical
item plus a sample) shown with the arm relabelled "System 1/2" via a per-paper
random mapping, for a human to verify. The de-blinding key is saved separately.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Literal, cast

from pydantic import BaseModel, Field

from .types import (
    AdjudicatedFinding,
    ArmOutput,
    Comparison,
    Locality,
    PaperResult,
    Severity,
)

logger = logging.getLogger("whetstone.bench")


# ── Blinding (remove identity bias) + de-blinding ──────────────────────────
# The judge never learns which review is whetstone (the system under test) and
# which is the whole-document baseline: both are shown as anonymous "System 1"
# / "System 2", in a per-paper randomised order. Outputs come back in System
# 1/2 terms and are de-blinded here via the mapping. Arm "A" = whetstone,
# "B" = whole-document throughout.


def _mapping(slug: str, seed: int, *, swap: bool) -> dict[str, str]:
    """Per-paper system→arm mapping. ``swap`` flips it for the order-shift run."""
    a_is_system_1 = random.Random(f"{slug}:{seed}").random() < 0.5
    if swap:
        a_is_system_1 = not a_is_system_1
    return (
        {"system_1": "A", "system_2": "B"}
        if a_is_system_1
        else {"system_1": "B", "system_2": "A"}
    )


def _deblind_bucket(blind: str, m: dict[str, str]) -> str:
    """system_1_only / system_2_only / both → a_only / b_only / both."""
    if blind == "both":
        return "both"
    arm = m[blind[:-5]]  # strip "_only" → system_1 / system_2
    return "a_only" if arm == "A" else "b_only"


def _deblind_verdict(blind: str, m: dict[str, str]) -> str:
    """system_1 / system_2 / comparable → whetstone / whole-doc / comparable."""
    if blind == "comparable":
        return "comparable"
    return "whetstone" if m[blind] == "A" else "whole-doc"


def _deblind_text(text: str, m: dict[str, str]) -> str:
    """Replace 'System 1/2' in the judge's prose with the real system names."""
    out = text
    for sys in ("system_1", "system_2"):
        name = "whetstone" if m[sys] == "A" else "the whole-document review"
        out = re.sub(sys.replace("_", " "), name, out, flags=re.IGNORECASE)
    return out


def _reblind_bucket_label(bucket: str, m: dict[str, str]) -> str:
    """a_only / b_only / both → 'System N only' / 'both systems' for a prompt."""
    if bucket == "both":
        return "both systems"
    arm = "A" if bucket == "a_only" else "B"
    sys = "System 1" if m["system_1"] == arm else "System 2"
    return f"{sys} only"


# ── Alignment / bucketing (blinded, single run) ────────────────────────────

_RUBRIC = """RUBRIC:
- raised_by: which review(s) raised this issue. "both" = both raised it (match
  issues that are the same even when worded differently); "system_1_only" /
  "system_2_only" = only that one.
- severity: "critical" = a problem that undermines a central claim, result, or
  the soundness of the work (unsupported headline claim, methodological flaw,
  internal contradiction, missing control the conclusions depend on).
  "minor" = style, clarity, or a peripheral point.
- locality: "cross_section" = recognising it requires reading and connecting
  MULTIPLE sections (e.g. abstract claim vs results). "local" = visible within
  a single section."""

_JUDGE_PROMPT = (
    """You are aligning two anonymous, independent reviews of the SAME \
manuscript (System 1 and System 2) and classifying every distinct issue they \
raise. The systems are anonymised on purpose — judge only on the content, not \
on any guess about which tool produced which review.

Produce ONE entry per distinct issue: its text, which system(s) raised it, \
and its severity and locality.

"""
    + _RUBRIC
)


class _BlindFinding(BaseModel):
    text: str
    raised_by: Literal["system_1_only", "system_2_only", "both"]
    severity: Severity
    locality: Locality


class _BlindAdjudicationList(BaseModel):
    items: list[_BlindFinding] = Field(default_factory=list)


def _format_arm(out: ArmOutput) -> str:
    if not out.findings:
        return "  (no findings)"
    return "\n".join(f"  - {f.title}: {f.detail}" for f in out.findings)


async def adjudicate(
    arm_a: ArmOutput, arm_b: ArmOutput, *, model: str, slug: str, seed: int = 1
) -> list[AdjudicatedFinding]:
    """Align + bucket the two arms, blinded as System 1/2; de-blind to a/b."""
    from andamentum.core.agents import AgentDefinition, build_pydantic_ai_agent
    from andamentum.core.models import resolve_model

    m = _mapping(slug, seed, swap=False)
    arms = {"A": arm_a, "B": arm_b}
    defn = AgentDefinition(
        name="bench_judge",
        prompt=_JUDGE_PROMPT,
        output_model=_BlindAdjudicationList,
        retries=2,
        output_retries=2,
    )
    agent = build_pydantic_ai_agent(defn, resolve_model(model))
    prompt = (
        f"SYSTEM 1:\n{_format_arm(arms[m['system_1']])}\n\n"
        f"SYSTEM 2:\n{_format_arm(arms[m['system_2']])}\n\n"
        "Classify every distinct issue."
    )
    result = await agent.run(prompt)
    items = cast(_BlindAdjudicationList, result.output).items
    out = [
        AdjudicatedFinding(
            text=bf.text,
            bucket=_deblind_bucket(bf.raised_by, m),  # type: ignore[arg-type]
            severity=bf.severity,
            locality=bf.locality,
        )
        for bf in items
    ]
    logger.info("[judge] %d adjudicated issue(s) (blinded)", len(out))
    return out


class _VerdictMatch(BaseModel):
    matches: bool = Field(
        description="True if the candidate summary covers the same central "
        "problems as the reference weaknesses."
    )


async def judge_verdict_match(a_verdict: str, b_verdict: str, *, model: str) -> bool:
    """Does the CANDIDATE summary cover the REFERENCE weaknesses? Directional by
    design (candidate = whetstone's synthesis, reference = whole-doc's top-3),
    but identity-blind — the judge is never told which tool produced which, so
    no priors about 'whetstone' or 'whole-document' leak in."""
    from andamentum.core.agents import AgentDefinition, build_pydantic_ai_agent
    from andamentum.core.models import resolve_model

    defn = AgentDefinition(
        name="bench_verdict_match",
        prompt=(
            "You are given a CANDIDATE summary of a manuscript's main problems "
            "and a REFERENCE list of its key weaknesses. Decide whether the "
            "candidate identifies the same central weaknesses as the reference "
            "(it may add more, but must cover the core ones)."
        ),
        output_model=_VerdictMatch,
        retries=2,
        output_retries=2,
    )
    agent = build_pydantic_ai_agent(defn, resolve_model(model))
    result = await agent.run(
        f"CANDIDATE SUMMARY:\n{a_verdict}\n\nREFERENCE WEAKNESSES:\n{b_verdict}"
    )
    return bool(result.output.matches)  # type: ignore[attr-defined]


# ── Comparative verdict (blinded, two-run order-swap consistency) ──────────

_COMPARE_PROMPT = """You are comparing two anonymous reviews (System 1 and \
System 2) of the same manuscript to say which would be MORE USEFUL to the \
author, and why. The systems are anonymised on purpose — judge only on \
content.

You are given: the FULL MANUSCRIPT, the already-aligned issue list (each issue \
tagged with which system raised it, its severity, and whether it is \
cross-section or local), and each system's overall verdict.

USE THE MANUSCRIPT to check each system's issues. An issue that the manuscript \
does not actually support — off-base, a misreading, or something the text \
already addresses — counts AGAINST the review that raised it. Weight your \
verdict by BOTH usefulness AND accuracy: a review that raises many off-base or \
unsupported comments is less useful even if it raises more of them.

Decide `more_useful`:
  • "system_1"   — System 1's review is more useful (caught real critical / \
cross-section issues the other missed, with acceptable accuracy and noise).
  • "system_2"   — System 2's review is more useful.
  • "comparable" — neither is clearly more useful.

In `reasoning` (3-4 sentences), justify the call by CITING specific issues \
(what each caught/missed, and any that are off-base against the manuscript) \
and the signal-to-noise picture. Refer to the systems only as "System 1" / \
"System 2". Do not invent issues."""


class _BlindVerdict(BaseModel):
    more_useful: Literal["system_1", "system_2", "comparable"]
    reasoning: str = ""


_MAX_PAPER_CHARS = 100_000


def _load_paper_text(result: PaperResult) -> str:
    """The harvested manuscript markdown, so the comparator can verify whether
    flagged issues are actually supported by the paper. Empty if unavailable."""
    from pathlib import Path

    path = result.paper.markdown_path
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8")[:_MAX_PAPER_CHARS]
    logger.warning(
        "[compare] %s: no manuscript text — comparator can't verify accuracy",
        result.paper.slug,
    )
    return ""


async def _compare_once(
    result: PaperResult, m: dict[str, str], *, model: str, paper_text: str
) -> _BlindVerdict:
    from andamentum.core.agents import AgentDefinition, build_pydantic_ai_agent
    from andamentum.core.models import resolve_model

    issues = (
        "\n".join(
            f"  - [{_reblind_bucket_label(f.bucket, m)}|{f.severity}|{f.locality}] {f.text}"
            for f in result.adjudications
        )
        or "  (no aligned issues)"
    )
    arms = {"A": result.arm_a, "B": result.arm_b}
    defn = AgentDefinition(
        name="bench_compare",
        prompt=_COMPARE_PROMPT,
        output_model=_BlindVerdict,
        retries=2,
        output_retries=2,
    )
    agent = build_pydantic_ai_agent(defn, resolve_model(model))
    prompt = (
        f"FULL MANUSCRIPT:\n--- BEGIN ---\n{paper_text or '(unavailable)'}\n"
        f"--- END ---\n\n"
        f"ALIGNED ISSUES (attributed to System 1 / System 2):\n{issues}\n\n"
        f"System 1's verdict:\n{arms[m['system_1']].verdict or '—'}\n\n"
        f"System 2's verdict:\n{arms[m['system_2']].verdict or '—'}\n\n"
        "Which review is more useful to the author, and why? Discount any "
        "issues the manuscript does not support."
    )
    res = await agent.run(prompt)
    return cast(_BlindVerdict, res.output)


async def compare_reviews(
    result: PaperResult, *, model: str, seed: int = 1
) -> Comparison:
    """Grounded comparative verdict, blinded and run TWICE with the review order
    swapped. The judge also sees the FULL MANUSCRIPT, so it can discount
    off-base comments. If the de-blinded verdicts agree, that's the call; if
    they flip under the swap, it's position-sensitive → ``inconsistent``."""
    slug = result.paper.slug
    paper_text = _load_paper_text(result)
    runs = []
    for swap in (False, True):
        m = _mapping(slug, seed, swap=swap)
        bv = await _compare_once(result, m, model=model, paper_text=paper_text)
        runs.append((m, _deblind_verdict(bv.more_useful, m), bv.reasoning))

    (m0, v0, r0), (_m1, v1, _r1) = runs
    reasoning0 = _deblind_text(r0, m0)
    if v0 == v1:
        logger.info("[compare] %s → %s (order-consistent)", slug, v0)
        return Comparison(more_useful=v0, reasoning=reasoning0, order_consistent=True)
    logger.info("[compare] %s → INCONSISTENT (%s vs %s under swap)", slug, v0, v1)
    return Comparison(
        more_useful="inconsistent",
        order_consistent=False,
        reasoning=(
            f"Position-sensitive: the judge favoured '{v0}' in one order and "
            f"'{v1}' when the two reviews were swapped, so the preference is not "
            f"reliable. First run's reasoning: {reasoning0}"
        ),
    )


# ── Blinded human worksheet ─────────────────────────────────────────────────

_BUCKET_TO_SYSTEM = {"a_only": "A", "b_only": "B"}


def build_worksheet(
    result: PaperResult, *, seed: int, sample_minor: int = 3
) -> tuple[str, dict[str, str]]:
    """Markdown worksheet + de-blinding key for one paper.

    Includes every ``b_only`` critical item (the headline subset) plus a small
    sample of others, with the arm relabelled System 1/2 via a per-paper random
    mapping so the human's judgement is unbiased. The key maps System↔arm.
    """
    rng = random.Random(f"{result.paper.slug}:{seed}")
    # Per-paper blind mapping of arm → System label.
    flip = rng.random() < 0.5
    arm_to_system = (
        {"A": "System 1", "B": "System 2"}
        if flip
        else {
            "A": "System 2",
            "B": "System 1",
        }
    )
    key = {v: k for k, v in arm_to_system.items()}  # System label → arm

    headline = [
        f
        for f in result.adjudications
        if f.bucket == "b_only" and f.severity == "critical"
    ]
    others = [f for f in result.adjudications if f not in headline]
    rng.shuffle(others)
    selected = headline + others[:sample_minor]

    lines = [
        f"## {result.paper.slug}",
        "",
        "For each issue, mark whether you agree it is real, and your own "
        "severity/locality. Systems are anonymised.",
        "",
    ]
    for i, f in enumerate(selected, 1):
        if f.bucket == "both":
            who = "both systems"
        else:
            who = arm_to_system[_BUCKET_TO_SYSTEM[f.bucket]] + " only"
        lines += [
            f"### {i}. {f.text}",
            f"- raised by: **{who}**",
            f"- judge tags: {f.severity}, {f.locality}",
            "- your verdict (real? severity? locality?): ____",
            "",
        ]
    return "\n".join(lines), key
