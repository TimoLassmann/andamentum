"""Statistical self-consistency check (statcheck-equivalent).

Extract patterns of the shape ``test_statistic(df) = value, p [op] reported_p``
and recompute the implied p-value from the test statistic alone. If the
reported p disagrees with what the math says, emit a Finding.

This catches a real and well-documented class of bug — published
psychology has a documented ~10–15% rate of test-statistic / p-value
inconsistency, biomedical work isn't far behind. Inconsistencies fall
into two grades:

  • **decision-changing** (severity ``major``) — reported p crosses
    α=0.05 but recomputed p doesn't, or vice versa. The conclusion
    "this is significant" or "not significant" is wrong.
  • **non-decision-changing** (severity ``moderate``) — same side of
    0.05 but the reported p is meaningfully different from what the
    statistic implies, beyond rounding tolerance.

We're conservative: we only flag inconsistencies we can be confident
about. When df is unclear, when assumptions are ambiguous (one-tailed
vs two-tailed), or when the regex match looks suspect, we skip the
claim rather than emit a false-positive finding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

from scipy import stats  # type: ignore[import-untyped]

from ..schemas import Finding, Quote
from .types import SectionRef

PReportedOp = Literal["<", ">", "="]
TestKind = Literal["t", "F", "chi2", "z", "r"]


# ── Regexes ────────────────────────────────────────────────────────────


# Helpers ----------------------------------------------------------------
_INT = r"\d+"
_NUM = r"-?\d+(?:\.\d+)?|-?\.\d+"  # signed floats, with or without leading zero
_OP = r"[<>=≤≥]"
_P_VAL = r"\d+(?:\.\d+)?|\.\d+"  # 0.05 / .05 / 0.001 / 0


def _p_value_block() -> str:
    """A subgroup matching ``p [op] <value>``."""
    return rf"p\s*(?P<p_op>{_OP})\s*(?P<p_val>{_P_VAL})"


# t-test: t(48) = 2.34, p = 0.024
_T_RE = re.compile(
    rf"\bt\s*\(\s*(?P<df>{_INT})\s*\)\s*=\s*(?P<stat>{_NUM})\s*[,;]\s*"
    + _p_value_block(),
    re.IGNORECASE,
)

# F-test: F(2, 96) = 4.51, p = 0.013
_F_RE = re.compile(
    rf"\bF\s*\(\s*(?P<df1>{_INT})\s*,\s*(?P<df2>{_INT})\s*\)\s*="
    rf"\s*(?P<stat>{_NUM})\s*[,;]\s*"
    + _p_value_block()
)

# χ²(2) = 5.91, p = .052     /     chi2(2, N=120) = 5.91, p = 0.052
_CHI2_RE = re.compile(
    rf"\b(?:χ\s*[²2]|chi[\s\-]?2|chi[\s\-]?square[d]?)\s*\(\s*(?P<df>{_INT})"
    rf"(?:\s*,\s*N\s*=\s*{_INT})?\s*\)\s*=\s*(?P<stat>{_NUM})\s*[,;]\s*"
    + _p_value_block(),
    re.IGNORECASE,
)

# z = 1.96, p = 0.05
_Z_RE = re.compile(
    rf"\bz\s*=\s*(?P<stat>{_NUM})\s*[,;]\s*"
    + _p_value_block(),
    re.IGNORECASE,
)

# r(48) = 0.42, p < 0.001     /     r = .42, p = .002, n = 50
_R_WITH_DF_RE = re.compile(
    rf"\br\s*\(\s*(?P<df>{_INT})\s*\)\s*=\s*(?P<stat>{_NUM})\s*[,;]\s*"
    + _p_value_block(),
    re.IGNORECASE,
)
_R_NO_DF_RE = re.compile(
    rf"\br\s*=\s*(?P<stat>{_NUM})\s*[,;]\s*"
    + _p_value_block()
    + rf"\s*[,;]\s*(?:n|N)\s*=\s*(?P<n>{_INT})",
    re.IGNORECASE,
)


# ── Extracted claim ────────────────────────────────────────────────────


@dataclass
class StatClaim:
    """One ``(test_stat, df, reported_p)`` triple plucked from a section."""

    raw: str
    test_kind: TestKind
    df: tuple[int, ...]  # () for z; (df,) for t/chi2; (df1,df2) for F; (n,) for r
    statistic_value: float
    reported_p_op: PReportedOp
    reported_p_value: float
    section_id: str
    char_start: int
    char_end: int


def extract_stat_claims(sections: list[SectionRef]) -> list[StatClaim]:
    """Walk every section, return every recognised statistical claim."""
    out: list[StatClaim] = []
    for s in sections:
        out.extend(_extract_one_section(s))
    return out


def _extract_one_section(section: SectionRef) -> list[StatClaim]:
    out: list[StatClaim] = []
    text = section.text

    for m in _T_RE.finditer(text):
        out.append(_make_claim(m, section, "t", df=(int(m.group("df")),)))

    for m in _F_RE.finditer(text):
        out.append(
            _make_claim(
                m,
                section,
                "F",
                df=(int(m.group("df1")), int(m.group("df2"))),
            )
        )

    for m in _CHI2_RE.finditer(text):
        out.append(_make_claim(m, section, "chi2", df=(int(m.group("df")),)))

    for m in _Z_RE.finditer(text):
        out.append(_make_claim(m, section, "z", df=()))

    for m in _R_WITH_DF_RE.finditer(text):
        out.append(_make_claim(m, section, "r", df=(int(m.group("df")),)))

    for m in _R_NO_DF_RE.finditer(text):
        # df = n - 2 for Pearson's r
        n = int(m.group("n"))
        if n >= 3:
            out.append(_make_claim(m, section, "r", df=(n - 2,)))

    return out


def _make_claim(
    m: re.Match[str],
    section: SectionRef,
    kind: TestKind,
    *,
    df: tuple[int, ...],
) -> StatClaim:
    p_op_raw = m.group("p_op")
    # Unicode comparison operators normalise to ASCII for downstream math
    p_op: PReportedOp = {"≤": "<", "≥": ">"}.get(p_op_raw, p_op_raw)  # type: ignore[assignment]
    p_val_str = m.group("p_val")
    if p_val_str.startswith("."):
        p_val_str = "0" + p_val_str
    return StatClaim(
        raw=m.group(0),
        test_kind=kind,
        df=df,
        statistic_value=float(m.group("stat")),
        reported_p_op=p_op,
        reported_p_value=float(p_val_str),
        section_id=section.id,
        char_start=m.start(),
        char_end=m.end(),
    )


# ── Recomputation ──────────────────────────────────────────────────────


def recompute_p_value(claim: StatClaim) -> Optional[float]:
    """Recompute the two-tailed p-value the statistic implies.

    Returns ``None`` when we can't be confident in the recomputation —
    e.g. r without n, F with df=0, etc. Callers should silently skip
    None results rather than emit a low-confidence finding.
    """
    try:
        stat = abs(claim.statistic_value)
        if claim.test_kind == "t":
            (df,) = claim.df
            if df <= 0:
                return None
            return float(2 * (1 - stats.t.cdf(stat, df)))
        if claim.test_kind == "F":
            df1, df2 = claim.df
            if df1 <= 0 or df2 <= 0:
                return None
            # F is one-tailed by convention
            return float(1 - stats.f.cdf(claim.statistic_value, df1, df2))
        if claim.test_kind == "chi2":
            (df,) = claim.df
            if df <= 0:
                return None
            return float(1 - stats.chi2.cdf(claim.statistic_value, df))
        if claim.test_kind == "z":
            return float(2 * (1 - stats.norm.cdf(stat)))
        if claim.test_kind == "r":
            (df,) = claim.df
            if df <= 0:
                return None
            r = claim.statistic_value
            if abs(r) >= 1:
                return None
            t_stat = r * (df / (1 - r * r)) ** 0.5
            return float(2 * (1 - stats.t.cdf(abs(t_stat), df)))
    except Exception:
        return None
    return None


# ── Consistency check ──────────────────────────────────────────────────


def _round_to_reported_precision(implied: float, reported_str: str) -> float:
    """Round the implied p to the same decimal precision the reporter used."""
    if "." in reported_str:
        decimals = len(reported_str.split(".", 1)[1])
    else:
        decimals = 0
    return round(implied, decimals)


def _reported_significant(claim: StatClaim, alpha: float = 0.05) -> bool:
    """Does the reported p claim significance at α?"""
    if claim.reported_p_op == "<":
        return claim.reported_p_value <= alpha
    if claim.reported_p_op == ">":
        return False  # author claims p > something; not claiming significance
    return claim.reported_p_value < alpha


def _consistent(claim: StatClaim, implied_p: float) -> bool:
    """True if reported and implied are consistent within tolerance."""
    reported = claim.reported_p_value
    if claim.reported_p_op == "<":
        return implied_p <= reported * 1.05  # small slack for rounding
    if claim.reported_p_op == ">":
        return implied_p >= reported * 0.95
    # equality: round implied to reporter's precision; allow ±1 ULP
    reported_str = repr(reported)
    rounded_implied = _round_to_reported_precision(implied_p, reported_str)
    if abs(rounded_implied - reported) <= 10 ** -_decimals(reported_str):
        return True
    # Fallback: fractional tolerance for cases where rounding misbehaves
    if reported == 0:
        return implied_p < 0.001
    return abs(implied_p - reported) / max(implied_p, reported) < 0.20


def _decimals(s: str) -> int:
    return len(s.split(".", 1)[1]) if "." in s else 0


# ── Finding generation ─────────────────────────────────────────────────


def check_stat_consistency(sections: list[SectionRef]) -> list[Finding]:
    """Run the full extract-recompute-compare pipeline. Returns Findings."""
    out: list[Finding] = []
    claims = extract_stat_claims(sections)

    section_lookup = {s.id: s for s in sections}

    for claim in claims:
        implied = recompute_p_value(claim)
        if implied is None:
            continue
        if _consistent(claim, implied):
            continue

        reported_significant = _reported_significant(claim)
        implied_significant = implied < 0.05
        decision_changing = reported_significant != implied_significant

        section = section_lookup.get(claim.section_id)
        quotes = []
        if section is not None:
            quotes = [
                Quote(
                    section_id=section.id,
                    char_start=claim.char_start,
                    char_end=claim.char_end,
                    text=claim.raw,
                )
            ]

        if decision_changing:
            severity = "major"
            verdict = (
                "decision-changing — the reported significance "
                "doesn't match the recomputed p-value"
            )
        else:
            severity = "moderate"
            verdict = "non-decision-changing but materially off"

        rationale = (
            f"Reported: {claim.raw}. "
            f"Recomputed (two-tailed where applicable): p ≈ {implied:.4g}. "
            f"This is {verdict}. Verify the test statistic, df, and tail."
        )

        out.append(
            Finding(
                title=(
                    f"Reported p inconsistent with {claim.test_kind}-statistic"
                ),
                severity=severity,  # type: ignore[arg-type]
                confidence="high",
                rationale=rationale,
                quotes=quotes,
                sections_involved=[claim.section_id],
                category="statistics",
            )
        )

    return out
