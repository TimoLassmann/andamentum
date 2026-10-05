"""Statcheck-style recomputation of APA results, and GRIM (spec section 4, Tier 0).

Recognised, paragraph by paragraph (``-`` and the unicode minus both read):

- ``t(df) = x, p ...``
- ``F(df1, df2) = x, p ...``
- ``chi2(df) = x, p ...`` (also the symbol and ``chi-square``; an inner
  ``N = ...`` is allowed)
- ``r(df) = x, p ...``
- ``z = x, p ...``

The p value must be reported (``p = .046``, ``p < .001``, ``p > .05``). It is
recomputed with scipy (two-tailed for t, r and z). The test statistic is
treated as rounded, so the recomputed p is a range, as statcheck does; a
reported p is consistent when its own rounding interval overlaps that range.
A paragraph that says "one-tailed" or "one-sided" also accepts the halved p.
Inconsistent: error when the decision at .05 flips, warning otherwise.

GRIM: a mean ``M = x`` in a paragraph that states exactly one sample size
(``n = 20``) and exactly one item count (``5-item``, ``single-item``) is
tested for integer data: the sum of ``n * items`` integers must be able to
give the mean at its reported decimals. Skipped when the paragraph states no
n or no item count, states several different ones, or when ``n * items`` is
so large that every mean is attainable. Warning only.

Coverage limits: no regression coefficients, no ``t`` with unreported or
inexact p (``p = ns``), no results split across sentences or tables, no
corrections (Greenhouse-Geisser) or effect-size consistency, df given as
decimals only for t. Anything not matching is skipped silently, not flagged.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from scipy import stats

from ..kernel import (
    CheckContext,
    CheckSpec,
    Diagnostic,
    DocumentView,
    register,
)

ALPHA = 0.05

_NUM = r"[-−]?\d*\.?\d+"
_P = r"[pP]\s*(?P<op>[=<>≤≥])\s*(?P<p>(?:\d+)?\.\d+|\d+(?:\.\d+)?)"
_TAIL = r"\s*[,;]?\s*(?:\(?N\s*=\s*\d+\)?\s*[,;]?\s*)?"
_DF = r"\d+(?:\.\d+)?"

_PATTERNS: dict[str, re.Pattern[str]] = {
    "t": re.compile(rf"\bt\s*\(\s*(?P<df>{_DF})\s*\)\s*=\s*(?P<x>{_NUM}){_TAIL}{_P}"),
    "F": re.compile(
        rf"\bF\s*\(\s*(?P<df1>{_DF})\s*,\s*(?P<df2>{_DF})\s*\)\s*=\s*(?P<x>{_NUM}){_TAIL}{_P}"
    ),
    "chi2": re.compile(
        r"(?:χ\s*²?|χ\s*2|chi-?square[d]?|chi2|X\s*2|Χ\s*2)"
        rf"\s*\(\s*(?P<df>{_DF})(?:\s*,\s*N\s*=\s*\d+)?\s*\)\s*=\s*(?P<x>{_NUM}){_TAIL}{_P}",
        re.IGNORECASE,
    ),
    "r": re.compile(rf"\br\s*\(\s*(?P<df>{_DF})\s*\)\s*=\s*(?P<x>{_NUM}){_TAIL}{_P}"),
    "z": re.compile(rf"\bz\s*=\s*(?P<x>{_NUM}){_TAIL}{_P}"),
}

_ONE_TAILED = re.compile(r"one[- ](?:tailed|sided)", re.IGNORECASE)
_MEAN = re.compile(r"\bM\s*=\s*(?P<m>\d+\.\d+)")
_N = re.compile(r"\b[nN]\s*=\s*(?P<n>\d+)")
_ITEMS = re.compile(
    r"\b(?:(?P<k>\d+)[- ]items?\b|(?P<single>single[- ]item)\b)", re.IGNORECASE
)


def _f(text: str) -> float:
    return float(text.replace("−", "-"))


def _decimals(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def _p_value(kind: str, x: float, dfs: tuple[float, ...]) -> float:
    """Two-tailed p for the statistic (absolute value for the symmetric tests)."""
    if kind == "t":
        return float(2 * stats.t.sf(abs(x), dfs[0]))
    if kind == "F":
        return float(stats.f.sf(x, dfs[0], dfs[1]))
    if kind == "chi2":
        return float(stats.chi2.sf(x, dfs[0]))
    if kind == "r":
        r = abs(x)
        if r >= 1:
            return 0.0
        t = r * math.sqrt(dfs[0] / (1 - r * r))
        return float(2 * stats.t.sf(t, dfs[0]))
    if kind == "z":
        return float(2 * stats.norm.sf(abs(x)))
    raise ValueError(kind)


def _p_range(
    kind: str, x_text: str, dfs: tuple[float, ...]
) -> tuple[float, float, float]:
    """``(low, high, point)`` of the p that statistic rounding allows."""
    x = _f(x_text)
    half = 0.5 * 10 ** -_decimals(x_text)
    lo_x, hi_x = (x - half, x + half)
    if kind in ("F", "chi2"):
        lo_x, hi_x = max(lo_x, 0.0), max(hi_x, 0.0)
    elif x >= 0:
        lo_x = max(lo_x, 0.0)
    else:
        lo_x, hi_x = x - half, min(x + half, 0.0)
    ps = [_p_value(kind, v, dfs) for v in (lo_x, hi_x)]
    return min(ps), max(ps), _p_value(kind, x, dfs)


@dataclass(frozen=True)
class _Reported:
    op: str
    text: str

    @property
    def value(self) -> float:
        return float(self.text)

    def interval(self) -> tuple[float, float]:
        half = 0.5 * 10 ** -_decimals(self.text)
        return self.value - half, self.value + half

    def significant(self) -> bool | None:
        """The decision the report states, None when it states none."""
        if self.op in ("=",):
            return self.value <= ALPHA
        if self.op in ("<", "≤"):
            return True if self.value <= ALPHA else None
        return False if self.value >= ALPHA else None


def _consistent(reported: _Reported, low: float, high: float) -> bool:
    if reported.op == "=":
        r_lo, r_hi = reported.interval()
        return low <= r_hi and high >= r_lo
    if reported.op in ("<", "≤"):
        return low < reported.value or (reported.op == "≤" and low <= reported.value)
    return high > reported.value or (reported.op == "≥" and high >= reported.value)


class StatisticsCheck:
    spec = CheckSpec(id="statistics", tier=0, scope="paragraph", needs=[])
    version = "1"

    def run(self, view: DocumentView, ctx: CheckContext) -> list[Diagnostic]:
        out: list[Diagnostic] = []
        for paragraph in view.paragraphs:
            out.extend(self._results(paragraph.id, paragraph.text))
            out.extend(self._grim(paragraph.id, paragraph.text))
        return out

    def _diag(
        self,
        severity: str,
        message: str,
        *,
        paragraph_id: str,
        start: int,
        end: int,
        evidence: str,
        data: dict[str, object],
    ) -> Diagnostic:
        return Diagnostic(
            check=self.spec.id,
            tier=self.spec.tier,
            severity=severity,  # type: ignore[arg-type]
            paragraph_id=paragraph_id,
            start=start,
            end=end,
            message=message,
            evidence=[evidence],
            data=data,
        )

    def _results(self, paragraph_id: str, text: str) -> list[Diagnostic]:
        out: list[Diagnostic] = []
        one_tailed = bool(_ONE_TAILED.search(text))
        taken: list[tuple[int, int]] = []
        for kind, pattern in _PATTERNS.items():
            for match in pattern.finditer(text):
                span = match.span()
                if any(span[0] < b and a < span[1] for a, b in taken):
                    continue
                taken.append(span)
                groups = match.groupdict()
                dfs = tuple(
                    float(groups[k]) for k in ("df", "df1", "df2") if groups.get(k)
                )
                low, high, point = _p_range(kind, groups["x"], dfs)
                reported = _Reported(op=groups["op"], text=groups["p"])
                ok = _consistent(reported, low, high)
                if not ok and one_tailed:
                    ok = _consistent(reported, low / 2, high / 2)
                if ok:
                    continue
                decision = reported.significant()
                computed = (point / 2 if one_tailed else point) < ALPHA
                flips = decision is not None and decision != computed
                out.append(
                    self._diag(
                        "error" if flips else "warning",
                        (
                            f"Reported p {reported.op} {reported.text} does not match the "
                            f"p recomputed from the {kind} statistic ({point:.4g})"
                            + ("; the decision at .05 changes." if flips else ".")
                        ),
                        paragraph_id=paragraph_id,
                        start=span[0],
                        end=span[1],
                        evidence=match.group(0),
                        data={
                            "test": kind,
                            "reported_p": f"{reported.op}{reported.text}",
                            "recomputed_p": point,
                            "decision_flips": flips,
                        },
                    )
                )
        return out

    def _grim(self, paragraph_id: str, text: str) -> list[Diagnostic]:
        means = list(_MEAN.finditer(text))
        ns = {int(m["n"]) for m in _N.finditer(text)}
        items = {1 if m["single"] else int(m["k"]) for m in _ITEMS.finditer(text)}
        if not means or len(ns) != 1 or len(items) != 1:
            return []
        n, k = next(iter(ns)), next(iter(items))
        if n < 1 or k < 1:
            return []
        total = n * k
        out: list[Diagnostic] = []
        for match in means:
            digits = _decimals(match["m"])
            if total >= 10**digits:
                continue
            if _grim_possible(match["m"], total):
                continue
            out.append(
                self._diag(
                    "warning",
                    (
                        f"Mean {match['m']} cannot arise from integer data with "
                        f"n = {n} and {k} item(s) (GRIM)."
                    ),
                    paragraph_id=paragraph_id,
                    start=match.start(),
                    end=match.end(),
                    evidence=match.group(0),
                    data={"mean": match["m"], "n": n, "items": k},
                )
            )
        return out


def _grim_possible(mean_text: str, total: int) -> bool:
    mean = Decimal(mean_text)
    digits = _decimals(mean_text)
    quantum = Decimal(1).scaleb(-digits)
    centre = int(mean * total)
    for s in (centre - 1, centre, centre + 1, centre + 2):
        if s < 0:
            continue
        rounded = (Decimal(s) / Decimal(total)).quantize(
            quantum, rounding=ROUND_HALF_UP
        )
        if rounded == mean:
            return True
    return False


register(StatisticsCheck())
