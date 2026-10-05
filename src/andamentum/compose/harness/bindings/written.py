"""A number as written in the text: parse it, compare it to a stored value, write a new one like it.

Precision is what the text states: ``23.4%`` has one decimal and a percent
sign, ``1,234`` a thousands separator, ``−0.5`` a typographic minus, ``.03`` no
leading zero. :func:`format_like` writes another value in the same shape, and
:func:`scale_for` says whether a stored value is the written one directly or as
a fraction of 100 (``0.234`` stored, ``23.4%`` written).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

_WRITTEN = re.compile(
    r"^(?P<sign>[-−+]?)(?P<whole>\d[\d,]*)?(?:\.(?P<fraction>\d+))?(?P<percent>%?)$"
)
_MINUS = {"-", "−"}


@dataclass(frozen=True)
class WrittenNumber:
    text: str
    value: float
    decimals: int
    percent: bool
    thousands: bool
    sign: str  # "", "-", "−" or "+"
    leading_zero: bool


def parse_written(text: str) -> WrittenNumber | None:
    """The shape of *text*, or None when it is not a plain number."""
    match = _WRITTEN.match(text.strip())
    if match is None or (match["whole"] is None and match["fraction"] is None):
        return None
    whole = match["whole"] or ""
    fraction = match["fraction"] or ""
    magnitude = float(f"{whole.replace(',', '') or '0'}.{fraction or '0'}")
    sign = match["sign"]
    return WrittenNumber(
        text=text.strip(),
        value=-magnitude if sign in _MINUS else magnitude,
        decimals=len(fraction),
        percent=bool(match["percent"]),
        thousands="," in whole,
        sign=sign,
        leading_zero=bool(whole),
    )


def tolerance(written: WrittenNumber) -> float:
    """Half a unit in the last written place: what the text's precision allows."""
    return 0.5 * 10**-written.decimals + 1e-9


def scale_for(written: WrittenNumber, stored: float) -> float | None:
    """1.0 when *stored* is the written value, 100.0 when it is that value as a
    fraction of 100 (percent only), None when it is neither within precision.

    A written zero never matches a stored non-zero value: ``0`` in "week 0"
    would otherwise bind to a p-value of .003, and a refresh would rewrite it.
    """
    scales = (1.0, 100.0) if written.percent else (1.0,)
    for scale in scales:
        if written.value == 0 and stored != 0:
            return None
        if abs(stored * scale - written.value) <= tolerance(written):
            return scale
    return None


def format_like(template: WrittenNumber, shown: float) -> str:
    """*shown* written with the template's decimals, separator, sign and unit."""
    quantum = Decimal(1).scaleb(-template.decimals)
    rounded = Decimal(repr(shown)).quantize(quantum, rounding=ROUND_HALF_UP)
    negative = rounded < 0
    digits = f"{abs(rounded):,.{template.decimals}f}"
    if not template.thousands:
        digits = digits.replace(",", "")
    if not template.leading_zero and digits.startswith("0."):
        digits = digits[1:]
    if negative and rounded != 0:
        sign = template.sign if template.sign in _MINUS else "-"
    else:
        sign = "+" if template.sign == "+" and rounded != 0 else ""
    return f"{sign}{digits}{'%' if template.percent else ''}"
