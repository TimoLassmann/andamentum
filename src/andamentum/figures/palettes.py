"""Journal color palettes — Python equivalent of R's ggsci package.

Provides qualitative color palettes used by major scientific journals.
Each palette is a list of hex color strings. Use get_palette() to retrieve
colors by name, with automatic cycling if more colors are needed than
the palette provides.
"""

from __future__ import annotations

from .errors import FiguresError

# ── Qualitative palettes (categorical data) ──────────────────────────────────

NPG: list[str] = [
    "#E64B35",
    "#4DBBD5",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#8491B4",
    "#91D1C2",
    "#DC0000",
    "#7E6148",
    "#B09C85",
]

NEJM: list[str] = [
    "#BC3C29",
    "#0072B5",
    "#E18727",
    "#20854E",
    "#7876B1",
    "#6F99AD",
    "#FFDC91",
    "#EE4C97",
]

LANCET: list[str] = [
    "#00468B",
    "#ED0000",
    "#42B540",
    "#0099B4",
    "#925E9F",
    "#FDAF91",
    "#AD002A",
    "#ADB6B6",
    "#1B1919",
]

JAMA: list[str] = [
    "#374E55",
    "#DF8F44",
    "#00A1D5",
    "#B24745",
    "#79AF97",
    "#6A6599",
    "#80796B",
]

AAAS: list[str] = [
    "#3B4992",
    "#EE0000",
    "#008B45",
    "#631879",
    "#008280",
    "#BB0021",
    "#5F559B",
    "#A20056",
    "#808180",
    "#1B1919",
]

D3: list[str] = [
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
]

# Okabe & Ito — recommended by Nature for colorblind accessibility
OKABE_ITO: list[str] = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]

# ── andamentum design-system palette ────────────────────────────────────────
#
# The six hexes are design tokens from docs/design/components.css. Order
# matters: the first colour is the one series most charts have. Charts use
# the tones as ordinary categorical colours; where the data carries a
# meaning (a "wrong" outcome, a "confirmed" result) callers map a series to
# a tone by name via ``series_colors``.
#
# Position | Token               | Meaning
# 1        | --am-info-accent    | slate, the subject
# 2        | --am-warn-accent    | ochre, caution
# 3        | --am-success-accent | olive, confirmed
# 4        | --am-danger-accent  | rust, danger
# 5        | --am-ink-3          | grey, a series with no tone
# 6        | --am-ink-5          | lighter grey

ANDAMENTUM_TONES: dict[str, str] = {
    "slate": "#6e7a8c",
    "ochre": "#9a8559",
    "olive": "#5e7a6a",
    "rust": "#9c4a36",
    "grey": "#555555",
    "grey-light": "#999999",
}

# The semantic names the design system gives the four toned accents. Accepted
# as aliases in ``series_colors`` so a caller can write ``right=success``.
ANDAMENTUM_TONE_ALIASES: dict[str, str] = {
    "info": "slate",
    "warn": "ochre",
    "success": "olive",
    "danger": "rust",
}

ANDAMENTUM: list[str] = list(ANDAMENTUM_TONES.values())

# The palette carries four *tones*; the two greys are for series with no
# meaning. Past four series the chart stops reading as one ink set.
ANDAMENTUM_TONE_COUNT = 4

# ── Palette registry ─────────────────────────────────────────────────────────

PALETTES: dict[str, list[str]] = {
    "npg": NPG,
    "nejm": NEJM,
    "lancet": LANCET,
    "jama": JAMA,
    "aaas": AAAS,
    "d3": D3,
    "okabe_ito": OKABE_ITO,
    "andamentum": ANDAMENTUM,
}


def get_palette(name: str, n: int | None = None) -> list[str]:
    """Get a color palette by name.

    Args:
        name: Palette name (npg, nejm, lancet, jama, aaas, d3, okabe_ito,
            andamentum).
        n: Number of colors needed. If greater than palette length, colors cycle.
           If None, returns the full palette. The ``andamentum`` palette never
           cycles: asking for more than its six colours raises.

    Returns:
        List of hex color strings.

    Raises:
        FiguresError: If palette name is not recognized, or if ``andamentum``
            is asked for more colours than it has.
    """
    key = _palette_key(name)
    colors = PALETTES[key]

    if n is None:
        return list(colors)

    if n <= len(colors):
        return colors[:n]

    if key == "andamentum":
        raise FiguresError(
            f"The andamentum palette has {len(colors)} colours and "
            f"{n} were requested. It does not cycle: merge or facet the series."
        )

    # Cycle through palette for more colors than available
    return [colors[i % len(colors)] for i in range(n)]


def palette_capacity_note(name: str, n: int) -> str | None:
    """Advisor note when a palette is stretched past what it reads well at.

    Only the ``andamentum`` palette has a declared tone count. Past four
    series it still has two greys, so the chart renders, but the caller is
    told to merge or facet. Past six ``get_palette`` raises instead.

    Args:
        name: Palette name.
        n: Number of distinct colours the chart needs.

    Returns:
        A note string, or None when the palette covers ``n`` comfortably.
    """
    if _palette_key(name) != "andamentum":
        return None
    if n > ANDAMENTUM_TONE_COUNT:
        return (
            f"The andamentum palette carries {ANDAMENTUM_TONE_COUNT} tones and "
            f"this chart needs {n} colours. Merge or facet the series."
        )
    return None


def resolve_tone(name: str) -> str:
    """Map a design-system tone name to its hex.

    Accepts the six tone names (slate, ochre, olive, rust, grey, grey-light)
    and the four semantic aliases (info, warn, success, danger). Hex values
    are refused: the point of naming a tone is that the report catalogue
    can check the output against the token set.

    Raises:
        FiguresError: If the name is not a known tone.
    """
    key = name.strip().lower().replace("_", "-")
    key = ANDAMENTUM_TONE_ALIASES.get(key, key)
    if key not in ANDAMENTUM_TONES:
        accepted = ", ".join(list(ANDAMENTUM_TONES) + list(ANDAMENTUM_TONE_ALIASES))
        raise FiguresError(f"Unknown tone '{name}'. Accepted: {accepted}")
    return ANDAMENTUM_TONES[key]


def assign_series_colors(
    names: list[str],
    palette: list[str],
    series_colors: dict[str, str] | None,
) -> list[str]:
    """One colour per name: a mapped tone where given, else the next palette colour.

    Palette colours are handed out in order to the names without a
    mapping, skipping any hex a mapping already uses, so two series never
    share a colour by accident.

    Args:
        names: Series (or category) names in plot order.
        palette: The active palette's colours.
        series_colors: ``{name: tone_name}`` for the names with a meaning.

    Returns:
        A list of hexes aligned with ``names``.

    Raises:
        FiguresError: If a mapped name is not in ``names`` or a tone is unknown.
    """
    mapping = series_colors or {}
    unknown = [k for k in mapping if k not in names]
    if unknown:
        raise FiguresError(
            f"series_colors names {unknown} are not in the data. "
            f"Series present: {names}"
        )
    fixed = {k: resolve_tone(v) for k, v in mapping.items()}
    taken = set(fixed.values())
    free = [c for c in palette if c not in taken]
    if not free:
        free = list(palette)

    out: list[str] = []
    next_free = 0
    for nm in names:
        if nm in fixed:
            out.append(fixed[nm])
        else:
            out.append(free[next_free % len(free)])
            next_free += 1
    return out


def _palette_key(name: str) -> str:
    key = name.lower().replace("-", "_")
    if key not in PALETTES:
        available = ", ".join(sorted(PALETTES.keys()))
        raise FiguresError(f"Unknown palette '{name}'. Available: {available}")
    return key


def list_palettes() -> dict[str, int]:
    """List available palettes and their color counts.

    Returns:
        Dict mapping palette name to number of colors.
    """
    return {name: len(colors) for name, colors in PALETTES.items()}
