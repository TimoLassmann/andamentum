"""Journal color palettes — Python equivalent of R's ggsci package.

Provides qualitative color palettes used by major scientific journals.
Each palette is a list of hex color strings. Use get_palette() to retrieve
colors by name, with automatic cycling if more colors are needed than
the palette provides.
"""

from __future__ import annotations

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

# ── Palette registry ─────────────────────────────────────────────────────────

PALETTES: dict[str, list[str]] = {
    "npg": NPG,
    "nejm": NEJM,
    "lancet": LANCET,
    "jama": JAMA,
    "aaas": AAAS,
    "d3": D3,
    "okabe_ito": OKABE_ITO,
}


def get_palette(name: str, n: int | None = None) -> list[str]:
    """Get a color palette by name.

    Args:
        name: Palette name (npg, nejm, lancet, jama, aaas, d3, okabe_ito).
        n: Number of colors needed. If greater than palette length, colors cycle.
           If None, returns the full palette.

    Returns:
        List of hex color strings.

    Raises:
        ValueError: If palette name is not recognized.
    """
    key = name.lower().replace("-", "_")
    if key not in PALETTES:
        available = ", ".join(sorted(PALETTES.keys()))
        raise ValueError(f"Unknown palette '{name}'. Available: {available}")

    colors = PALETTES[key]

    if n is None:
        return list(colors)

    if n <= len(colors):
        return colors[:n]

    # Cycle through palette for more colors than available
    return [colors[i % len(colors)] for i in range(n)]


def list_palettes() -> dict[str, int]:
    """List available palettes and their color counts.

    Returns:
        Dict mapping palette name to number of colors.
    """
    return {name: len(colors) for name, colors in PALETTES.items()}
