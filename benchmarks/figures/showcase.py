#!/usr/bin/env python3
"""Benchmark and showcase for mosaic-figures.

Generates one figure per plot type using realistic scientific data,
saves them to benchmarks/output/, and creates an HTML gallery for
visual inspection.

Run:
    uv run python benchmarks/showcase.py

Output:
    benchmarks/output/*.pdf       — individual figures
    benchmarks/output/*.png       — PNG copies for the gallery
    benchmarks/output/gallery.html — visual gallery
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from andamentum.figures import figure

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

rng = random.Random(42)


def gauss(mu: float, sigma: float, n: int) -> list[float]:
    return [rng.gauss(mu, sigma) for _ in range(n)]


def uniform(lo: float, hi: float, n: int) -> list[float]:
    return [rng.uniform(lo, hi) for _ in range(n)]


# ── 1. Bar chart: Gene expression across treatment groups ────────────────────


def showcase_bar():
    return figure(
        data={
            "Treatment": ["Vehicle", "10 nM", "100 nM", "1 μM", "10 μM"],
            "Expression": [1.0, 1.8, 3.5, 5.2, 4.8],
            "SEM": [0.2, 0.3, 0.5, 0.4, 0.6],
        },
        kind="bar",
        x="Treatment",
        y="Expression",
        error="SEM",
        error_type="sem",
        title="VEGF Expression by Drug Concentration",
        y_label="Relative Expression (fold change)",
        x_label="Drug Concentration",
        style="npg",
        journal="nature",
        output=OUTPUT / "01_bar_expression.pdf",
    )


# ── 2. Line chart: Dose-response with confidence bands ──────────────────────


def showcase_line():
    time = [0, 2, 4, 8, 12, 24, 48]
    drug_a = [100, 92, 78, 55, 35, 18, 8]
    drug_b = [100, 97, 91, 82, 70, 55, 40]
    control = [100, 99, 98, 97, 96, 95, 94]
    return figure(
        data={
            "Time (hours)": time,
            "Compound A": drug_a,
            "Compound B": drug_b,
            "DMSO Control": control,
        },
        kind="line",
        x="Time (hours)",
        y=["Compound A", "Compound B", "DMSO Control"],
        title="Cell Viability After Treatment",
        y_label="Cell Viability (%)",
        style="nejm",
        journal="nature",
        output=OUTPUT / "02_line_viability.pdf",
    )


# ── 3. Scatter plot: Correlation between two biomarkers ──────────────────────


def showcase_scatter():
    n = 150
    x = gauss(50, 15, n)
    # Correlated with noise
    y = [xi * 0.8 + rng.gauss(10, 8) for xi in x]
    return figure(
        data={"CD4+ T cells (%)": x, "IL-6 (pg/mL)": y},
        kind="scatter",
        x="CD4+ T cells (%)",
        y="IL-6 (pg/mL)",
        title="CD4+ T Cell Proportion vs IL-6 Levels",
        style="lancet",
        journal="nature",
        output=OUTPUT / "03_scatter_biomarker.pdf",
    )


# ── 4. Box plot: Alignment accuracy across methods ──────────────────────────


def showcase_box():
    methods = ["Kalign", "MAFFT", "MUSCLE", "ClustalΩ", "T-Coffee"]
    data: dict[str, list] = {"Method": [], "F1 Score": []}
    for m, mu in zip(methods, [0.92, 0.89, 0.85, 0.82, 0.88]):
        n = 50
        scores = [min(1.0, max(0, rng.gauss(mu, 0.05))) for _ in range(n)]
        data["Method"].extend([m] * n)
        data["F1 Score"].extend(scores)
    return figure(
        data=data,
        kind="box",
        x="Method",
        y="F1 Score",
        title="Alignment Accuracy on BAliBASE",
        y_label="F1 Score",
        style="aaas",
        journal="nature",
        output=OUTPUT / "04_box_alignment.pdf",
    )


# ── 5. Violin plot: Expression distributions across tissues ──────────────────


def showcase_violin():
    tissues = ["Brain", "Liver", "Kidney", "Heart", "Lung"]
    data: dict[str, list] = {"Tissue": [], "log2(TPM+1)": []}
    for t, mu, sigma in zip(
        tissues, [8.5, 5.2, 6.8, 4.1, 7.3], [1.5, 2.0, 1.8, 1.2, 2.5]
    ):
        n = 300
        vals = gauss(mu, sigma, n)
        data["Tissue"].extend([t] * n)
        data["log2(TPM+1)"].extend(vals)
    return figure(
        data=data,
        kind="violin",
        x="Tissue",
        y="log2(TPM+1)",
        title="TP53 Expression Across Tissues",
        y_label="log₂(TPM + 1)",
        style="d3",
        journal="nature",
        output=OUTPUT / "05_violin_expression.pdf",
    )


# ── 6. Strip plot: Small clinical trial (individual patient data) ────────────


def showcase_strip():
    arms = ["Placebo", "Low Dose", "High Dose"]
    data: dict[str, list] = {"Arm": [], "Δ Tumor Volume (%)": []}
    for arm, mu in zip(arms, [5, -15, -35]):
        n = 6  # small trial
        vals = gauss(mu, 12, n)
        data["Arm"].extend([arm] * n)
        data["Δ Tumor Volume (%)"].extend(vals)
    return figure(
        data=data,
        kind="strip",
        x="Arm",
        y="Δ Tumor Volume (%)",
        title="Phase I Tumor Response (n=6 per arm)",
        y_label="Change in Tumor Volume (%)",
        style="jama",
        journal="nature",
        output=OUTPUT / "06_strip_clinical.pdf",
    )


# ── 7. Swarm plot: Moderate-sample immune cell counts ────────────────────────


def showcase_swarm():
    groups = ["Healthy", "Mild", "Severe"]
    data: dict[str, list] = {"Group": [], "CD8+ cells/μL": []}
    for g, mu, sigma in zip(groups, [800, 500, 200], [150, 120, 80]):
        n = 40
        vals = [max(0, v) for v in gauss(mu, sigma, n)]
        data["Group"].extend([g] * n)
        data["CD8+ cells/μL"].extend(vals)
    return figure(
        data=data,
        kind="swarm",
        x="Group",
        y="CD8+ cells/μL",
        title="CD8+ T Cell Counts by Disease Severity",
        y_label="CD8+ T cells (cells/μL)",
        style="okabe_ito",
        journal="nature",
        output=OUTPUT / "07_swarm_immune.pdf",
    )


# ── 8. Histogram: P-value distribution (well-calibrated test) ───────────────


def showcase_histogram():
    # Mix of uniform (null) and left-skewed (true positives)
    null_pvals = uniform(0, 1, 800)
    signal_pvals = [rng.betavariate(0.3, 5) for _ in range(200)]
    return figure(
        data={"p-value": null_pvals + signal_pvals},
        kind="histogram",
        y="p-value",
        title="P-value Distribution (10,000 tests)",
        y_label="p-value",
        x_label="p-value",
        style="aaas",
        journal="nature",
        output=OUTPUT / "08_histogram_pvalues.pdf",
    )


# ── 9. Heatmap: Drug sensitivity correlation matrix ──────────────────────────


def showcase_heatmap():
    drugs = ["Cisplatin", "Doxorubicin", "Paclitaxel", "Gemcitabine", "5-FU"]
    n = len(drugs)
    # Generate a plausible correlation matrix
    matrix: dict[str, list[float]] = {}
    for i, d in enumerate(drugs):
        row: list[float] = []
        for j in range(n):
            if i == j:
                row.append(1.0)
            elif j < i:
                row.append(matrix[drugs[j]][i])  # symmetric
            else:
                row.append(round(rng.uniform(0.1, 0.85), 2))
        matrix[d] = row
    return figure(
        data=matrix,
        kind="heatmap",
        title="Drug Sensitivity Correlation",
        style="npg",
        journal="nature",
        output=OUTPUT / "09_heatmap_correlation.pdf",
    )


# ── 10. Auto-detection showcase ──────────────────────────────────────────────


def showcase_auto():
    """Let the advisor pick the chart type — demonstrates kind='auto'."""
    return figure(
        data={
            "Condition": ["WT"] * 25 + ["KO"] * 25 + ["Rescue"] * 25,
            "Migration Rate (μm/h)": gauss(12, 3, 25)
            + gauss(5, 2, 25)
            + gauss(10, 3, 25),
        },
        # kind="auto" is the default — advisor should pick "box"
        title="Cell Migration Rate",
        y_label="Migration Rate (μm/h)",
        style="npg",
        journal="nature",
        output=OUTPUT / "10_auto_migration.pdf",
    )


# ── 11. Showcase mode (presentation) ────────────────────────────────────────


def showcase_presentation():
    return figure(
        data={
            "Quarter": ["Q1", "Q2", "Q3", "Q4"],
            "Revenue": [2.1, 3.4, 4.8, 5.2],
        },
        kind="bar",
        x="Quarter",
        y="Revenue",
        title="Revenue Growth",
        y_label="Revenue ($M)",
        style="d3",
        mode="showcase",
        output=OUTPUT / "11_showcase_presentation.pdf",
    )


# ── Gallery generator ────────────────────────────────────────────────────────


def generate_gallery(results: list[tuple[str, object]]) -> None:
    """Generate an HTML gallery of all showcase figures."""
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<title>mosaic-figures Showcase</title>",
        "<style>",
        "body { font-family: -apple-system, system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }",
        "h1 { color: #333; }",
        ".grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 24px; }",
        ".card { background: white; border-radius: 8px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }",
        ".card h3 { margin: 0 0 8px; color: #333; font-size: 14px; }",
        ".card img { width: 100%; border: 1px solid #eee; border-radius: 4px; }",
        ".card .legend { font-size: 12px; color: #666; margin-top: 8px; font-style: italic; }",
        ".card .meta { font-size: 11px; color: #999; margin-top: 4px; }",
        ".card .notes { font-size: 11px; color: #c44; margin-top: 4px; }",
        "</style>",
        "</head><body>",
        "<h1>mosaic-figures Showcase</h1>",
        f"<p>{len(results)} figures generated. Each demonstrates a different plot type with realistic scientific data.</p>",
        '<div class="grid">',
    ]

    for name, result in results:
        r = result  # type: ignore
        pdf_name = Path(r.path).name
        html_parts.append('<div class="card">')
        html_parts.append(f"<h3>{name}</h3>")
        html_parts.append(
            f'<object data="{pdf_name}" type="application/pdf" width="100%" height="300px">'
        )
        html_parts.append(f'<p><a href="{pdf_name}">View PDF</a></p>')
        html_parts.append("</object>")
        html_parts.append(f'<div class="legend">{r.legend}</div>')
        html_parts.append(
            f'<div class="meta">Kind: {r.kind} | Palette: {r.palette} | {r.data_summary} | {r.width_inches}" × {r.height_inches}"</div>'
        )
        if r.advisor_notes:
            html_parts.append(
                f'<div class="notes">⚠ {" | ".join(r.advisor_notes)}</div>'
            )
        html_parts.append("</div>")

    html_parts.extend(["</div>", "</body></html>"])

    gallery_path = OUTPUT / "gallery.html"
    gallery_path.write_text("\n".join(html_parts))
    print(f"\nGallery: {gallery_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    showcases = [
        ("Bar: Gene Expression", showcase_bar),
        ("Line: Cell Viability", showcase_line),
        ("Scatter: Biomarker Correlation", showcase_scatter),
        ("Box: Alignment Accuracy", showcase_box),
        ("Violin: Tissue Expression", showcase_violin),
        ("Strip: Clinical Trial (n=6)", showcase_strip),
        ("Swarm: Immune Cell Counts", showcase_swarm),
        ("Histogram: P-value Distribution", showcase_histogram),
        ("Heatmap: Drug Correlation", showcase_heatmap),
        ("Auto-detect: Cell Migration", showcase_auto),
        ("Showcase Mode: Business", showcase_presentation),
    ]

    results: list[tuple[str, object]] = []
    for name, fn in showcases:
        print(f"  Generating: {name}...", end=" ", flush=True)
        result = fn()
        results.append((name, result))
        print(f"✓ {result.kind} → {result.path}")
        if result.advisor_notes:
            for note in result.advisor_notes:
                print(f"    ⚠ {note}")

    generate_gallery(results)

    print(f"\n{'=' * 60}")
    print(f"  {len(results)} figures generated in {OUTPUT}/")
    print("  Open gallery.html to inspect all figures visually.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
