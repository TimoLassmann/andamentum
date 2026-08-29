#!/usr/bin/env python3
"""Build code_index.html — a navigable index of the experiment's code & data.

A companion to report.html (the scientific write-up). This page links to every
file in the folder with a one-line description, grouped by role, so a new reader
can find their way around. Links are RELATIVE, so they resolve when the folder is
unzipped and browsed locally. Reuses the report's stylesheet for a matching look.

    uv run python -m experiments.dirichlet_confidence.build_code_index
"""

from __future__ import annotations

from pathlib import Path

from experiments.dirichlet_confidence.build_report import STYLE

HERE = Path(__file__).resolve().parent
OUT = HERE / "code_index.html"

# (relative path, description). Grouped below.
RUN_FILES = [
    ("band_experiment.py", "<strong>Experiment 1</strong> — the band as an error-detector. Gathers (or reuses) the cached histograms and scores the disagreement band; has <code>gather</code> (API) and <code>analyze</code> (offline) subcommands."),
    ("compare_baselines.py", "<strong>Experiment 2</strong> — band vs. self-consistency / semantic entropy / predictive entropy, with paired-bootstrap comparisons. Writes <code>results/baselines.json</code>."),
    ("ablation_1v5.py", "<strong>Experiment 3</strong> — one call vs. five; isolates the disagreement (mutual-information) term. Writes <code>results/ablation_1v5.json</code>."),
    ("call_variability.py", "<strong>Experiment 4</strong> — how much the five histograms actually vary (TV distance, vote agreement, worked examples). Prints to console."),
]
REPORT_FILES = [
    ("make_figures.py", "Renders every figure (fig1–fig5), including Experiment 5's residual-composition analysis, via <code>andamentum.figures</code>."),
    ("build_report.py", "Generates <code>report.html</code> — the scientific write-up, with figures base64-embedded."),
    ("build_code_index.py", "Generates this page."),
    ("Snakefile", "The reproducibility DAG: cache → analysis JSON → figures → report. Run from the repo root with <code>snakemake -s …/Snakefile</code>."),
]
CORE_FILES = [
    ("mixture.py", "The mixture-of-Dirichlets maths: per-row smoothing, concentration fitting, and the within/between variance decomposition the band is read from."),
    ("metrics.py", "Scoring: error-detection AUROC, risk–coverage curves, ECE, Brier, sampled-NLL."),
    ("phrasings.py", "The five prompt templates (the phrasing arm) for SciFact claim verification."),
    ("scifact.py", "SciFact loader → balanced <code>(claim, evidence, label)</code> subsets; the source of the gold labels."),
    ("elicit.py", "Single-call elicitation helper (one prompt → one histogram)."),
    ("elicitation.py", "Multi-model / multi-phrasing elicitation harness (used by the gather path)."),
    ("cache_io.py", "Append-only JSONL cache reader/writer for elicited histograms."),
    ("__init__.py", "Package marker."),
]
DOC_FILES = [
    ("report.html", "▶ <strong>Start here.</strong> The full scientific write-up with all figures embedded."),
    ("README.md", "The original folder README."),
    ("FINDINGS.md", "The original running findings log."),
    ("HANDOVER_README.md", "Orientation for this bundle: what to read first, dependencies, how to run."),
]
DATA_FILES = [
    ("data/scifact/", "The SciFact data (claims + corpus). Source of claims and gold labels; needed to reproduce the exact 150-claim subset (seed 42)."),
    ("cache/", "The cached model answers — <code>band_experiment*.jsonl</code>, 150 claims × 5 calls × 2 arms × 3 models. Every offline analysis reads these; no API needed."),
    ("results/", "Machine-readable results — <code>baselines.json</code>, <code>ablation_1v5.json</code>, and the per-model band scores."),
    ("figures/", "The rendered PNG figures embedded in the report."),
    ("tests/", "Unit tests for the maths (mixture, metrics, scifact loader)."),
]


def _stat(rel: str) -> str:
    p = HERE / rel
    if rel.endswith("/"):
        if not p.exists():
            return "—"
        n = sum(1 for _ in p.rglob("*") if _.is_file())
        return f"{n} files"
    if not p.exists():
        return "—"
    kb = p.stat().st_size / 1024
    if rel.endswith((".py",)):
        lines = sum(1 for _ in p.open(encoding="utf-8", errors="ignore"))
        return f"{lines} lines"
    return f"{kb:.0f} KB"


def _rows(items: list[tuple[str, str]]) -> str:
    out = []
    for rel, desc in items:
        exists = (HERE / rel).exists()
        if exists:
            link = f'<a href="{rel}"><code>{rel}</code></a>'
        else:
            link = f"<code>{rel}</code>"
        out.append(f"<tr><td>{link}</td><td>{desc}</td><td class='num'>{_stat(rel)}</td></tr>")
    return "\n".join(out)


def _section(title: str, items: list[tuple[str, str]]) -> str:
    return (f"<h2>{title}</h2>\n<table>\n"
            "<thead><tr><th>File</th><th>What it is</th><th class='num'>Size</th></tr></thead>\n"
            f"<tbody>\n{_rows(items)}\n</tbody>\n</table>")


HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>dirichlet_confidence — Code Index</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&display=swap" rel="stylesheet">
<style>{STYLE}</style>
</head>
<body>
<div class="article">

<h1>Code index — <code>dirichlet_confidence</code></h1>
<p class="subtitle">A map of the LLM-confidence reliability experiments. The science is in <a href="report.html">report.html</a>; this page is the developer's-eye view — every file, what it does, and how they fit together. Links are relative, so they work from the unzipped folder.</p>

<div class="note accent">
<strong>Read <a href="report.html">report.html</a> first</strong> for the full story (the question, five experiments, conclusions, novelty scan, next steps). Then come here to find the code behind any result.
</div>

<div class="note">
<strong>How to run.</strong> The scripts are part of the <code>andamentum</code> package and run as modules from the repo root. Place this folder at <code>andamentum/experiments/dirichlet_confidence/</code>, then <code>uv sync --extra dev</code>. Everything except the <code>gather</code> step is pure re-analysis of the cached answers — no API key, no cost. One-command rebuild: <code>snakemake -s experiments/dirichlet_confidence/Snakefile --cores 1</code>. Figures need <code>andamentum.figures</code>; re-gathering needs <code>andamentum.core</code> + an <code>OPENAI_API_KEY</code> in a <code>.env</code>.
</div>

{_section("Run these — the five experiments", RUN_FILES)}
{_section("Reproducibility &amp; reporting", REPORT_FILES)}
{_section("Core maths &amp; data loaders", CORE_FILES)}
{_section("Documents", DOC_FILES)}
{_section("Data, cache &amp; outputs", DATA_FILES)}

<p class="meta-stamp">
<code>dirichlet_confidence</code> · generated by <code>build_code_index.py</code> · companion to <code>report.html</code> · shared style with the andamentum <code>disco_flow.html</code>.
</p>

</div>
</body>
</html>
"""


def main() -> None:
    OUT.write_text(HTML, encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
