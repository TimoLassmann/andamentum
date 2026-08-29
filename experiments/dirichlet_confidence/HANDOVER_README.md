# Handover — LLM Confidence Reliability experiments

This bundle is the `dirichlet_confidence` experiment folder from the **andamentum**
repository: a study of whether sampling an LLM multiple times and measuring its
self-disagreement gives a reliable signal of *when its answer is wrong*, and how
that compares to simply reading the model's stated confidence once.

## Start here

1. **`report.html`** — the full scientific write-up. Open it in any browser; it is
   self-contained (all figures embedded). Read this first.
2. **`code_index.html`** — a navigable map of every code/data file with a description.

## What's inside

- **5 experiments**, each a runnable script: `band_experiment.py` (band as detector),
  `compare_baselines.py` (band vs. standard scores), `ablation_1v5.py` (1 call vs. 5),
  `call_variability.py` (do the 5 calls differ?), and `make_figures.py` (the plots,
  incl. the residual-composition analysis).
- **Cached model answers** in `cache/band_experiment*.jsonl` — 150 SciFact claims ×
  5 calls × 2 arms × 3 models. Every analysis re-reads these; **no API key needed**.
- **Results** in `results/*.json`, **figures** in `figures/`, data in `data/scifact/`.
- **`Snakefile`** — the dependency DAG (cache → analysis → figures → report).

## How to run

These scripts are part of the `andamentum` package and run as modules from the repo
root. To execute them:

1. Clone/obtain the `andamentum` repo and place this folder at
   `andamentum/experiments/dirichlet_confidence/`.
2. From the repo root: `uv sync --extra dev`.
3. Rebuild everything (offline, no cost):
   ```
   snakemake -s experiments/dirichlet_confidence/Snakefile --cores 1
   ```
   or run a single experiment, e.g.:
   ```
   uv run python -m experiments.dirichlet_confidence.compare_baselines
   uv run python -m experiments.dirichlet_confidence.ablation_1v5
   uv run python -m experiments.dirichlet_confidence.make_figures
   ```

Everything except `gather` is pure re-analysis of the cached answers — deterministic,
no network, no cost. **`rule gather`** (and `band_experiment.py gather`) is the only
step that queries a model; it needs `OPENAI_API_KEY` in a `.env` and **costs API calls**.
It is deliberately *not* part of the default Snakemake target.

## Dependencies

- Re-analysis (`compare_baselines`, `ablation_1v5`, `call_variability`): `numpy`, `scipy`.
- Figures (`make_figures`): also `matplotlib` via `andamentum.figures`.
- Re-gathering (`gather`): also `andamentum.core`, `pydantic-ai`, `openai`, and a key.

## The one-paragraph result

A single verbalized-confidence call is as good an abstention signal as far more
expensive multi-call methods (self-consistency, semantic entropy, the Dirichlet
disagreement band, mutual information) — the band did **not** win. But multi-call
predictive entropy specifically removes one dangerous, otherwise-invisible error
class: *confident paraphrase-flips* (the answer changes when you reword the question,
yet each answer looks confident). Framed as a **reproducibility** rather than accuracy
concern, that is the one place the extra calls earn their cost. See `report.html` §9–11
for conclusions, the literature/novelty scan, and recommended next steps.
