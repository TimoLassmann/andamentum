# Mixture-of-Dirichlets confidence (standalone experiment)

An independent, self-contained implementation of the verbalized-confidence
aggregation method described in `confidence_method_walkthrough.md`. It turns a
noisy, prompt-sensitive verbalized number from an LLM into a probability
distribution whose **spread is decomposed into interpretable parts** (phrasing
brittleness vs model-family disagreement), and validates the whole thing against
SciFact ground truth.

It lives **outside** `src/andamentum/` on purpose. Whether this produces
trustworthy confidence is an open empirical question; it is developed and
benchmarked here in isolation, and only if it clearly works would lifting it
into `andamentum.epistemic` be considered.

## What's here

| File | Purpose |
|---|---|
| `mixture.py` | The maths. `repair_row`, `MixtureOfDirichlets`, `build_mixture`, `fit_concentration_s`. Pure numpy/scipy, no I/O, no andamentum deps. |
| `metrics.py` | Accuracy, ECE, Brier, predictive NLL, sampled-NLL, reliability curve. |
| `phrasings.py` | The K=5 prompt phrasings (4 direct + 1 chain-of-thought) and the shared output contract. |
| `elicitation.py` | Verbalized-distribution elicitation via `andamentum.core.agents` (one runner per model, temperature handled gracefully for reasoning models). |
| `scifact.py` | Loader → flat `(claim, evidence, label)` examples. Handles the canonical AllenAI directory layout and a pre-flattened JSONL. |
| `benchmark.py` | Cached elicitation + the A/B/C/D + per-model scoring. |
| `cli.py` | Command-line entry point + readable report. |
| `tests/` | 41 offline tests (no network): the math, metrics, loader, parsing, and the full scoring pipeline on synthetic data. |

## The method in one paragraph

Elicit `M models × K phrasings` of a categorical distribution over
{SUPPORT, CONTRADICT, NOINFO}; smooth zeros with a weak prior (`alpha0`); pool
the `M×K` rows as a **mixture** of Dirichlets (not an average — the spread
survives) with a shared concentration `s`; fit `s` on labelled dev data by
minimising *sampled* NLL (the predictive mean is insensitive to `s`); then
decompose each prediction's variance into `within_component` (elicitation noise),
`within_model` (phrasing brittleness) and `between_model` (model-family
disagreement).

## Running the benchmark

```bash
uv run python -m experiments.dirichlet_confidence.cli \
    --scifact-path /path/to/scifact \
    --n-claims 60 \
    --models ollama:gemma3:27b openai:gpt-5.1 openai:gpt-5.1-codex openai:gpt-5.1-codex-mini \
    --out experiments/dirichlet_confidence/results/run1.json \
    --verbose
```

- `--scifact-path` is a directory containing `corpus.jsonl` + a `claims_*.jsonl`,
  **or** a flat `.jsonl` with `claim` / `evidence` / `label` fields.
- Default models are one local + the three gpt-5.1 variants. All three OpenAI ids
  are confirmed available; `gemma3:27b` must be pulled (`ollama pull gemma3:27b`)
  or swapped out — the model list is fully configurable.
- Elicitations are cached under `--cache-dir` keyed by (claim, model, phrasing),
  so an interrupted or repeated run re-pays for nothing already gathered.
- API keys are read from `.env` via `python-dotenv`.

The report compares: each model asked **directly** (single prompt + its
multi-phrasing mean), then configs **A** (single prompt), **B** (multi-phrasing
mean), **C** (full point mixture), **D** (Dirichlet mixture at fitted `s`).
C and D tie on accuracy/ECE/Brier by construction; D's win, if any, shows up in
**sampled-NLL** and the variance decomposition.

## Tests

```bash
uv run pytest experiments/dirichlet_confidence/tests/ -q
```

These are not collected by the repo's default `uv run pytest` (the package is
outside `testpaths`), keeping the experiment fully decoupled.
