# Whetstone evaluation

A decision-grade comparison of whetstone's chunked review pipeline against a
single whole-document read by the **same** frontier model — to answer one
question:

> Holding the model constant, does whetstone's chunked architecture miss
> **critical** issues that a whole-document read catches?

Full design, rubric, and methodology:
**`docs/.internal/plans/2026-05-21-whetstone-benchmark-prd.md`**.

## Layout (planned)

| Path | What | Tracked? |
|---|---|---|
| `cli.py` / `runner.py` / `loader.py` / `report.py` | harness code | committed |
| `corpus/` | downloaded bioRxiv/arXiv v1 PDFs + harvested markdown | gitignored |
| `runs/` | both arms' outputs + adjudication results | gitignored |

This directory is **not shipped** in the wheel (the wheel is `src/andamentum`
only), so eval code stays versioned with the package without bloating the
distribution. Downloaded papers and run outputs live under `corpus/` and
`runs/` and never enter git.

## Arms

- **A** — whetstone `review_document(text, model=M)`.
- **B** — whole document in M's context, one critical-review prompt.
- **C** *(optional)* — whetstone on a small local model, to quantify the
  local/privacy tax. Reported separately, never the primary A/B.

Both arms consume the **same harvested markdown** so extraction differences
don't confound the comparison.

## Status

PRD drafted; harness not yet built. Pilot target: 5 bioRxiv v1 papers.
