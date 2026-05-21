# Whetstone evaluation — PRD

**Status:** Draft. User review required before building the harness.
**Branch:** `whetstone-iterative-review`
**Author:** Claude + user discussion 2026-05-21.
**Home:** `benchmarks/whetstone/` (committed, never shipped — wheel is
`src/andamentum` only).

---

## 1. Why

Whetstone can be tinkered with forever without a verdict. The architecture
was deliberately refactored from "whole document in a foundation model's
context" to a chunked, section-by-section pipeline so it runs on small, local,
private models — but that trades away global, cross-section reasoning. We need
a **decision-grade** answer to one question, not endless tinkering:

> Holding the model constant, does whetstone's chunked pipeline miss
> **critical** issues that a single whole-document read catches?

The output is not a leaderboard number — it's a **prioritised gap list** that
tells us what to fix (architecture vs lens bugs vs noise) or tells us the
architecture is already sound and the only cost is model size.

## 2. The core design decision: hold the model constant

Run **both arms on the same frontier model M**, so any gap is purely
*architecture* (chunked vs whole-context), not model capability.

- **Arm A — whetstone**: the full pipeline on model M.
- **Arm B — whole-document baseline**: one prompt, the entire document in M's
  context, asked for a critical review in the same output shape (findings +
  a top-3 "central weaknesses" verdict).
- **Arm C — whetstone on a small local model** *(optional, secondary)*: only to
  quantify the local/privacy tax, reported separately. Never the primary A/B.

Both arms consume **identical harvested text** (see §4) so extraction
differences can't confound the result.

## 3. Corpus

**bioRxiv v1 preprints** are the primary source: they have version history, so
the earliest version is a genuine pre-peer-review draft with real issues still
in it (polished published papers have too few remaining flaws to discriminate).
Uniform PDF format, public author-posted (no confidentiality concern, and
whetstone's own confidentiality guard won't trip).

- **Selection rule:** papers with ≥2 versions where the revision is
  *substantive* (not a typo pass); a spread of subfields; sane length (skip
  80-page genomics monsters for the pilot).
- **Open decision:** bioRxiv-only (bio-concentrated — a named limitation) vs
  **bioRxiv + a few arXiv** (CS/stats/physics, also versioned) for breadth,
  since whetstone is meant to be domain-agnostic. Recommendation: ~14 bioRxiv
  + 6 arXiv for the full run; pilot can be bioRxiv-only.
- **Weak ground-truth enrichment (optional, post-pilot):** the **v1 → later
  version diff** reveals issues the authors actually fixed. Use it as
  *qualitative corroboration* on the interesting findings ("did either arm flag
  something that was later revised?"), NOT as a hard precision metric — most
  diffs are cosmetic, many real fixes weren't reviewer-prompted, and many real
  issues never get fixed.

## 4. Pipeline (per paper)

```
fetch v1 PDF ─► harvest.extract → markdown (ONCE, shared)
                     │
                     ├─► Arm A: whetstone review_document(markdown, model=M)
                     └─► Arm B: whole-doc critical review (markdown, model=M)
                              │
                              ▼
                    align + bucket findings (judge model)
                              │
                              ▼
                    blinded human adjudication (subset)
                              │
                              ▼
                         per-paper + aggregate readout
```

## 5. Measurement — comparative, against a pre-registered rubric

There is **no ground truth** for a paper's "true" flaws, so the unit of
analysis is the **difference between the two reviews**, judged against a rubric
fixed *before* looking at results.

For each finding, bucket: **both / A-only / B-only**. Tag each:
- **severity:** critical vs minor (rubric defines "critical": e.g. an
  unsupported central claim, a methodological flaw invalidating a result, a
  cross-section contradiction, a missing control/comparison the conclusions
  depend on).
- **locality:** cross-section (needs whole-document reasoning) vs local
  (within one section — whetstone *should* have caught it).

**Headline metric:** count of **critical, cross-section** issues that B found
and A missed — that is literally the research question, quantified.

**Secondary:**
- Noise each side (false/low-value flags) — does whetstone over-flag (the known
  volume problem)? Does whole-doc?
- Did A's **synthesis verdict** match B's **top-3** central weaknesses? (the
  "did it find the central problem" test.)
- A-only findings: real issues whole-doc missed, or noise?

## 6. Adjudication — credible despite a judge model

Aligning two finding-lists and rating criticality is subjective, and an LLM
judging LLM output is circular. So:
- A **strong judge model** does the matching + bucketing pass at scale.
- A **human adjudicates a blinded subset** — findings shown as "System 1 / 2"
  with no labels (blinding matters *because we built whetstone*) — covering at
  minimum **every B-only "critical" item** plus a random sample, to validate
  the judge. Disagreement rate between judge and human is itself reported.

## 7. Harness location & data hygiene

- **Code:** `benchmarks/whetstone/` — committed, versioned with the code that
  produced it, never in the wheel. Mirrors `benchmarks/chunker/` layout
  (`cli.py`, `runner.py`, `loader.py`, `report.py`, `README.md`).
- **Data + outputs:** gitignored. `benchmarks/whetstone/.gitignore` ignores
  `corpus/` (downloaded PDFs/markdown) and `runs/` (both arms' outputs +
  adjudication). Co-located with the code for convenience; never in git.
- **No ambient state:** harness takes explicit `--corpus-dir` / `--out-dir`
  (default to the gitignored subdirs, overridable to a path outside the repo).
  No env vars, no hidden home-dir writes.

## 8. Plan

1. **Pilot — 5 bioRxiv v1 papers.** Build the harness, calibrate the rubric,
   and answer fast/cheap whether a gap is even visible. If A(M) ≈ B(M) on 5,
   we may have the answer already.
2. **Full run — 20 papers** (corpus mix per §3 decision), full adjudication.
3. **Readout → prioritised gap list:** B-only & cross-section → architecture
   work (e.g. the substantiation pass); B-only & local → lens/prompt bugs;
   A-only noise → the volume problem.

## 9. Honest caveats (state in the writeup)
- Directional, not statistical — 20 papers shows patterns, not p-values. Enough
  to *decide*, not to publish a benchmark.
- Comparative, not absolute — no gold standard of "true" flaws.
- Third-party preprints are used for internal evaluation; not whetstone's stated
  use (own drafts).
- Judge-model circularity, mitigated by blinded human spot-check.

## 10. Resolved decisions (user, 2026-05-21)
1. **Corpus breadth — bioRxiv + a few arXiv.** ~14 bioRxiv + ~6 arXiv for the
   full run, so the result isn't bio-only; whetstone is domain-agnostic.
2. **Model is an argument — no hardcoded M.** The harness takes `--model` and
   resolves it through the existing infrastructure (`core.models.resolve_model`
   / `resolve_model_from_args`, the same path the CLIs use), so any pydantic-ai
   id works — `ollama:…`, `openai:…`, `bedrock:…`. **Both arms run on the same
   resolved model**: Arm A passes the string to `review_document(model=…)`
   (whetstone resolves internally); Arm B builds a pydantic-ai agent via
   `core` from the same string. The judge/adjudication model is a separate
   `--judge-model` argument (same resolution path).
3. **A vs B only for now.** No Arm C (small-local tax) and no v1→later-version
   diff enrichment in the first study — keep it to whetstone vs whole-document
   on one shared model. Both remain documented as future extensions.
