# Whetstone comment consolidation — PRD

**Status:** Draft. User review required before Phase 2 (design options).
**Branch:** `whetstone-iterative-review`
**Author:** Claude + user discussion 2026-05-21.

---

## 1. Background

After the `whetstone-review-fixes` work landed, comments now render in Word
and anchor to their flagged text. The next problem the user raised: on a real
manuscript the review produces **too many comments, many redundant**. Two
flavours:

1. **Mechanical flood.** A single proofread run emitted ~96 deterministic
   findings, ~73 of them passive-voice — one Word comment per instance.
2. **Cross-lens / cross-section redundancy.** Several lenses independently
   flag the same sentence; the same concern shows up anchored to two
   different spans (e.g. an overclaim noted on both the abstract and the
   conclusion). Each survives as its own comment.

The user wants comments that are **high quality and not redundant**, via a
mechanism that is **minimal, general, and elegant** — no domain-specific
rules, works on any draft.

## 2. Diagnosis — why volume and redundancy happen (from the code)

The finding lifecycle today:

```
CriticalRead → ReflectAndInvestigate → NoveltyCheck → EditSections
            → Challenge → AuthorQuestions → Synthesise
```

Two structural causes, both verified in the source:

**2.1 Deterministic findings bypass the entire quality pipeline.**
`nodes/chunk_and_scan.py:135` appends proofread output straight into
`state.deterministic_findings`. `nodes/synthesise.py:34` concatenates that
list *raw* with the challenged findings at the very end. Deterministic
findings never face reflection, never face challenge, are never deduped or
aggregated. The proofread adapter (`structural/proofread_adapter.py:280`)
emits **one Finding per instance**, so 73 passive-voice constructions become
73 comments. This is the dominant volume driver and it is purely mechanical.

**2.2 Nothing ever inspects findings as a set.**
There is no dedup / cluster / merge operator anywhere in the module (grep
confirms). `Finding.category` (`schemas.py:60`) is a tag nobody acts on.
`Challenge` (`nodes/challenge.py`) refutes each finding **in isolation** — it
has no cross-finding view, so when two lenses flag the same sentence both
survive. Redundancy among LLM findings is invisible to the system.

The existing `ReflectAndInvestigate` loop *can* drop/refine findings, but it
is framed around investigation, runs on LLM findings only, and does not
operate on the union with deterministic findings.

## 3. Goals / non-goals

**Goals**
- Collapse redundant comments into one, treating cross-lens agreement as a
  confidence signal rather than noise.
- Aggregate mechanical findings (proofread style nitpicks) instead of
  enumerating one comment per instance.
- Route deterministic findings *through* the same consolidation step so they
  stop bypassing the pipeline.
- General mechanism: no domain rules, keyed only off data every `Finding`
  already carries (anchored quote span, category, rationale).

**Non-goals**
- No new "verify the comment" loop — `Challenge` and `ReflectAndInvestigate`
  already verify. This work is about the finding *set*, not re-checking
  individual findings.
- No change to how findings are *born* (lenses, reflection) beyond the
  consolidation step.
- Not building a generic clustering library — one purpose-built node.

## 4. Locked decisions (user, 2026-05-21)

1. **Ollama is required.** Embedding-based similarity is always on. If Ollama
   is unreachable the consolidation step **errors loudly** — no silent skip,
   no anchor-overlap-only fallback (per `CONSTITUTION.md`: fail fast, fail
   loud).
2. **Must work with smaller models.** The LLM adjudication step uses a **flat
   binary** schema (`same | distinct`), never a structured cluster partition.
   Merge groups are rebuilt deterministically (union-find) from the pairwise
   verdicts.
3. **Embed the claim**, i.e. the finding's `title + rationale` — not the
   quote. The quote is the anchor; embedding it would only re-discover
   anchor overlaps.

## 5. Proposed mechanism — a single `Consolidate` node

Placed **after `Challenge`, before `Synthesise`**, operating on the **union**
of `deterministic_findings + challenged_findings`. Three tiers:

### Tier 1 — Substrate (deterministic, cheap). Proposes candidates, never decides.

Build an undirected graph over findings with two kinds of edge:

- **Anchor overlap** — same `section_id` and overlapping quote char-spans.
  Catches "two lenses flagged the same sentence."
- **Semantic similarity** — embed each finding's `title + rationale` via
  `core.embeddings.embed_texts(model=<embedding model>)`; cosine ≥ threshold
  (conservative, to keep edges sparse) → edge. Catches "same concern,
  different anchor." `core.embeddings.cosine_similarity` provides the metric.

Connected components of this graph are **candidate clusters**. Pure recall;
no judgment yet.

### Tier 2 — Adjudication (LLM, precise, small-model-safe).

The substrate over-proposes on purpose. For each candidate **edge** that is
not short-circuited (see Tier 3), one flat binary call:

> Given finding A and finding B (rationale + quote + section for each):
> are these **the same issue**, or **distinct** issues that merely co-locate?

Output schema is flat: `{ relation: "same" | "distinct" }`. Union-find over
the `same` verdicts reconstructs merge groups transitively, so a cluster of
size *k* needs at most its candidate-edge count of calls (often just 1 for a
pair), and never a partition. The model only ever sees two short findings per
call — trivially within a small model's reach.

This is exactly the user's framing: substrate says "these *might* be the
same"; the LLM decides "same section but actually different" vs "genuinely
one issue."

### Tier 3 — Merge (deterministic to start).

- **Homogeneous deterministic short-circuit.** A candidate edge where both
  findings are `source="deterministic"` with the same `category` (e.g. two
  `style:passive`) is merged **without** an LLM call — `category` already
  proves they are the same kind. All passive-voice in a section roll up to
  one comment ("N passive-voice constructions in this section"), anchored at
  the first instance, body listing each.
- **Confirmed-same LLM groups.** Keep the highest-severity / highest-
  confidence member as canonical; union the `perspective`s and `category`s;
  **raise confidence by corroboration** when ≥2 *independent* lenses agree
  (the Reichenbach common-cause borrow from epistemic); note "flagged by N
  perspectives." Merge prose stays deterministic for v1 (an optional small
  LLM "write one unified comment" call is a later refinement if merged
  bodies read poorly).
- **Distinct** → both findings survive, co-located.

## 6. Cost & dependencies

- **Embeddings:** ~150 short claims, local Ollama (`embeddinggemma:latest`
  via `core.embeddings`), batched — cheap. Pairwise cosine over ~150 vectors
  is trivial.
- **LLM calls:** bounded by candidate-edge count (sparse by threshold),
  minus the deterministic short-circuit — typically far fewer than the
  per-finding `Challenge` pass already costs.
- **New runtime dependency:** Ollama becomes required for the review path
  (today it is only needed for over-budget section splitting). Accepted per
  decision 4.1.

## 7. Resolved decisions (user, 2026-05-21)

1. **Embedding model threading — overridable, sensible default.** The
   embedding model flows as an explicit `embedding_model=` kwarg from CLI →
   `review_document` → `Consolidate`, mirroring how `model=` flows today,
   defaulting to `embeddinggemma:latest`. No hidden config, user-overridable.
2. **Similarity threshold — strict to start.** Begin with a high cosine
   threshold so only clearly-similar pairs become candidate edges (fewer LLM
   calls, lower mis-merge risk). Loosen empirically only if real duplicates
   slip through.
3. **Mechanical aggregation — hybrid, via a general count threshold.** Roll
   up a style category within a section into one summary comment **only when
   its instance count in that section exceeds a threshold** (e.g. ≥3);
   below that, keep individual pinpoint comments. This achieves "roll up
   high-volume passive-voice, pinpoint rare duplicate-word typos" *without* a
   hard-coded per-category allowlist — it keys off count, not category
   identity (high-volume categories naturally roll up; rare ones stay
   precise). Threshold is tunable.
4. **Confidence boost — raise confidence AND note who agreed.** Corroboration
   by ≥2 independent perspectives bumps the merged finding's `confidence`
   tier (ranking signal) and records the contributing perspectives so the
   rendered comment can show "raised by N perspectives" (visible context).
   Implementation detail (reuse `perspective`/a new field) deferred to design.

## 8. Success criteria

- On the real manuscript: total comment count drops substantially (target:
  the ~73 passive-voice collapse to a handful of per-section roll-ups), with
  no loss of distinct substantive findings.
- Two lenses flagging the same sentence with the same concern → one comment
  noting both perspectives, at higher confidence.
- Two findings on the same span with *different* concerns → both survive.
- Ollama down → the run errors loudly at the consolidation step.
- Canonical green state holds (`pytest` / `pyright` / `ruff`).
