# Whetstone cross-section claim substantiation — PRD

**Status:** Draft. User review required before Phase 2 (design).
**Branch:** `whetstone-iterative-review`
**Author:** Claude + user discussion 2026-05-21.

---

## 1. Background & motivation

The user's gold standard is a foundation model with the *whole manuscript*
in context — it catches issues that only exist **across** sections: a claim
in the abstract that the results never deliver, a number that disagrees
between abstract and table, an evaluation that doesn't test the headline
claim. whetstone's section-by-section pipeline (with a one-line document-map
preamble) was built so the work runs on **small, local, private** models —
but that traded away most global, cross-section reasoning. The synthesis
step only *aggregates* local findings; it never reads the document and forms
a holistic judgement, and no step asks "is every claim actually backed up?"

This feature recovers cross-section reasoning **within a small model's
context budget** via map-reduce, and fits the codebase's existing
demand/escalation philosophy.

Prior art: a per-section `claim_evidence` lens already exists
(`agents/lens_prompts.py`), scoped to results sections — "is each empirical
claim here anchored to a figure/number *here*". This feature is its
**document-wide** generalisation, built on a shared digest rather than
duplicating it.

## 2. The mechanism — map / reduce / (trust)

### Map — per-section structured digest (fits any context)

For each section, a single-job extractor produces a compact digest:
- **Claims** — every empirical assertion the section makes.
- **Evidence** — internal support: numbers, results, figure/table references.
- **Citation presence** — whether each claim is accompanied by a citation
  (Pandoc `[@key]` / numeric `[N]` markers).
- **Terms defined**.

**Quality guarantee (decision 4 — "drop it"):** every extracted item carries
a **verbatim quote** and is **discarded if it can't be anchored** to real
section text. The digest is therefore hallucination-free — model-invented
claims cannot enter it. Flat single-job schema (small-model-fillable); one
section at a time (context is never the bottleneck).

### Reduce — global reasoning over the compact digest

Reason across all sections' digests (the digest is small even when the raw
document is 55k chars). For each claim, decide whether it is **substantiated**
— **by the paper's own data OR by a citation** (decision 2, "and/or"). A claim
is flagged only when it has **neither** internal supporting evidence **nor**
a citation.

Claim→evidence matching reuses the Consolidate pattern: an embedding
substrate pre-matches each claim to candidate evidence items, and the LLM
confirms "does this evidence support this claim?". A claim with no confirmed
support and no citation → **unsubstantiated claim** finding. The same pass
surfaces number/term inconsistencies across sections.

### Trust the digest (decision 3) — no mandatory full-text re-read

Findings are emitted **directly from the digest reasoning**, without pulling
full section text to confirm. This is the fast/cheap path the user chose.
The accepted residual error is "evidence existed but the digest missed it."

**Mitigations (so this doesn't reintroduce the comment flood):**
- Digest-derived findings are tagged **lower confidence** and labelled
  *"flagged by cross-section scan — not verified against full text"* so they
  are visibly second-tier, never presented as confirmed.
- They flow through **Consolidate** (dedupe + roll-up) and the existing
  severity/priority gating like any other finding.
- The "and/or citation" bar is lenient, which itself suppresses false
  positives.
- (Optional future toggle: a `--verify-claims` flag that turns on full-text
  confirmation for users who prefer precision over speed. Not in v1.)

### Holistic verdict

The digest is also handed to **synthesis**, so it can finally produce a
top-level structural judgement ("the central gap is X") instead of only
aggregating local findings.

## 3. Decisions (user, 2026-05-21)

1. **Claim scope — every empirical assertion** (not just headline claims).
   Thorough; volume controlled by the §2 mitigations + Consolidate.
2. **Substantiation bar — own data AND/OR citation.** A claim survives if it
   has internal evidence *or* a supporting citation; flagged only when it has
   neither.
3. **Verify first — trust the digest.** No mandatory full-text re-read;
   findings emitted from digest reasoning, tagged lower-confidence.
4. **Digest grounding — drop unanchorable items.** Verbatim-anchored or
   discarded → hallucination-free digest.

## 4. How it sits in the architecture

- **Map:** a lightweight per-section digest extractor. Open question (§6):
  fold into the existing per-section read to amortise cost, or a dedicated
  pass that enriches `document_map` regardless of lens configuration.
- **Reduce:** a new global node (working name `ReconcileClaims`) after
  `CriticalRead` (or folded into `ReflectAndInvestigate`, which already does
  cross-section work). Emits anchored findings.
- **Downstream:** findings → existing anchoring → `Consolidate` → renderers.
  Digest → `Synthesise` for the holistic verdict.
- **Model-agnostic:** the digest path benefits every model; it is *not*
  gated on model size. A genuine whole-document holistic pass for
  large-context models is a separate, later lane (out of scope here).

## 5. Cost

- Map: ~1 extraction call per section (24 on the test manuscript), or free if
  amortised into the existing per-section read.
- Reduce: embedding of claims+evidence (local, cheap) + one short LLM confirm
  per candidate claim→evidence pair; claims with no candidate support need no
  confirm call.
- Verify: 0 (trusting the digest).

## 6. Open questions for design phase

1. **Digest cost placement** — side-output of the existing per-section lens
   read (cheaper, couples to `CriticalRead`) vs a dedicated extraction pass
   (clean separation, +1 call/section, runs regardless of lens config).
2. **Embedding reuse** — the claim→evidence substrate reuses
   `core.embeddings`; confirm it shares the `embedding_model` knob added for
   Consolidate.
3. **Anchor of an unsubstantiated-claim finding** — anchor on the claim's
   own quote (where the over-reach is) — confirm that's the right surface vs
   the (absent) evidence.
4. **Number/term inconsistency** — ship in v1 alongside substantiation, or
   land substantiation first and add consistency as a fast-follow.

## 7. Success criteria

- On the real manuscript: surfaces genuine unsubstantiated headline claims
  (e.g. a robustness claim with no robustness experiment) that the
  section-by-section pass missed.
- Digest is hallucination-free (every item anchors to verbatim text).
- Digest-derived findings are visibly tagged lower-confidence and don't
  balloon the comment count (Consolidate + gating hold volume).
- Synthesis emits a holistic verdict drawn from the digest.
- Canonical green (`pytest` / `pyright` / `ruff`).
