# Whetstone v3 — whole-document, digest-focused, SPECS-criterion review

**Status:** Design. User review required before build.
**Author:** Claude + user discussion 2026-05-22.
**Supersedes:** the per-section `CriticalRead` pipeline (the chunked v2 core).

---

## 1. Why

Whetstone v2 reviews **section by section** — a workaround for small-context
models that has become the root cause of its two worst failures:

- **The minor-issue flood** — per-section × per-lens multiplication (24
  sections × N lenses × "0–3 each") → dozens of equal-weight comments.
- **Cross-section blindness** — a lens that sees one chunk cannot judge the
  story arc, whether the abstract is delivered later, or significance. Every
  patch this session (doc-map preamble, ReconcileClaims, the multi-section
  consistency lens, Consolidate) is an epicycle compensating for this.

The AAAI-26 AI Review pilot (arXiv:2604.13940) — deployed at conference scale,
preferred over human reviews on technical accuracy — uses the opposite shape:
**whole-document, criterion-targeted stages (SPECS: Story, Presentation,
Evaluations, Correctness, Significance), accumulating context, then synthesis +
self-critique.** v3 adopts that shape and resolves the weak-model problem with
a pre-built, verified **document digest** that lets even gemma-class models
reason across the whole paper without ever holding a big input.

## 2. Core ideas

1. **Whole-document-first.** Criterion stages reason over the whole paper, not
   one section. Chunking is demoted to a fallback for documents that exceed the
   model's context (theses, books).
2. **A radically simple, verified document model (digest) is the universal
   focusing layer.** The ONLY thing the LLM extracts is **claims, as verbatim
   spans** — "copy the sentences that assert something." No paraphrase (the span
   *is* the claim, and is verifiable), no classification enums (criterion stages
   judge relevance themselves). Everything else is **deterministic or
   read-on-demand**: section **gists** (title + first sentence) and **citations**
   (regex over `[N]`/`[@key]` markers + the reference section) are deterministic;
   `has_citation` is a regex over the claim's span; **support / evidence / links
   / equations are NOT pre-extracted** — a criterion stage establishes support by
   reading the *real source text* when it needs it (more grounded than a lossy
   second extraction, and recoverable in the gap loop). General by construction:
   any document has claims; technical facets simply don't appear when absent.
   Document model: `claims[{id, quote, span}]`, `gists[{section_id, title,
   gist}]`, `citations[…]`.
3. **Stages consume projections, not the whole model.** Each criterion reads a
   *slice* of the document model relevant to it (with its dependencies — e.g.
   claims **plus the support they link to**), and each atomic question can drill
   to a minimal per-question slice (the claim + its linked support + the located
   source span). The document model is a **shared store that stages query**, not
   a blob handed to each. Smaller, exact context is *more* reliable on a weak
   model, not just cheaper. Strong models may also receive full text as a
   backstop.
4. **String-match verification + retry (the reliability spine).** Every span an
   LLM emits — every claim AND every finding — is located in the source by
   normalised string matching (strip markdown; the same primitive as docx
   comment anchoring), **scoped to the item's origin section** so there's no
   occurrence ambiguity. On a miss, the agent is asked to re-quote the exact span
   for *that* item — **≤3 round trips**, then dropped. Hallucinations cannot
   survive into the digest or the review.
5. **Atomic, criterion-specific questions.** A criterion stage is not one
   "review this" call — it is a set of *separate, atomic, checkable questions*
   ("Is there a baseline? Does claim C1 have support?"). Atomic questions are
   far more reliable on weak models and give cleaner, gateable findings.
6. **Criteria are a pluggable set, NOT baked into the graph.** The graph and
   digest are domain-general; the *criterion set* is data. **SPECS (Story,
   Presentation, Evaluations, Correctness, Significance) is the DEFAULT set for
   academic documents — one configuration among several.** The active set is
   chosen by document type (academic / external-communication / general) or
   supplied by the user. This **unifies** whetstone's existing modes — custom
   criteria, journal guidelines, expert panel — under one architecture; each is
   just "a criterion set run over the document model". The graph never names
   "Story" or "Evaluations".
7. **A gap-analysis loop re-grounds in the source before synthesis.** Reviewing
   the representation alone is unsafe — a claim extraction missed is invisible
   from inside the digest. So between the criterion stages and synthesis sits a
   bounded loop (deep_research / epistemic `Demand` style): find coverage gaps,
   emit demands, satisfy them by **re-reading the original source** of the gap
   area, verify, and loop until no demands or a round cap. This makes lossy
   extraction *recoverable* and folds the novelty check in as a routed demand.
8. **Clean deterministic / agent node split.** Every graph node is EITHER
   deterministic (pure transform, fully unit-testable) OR an agent node (one
   job, mockable). No node mixes the two. Mirrors epistemic's P1.
9. **One review, not a pile.** Output is a synthesised, structured review
   (synopsis / strengths / weaknesses); Presentation is a *bounded summary*, not
   96 balloons; anchored comments are a secondary detail layer. This is the
   flood fix, natural once stages are whole-document.

## 3. The pydantic-graph (nodes; D = deterministic, A = agent)

```
Harvest (D,IO)
  └─► Sectionize (D)               = the existing chunker (heading-aware, size-banded, offsets)
        └─► ExtractDigest (A)            map: per-UNIT verbatim CLAIM spans only  ← only per-section node
              └─► VerifyDigest (D+A)      locate each claim (section-scoped); retry-quote on miss ≤3; drop misses
                    └─► BuildDocumentModel (D)   located claims + deterministic gists/citations/has_citation → shared store
                          └─► CriterionReview (A, fanned over the ACTIVE criterion set, parallel)
                                  one generic node, one instance per criterion in the set;
                                  each reads its PROJECTION of the model + asks atomic questions
                                  (academic default set = SPECS; other sets for other doc types / user)
                                └─► VerifyFindings (D)   locate every finding quote; drop hallucinations
                                      └─► ┌──────── GAP LOOP (≤ round cap) ────────┐
                                          │ CoverageMap (D)    gaps: sections/claims/criteria under-covered
                                          │   └─► GapAnalysis (A)   reads coverage + gists + findings +
                                          │         │               ORIGINAL doc structure → emits Demands
                                          │         │               (empty → exit loop)
                                          │         └─► RouteDemands (D)   demand → minimal satisfier
                                          │               └─► SatisfyDemands (A)  RE-READ original source of
                                          │                     │                 the gap area; new claims /
                                          │                     │                 follow-ups / novelty search
                                          │                     └─► VerifyNew (D)   locate/drop (same primitive)
                                          │                           └─► LoopControl (D)  demands & under cap? ↺
                                          └──────────────────────────────────────────┘
                                      └─► GateAndAggregate (D)   importance-gate; summarise presentation; dedup
                                            └─► Synthesise (A)        one structured review
                                                  └─► SelfCritique (A)   reread vs model/source; flag unsupported
                                                        └─► Revise (A)     apply critique → final review
                                                              └─► End[ReviewResult]
Render (D)  — md / html / docx, post-result (reuses existing renderers)
```

**Node responsibilities:**

| Node | D/A | Job (single) | Section or whole-paper |
|---|---|---|---|
| Harvest | D (I/O) | source → markdown (reuse `harvest`) | whole |
| Sectionize | D | **the existing chunker** (`extract_units`): heading-aware, size-banded units with breadcrumbs + char offsets | makes the units |
| ExtractDigest | A | per **unit**: return the **verbatim claim spans** only — "copy the sentences that assert something." One field, no enums, no paraphrase (maximally weak-model-fillable) | **per-section (only one)** |
| VerifyDigest | D (+A retry) | normalised string-match each claim quote, origin-section-scoped; on miss, agent re-quotes (≤3); attach span; **drop unverifiable** | whole |
| BuildDocumentModel | D | assemble located claims + deterministic `has_citation` (regex), deterministic section gists (title + first sentence), deterministic citations (markers + reference section) → the shared document model. **No support/links/equations pre-extracted.** | whole |
| CriterionReview | A | **one generic node, instantiated per criterion in the active set**; reads its *projection* of the model + asks **atomic questions** (+ located sections / full text per budget); emits findings with verbatim quotes. Presentation also folds in the deterministic proofread as a summary. | whole (via projection) |
| VerifyFindings | D | locate every finding quote; **drop hallucinated/unanchorable** | whole |
| CoverageMap | D | light input for reflection: what was *examined* + which findings exist (not "coverage of all sections") | whole |
| GapAnalysis | A | **re-examination**: which existing findings to re-verify against the real text, plus what may have been missed; emit flat `Demand`s; none → exit loop | whole |
| RouteDemands | D | route each demand to its minimal satisfier (re-verify finding / re-read section / targeted question / external novelty search) | whole |
| SatisfyDemands | A | **re-read the ORIGINAL source**: re-check a finding's veracity against the text, extract missed claims, answer a follow-up, or run a novelty search | targeted section(s) |
| VerifyNew | D | locate/drop the new items (same `locate`) | whole |
| LoopControl | D | loop while demands raised AND under the round cap; else proceed | whole |
| GateAndAggregate | D | importance-gate (criterion + severity); aggregate Presentation/style into a bounded summary; dedup near-duplicates | whole |
| Synthesise | A | gated findings + document model → one structured review (synopsis/strengths/weaknesses) | whole |
| SelfCritique | A | reread the review against the document model/source; flag unsupported claims, factual errors, weak citations | whole |
| Revise | A | apply the self-critique → final review | whole |
| Render | D | reuse `render_markdown` / `render_html` / `render_docx` | whole |

Only **ExtractDigest** works section-by-section (cheap, parallel, weak-model-sized).
Every reasoning node works over the whole paper via the compact shared model.

**The active criterion set is data, not graph structure.** A criterion =
`{name, atomic questions, which digest facets it reads}`. SPECS is the academic
default; document type or the user selects the set; custom-criteria / guidelines
/ panel modes are all just alternative sets.

**Shared deterministic primitive:** `locate(quote, source) -> Span | None`
(normalised string match) — generalised from `whetstone/docx/anchor.py`. Used
by VerifyDigest, VerifyFindings, and the renderers. One reliability mechanism,
three call sites.

## 4. One reasoning surface for all models (no capability detection)

**The digest is the primary reasoning surface for *everyone*.** We do NOT try to
detect model capability (a big context window ≠ strong reasoning). Instead:

- Every criterion stage reasons over the **document model + per-question slices**
  (claim + its located source section). Small, exact context — what a weak model
  needs and a strong model still benefits from.
- **Raw full text is added only when it fits** the model's context, as extra
  grounding. That's the only thing that varies — a fit check, not a capability
  guess.
- **Over-budget document:** the `chunker` already size-bands extraction units, so
  the digest stays compact regardless of document length; the reduce is always
  whole-document over that compact model.

No "weak vs strong mode," no hard switch — one surface, optionally enriched with
raw text when it fits.

## 5. What's reused vs retired

**Reused (rendering = 100% reuse):** `ReviewResult` is the output contract, so
**all three renderers + the docx track-changes/anchoring machinery consume v3's
output unchanged** — v3 just has to fill a `ReviewResult`. Also reused:
`harvest`, the `chunker` (sectionize), `anchor.py` → `locate` (now the
verification spine), the `digest_extractor`/claim machinery (→ ExtractDigest),
the section classifier, `deep_research` (novelty, now a routed gap demand),
`AuthorQuestions`, and the `Synthesise`/`Challenge` *node skeletons*.

**Reframed (reused skeleton, new agent contract):** `Synthesise`'s agent shifts
from must-fix/should-fix buckets to synopsis/strengths/weaknesses; `Challenge`'s
refute logic is re-aimed into `SelfCritique` (review-level, not per-finding). A
small `Revise` step is new; possibly one structured-review field on
`ReviewResult` (or reuse the `summary` string).

**Unified, not retired:** today's **custom-criteria**, **guidelines**, and
**panel** modes all become *alternative criterion sets* run by the one generic
CriterionReview node — so v3 generalises whetstone's modes rather than narrowing
to academic papers.

**Reborn:** `ReflectAndInvestigate`'s gap-finding function returns as the **gap
loop** — improved: demand-routed, source-grounded, capped, clean D/A nodes.

**Retired:** the per-section `CriticalRead` loop, the seven persona lenses
(folded into the pluggable criterion sets), the document-map preamble, most of
`Consolidate` (GateAndAggregate replaces it), `ReconcileClaims` (its logic is
now a claim↔support check over the document model).

## 6. Build phases

0. **`locate` primitive** — generalise `anchor.py` into a clean
   verify-and-locate util + tests. Foundation for everything.
1. **Verified document model** — ExtractDigest (A) → VerifyDigest (D) →
   BuildDocumentModel (D). Load-bearing reliability piece; fully testable.
2. **Criterion stages** — the one generic CriterionReview node + the academic
   **SPECS** criterion set (atomic questions over the document-model projection)
   + VerifyFindings (D). Other criterion sets come later.
3. **Gap loop** — CoverageMap (D) → GapAnalysis (A) → RouteDemands (D) →
   SatisfyDemands (A) → VerifyNew (D) → LoopControl (D), capped. Re-grounds in
   the source; novelty search rides in as a demand.
4. **Output reframe** — GateAndAggregate (D) + Synthesise/SelfCritique/Revise
   (A) into one `ReviewResult`. This is where the flood dies; rendering is reuse.
5. **Raw-text-if-it-fits** — wire the fit check that adds full text to a
   stage's context when budget allows (no capability detection).
6. **Wire + migrate** — renderers, retire the old pipeline, validate with a
   SPECS-style perturbation benchmark (inject known errors → measure recall vs
   the old chunked pipeline).

## 7. Locked decisions

- **Digest = claims-only.** The LLM extracts only verbatim claim spans; gists,
  citations, and `has_citation` are deterministic; support/links/equations are
  read-on-demand, not pre-extracted. No enums, no paraphrase.
- **`locate` = normalised match, origin-section-scoped, with ≤3 agent re-quote
  retries on miss, then drop.** Used by VerifyDigest, VerifyFindings, renderers.
- **Gap loop = re-examination**, not coverage: re-verify findings against the
  real source + reflect for misses; novelty search is a routed demand; bounded
  by a round cap (≈2–3, overridable) — the termination guarantee.
- **Digest-primary for all models; add raw text only if it fits.** No capability
  detection, no weak/strong mode.
- **Build v3 as a new graph alongside v2, swap at the end.** `Consolidate` and
  `ReconcileClaims` retire (ideas survive: gating; claim↔support-by-reading).
- **One generic CriterionReview node over a pluggable criterion set;** SPECS is
  the academic default, auto-selected by document type, user-overridable.
- **Primary output = the synthesised structured review** (synopsis / strengths /
  weaknesses); anchored comments are a secondary detail layer.

## 8. Phase-local decisions (settle when we open the phase)

- Extraction granularity — bound to load-bearing claims (avoid the 314-claim
  re-flood). (Phase 1)
- The concrete SPECS atomic questions + criterion-projection mapping. (Phase 2)
- Presentation = agent structural findings + deterministic proofread summary,
  combined without re-flooding. (Phase 2/4)
- Gap-loop demand memory (don't re-emit satisfied demands) + call-count sanity
  on weak local models. (Phase 3)
- `ReviewResult` structured-review field vs markdown-in-`summary`; whether to
  port the opt-in Editor (track-changes) and AuthorQuestions. (Phase 4)
- Validation: build the SPECS-perturbation benchmark to prove v3 > v2, or
  eyeball. (Phase 6)

## 9. Honest limits

- The digest is lossy — extraction recall is the ceiling for weak mode;
  VerifyFindings catches false positives, not misses. Invest in extraction
  reliability (atomic, anchored, partly deterministic).
- Weak model + digest = *good enough*, not frontier-equal.
- Correctness-by-execution (a code interpreter for equations/algorithms) is a
  real capability gap and a separate, later upgrade — out of scope for v3 core.
- This is a genuine rearchitecture of the core, though the leaf machinery and
  renderers are reused.
