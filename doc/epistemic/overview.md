# Epistemic system — overview

A formal-epistemology pipeline that takes a question, decomposes it, gathers evidence, and scrutinises each claim from multiple philosophical angles. It produces an answer with calibrated confidence, or suspends judgment when the inquiry doesn't converge.

This document is the markdown counterpart to [`epistemic_flow.html`](./epistemic_flow.html). The HTML is the rendered, human-readable version with diagrams; this is the same content as plain text suitable for grep, diff, and feeding to LLMs.

---

## Contents

1. [Architecture at a glance](#1-architecture-at-a-glance)
2. [The seven primitives](#2-the-seven-primitives)
3. [The graph (23 nodes)](#3-the-graph-23-nodes)
4. [The claim lifecycle](#4-the-claim-lifecycle)
5. [The inquiry cycle (Peirce)](#5-the-inquiry-cycle-peirce)
6. [Adversarial check (Popper / Lakatos)](#6-adversarial-check-popper--lakatos)
7. [The IBE chain (Lipton)](#7-the-ibe-chain-lipton)
8. [Convergence (Reichenbach)](#8-convergence-reichenbach)
9. [Lazy escalation](#9-lazy-escalation)
10. [Confidence scoring](#10-confidence-scoring)
11. [Question-type routing](#11-question-type-routing)
12. [Threshold reference](#12-threshold-reference)
13. [Code structure](#13-code-structure)
14. [Operations catalogue](#14-operations-catalogue)
15. [Providers catalogue](#15-providers-catalogue)

---

## 1. Architecture at a glance

Five questions structure the pipeline:

1. **What is the question?** Classify, clarify, decompose into sub-claims.
2. **What evidence exists?** Pick providers, write queries, fetch and extract passages.
3. **What does the evidence say about each claim?** Scrutinise, judge, surface uncertainties, investigate where doubts remain.
4. **Has the claim survived adversarial challenge?** Multi-angle verification, cross-domain convergence, IBE selection of best explanation.
5. **What's the integrated verdict?** Combine sub-claim verdicts; emit a calibrated posterior or suspend judgment.

A graph of 23 nodes routes the work. Each node reads explicit state, dispatches one or more operations, and returns one of its declared successors. Operations read entities, do work (LLM calls, computation), and write results back; the graph alone decides what runs next.

### High-level pipeline

```
question
   ↓
PrepareObjective  (classify · clarify · concept-analysis)
   ↓
Decompose         (question → sub-claims)
   ↓
PlanEvidence + ExtractEvidence  (provider tournament K=2)
   ↓
CreateClaims
   ↓
[scrutiny cycle]  Scrutinize ⇄ Investigate ⇄ ResolveUncertainties
   ↓
PromoteToSupported → ClusterEvidence → RunVerification
   ↓
[IBE chain]       EnumerateCandidates → ScoreLoveliness → ScoreLikeliness
                  → SelectBestExplanation → PromoteSupported
   ↓
CombineClaimVerdicts → CheckCompletion → CheckSynthesisDemand
   ↓                                          ↓
Synthesize                              SynthesizeInsufficient
```

Every successor a node can return is declared in its `run()` return type, enforced by both the type-checker and the graph runtime. The set of nodes and edges is also exposed as a Python value (`topology()` in `graph/topology.py`), which test suites use to assert reachability properties.

---

## 2. The seven primitives

Every persistent thing the system reasons over is one of seven entity types. They share a common base (`EpistemicEntity`) with id, timestamps, and provenance fields. All seven live in `epistemic/entities/`.

| Entity | Role | Key fields |
|---|---|---|
| **Objective** | The research question and what's been done about it. | `question`, `question_type`, `decomposition`, `artefact_id`, `snapshot_id`, `claim_to_verify` |
| **Evidence** | A piece of source content extracted from a provider. | `source_type`, `source_ref`, `extracted_content`, `quality_score`, `support_judgment`, `invalidated` |
| **Claim** | A scoped proposition under investigation. | `statement`, `scope`, `stage`, `scrutiny_verdict`, `integrated_assessment`, `integrated_confidence`, `adversarial_balance`, `convergence_verdict`, `cycle_capped`, `integration_candidates` |
| **Uncertainty** | An identified doubt, gap, or open question about a claim. | `description`, `uncertainty_type`, `blocking`, `resolved`, `resolution` |
| **Decision** | The "what would change my mind?" record at ACTIONABLE stage. | `criteria`, `actionable_threshold`, `what_would_change_mind` |
| **Snapshot** | An immutable freeze of the objective's state at synthesis time. | `artefact_id`, `frozen_at`, `claim_ids`, `evidence_ids` |
| **Artefact** | The synthesised report (markdown / HTML). | `artefact_type` ∈ {`summary`, `insufficient`}, `content`, `title` |

Every field on these entities represents something real: a verdict, a score, a stage. Nothing exists purely to signal scheduling.

The final report is reachable from `Objective` via two routes: directly through `Objective.artefact_id → Artefact.content`, and through the immutable freeze `Objective.snapshot_id → Snapshot.artefact_id → Artefact.content`. Both point at the same `Artefact`.

`Prediction` entities (in `entities/prediction.py`) record the ROBUST stage's falsifying commitments but are not part of the seven core primitives.

---

## 3. The graph (23 nodes)

The graph is a pydantic-graph DAG over typed nodes. Each node declares four contract fields:

- `reads` — state field names from `EpistemicGraphState` the node will read
- `writes` — state fields the node will mutate
- `operations` — operation classes the node dispatches
- `post_invariants` — predicates that must hold after the node runs

Successors are the return type of the node's `run()` method.

### Successor map

```
PrepareObjective         → Decompose
Decompose                → PlanEvidence
PlanEvidence             → ExtractEvidence
ExtractEvidence          → CreateClaims | Scrutinize
CreateClaims             → Scrutinize
Scrutinize               → AbandonOrDemote | Investigate | ResolveUncertainties
Investigate              → ExtractNewEvidence
ExtractNewEvidence       → Scrutinize
AbandonOrDemote          → PromoteToSupported | Scrutinize
ResolveUncertainties     → EnumerateCandidates | PromoteToSupported | ResolveUncertainties | Scrutinize
PromoteToSupported       → CheckCompletion | ClusterEvidence
ClusterEvidence          → RunVerification
RunVerification          → EnumerateCandidates | ResolveUncertainties
EnumerateCandidates      → ScoreLoveliness
ScoreLoveliness          → ScoreLikeliness
ScoreLikeliness          → SelectBestExplanation
SelectBestExplanation    → PromoteSupported
PromoteSupported         → CombineClaimVerdicts
CombineClaimVerdicts     → CheckCompletion
CheckCompletion          → CheckSynthesisDemand | SynthesizeInsufficient
CheckSynthesisDemand     → Scrutinize | Synthesize | SynthesizeInsufficient
Synthesize               → (terminal)
SynthesizeInsufficient   → (terminal)
```

### Node contracts (reads / writes / operations)

#### Setup

- **PrepareObjective**
  - reads: `objective_id`, `skip_preplanning`
  - writes: `question_type`
  - ops: `ClarifyQuestion`, `ClassifyQuestion`, `ConceptualAnalysis`
- **Decompose**
  - reads: `objective_id`
  - ops: `DecomposeQuestion`

#### Evidence gathering

- **PlanEvidence**
  - reads: `objective_id`
  - ops: `PlanTask`
- **ExtractEvidence**
  - reads: `claims_created`, `objective_id`
  - writes: `retrieval_failed`
  - ops: `ExtractEvidence`
- **CreateClaims**
  - reads: `objective_id`
  - writes: `claim_ids`, `claims_created`
  - ops: `MultiSeedClaim`, `ProposeClaims`, `SeedClaim`
- **ExtractNewEvidence**
  - reads: `objective_id`
  - writes: `retrieval_failed`
  - ops: `ExtractEvidence`

#### Inquiry cycle

- **Scrutinize**
  - reads: `claims_needing_rescrutiny`, `investigation_counts`, `objective_id`, `scrutiny_resolve_cycles`
  - writes: `claims_needing_rescrutiny`, `scrutiny_resolve_cycles`
  - ops: `ScrutiniseClaim`
- **Investigate**
  - reads: `investigation_counts`, `objective_id`
  - writes: `claims_needing_rescrutiny`, `claims_needing_tms`, `investigation_counts`
  - ops: `InvestigateClaim`
- **AbandonOrDemote**
  - reads: `investigation_counts`, `objective_id`
  - writes: `terminal_claims`, `verification_done`
  - ops: `AbandonStaleClaim`, `DemoteClaim`, `PromoteAsRefuted`, `SoftPromote`
- **ResolveUncertainties**
  - reads: `claims_needing_rescrutiny`, `objective_id`, `scrutiny_resolve_cycles`
  - writes: `claims_needing_rescrutiny`, `scrutiny_resolve_cycles`
  - ops: `DeduplicateConcerns`, `ResolveUncertainty`
- **PromoteToSupported**
  - reads: `objective_id`, `verification_done`
  - writes: `terminal_claims`
  - ops: `PromoteClaim`, `SetRoutingDefaults`

#### Verification

- **ClusterEvidence**
  - reads: `objective_id`
  - ops: (deterministic — convergence detector)
- **RunVerification**
  - reads: `claims_needing_rescrutiny`, `objective_id`, `question_type`
  - ops: `AdversarialSearch`, `AnalyzeArgument`, `AssessConvergence`, `ContrastiveEvaluation`, `CrossClaimConsistency`, `ValidateDeductively`, `VerifyComputationally`

#### IBE chain

- **EnumerateCandidates**
  - reads: `objective_id`
  - ops: `EnumerateCandidates`
- **ScoreLoveliness**
  - reads: `objective_id`
  - ops: `ScoreLoveliness`
- **ScoreLikeliness**
  - reads: `objective_id`
  - ops: `ScoreLikeliness`
- **SelectBestExplanation**
  - reads: `objective_id`
  - ops: `SelectBestExplanation`
- **PromoteSupported**
  - reads: `objective_id`
  - writes: `verification_done`
  - ops: `GeneratePrediction`, `PromoteClaim`, `RecordDecision`

#### Combination & demand

- **CombineClaimVerdicts**
  - reads: `objective_id`
  - ops: (deterministic — combiner in `graph/combination.py`)
- **CheckCompletion**
  - reads: `errors`, `failed`, `objective_id`, `operations_log`, `quarantined`, `retrieval_failed`, `successful`
  - writes: `synthesis_insufficient_reason`
  - ops: (deterministic gates)
- **CheckSynthesisDemand**
  - reads: `claims_needing_rescrutiny`, `objective_id`, `scrutiny_resolve_cycles`
  - writes: `claims_needing_rescrutiny`
  - ops: (deterministic gates plus Demand LLM agent)

#### Terminals

- **Synthesize**
  - reads: same as `CheckCompletion`
  - ops: `FreezeSnapshot`, `SynthesizeReport` (writer ⇄ validator loop)
- **SynthesizeInsufficient**
  - reads: same as `Synthesize` plus `synthesis_insufficient_reason`
  - ops: `FreezeSnapshot`, `SynthesizeInsufficientReport`

### Loops

Three loops are visible in the topology, all bounded by `PEIRCE_CYCLE_CAP = 3`:

- **Investigation cycle:** `Scrutinize → Investigate → ExtractNewEvidence → Scrutinize`. Bounded by `investigation_counts[claim_id]`.
- **Scrutiny–resolve cycle:** `Scrutinize → ResolveUncertainties → Scrutinize`. Bounded by `scrutiny_resolve_cycles[claim_id]`.
- **Demand loop:** `CheckSynthesisDemand → Scrutinize` when synthesis isn't satisfied. Bounded by the same `scrutiny_resolve_cycles` counter.

A claim that hits the cap on any of these is marked `cycle_capped=True`. The combiner and posterior calculator treat capped claims with reduced weight rather than discarding them.

---

## 4. The claim lifecycle

Claims advance through five stages. Promotion requires passing a deterministic gate (defined in `gates.py:STAGE_GATES`). Gates are routing-aware: only verification tracks marked PRIMARY or SECONDARY for the question type are required.

| Stage | Requirements |
|---|---|
| **HYPOTHESIS** | Just proposed. No evidence required. |
| **SUPPORTED** | ≥ 1 supporting evidence + uncertainties listed + scrutiny passed. |
| **PROVISIONAL** | ≥ 2 evidence + scrutiny passed + contested-OK (adversarial balance not REFUTED). |
| **ROBUST** | Convergence ≥ STRONG + adversarial SURVIVED + predictions made. |
| **ACTIONABLE** | Decision criteria met + "what would change my mind" recorded. |

Each gate is the conjunction of:

1. **Counted preconditions**: evidence count, source diversity, uncertainty count. Pure data lookups.
2. **Track requirements**: verification tracks the question type marks PRIMARY for this stage. SECONDARY tracks contribute when present but don't gate; SKIP tracks are not required.
3. **Quality breakpoints**: adversarial balance band, convergence verdict, etc.

Routing-awareness means a normative claim is not refused promotion because its computational-verification track is empty; that track is SKIP for normative questions.

### Stage demotion

If new evidence invalidates a key supporting source, or adversarial balance drops below `ADVERSARIAL_REFUTED_THRESHOLD`, or contradicting evidence outweighs supporting evidence, the claim is demoted (Truth Maintenance System cascade). Demotion resets verification flags so the claim can be re-investigated, but persistent results like `adversarial_balance` and `integration_candidates` are preserved.

---

## 5. The inquiry cycle (Peirce)

Peirce's framework: inquiry is bounded. At some point the system declares "we have circled this enough" and accepts the current state, even when it isn't fully resolved. Three loops in the graph encode this commitment, all sharing the cap `PEIRCE_CYCLE_CAP = 3`.

### Why one cap, three loops?

All three are conceptually the same Peircean "fix belief in bounded inquiry" commitment. Tuning the cap is one decision, not three. Operational caps that aren't Peirce-grounded — like `MAX_VALIDATION_ROUNDS = 3` for the writer-validator loop in synthesis — keep their own names and homes.

### Cycle-capping

A capped claim:

- Has `cycle_capped=True` set on the entity.
- Skips further scrutiny rounds; the demand loop in `CheckSynthesisDemand` will not add it back.
- Is excluded from IBE certification.
- Contributes to the aggregated posterior with reduced weight (`CYCLE_CAP_CONFIDENCE_PENALTY = 0.7` as a pull-toward-neutral on the counting fallback path; multiplicative on confidence in the integration path).
- Surfaces in the synthesis report under "limitations" with its capped status named.

Cycle-capping is the system's signal that an inquiry has reached its bounded limit. The architecture trades occasional honest suspension on a hard claim for never confidently ratifying a claim that repeatedly failed scrutiny.

---

## 6. Adversarial check (Popper / Lakatos)

For each claim that has reached SUPPORTED stage, the system actively searches for counter-evidence. The result is `adversarial_balance` ∈ [0, 1] — the fraction of the claim's effective evidence that survives after counter-evidence is weighed in.

### Pipeline

1. Generate adversarial queries from the claim ("evidence against X", "X refuted", "limitations of X").
2. Search the same providers used for the original evidence so the adversarial source pool isn't biased toward sources that already supported the claim.
3. Extract counter-claims from the adversarial sources.
4. The judgement agent assesses each counter-claim impartially, on what the source actually says rather than the desired direction.
5. Compute `adversarial_balance` as `weighted_supports / (weighted_supports + weighted_contradicts)`.

### Soft tri-state

| Band | Range | Reading |
|---|---|---|
| **REFUTED** | balance < 0.3 | Popper-falsified — claim must demote/abandon |
| **CONTESTED** | 0.3 ≤ balance < 0.7 | Lakatosian middle — cannot promote past PROVISIONAL |
| **SURVIVED** | balance ≥ 0.7 | Popperian corroboration — required for ROBUST/ACTIONABLE |
| **SUSPICIOUS** (diagnostic) | balance ≥ 0.95 | Search itself was insufficient — flag for review, not a decision |

The thresholds are symmetric around 0.5 with ±0.2 distance, giving a contested band of width 0.4. The width is wide because small differences around 0.5 don't license decisive directional commitments. `ADVERSARIAL_SUSPICIOUS_THRESHOLD = 0.95` is a meta-diagnostic, not a decision threshold: balances above 0.95 suggest the adversarial search itself was insufficient.

---

## 7. The IBE chain (Lipton)

For each SUPPORTED claim that reaches verification, Lipton's inference-to-best-explanation runs as a five-node chain. The chain produces `integrated_assessment` ∈ {`supports`, `contradicts`, `insufficient`} and `integrated_confidence` ∈ [0, 1]. These feed the posterior calculator's directional verdict.

```
EnumerateCandidates → ScoreLoveliness → ScoreLikeliness → SelectBestExplanation → PromoteSupported
```

### What each node does

1. **EnumerateCandidates** produces 4–6 candidate explanations of the evidence with balanced supports and contradicts framings. Balanced enumeration ensures the loveliness scorer compares real alternatives rather than variations of one framing.
2. **ScoreLoveliness** scores each candidate's theoretical virtue: simplicity, scope, depth, unification, internal coherence, fit with background knowledge. *Loveliness* is Lipton's term for "how much we'd want this explanation to be true if it explained the data". Range: [0, 1].
3. **ScoreLikeliness** estimates `P(observed evidence | candidate is true)`. Range: [0, 1].
4. **SelectBestExplanation** picks the candidate with highest `loveliness × likeliness`, then applies the framing-tie cap on `integrated_confidence`.
5. **PromoteSupported** commits the verdict to `integrated_assessment`; if the chosen verdict crosses ROBUST gate requirements, generates the falsifying prediction.

### The framing-tie cap

When the chosen candidate's loveliness only narrowly exceeds the best opposing candidate's, the abductive chain has no principled tie-breaker; both stories are coherent. The cap dampens `integrated_confidence` in proportion to the gap:

```
cap = 0.5 + (loveliness_gap / FRAMING_TIE_SATURATION_GAP) × 0.5
```

where `FRAMING_TIE_SATURATION_GAP = 0.4` mirrors the width of the adversarial CONTESTED band. Behaviour:

- Gap ≥ 0.4: cap = 1.0 (no dampening)
- Gap = 0.0: cap = 0.5 (severe dampening, perfect tie)
- Linear in between

The cap is smooth: it dampens confidence proportionally rather than gating verdicts at a discrete threshold. Counting evidence still contributes to the posterior even when the framing-tie cap engages.

### K-agreement check

The IBE chain is stochastic: small noise in the loveliness or likeliness scores can flip the argmax in `SelectBestExplanation`, so two runs of the chain on the same claim may commit to opposite verdicts. The framing-tie cap dampens confidence, but it can't dampen direction — argmax is discrete.

The K-agreement check addresses this. After the chain has run once, `SelectBestExplanationOperation` runs the full chain (Enumerate → Loveliness → Likeliness → Select) `K − 1` additional times in memory. The K independent verdicts are then aggregated:

- If all K runs agree on canonical direction (`supports` / `contradicts` / `insufficient`), the verdict is committed; `integrated_confidence` is set to the minimum across the K runs.
- If the runs disagree, `integrated_assessment` falls back to `insufficient` and `integrated_confidence` to 0.5 — independent reasoning passes failed to converge, so a single-run argmax is not certified.

K is tunable per run via the `ibe_agreement_k` parameter on `run_epistemic_graph`. The default is `IBE_AGREEMENT_K_DEFAULT = 2` — the minimum sample size at which "agreement" carries information, paralleling the K=2 commitment in the provider tournament. Higher K trades LLM cost for stricter agreement; setting K=1 disables the check (legacy single-run behaviour).

K=2 doubles the IBE chain's LLM-call count for any claim that reaches verification. In typical runs this is roughly +10–15% of total LLM cost per case. K=20 is permitted; the cost scales linearly.

---

## 8. Convergence (Reichenbach)

Reichenbach's common-cause principle: agreement across causally-independent sources is evidence the agreement isn't an artefact of any one source's biases. Cross-domain convergence is detected as part of verification.

### Pipeline

The convergence detector (`convergence_detector.py`) runs in `ClusterEvidence` and `RunVerification`:

1. Cluster evidence by content embedding (HDBSCAN; cosine threshold 0.7).
2. Compute inter-cluster distance — methodological similarity measure.
3. Score independence within each cluster (fraction of pairs that are methodologically independent).
4. Compute strength from cluster count, inter-domain distance, and representative quality.
5. Map strength to verdict: NO_EVIDENCE / SINGLE_DOMAIN / PARTIAL / CONVERGENT.

The CONVERGENT label triggers the **fast-path-to-IBE** in `RunVerification`: a SUPPORTED claim with at least one CONVERGENT sibling skips `ResolveUncertainties` and goes straight to `EnumerateCandidates`. Converged claims have already cleared the bar that uncertainty-resolution exists to clear.

### Convergence thresholds

| Constant | Value | Role |
|---|---|---|
| `CONVERGENCE_STRONG_THRESHOLD` | 0.7 | Strength score at or above this maps to CONVERGENT |
| `CONVERGENCE_INTRA_DIVERSITY_THRESHOLD` | 0.5 | Min fraction of independent within-cluster pairs |
| `CONVERGENCE_INTER_DOMAIN_DISTANCE_LOW` | 0.3 | Below: clusters too similar — `shared_error_modes` warning |

---

## 9. Lazy escalation

Each layer asks for what it's missing; breadth comes from demand. Planning, investigation, scrutiny, and synthesis don't pre-commit to a wide search. When a layer detects insufficient evidence, it emits a `Demand`, and the graph routes that demand to the layer best placed to satisfy it minimally.

### The Demand object

Three flat fields, chosen for small-LLM compatibility (in `epistemic/demand.py`):

```python
class Demand(BaseModel):
    needs_more: bool
    justification: str
    target_hint: str  # may be empty
```

The demand is not layer-specific. One uniform shape travels everywhere; each consumer interprets the freeform fields in its own context.

### Iterative provider tournament

Round 1 of research-mode investigation uses a provider tournament at the objective level:

- `epistemic_rank_providers` is called K = `RESEARCH_MODE_PROVIDER_K` = 2 times with a shrinking candidate pool to pick K distinct providers.
- Every sub-claim then queries all K. K=2 is the minimum that lets the cross-domain convergence detector fire per claim.
- Round 2+ escalates with the next-best unused provider per sub-claim only when scrutiny demands more (the `state.providers_used_per_sub` set tracks what's been queried).

### Synthesis demand loop

`CheckSynthesisDemand` runs cheap deterministic gates first, and a Demand-emitting LLM agent only when the gates are inconclusive:

1. **Gate 1: open-research check** — is the question entirely unresolved?
2. **Gate 2: no-combined-verdict check** — does the combiner have nothing to combine?
3. **Gate 3: stranded-claims check** — are there non-terminal claims with eligible cycle budget remaining?
4. **Gate 4: decisive-posterior check** — is the posterior already at or beyond `POSTERIOR_DECISIVE_THRESHOLD = 0.85`?
5. **LLM judgment (fallback)** — emit a `Demand`; if `needs_more=True`, route eligible claims back to `Scrutinize`; otherwise proceed to `Synthesize`.

The termination guarantee is the per-sub-claim cap (`scrutiny_resolve_cycles[claim_id] < PEIRCE_CYCLE_CAP`), not a global give-up budget. The loop terminates as soon as no claim can make progress, even if the satisfaction LLM keeps saying `needs_more`.

---

## 10. Confidence scoring

The aggregated posterior is computed by `compute_posterior` in `confidence.py`. It blends two signals via weighted model averaging:

- **Counting posterior** — Bayesian count of effective supporting vs contradicting evidence per claim, log-odds composed across claims, normalised.
- **Integration posterior** — derived from the IBE chain's `integrated_assessment` weighted by `integrated_confidence`.

The blending weight reflects how much the IBE chain certified. When all claims have an integrated assessment, the integration signal dominates. When none do, the counting signal carries the verdict. When some do and some don't, the blend interpolates.

### Posterior penalties

When a process flag fires (the inquiry didn't converge cleanly), the verdict it produced is provisional. The penalty surfaces that signal with reduced weight, rather than discarding it.

| Flag | Constant | Value | Mechanism |
|---|---|---|---|
| `cycle_capped=True` on any contributing claim | `CYCLE_CAP_CONFIDENCE_PENALTY` | 0.7 | Multiplier on confidence (integration path) or pull-toward-neutral on posterior (counting path) |
| `state.retrieval_failed=True` | `RETRIEVAL_FAILED_CONFIDENCE_PENALTY` | 0.7 | Same shape, same value. Stacks multiplicatively |

A doubly-provisional verdict gets `0.7 × 0.7 = 0.49` of its full distance from 0.5.

### Verdict labelling

| Posterior | Label |
|---|---|
| ≥ 0.66 (`POSTERIOR_DIRECTIONAL_BREAKPOINT`) | **supports** |
| ≤ 0.34 (1 − breakpoint) | **contradicts** |
| 0.34 < p < 0.66 | **insufficient** |

The directional breakpoint is wider than 0.5 to keep the label calibrated with the underlying evidence strength. `POSTERIOR_DECISIVE_THRESHOLD = 0.85` is a separate, stricter threshold used only by `CheckSynthesisDemand` Gate 4 to decide whether more inquiry could change the headline.

---

## 11. Question-type routing

Not every verification track applies to every question. The system classifies the question into one of seven types in `PrepareObjective`, and the type dictates which verification tracks are PRIMARY (gating), SECONDARY (contributing), *If applicable*, or SKIP (irrelevant).

### Seven question types

| Type | Example | Decomposes? |
|---|---|---|
| **Empirical** | Is APOE4 associated with Alzheimer's risk? | Yes |
| **Causal** | Does smoking cause lung cancer? | Yes |
| **Comparative** | Is drug A more effective than drug B? | Yes |
| **Definitional** | What is a metalloproteinase? | No (single-claim) |
| **Predictive** | Will treatment X reduce mortality? | Yes |
| **Methodological** | Is the proposed RCT design adequate? | Yes |
| **Verificatory** | Does paper P support claim C? | No (seed claim is the question) |

### Routing matrix

P = Primary (gates promotion), S = Secondary (contributes when present), A = If applicable, — = Skipped.

| Track | Empirical | Causal | Comparative | Definitional | Predictive | Methodological | Verificatory |
|---|---|---|---|---|---|---|---|
| Adversarial | P | P | P | S | P | P | P |
| Convergence | P | P | P | S | S | S | P |
| Deductive | — | S | — | P | — | P | — |
| Computational | A | A | A | — | P | A | — |
| Contrastive | S | P | P | — | S | S | S |
| Cross-claim consistency | S | S | S | S | S | S | — |
| Argument analysis | S | S | S | P | S | P | S |

The classification is set once in `PrepareObjective` and stored on `Objective.question_type`. `SetRoutingDefaultsOperation`, run when a claim crosses into SUPPORTED, reads the type and pre-marks SKIP tracks as not-applicable on the claim itself, so downstream verifiers don't waste calls.

### Verificatory mode

When `Objective.claim_to_verify` is set, decomposition is skipped. `CreateClaims` seeds the single specified claim and the rest of the pipeline runs normally. A Pydantic `model_validator` on `Objective` refuses constructions that set both `claim_to_verify` and `decomposition`; they are mutually exclusive seed modes.

---

## 12. Threshold reference

Every decision-relevant numeric threshold lives in `epistemic/thresholds.py`. Each constant is named after the philosophical commitment it encodes; every site across the codebase imports from this single file.

### Adversarial balance (Popper / Lakatos)

| Constant | Value | Read by |
|---|---|---|
| `ADVERSARIAL_REFUTED_THRESHOLD` | 0.3 | stage demotion · posterior calculation · reporters |
| `ADVERSARIAL_SURVIVED_THRESHOLD` | 0.7 | stage gates · refire-skip logic · synthesis writer · reporters |
| `ADVERSARIAL_SUSPICIOUS_THRESHOLD` | 0.95 | balance interpreter (diagnostic) |
| `FRAMING_TIE_SATURATION_GAP` | 0.4 (derived) | `integration._framing_tie_cap` |

### Inquiry cycling (Peirce)

| Constant | Value | Read by |
|---|---|---|
| `PEIRCE_CYCLE_CAP` | 3 | investigation · scrutiny–resolve · uncertainty depth |
| `IBE_AGREEMENT_K_DEFAULT` | 2 | default for `EpistemicGraphState.ibe_agreement_k` — number of independent IBE chain runs whose verdicts must agree |

### Output-layer provenance

| Constant | Value | Read by |
|---|---|---|
| `CYCLE_CAP_CONFIDENCE_PENALTY` | 0.7 | posterior aggregation when any claim is cycle-capped |
| `RETRIEVAL_FAILED_CONFIDENCE_PENALTY` | 0.7 | posterior aggregation when retrieval failed |

### Posterior breakpoints

| Constant | Value | Read by |
|---|---|---|
| `POSTERIOR_DIRECTIONAL_BREAKPOINT` | 0.66 | `combination._verdict_label` |
| `POSTERIOR_DECISIVE_THRESHOLD` | 0.85 | `CheckSynthesisDemand` Gate 4 |

### Convergence (Reichenbach / Mill)

| Constant | Value | Read by |
|---|---|---|
| `CONVERGENCE_STRONG_THRESHOLD` | 0.7 | convergence verdict — gates IBE fast-path |
| `CONVERGENCE_INTRA_DIVERSITY_THRESHOLD` | 0.5 | cluster diversity check |
| `CONVERGENCE_INTER_DOMAIN_DISTANCE_LOW` | 0.3 | `shared_error_modes` warning |

The constants are deliberately few. Adding a new threshold requires the same justification (theoretical basis plus the sites that read it) as the existing set. Bare numeric literals in decision logic across the rest of `epistemic/` are an architectural smell.

---

## 13. Code structure

Top-level layout under `src/andamentum/epistemic/`:

| Path | Role |
|---|---|
| `thresholds.py` | Single source of truth for every decision-relevant numeric threshold. |
| `entities/` | The seven primitives plus `Prediction` and decomposition types. Pure pydantic data classes. |
| `operations/` | 17 operation classes (`BaseOperation` subclasses). Pure transforms. |
| `graph/` | The 23-node DAG. `nodes.py` · `state.py` · `topology.py` · `base.py`. |
| `gates.py` | Stage promotion gates. `STAGE_GATES` dict plus `validate_promotion` (routing-aware). |
| `providers/` | 10 evidence providers plus `CONTRIBUTING.md`. Each returns `list[GatheredEvidence]`; never raises. |
| `convergence_detector.py` | Reichenbach common-cause cluster analysis. |
| `confidence.py` | Posterior aggregation; counting plus integration via weighted model averaging. |
| `demand.py` | The `Demand` Pydantic model used everywhere lazy-escalation happens. |
| `repository.py` | `EpistemicRepository` wraps `StorageBackend`; in-memory backend in `storage.py`. |
| `runner.py` | `DefaultAgentRunner` wrapping `core.AgentRunner` with the epistemic agent registry. |
| `cli.py` | `andamentum-epistemic` entry point. Two modes: `ask` and `verify`. |
| `tests/` | Tests live next to the code they test. `test_topology.py` asserts reachability properties. |

### Entry point

```python
from andamentum.epistemic.graph import run_epistemic_graph

result = await run_epistemic_graph(
    question="...",
    mode="research",  # or "verify"
    model="anthropic:claude-haiku-4-5",
    embedding_model="ollama:embeddinggemma:latest",
    ibe_agreement_k=2,  # default; raise for stricter agreement
)
```

The CLI `andamentum-epistemic ask "<question>"` wraps this. `verify "<claim>"` sets `Objective.claim_to_verify` and skips decomposition. Both resolve `--model` from the argument or `$ANDAMENTUM_MAIN_LLM_MODEL`, and exit with an error if neither is set.

---

## 14. Operations catalogue

One module per family in `operations/`. Total: 17 modules.

| Module | Operations |
|---|---|
| `preplanning` | `ClarifyQuestionOperation`, `ClassifyQuestionOperation`, `ConceptualAnalysisOperation`, `DecomposeQuestionOperation`, `PlanTaskOperation` |
| `seed_claim`, `multi_seed_claim`, `claims` | `SeedClaimOperation`, `MultiSeedClaimOperation`, `ProposeClaimsOperation`, plus `top_n_representatives` (claim-relevance rerank, async) |
| `evidence` | `ExtractEvidenceOperation` (provider dispatch plus passage extraction plus relevance rerank) |
| `scrutiny`, `investigation` | `ScrutiniseClaimOperation`, `InvestigateClaimOperation` |
| `uncertainty`, `concerns` | `ResolveUncertaintyOperation`, `DeduplicateConcernsOperation` |
| `verification` | `AdversarialSearchOperation`, `AnalyzeArgumentOperation`, `AssessConvergenceOperation`, `ContrastiveEvaluationOperation`, `CrossClaimConsistencyOperation`, `ValidateDeductivelyOperation`, `VerifyComputationallyOperation` |
| `integration` | `EnumerateCandidatesOperation`, `ScoreLovelinessOperation`, `ScoreLikelinessOperation`, `SelectBestExplanationOperation` (plus framing-tie cap helper) |
| `stage_management` | `PromoteClaimOperation`, `DemoteClaimOperation`, `PromoteAsRefutedOperation`, `SoftPromoteOperation`, `AbandonStaleClaimOperation`, `SetRoutingDefaultsOperation`, `GeneratePredictionOperation`, `RecordDecisionOperation` |
| `synthesis` | `FreezeSnapshotOperation`, `SynthesizeReportOperation` (writer ⇄ validator), `SynthesizeInsufficientReportOperation` |
| `belief_maintenance` | TMS cascade after evidence invalidation |
| `analysis`, `identifier_extraction`, `cleanup` | Helpers used during scrutiny and investigation |

Every operation inherits from `BaseOperation` and conforms to the same shape: takes an `OperationInput`, does work, returns `OperationResult`.

---

## 15. Providers catalogue

Providers retrieve and structure evidence. They never assess quality (`quality_score=None` always), never truncate content, and return `list[GatheredEvidence]` (empty list on error, never raise). The full specification is in `providers/CONTRIBUTING.md`.

| Provider | Source | Domain |
|---|---|---|
| `arxiv` | arXiv preprints | physics, CS, math, quantitative biology |
| `biorxiv` | bioRxiv preprints | biology, medicine |
| `chembl` | ChEMBL | compound bioactivity |
| `clinicaltrials` | ClinicalTrials.gov | clinical trials |
| `cochrane` | Cochrane Library | systematic reviews |
| `europepmc` | Europe PMC | life sciences literature |
| `monarch` | Monarch Initiative | genotype–phenotype |
| `open_targets` | Open Targets | drug-target evidence |
| `openalex` | OpenAlex | cross-disciplinary scholarly |
| `pubmed` | PubMed / NCBI | biomedical literature |

Total: 10 providers. The iterative tournament selects K=2 distinct providers per sub-claim by default, escalating up to all 10 only when scrutiny demands more.
