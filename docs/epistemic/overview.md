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
PlanEvidence + ExtractEvidence  (description-driven dispatch — one agent call per provider, commit or abstain)
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
| **Evidence** | A piece of source content extracted from a provider. | `source_type`, `source_ref`, `extracted_content`, `quality_score`, `support_judgment`, `judgment_distribution` (verbalized 3-way belief distribution behind the verdict; `judgment_confidence` / `judgment_entropy` / `judgment_one_hot` are computed from it), `invalidated` |
| **Claim** | A scoped proposition under investigation. | `statement`, `scope`, `stage`, `scrutiny_verdict`, `integrated_assessment`, `integrated_confidence`, `adversarial_balance`, `convergence_verdict`, `cycle_capped`, `cap_evaluation`, `integration_candidates` |
| **Uncertainty** | An identified doubt, gap, or open question about a claim. | `description`, `uncertainty_type`, `blocking`, `resolved`, `resolution` |
| **Decision** | The "what would change my mind?" record at ACTIONABLE stage. | `criteria`, `actionable_threshold`, `what_would_change_mind` |
| **Snapshot** | An immutable freeze of the objective's state at synthesis time. | `artefact_id`, `frozen_at`, `claim_ids`, `evidence_ids` |
| **Artefact** | The synthesised report (markdown / HTML). | `artefact_type` (free-form; the pipeline stamps `summary` or `insufficient`), `content`, `title` |

Every field on these entities represents something real: a verdict, a score, a stage. Nothing exists purely to signal scheduling.

The final report is reachable from `Objective` via two routes: directly through `Objective.artefact_id → Artefact.content`, and through the immutable freeze `Objective.snapshot_id → Snapshot.artefact_id → Artefact.content`. Both point at the same `Artefact`.

`Prediction` entities (in `entities/prediction.py`) record the ROBUST stage's falsifying commitments, and `IntentRecord` entities (in `entities/intent_record.py`) record investigation intents; neither is part of the seven core primitives.

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
PromoteToSupported       → ClusterEvidence | CombineClaimVerdicts
ClusterEvidence          → RunVerification
RunVerification          → EnumerateCandidates | ResolveUncertainties
EnumerateCandidates      → ScoreLoveliness
ScoreLoveliness          → ScoreLikeliness
ScoreLikeliness          → SelectBestExplanation
SelectBestExplanation    → PromoteSupported
PromoteSupported         → CombineClaimVerdicts
CombineClaimVerdicts     → CheckCompletion   (its SOLE predecessor)
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
  - ops: `ScrutiniseClaim`, `ConcludeAtCycleCap`
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
  - ops: `DeduplicateConcerns`, `ResolveUncertainty`, `ConcludeAtCycleCap`
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
  - reads: `ibe_agreement_k`, `objective_id`
  - ops: `SelectBestExplanation`
- **PromoteSupported**
  - reads: `objective_id`
  - writes: `verification_done`
  - ops: `GeneratePrediction`, `PromoteClaim`, `RecordDecision`

#### Combination & demand

- **CombineClaimVerdicts**
  - reads: `objective_id`, `retrieval_failed`
  - writes: `verdict` — the run's single `EpistemicVerdict`, and its SOLE writer
  - ops: (deterministic — `confidence.build_verdict`, which wraps
    `compute_posterior` and the combiner in `graph/combination.py`)
- **CheckCompletion**
  - reads: `errors`, `failed`, `objective_id`, `operations_log`, `quarantined`, `retrieval_failed`, `successful`, `verdict`
  - writes: `synthesis_insufficient_reason`
  - ops: (deterministic gates; a pure reader over `state.verdict.provenance` —
    raises when the verdict is absent, since that means a routing bug)
- **CheckSynthesisDemand**
  - reads: `claims_needing_rescrutiny`, `objective_id`, `scrutiny_resolve_cycles`, `verdict`
  - writes: `claims_needing_rescrutiny`
  - ops: (deterministic gates plus Demand LLM agent)

#### Terminals

- **Synthesize**
  - reads: same as `CheckCompletion`
  - ops: `FreezeSnapshot` (stamps `state.verdict` onto the Snapshot via the
    typed `OperationInput.verdict` channel), `SynthesizeReport`
    (writer ⇄ validator loop over the prose only)
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
2. **Track requirements**: verification tracks the question type marks PRIMARY or SECONDARY for this stage — both gate promotion when the stage's gate names them. Only SKIP and IF_APPLICABLE tracks are never required.
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
- Is excluded from the **IBE chain** — on LLM-cost grounds, not because its verdict would be discarded. Four abductive-deliberation calls are not warranted for a claim whose inquiry did not converge.
- **Receives a deterministic terminal verdict at the cap instead.** `ConcludeAtCycleCapOperation` (no LLM) re-reads the claim's eligible representative evidence through the same `compute_evidence_weight` engine scrutiny uses. When that reading is `strong` or `moderate` it writes `integrated_assessment` / `integrated_confidence` / `integrated_reasoning`; when it is `conflicting` or `weak` it writes no verdict. Either way the reading is recorded as a typed `Claim.cap_evaluation`, so a capped claim with no verdict is the positive record that the *evidence* did not decide — rather than a scheduling side effect wearing the word "insufficient". The verdict a reader sees is rendered from the run's `EpistemicVerdict` (§10), which carries the `CapEvaluation` on its `ClaimVerdict` entry, so no renderer has to reach back to the entity to explain it.
- The cap-derived confidence is `purity × mass_adequacy`, ceilinged at `MECHANICAL_VERDICT_MAX_CONFIDENCE = 0.9`. `purity` is the normalised directional margin `(dominant − minority) / (dominant + minority)`; `mass_adequacy` is `min(1, dominant / SCRUTINY_STRONG_MASS_THRESHOLD)`. The second factor exists because purity alone is 1.0 for *any* one-sided pool regardless of size, so a single unopposed source would report the system's maximum mechanical confidence — more than a 5-vs-1 pool. No new constant: the saturation point is the mass the system already calls "strong".
- Contributes to the aggregated posterior with reduced weight (`CYCLE_CAP_CONFIDENCE_PENALTY = 0.7` as a pull-toward-neutral on the counting fallback path; multiplicative on confidence in the integration path). With the cap-time verdict this is now a reachable path rather than a vacuous one. The two ceilings compose: a cap-derived single-claim posterior sits at most `0.9 × 0.7 / 2 = 0.315` from neutral, so it can never read as decisive.
- Carries the doubt of the judgments it stands on. `compute_evidence_weight` also reports `EvidenceWeight.doubt` (§ Gate 0), and the cap records it as `CapEvaluation.doubt`. This is the one site that mints a verdict with no LLM deliberation at all, so the uncertainty of the per-evidence judgments underneath is the only uncertainty it has; discarding it would contradict the premise that a judgment carries its doubt. `None` when no distribution-carrying item contributed directional mass — unmeasured, deliberately distinct from "measured as 0.0". It is a measurement only: a capped claim is not loop-back eligible, so it never escalates, and it is deliberately not written to `Claim.scrutiny_doubt`, which has exactly one writer.
- Owns only the verdict it wrote. `CapEvaluation.wrote_live_verdict` records whether *this* cap firing authored the live `integrated_assessment`; combined with a `stage == HYPOTHESIS` check (the cap never promotes), it is what lets a repeated firing revise or clear its own verdict while leaving an IBE- or `PromoteAsRefuted`-certified one untouched.
- Does **not** change stage, and is **never demoted**. `AbandonOrDemote` skips cycle-capped claims in its demote branch: `record_demotion` resets the whole Peirce-cycling state — which the cap exists to stop — and nulls `integrated_assessment`, which for a capped claim is the terminal verdict. The reachable sequence it closes is cap → `AbandonOrDemote` refute/soft-promotes to SUPPORTED → the next `Scrutinize` visit routes the failed-scrutiny SUPPORTED claim back → demotion erases the verdict.
- Surfaces in the synthesis report under "limitations" with its capped status named.

Cycle-capping is the system's signal that an inquiry has reached its bounded limit. The architecture trades occasional honest suspension on a hard claim for never confidently ratifying a claim that repeatedly failed scrutiny. What it must *not* trade away is signal it already has: a cap stops further LLM work, it does not zero out the evidence already gathered.

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

K is tunable per run via the `ibe_agreement_k` parameter on `run_epistemic_graph`. The default is `IBE_AGREEMENT_K_DEFAULT = 2` — the minimum sample size at which "agreement" carries information. Higher K trades LLM cost for stricter agreement; setting K=1 disables the check (legacy single-run behaviour).

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

### Description-driven dispatch

Evidence gathering runs through a single per-provider dispatch agent (`epistemic_dispatch_provider`):

- Each provider class self-describes via class attributes: `description` (scope and strengths/weaknesses), `query_guidance` (native query syntax catalogue), `query_examples` (representative claim → native-query pairs, with `None`-query entries teaching abstention), `output_kind`, plus `independence_group` and `provider_contract_version` (declared on the class; `independence_group` is not yet consumed by any pipeline stage).
- For each claim (or sub-claim, or follow-up intent), the dispatch agent is invoked once per provider in parallel. It either commits one or two native-syntax queries or abstains for that provider.
- Adding a new provider is a class-attribute task — no agent design, no taxonomy update.
- Investigation rounds use the same dispatch path: the `epistemic_investigate_claim` agent generates a methodological *angle* (intent), and the dispatch agent receives the *claim* as the subject and the *intent* as the angle modifier. Queries combine both, so retrieved evidence stays grounded in the claim's subject matter while exploring the intent's dimension.

### Synthesis demand loop

`CheckSynthesisDemand` runs cheap deterministic gates first, and a Demand-emitting LLM agent only when the gates are inconclusive:

0. **Gate 0: aggregate judgment doubt** — were the evidence judgments behind a claim's verdict internally unsure?
1. **Gate 1: open-research check** — is the question entirely unresolved?
2. **Gate 2: no-combined-verdict check** — does the combiner have nothing to combine?
3. **Gate 3: stranded-claims check** — are there non-terminal claims with eligible cycle budget remaining?
4. **Gate 4: decisive-posterior check** — is the posterior already at or beyond `POSTERIOR_DECISIVE_THRESHOLD = 0.85`?
5. **LLM judgment (fallback)** — emit a `Demand`; if `needs_more=True`, route eligible claims back to `Scrutinize`; otherwise proceed to `Synthesize`.

The termination guarantee is the per-sub-claim cap (`scrutiny_resolve_cycles[claim_id] < PEIRCE_CYCLE_CAP`), not a global give-up budget. The loop terminates as soon as no claim can make progress, even if the satisfaction LLM keeps saying `needs_more`.

### Gate 0 — doubt drives escalation

The pipeline makes many sampled LLM judgments per claim and feeds them to threshold-based gates, so small judgment noise becomes categorical output flips. Gate 0 is the one gate that asks about the **quality of the inputs** rather than about the conclusion.

- `scrutiny_weight.compute_evidence_weight` now reports `doubt` alongside the directional masses: the **directional-mass-weighted mean normalised entropy** of the judgments that produced them. Each item weighs by exactly the factor with which it moved the verdict. Items with no captured `judgment_distribution` (adversarial counter-evidence, pre-Tier-0 data) contribute mass but **not** doubt — absence of a measured confidence is not evidence of low confidence, the same exemption `GATE_MIN_JUDGMENT_CONFIDENCE` already encodes. When nothing was measured the value is `None`, never `0.0`; only `None` can never escalate.
- `ScrutiniseClaimOperation` persists it as `Claim.scrutiny_doubt` — a measurement recorded beside the verdict, not a scheduling signal (P3). The operation writes and returns; the graph decides whether anything follows.
- Gate 0 fires when the escalation has not already fired this run (`state.judgment_doubt_escalated`) **and** at least one *loop-back-eligible* claim carries a measured doubt at or above `JUDGMENT_DOUBT_ESCALATION_THRESHOLD`. On firing it emits a `Demand` and routes through the existing `_maybe_loop_back`.

Three properties fall out of that construction rather than out of convention:

- **It cannot manufacture a false "insufficient".** `_maybe_loop_back` routes to `SynthesizeInsufficient` precisely when its eligible set is empty; Gate 0 and `_maybe_loop_back` consult the same module-level `_loopback_eligible` predicate, so the gate only fires when that set is non-empty.
- **It terminates.** The escalation fires at most once per objective, and the loop it triggers is the existing one, already capped per claim by `SCRUTINY_RESOLVE_CYCLE_CAP`. Doubt buys exactly one extra round of inquiry — it is a suspicion-raiser, never a decision.
- **It is inert on unmeasured claims**, so no pre-existing behaviour changes.

Gate 0 runs **before Gate 1**, and the ordering is load-bearing: Gate 1 short-circuits to satisfied whenever there is no combination rule — every verify-mode and open-research run, i.e. exactly the runs this signal exists for. Placed after it, Gate 0 would be dead code on that whole arm. Running first also means it pre-empts the expensive reproducibility tripwire in Gate 4.

### Quality second opinion — a point value must not destroy evidence

The same principle applied one layer down. `ExtractEvidenceOperation._score_evidence` Path 2 blends four agent-produced floats into one quality score and, below `0.10`, used to invalidate the evidence outright — a categorical cliff driven by a single sample.

That destroy is now gated by `quality_doubt.confirm_low_quality`, one `llm_judge.judge_score` call (FAST path: single judge, temperature 0, reproducible; **one** criterion, so **one** model call). The evidence is invalidated only when the second opinion returns `overall == "fails"` **and** `needs_review` is False — the label decides, and `needs_review` already folds in llm_judge's own doubt threshold, so no new numeric cut is introduced. When the judge disagrees or is unsure, the evidence survives with its low `quality_score` intact and stays heavily down-weighted everywhere: attenuation, not destruction. The distribution is stored on `Evidence.quality_distribution`, whose presence *is* the marker that a second opinion ran. A missing `judge_model` raises rather than resuming the old unconditional destroy, and a judge failure propagates to `_execute_op` as a loud quarantine.

This is the only place `epistemic` uses `llm_judge`'s elicitation layer. `epistemic_judge_evidence` is deliberately **not** converted: it forces a four-step scope analysis and enforces `in_scope` ↔ argmax over `{supports, contradicts, no_bearing}` via a retry-triggering cross-field validator, none of which llm_judge's generic `meets/partial/fails` schema can express.

**The gate must be reachable from every construction path.** `judge_model` flows from `run_epistemic_graph` → `EpistemicDeps` → `_make_op`. Two graph nodes bypass `_make_op` because their operation takes a `providers` kwarg it cannot supply (`PlanEvidence._run_dispatch`, `ResolveUncertainties._run_investigate`), and `dispatch_and_persist_for_text` builds its own `ExtractEvidenceOperation` helper. All three are the *primary* evidence-gathering path, and all three must carry the same dependency set as `_make_op` — the helper now uses `BaseOperation.derive`, which copies the set rather than enumerating it, and `test_structural_wiring::TestInlineOperationConstructionsMatchMakeOp` pins the two node constructions equal to `_make_op` so the next dependency added cannot skip them. The wiring raise deliberately stays at the point of use rather than at operation construction: an item scored via Path 1 or Path 3 never reaches the gate, so asserting the dependency at construction would make a judge model mandatory for callers that never elicit one. See the decision recorded in `_confirm_or_retain_low_quality`'s docstring.

---

## 10. Confidence scoring

### The verdict object — one decision, rendered everywhere

`EpistemicVerdict` (`entities/verdict.py`) is the run's single decision
record. `CombineClaimVerdicts` — the sole predecessor of `CheckCompletion` —
builds it from the live database on every pass via `confidence.build_verdict`,
writes it to `EpistemicGraphState.verdict`, and both terminals stamp it onto
the frozen `Snapshot`. It stores the minimum (the `PosteriorReport`, the
per-claim `ClaimVerdict` entries with their `EvidenceRef`s, and a
`VerdictProvenance` of counts) and DERIVES `posterior` / `confidence` /
`direction` / `label` as computed fields, so a second number cannot drift
from the first.

Everything downstream RENDERS it rather than re-deciding: `CheckCompletion`'s
routing and all four insufficient-reason strings, `CheckSynthesisDemand`'s
four gates, the artefact's `> **Verdict:**` line
(`render_verdict_line`, code-owned — the writer agent has no verdict field),
the CLI, the offline report, and the HTML audit report. `verdict_label` in
`entities/verdict.py` is the ONLY place in the package that compares a
posterior against `POSTERIOR_DIRECTIONAL_BREAKPOINT`; `display_label` maps
the resulting *direction* onto the six-label display vocabulary and never
re-thresholds — it is a total function of the direction, with no terminal-state
override (see §10, *Display labels*). A source scan pins both properties, and
also pins that `audit_report` neither calls the mapping nor spells a label.

Reachability mirrors the artefact duality: 1-hop live via
`PipelineResult.verdict`, auditable freeze via
`objective.snapshot_id → snapshot.verdict`. The posterior is computed
**inside** the graph — there is no post-run recomputation, which is what
removes the live-vs-offline `terminal_state` divergence.

The aggregated posterior itself is computed by `compute_posterior` in `confidence.py`. It is **abductive-primary**: the IBE chain is the system's certifier of directional verdicts, and the posterior honours the chain's per-claim verdict rather than re-deriving direction from per-item counts. (An earlier design blended a sigmoid-of-counts with the integration verdict via a tunable mixing constant; that blend was removed — the constant couldn't be justified against held-out data.)

- **Integration path (the headline)** — each integrated claim maps its verdict to a probability: `supports` at confidence *c* → `0.5 + c/2`, `contradicts` → `0.5 − c/2`, `insufficient` → `0.5`. Decomposed runs aggregate rule-aware via `combine_claim_verdicts` (AND → min, OR → max, WEIGHTED_AND → weighted mean), so the headline agrees with `CombineClaimVerdicts`; open-research and seed-claim runs use a confidence-weighted average.
- **No-certified-verdict gate** — when active claims exist but none reached IBE certification, the posterior is forced to 0.5 with `terminal_state="oscillation_detected"` (or `retrieval_failed` when that's the more specific diagnostic), mirroring `CheckCompletion`'s routing to `SynthesizeInsufficient` under the same condition. Counting on uncertified evidence is structurally unsafe: its direction can flip run-to-run with no change in epistemic state.
- **Counting posterior (diagnostic + narrow fallback)** — a soft-vote count of supporting vs contradicting evidence, `sigmoid(Σ supporting − Σ contradicting)`. Always computed and reported for inspection (`counting_posterior`, `supporting_count`, `contradicting_count`); it carries the headline only in the counting-fallback branches (a UNION combination rule, which has no scalar verdict, or a run with no claims at all).

### Verbalized-confidence soft votes (Tier 0 / Tier 1)

The evidence-claim judge (`epistemic_judge_evidence`) emits a verbalized 3-way belief distribution over `{supports, contradicts, no_bearing}` rather than a bare label; the categorical `support_judgment` is its argmax, and the distribution plus derived signals (`judgment_confidence`, `judgment_entropy`, `judgment_one_hot` — pure math in `judgment_signal.py`) are stored on the `Evidence`.

The counting posterior consumes the distribution as a **soft vote** (`_evidence_counting_vote`): with distribution `[p_s, p_c, p_n]` and cluster weight `w = 1 + log(corroboration)`, an item contributes `w·p_s` to supporting and `w·p_c` to contradicting — a near-tie judgment nets ~0, a confident one nets ~±w. A one-hot distribution reproduces the old hard vote exactly, and evidence without a distribution (adversarial counter-evidence, legacy data) falls back to the hard `support_judgment` vote, so pre-Tier-0 behaviour is preserved as a limiting case.

### Posterior penalties

When a process flag fires (the inquiry didn't converge cleanly), the verdict it produced is provisional. The penalty surfaces that signal with reduced weight, rather than discarding it.

| Flag | Constant | Value | Mechanism |
|---|---|---|---|
| `cycle_capped=True` on any contributing claim | `CYCLE_CAP_CONFIDENCE_PENALTY` | 0.7 | Multiplier on confidence (integration path) or pull-toward-neutral on posterior (counting path) |
| `state.retrieval_failed=True` | `RETRIEVAL_FAILED_CONFIDENCE_PENALTY` | 0.7 | Same shape, same value. Stacks multiplicatively |
| Verdict derived mechanically, without the IBE chain | `MECHANICAL_VERDICT_MAX_CONFIDENCE` | 0.9 | Hard ceiling on `integrated_confidence`. Read by `PromoteAsRefutedOperation` (integer counts) and `ConcludeAtCycleCapOperation` (weighted directional mass) |

A doubly-provisional verdict gets `0.7 × 0.7 = 0.49` of its full distance from 0.5.

### Verdict labelling

| Posterior | Label |
|---|---|
| > 0.66 (`POSTERIOR_DIRECTIONAL_BREAKPOINT`) | **supports** |
| < 0.34 (1 − breakpoint) | **contradicts** |
| 0.34 ≤ p ≤ 0.66 | **insufficient** |

Both comparisons are **strict** — a posterior of exactly 0.66 or exactly 0.34 is `insufficient`. `entities.verdict.verdict_label` is the sole implementation; the audit report's legend prints the same strict band.

The directional breakpoint is wider than 0.5 to keep the label calibrated with the underlying evidence strength. `POSTERIOR_DECISIVE_THRESHOLD = 0.85` is a separate, stricter threshold used only by `CheckSynthesisDemand` Gate 4 to decide whether more inquiry could change the headline.

### Display labels

`verdict_label` produces a *direction*; `entities.verdict.display_label` maps that direction — and nothing else — onto the closed display vocabulary. It is a total function of the direction: no second threshold comparison, and (since 2026-08-09) no terminal-state override, so the objective headline and the per-claim badges in one document cannot disagree.

| Direction | Display label |
|---|---|
| `supports` | **Confirmed** (or *Confirmed with refinement*) |
| `contradicts` | **Refuted** |
| `insufficient` | **Inconclusive** |
| `no_verdict` (claim level: never adjudicated) | **Insufficient evidence** |
| `no_data` / `union` (objective level: no scalar verdict by construction) | **No single verdict** |

`no_data` is a question type `compute_posterior` declines to score; `union` is a UNION decomposition whose rule declares the sub-investigations are facets rather than a value to combine. Neither means "the evidence did not decide" — the claims may all be adjudicated — so both render as *No single verdict* with the per-claim results as the answer.

`retrieval_failed` no longer suppresses the direction in the label. `CheckCompletion` routes `retrieval_failed` + at least one adjudicated claim to the *directional* terminal by design, so the caveat is reported as a caveat: `render_verdict_line` appends "retrieval failed" to the same line and the audit report renders the terminal state explicitly.

The fallibilism terminal is the one exception, and it is a routing fact rather than a second derivation: `SynthesizeInsufficientReportOperation` renders through `render_insufficient_verdict_line`, which pins the label to *Insufficient evidence* because reaching that terminal is the system declaring it has no directional answer. Every provenance clause on the line still comes from the same object.

---

## 11. Question-type routing

Not every verification track applies to every question. The system classifies the question into one of seven types in `PrepareObjective`, and the type dictates which verification tracks are PRIMARY (gating), SECONDARY (contributing), *If applicable*, or SKIP (irrelevant).

### Seven question types

Defined in `routing.py:ROUTING_TABLE` and enumerated identically by the classifier agent (`epistemic_classify_question`):

| Type | Question shape | Example |
|---|---|---|
| **verificatory** | "Is P true?" — binary truth-claims; needs adversarial testing | Does X cause Y? |
| **explanatory** | "Why P?" / "How does P work?" — causal/mechanistic | What causes X? |
| **exploratory** | "What might be involved in P?" — hypothesis generation, breadth over depth | What are the effects of X? |
| **comparative** | "Is A better / more likely than B?" — ranking; symmetry of scrutiny matters | Is X a better treatment than Y? |
| **predictive** | "What will happen if P?" — forward projection; falsifiability decisive | Will X cause Y? |
| **compositional** | "What are the parts/factors of X?" — analytical decomposition | What factors contribute to X? |
| **normative** | "Should we do X?" — value-laden; separates facts from value commitments | Should we use X? |

Every type decomposes; what differs is the *kind* of sub-claim (falsifiable test claims for verificatory with an AND rule, mechanistic step-claims for explanatory, and so on — the decomposition agent's guidance is type-specific). The exception is verify mode (`Objective.claim_to_verify`), which skips decomposition entirely.

### Routing matrix

P = Primary, S = Secondary (both gate promotion when a stage's gate names the track), A = If applicable, — = Skipped. Only SKIP and IF_APPLICABLE tracks are never required.

| Track | verificatory | explanatory | exploratory | comparative | predictive | compositional | normative |
|---|---|---|---|---|---|---|---|
| Adversarial | P | S | — | S | S | — | S |
| Convergence | P | S | S | — | — | P | — |
| Deductive | S | P | — | S | P | — | P |
| Computational | A | A | — | — | P | — | — |
| Contrastive | — | P | — | P | — | — | — |
| Cross-claim consistency | — | — | P | P | — | P | P |
| Argument analysis | S | P | — | — | — | — | P |

Each routing profile also carries per-stage `gate_thresholds` overrides — e.g. verificatory's PROVISIONAL gate requires convergence, comparative's requires a contrastive evaluation, normative's requires fact–value separation at every stage. `routing.py` is pure data; it is the authoritative spec for conditional verification.

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
| `POSTERIOR_DIRECTIONAL_BREAKPOINT` | 0.66 | `entities.verdict.verdict_label` — the ONLY read in the package; every other consumer maps the resulting direction |
| `POSTERIOR_DECISIVE_THRESHOLD` | 0.85 | `CheckSynthesisDemand` Gate 4 |

### Convergence (Reichenbach / Mill)

| Constant | Value | Read by |
|---|---|---|
| `CONVERGENCE_STRONG_THRESHOLD` | 0.7 | convergence verdict — gates IBE fast-path |
| `CONVERGENCE_INTRA_DIVERSITY_THRESHOLD` | 0.5 | cluster diversity check |
| `CONVERGENCE_INTER_DOMAIN_DISTANCE_LOW` | 0.3 | `shared_error_modes` warning |

### Verbalized-confidence calibration

| Constant | Value | Read by |
|---|---|---|
| `JUDGMENT_ONE_HOT_THRESHOLD` | 0.95 | `judgment_signal.distribution_is_one_hot` — top-class mass at or above this marks the judgment's entropy as uninformative (the small-model degeneracy mode identified in the local-model calibration study) |
| `JUDGMENT_DOUBT_ESCALATION_THRESHOLD` | 0.5 | `graph.nodes.CheckSynthesisDemand._judgment_doubt_demand` (sole reader) — aggregate judgment doubt at or above this buys the objective one extra round of inquiry. Deliberately **identical** to `llm_judge.signals.NEEDS_REVIEW_DOUBT_THRESHOLD`: one doubt scale, one cutoff, pinned equal by `tests/test_doubt_scale_single_source.py` rather than by an import |

`epistemic.judgment_signal` and `llm_judge.signals` are one definition of doubt implemented twice, kept in step by a drift test. The import is not available in the other direction: `import andamentum.llm_judge.signals` executes the package `__init__` → `panel` → `elicit` → `pydantic_ai`, and `judgment_signal` is imported by `entities/evidence.py`, a pure-stdlib leaf. `argmax_verdict` is the one primitive deliberately *not* converged — `llm_judge.signals.argmax_label` uses a tie tolerance, epistemic uses exact `max`, and the near-tie behaviour feeds the `in_scope` validator and the promotion gates.

The constants are deliberately few. Adding a new threshold requires the same justification (theoretical basis plus the sites that read it) as the existing set. Bare numeric literals in decision logic across the rest of `epistemic/` are an architectural smell.

---

## 13. Code structure

Top-level layout under `src/andamentum/epistemic/`:

| Path | Role |
|---|---|
| `thresholds.py` | Single source of truth for every decision-relevant numeric threshold. |
| `entities/` | The seven primitives plus `Prediction`, decomposition types, `CapEvaluation` and `verdict.py`. Pure pydantic data classes. |
| `entities/verdict.py` | `EpistemicVerdict` / `ClaimVerdict` / `EvidenceRef` / `VerdictProvenance` / `PosteriorReport`, plus `verdict_label` (the one threshold read), `display_label` and `render_verdict_line`. Leaf module: pydantic + `thresholds` only. |
| `operations/` | 37 operation classes (`BaseOperation` subclasses) across 17 modules. Pure transforms. |
| `graph/` | The 23-node DAG. `nodes.py` · `state.py` · `topology.py` · `base.py` · `stages.py` (named stage runners with enforced exit invariants). |
| `gates.py` | Stage promotion gates. `STAGE_GATES` dict plus `validate_promotion` (routing-aware). |
| `providers/` | 11 evidence providers (10 domain APIs plus `web_search`) plus `CONTRIBUTING.md`. Each returns `list[GatheredEvidence]`; never raises. |
| `convergence_detector.py` | Reichenbach common-cause cluster analysis. |
| `confidence.py` | Posterior aggregation (abductive-primary, diagnostic counting posterior — see section 10) plus `build_verdict`, which assembles the run's `EpistemicVerdict` from the live DB. |
| `judgment_signal.py` | Pure math for the verbalized judgment distribution: normalisation, argmax verdict, confidence, entropy, one-hot detection. Leaf module, no epistemic deps. |
| `demand.py` | The `Demand` Pydantic model used everywhere lazy-escalation happens. |
| `repository.py` | `EpistemicRepository` wraps a `DocumentStore` (from `andamentum.document_store`). |
| `runner.py` | `DefaultAgentRunner` wrapping `core.AgentRunner` with the epistemic agent registry. |
| `cli.py` | `andamentum-epistemic` entry point. Primary modes: `ask` (research) and `verify` (seed-claim); further subcommands `run`, `agents`, `preflight`, `stage`, `inspect`, `confidence`. |
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

### Stage runners

`graph/stages.py` partitions the graph into six named stages — `preplanning`, `initial_evidence`, `scrutiny_and_investigation`, `verification`, `integration`, `synthesis` — each an (entry node, exit node) pair with an enforced **exit invariant**: an async predicate over the repository (e.g. every claim-linked evidence has a `support_judgment` before leaving scrutiny) that raises `StageInvariantError` if the stage boundary is not quiescent. Invariant violations crash loudly rather than letting a half-finished stage silently degrade the posterior toward 0.5. The `andamentum-epistemic stage <name>` subcommand runs a single stage, resuming from a checkpoint database via `--from-db` — useful for debugging one stage in isolation.

---

## 14. Operations catalogue

One module per family in `operations/`. Total: 37 operation classes across 17 modules.

| Module | Operations |
|---|---|
| `preplanning` | `ClarifyQuestionOperation`, `ClassifyQuestionOperation`, `ConceptualAnalysisOperation`, `DecomposeQuestionOperation` |
| `dispatch_gather` | `DispatchGatherOperation` (initial gather via description-driven dispatch) plus the module-level `dispatch_and_persist_for_text` helper shared with investigation |
| `seed_claim`, `multi_seed_claim`, `claims` | `SeedClaimOperation`, `MultiSeedClaimOperation`, `ProposeClaimsOperation`, plus `top_n_representatives` (claim-relevance rerank, async) |
| `evidence` | `ExtractEvidenceOperation` (residual stub-extraction path; the dispatch flow normally persists Evidence with content + scoring already attached, so this path is rarely exercised) |
| `scrutiny`, `investigation` | `ScrutiniseClaimOperation`, `InvestigateClaimOperation`, `GeneratePredictionOperation`, `RecordDecisionOperation` |
| `uncertainty`, `concerns` | `ResolveUncertaintyOperation`, `DeduplicateConcernsOperation` |
| `verification` | `AdversarialSearchOperation`, `AnalyzeArgumentOperation`, `AssessConvergenceOperation`, `ContrastiveEvaluationOperation`, `CrossClaimConsistencyOperation`, `ValidateDeductivelyOperation`, `VerifyComputationallyOperation` |
| `integration` | `EnumerateCandidatesOperation`, `ScoreLovelinessOperation`, `ScoreLikelinessOperation`, `SelectBestExplanationOperation` (plus framing-tie cap helper) |
| `stage_management` | `PromoteClaimOperation`, `DemoteClaimOperation`, `PromoteAsRefutedOperation`, `SoftPromoteOperation` |
| `synthesis` | `FreezeSnapshotOperation`, `SynthesizeReportOperation` (writer ⇄ validator), `SynthesizeInsufficientReportOperation` |
| `belief_maintenance` | TMS cascade after evidence invalidation, plus `SetRoutingDefaultsOperation` |
| `cleanup` | `AbandonStaleClaimOperation` |
| `analysis`, `identifier_extraction` | Helpers used during scrutiny and investigation |

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
| `web_search` | SearxNG web search | general-domain fallback |

Total: 11 providers. Default runs offer all 11 to the dispatch agent (`get_all_providers()`); `get_biomedical_providers()` returns the 10 domain APIs without `web_search` for biomedical runs that need the precision of the specialist set. There is no separate provider-selection tournament: the description-driven dispatch agent (section 9) is invoked once per provider and each either commits native-syntax queries or abstains.
