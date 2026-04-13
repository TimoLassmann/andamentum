# Epistemic System

A formal epistemology implementation for AI research — building knowledge that can be trusted, traced, and revised.

---

## Core Principle: Epistemic Decomposition

When we ask an LLM to research a topic, we get a monolith — confident prose with no traceability, no calibrated confidence, no revision path. If it's wrong, our only option is to regenerate and hope.

This isn't how reliable knowledge is produced. Science doesn't work because individual scientists are infallible. It works because the **process** — peer review, replication, adversarial challenge, explicit uncertainty — produces reliable knowledge from fallible participants.

The epistemic system applies this insight to LLM-based research:

> **Every epistemic judgment is made by a focused agent performing the narrowest possible task, and combined through deterministic, auditable rules.**

Instead of one LLM doing monolithic reasoning, we decompose the epistemic process into:

1. **Narrow judgment tasks** — each performed by a focused agent that sees only what's relevant and answers only what's asked
2. **Deterministic combination rules** — gates, thresholds, and calculations that combine judgments into outcomes
3. **Explicit state** — every claim, every piece of evidence, every uncertainty is a first-class object with provenance

LLMs are unreliable at sustained coherent reasoning over thousands of tokens. But they're remarkably good at narrow, well-specified judgment tasks: "Given this claim and this piece of text, is this relevant evidence? Rate 1-5 with justification." Each such task is something a human expert could do in two minutes. The system's reliability comes not from any individual judgment being perfect, but from the **structure** that combines many narrow judgments through transparent rules.

No single agent decides whether a claim is true. The verdict emerges from the structured interaction of many narrow judgments.

### The Analogy

| Scientific Process | Epistemic System |
|---|---|
| Individual researcher | Focused agent (narrow task) |
| Peer review | Scrutiny agent |
| Replication | Cross-domain convergence check |
| Devil's advocate | Adversarial search agent |
| Journal acceptance criteria | Stage gates |
| Retraction / correction | TMS cascading belief maintenance |
| The scientific community | The pattern-driven scheduler |

### What This Is Not

- **Not an ensemble of LLMs voting** — agents have different roles, not the same role repeated
- **Not chain-of-thought with extra steps** — the structure *is* the epistemology, not just a prompt technique
- **Not circular** — using a focused agent to evaluate evidence quality is analogous to peer review (humans evaluating other humans' work). The evaluation is narrow, auditable, and combined deterministically. Circularity would be asking the same agent to both produce and validate its own claims.

---

## The Design Test

Every component in the system falls into one of three categories:

| Category | What it does | Who does it | Example |
|---|---|---|---|
| **Judgment** | Requires understanding natural language to make a decision | Focused agent call | "Is this text relevant evidence for this claim?" |
| **Combination** | Merges existing judgments into a derived quantity | Deterministic arithmetic | `balance = supporting / (supporting + opposing)` |
| **Gate** | Decides whether an entity can advance based on existing signals | Deterministic threshold | "Need ≥2 evidence items with quality sum ≥0.5" |

The rule: **If a component requires understanding natural language, it is a judgment and must go through a focused agent. If it combines existing judgments, it must be deterministic and auditable.**

Keyword heuristics are a fourth category that should not exist. A keyword list checking for "p-value" or "systematic review" is neither genuine assessment (an agent could reason about whether statistical evidence is appropriate for the specific claim) nor a deterministic rule (it's a brittle approximation masquerading as logic). When you find a keyword heuristic in the codebase, replace it with a focused agent call.

### Compliance Checklist

When reviewing or writing code for the epistemic system, check:

1. **Does this function need to understand text to produce its output?**
   - YES → It must call a focused agent. No keyword matching. No regex classification.
   - NO → It should be deterministic (arithmetic, threshold, lookup).

2. **Does this agent call do exactly one thing?**
   - A focused agent should evaluate ONE aspect: relevance, or specificity, or quality — not all three at once.

3. **Is every input to a gate either a stored entity field or a deterministic computation?**
   - Gates must not contain implicit judgments. If a gate needs a quality score, that score must already exist on the entity, computed by a prior operation.

4. **Can I trace this output back to specific evidence?**
   - Every claim in the final artefact must link to evidence. Every piece of evidence must have a source reference.

---

## Seven Epistemic Primitives

All knowledge is represented using seven primitive types that form a complete vocabulary for describing the epistemic state of any research inquiry.

### 1. Objective

The research question. An objective captures the question and its current phase of investigation — from initial formulation through evidence collection, claim validation, and final synthesis. The objective is the root node connecting all other primitives. Each objective carries a `question_type` field (set by `classify_question`, one of seven types: verificatory, explanatory, exploratory, comparative, predictive, compositional, normative) that determines which verification tracks fire and what stage gate thresholds apply. The objective also maintains a `pending_concerns` buffer (list of dicts) for remaining concerns generated during uncertainty resolution rounds; these are batch-deduplicated by `DeduplicateConcernsOperation` before being promoted to uncertainty entities.

### 2. Evidence

An interpreted observation from a source. Evidence is not raw data — it is data that has been extracted, contextualized, and linked to the inquiry. Each piece records its source, the extracted content, known limitations, and whether extraction is complete. Evidence exists independently of claims; the same evidence may support, undermine, or be irrelevant to multiple claims. Each evidence entity also carries a `cluster_status` (unclustered, representative, corroborative, or deferred) set by the HDBSCAN deduplication pipeline, a `corroboration_count` recording how many semantically similar items the evidence represents, and a `support_judgment` field ("supports", "contradicts", or "no_bearing") set by inline judge calls during `ProposeClaimsOperation` or `ExtractEvidenceOperation`.

### 3. Claim

A scoped proposition with a defined lifecycle. Claims are the central currency of the system. Each has a statement, a scope (what it applies to), a stage (how well-substantiated it is), and explicit links to supporting evidence. Claims accumulate verification results as they progress through scrutiny, and track modification history for degeneracy detection. Additional fields: `contrastive_checked` and `consistency_checked` (for the two pairwise verification tracks), `routing_applied` (whether `SetRoutingDefaultsOperation` has pre-marked skipped tracks), and `saturated` (whether investigation has stopped producing new information, preventing infinite cycling).

### 4. Uncertainty

First-class doubt. Rather than hiding uncertainty in hedging language, the system represents it as an explicit entity with a type, scope, and resolution status. Some uncertainties are *blocking* — they prevent claims from advancing. Others are *non-blocking* — they serve as caveats without halting progress. This distinction is fundamental to making progress under imperfect information.

### 5. Decision

A commitment record. When a claim reaches sufficient maturity, a decision records what was decided, the justification, which claims informed it, and whether the decision is reversible.

### 6. Snapshot

An immutable freeze of the epistemic state at a point in time. Snapshots capture which claims, evidence, and uncertainties existed at the moment of synthesis, ensuring the output reflects a coherent state.

### 7. Artefact

The human-facing output with full traceability. Compiled from a snapshot, an artefact includes the final synthesis along with a trace map linking each paragraph to the claims and evidence that support it.

---

## The Claim Lifecycle

Claims progress through five stages of increasing epistemic maturity:

```
HYPOTHESIS → SUPPORTED → PROVISIONAL → ROBUST → ACTIONABLE
```

**Hypothesis** — Initial proposition, not yet scrutinized. The system makes no assumption about validity; it records the proposition and awaits evaluation.

**Supported** — Passed initial skeptic review. A scrutiny agent examined the claim for logical coherence, scope appropriateness, and basic evidential support. This does not mean the claim is true — only that it warrants investigation.

**Provisional** — Survived multiple independent verification methods: adversarial search, cross-domain convergence, deductive validation. Provisionally accepted but open to revision.

**Robust** — Supported by multiple independent evidence sources and has withstood sustained adversarial challenge. Reliable for practical purposes, though falsifiable in principle.

**Actionable** — Meets decision-readiness criteria. Evidence warrants commitment to real-world decisions.

The lifecycle is not a one-way escalator. Claims can be **demoted** when new evidence undermines them. Demotion is not failure — it is the system working as intended.

---

## Stage Gates

Each stage transition is governed by a **deterministic gate** — requirements that must be satisfied before a claim can advance. Gates are implemented in code, not evaluated by language models. The criteria for epistemic advancement must be transparent, reproducible, and immune to the persuasive fluency of LLM outputs.

| Transition | Default Requirements |
|---|---|
| **Hypothesis → Supported** | ≥1 evidence, quality sum ≥0.3, scrutiny passed, no blocking uncertainties, min_supporting_sources=1 |
| **Supported → Provisional** | ≥2 evidence, quality sum ≥0.5, all 7 verification track flags true, adversarial balance ≥0.4, no blocking uncertainties, min_supporting_sources=2 |
| **Provisional → Robust** | ≥3 evidence, quality sum ≥1.5, adversarial + convergence + deductive complete, independent evidence lines from ≥2 domains, min_supporting_sources=3 |
| **Robust → Actionable** | ≥3 evidence, quality sum ≥1.5, all verification tracks complete, decision criteria defined, min_supporting_sources=3 |

The `min_supporting_sources` gate counts independent evidence clusters with a `support_judgment` of "supports" (set by the judge module). This gate is only enforced when at least one evidence item has been judged, providing a graceful transition as the judge is integrated into existing inquiries.

The SUPPORTED → PROVISIONAL gate requires all seven verification track flags (`adversarial_checked`, `convergence_checked`, `deductive_checked`, `computational_checked`, `contrastive_checked`, `consistency_checked`, and `argument_analyzed`) to be true. Tracks that are not applicable to the question type are pre-marked `True` by `SetRoutingDefaultsOperation` before verification begins, so the gate condition is met without those tracks actually firing.

**Question-type parameterization**: Gate thresholds are overridden per question type via the routing config. For example, exploratory questions lower the SUPPORTED evidence threshold to 0.5, while verificatory questions raise the PROVISIONAL gate to require convergence. The `validate_promotion()` function reads overrides from the routing profile's `gate_thresholds` and falls back to defaults when no override exists.

Gates also check for blocking uncertainties and degeneracy warnings. If any blocking uncertainty is associated with a claim, promotion is denied regardless of other criteria. Quality sums exclude corroborative and deferred evidence (only representative evidence contributes).

**Important**: Gates consume signals that were produced by earlier operations. The gate itself is purely deterministic — it checks numeric thresholds against stored fields. The judgment about evidence quality, adversarial balance, and verification results happens in the operations that *set* those fields, each through focused agent calls.

---

## Quality Assessment: The Agent-Deterministic Split

This section describes how the system evaluates evidence quality and counterargument strength. This is where the core design principle has the most practical impact.

### The Three-Layer Pattern

Every quality assessment in the system follows the same pattern:

1. **Focused agent produces a judgment** — A small, auditable LLM call evaluates ONE aspect (relevance, specificity, source credibility, etc.) and produces a numeric score with justification
2. **Deterministic formula combines judgments** — Arithmetic combines individual scores into a composite (e.g., `quality = 0.4 * relevance + 0.3 * specificity + 0.3 * source_credibility`)
3. **Deterministic gate consumes the composite** — Threshold checks decide whether the entity can advance

The agent provides understanding. The formula provides transparency. The gate provides reproducibility. No step does the job of another.

### Evidence Quality

Evidence quality scoring combines:

- **Bibliometric signals** (deterministic) — Citation counts, journal DOAJ status, retraction checks. These are available for academic sources and are computed without LLM involvement.
- **Content assessment** (focused agent) — Relevance to the specific claim, methodological rigor, specificity of findings. These require understanding the text and must go through a focused agent call.

For sources without bibliometric metadata (web search, databases), content assessment from a focused agent is the primary quality signal. The system must not fall back to keyword heuristics when bibliometric signals are unavailable — it must use a focused agent instead.

### Counterargument Quality

When adversarial search finds potential counterarguments, each must be evaluated for:

- **Relevance** — Does this actually address the claim's specific assertions?
- **Specificity** — Is this a general objection or targeted to the claim?
- **Evidence backing** — Does the counterargument cite evidence, or is it speculative?
- **Source credibility** — Is the source authoritative for this domain?

Each of these evaluations requires understanding natural language. Each must be a focused agent call. The combined quality score is then a deterministic weighted sum, and the adversarial balance is deterministic arithmetic:

```
balance = supporting_weight / (supporting_weight + adversarial_weight)
```

### Domain Classification

Evidence is classified along four dimensions for convergence detection: methodology (experimental, observational, theoretical, computational), data source (primary, secondary, meta-analytic), temporal approach (cross-sectional, longitudinal, historical), and causal role (cause, effect, mechanism, correlation).

Classification requires understanding the evidence content. Each classification must be a focused agent call. The convergence calculation that uses these classifications is deterministic (clustering and distance metrics).

### Evidence-Claim Judgment

The judge module (`judge.py`) provides two focused evaluative LLM calls that are the only evaluative inputs to the confidence scoring model:

- **`judge_evidence()`** — Given a claim and a piece of evidence, returns a three-way classification: "supports", "contradicts", or "no_bearing", with reasoning.
- **`judge_independence()`** — Given two evidence items, returns a binary judgment: independent or not independent, with reasoning.

These run **inline** inside existing operations, not as separate operations:
- **`ProposeClaimsOperation`**: after each claim is created, all linked evidence is judged for support/contradict/no_bearing.
- **`ExtractEvidenceOperation`**: after extraction, if the evidence is already linked to a claim, it is judged immediately.

Judge verdicts feed into the posterior P(Y) score (see Confidence Scoring below) and the `min_supporting_sources` stage gate. The judge makes no domain-specific quality judgments — it evaluates only the relationship between a piece of evidence and a claim.

### What Must NOT Be Keyword-Based

The following evaluations require genuine understanding and must use focused agent calls:

- Evidence relevance to a claim
- Evidence quality / methodological rigor
- Counterargument relevance, specificity, and strength
- Evidence domain classification (methodology, data source, temporal approach, causal role)
- Adversarial query generation (extracting key concepts from a claim)
- Criticism classification (logical, empirical, methodological, scope-based)

### What Is Correctly Deterministic

The following are properly deterministic and should NOT use agent calls:

- Stage gate thresholds (evidence count, quality sum, verification flags, min_supporting_sources)
- Adversarial balance calculation (ratio of weights)
- Convergence detection (clustering algorithm on classified dimensions)
- Domain distance metrics (lookup table between classification categories)
- Degeneracy detection (modification counts and timestamps)
- Bibliometric quality scoring (citation counts, journal status)
- Answer confidence scoring (checklist pass/fail, logistic transform)
- Posterior P(Y) scoring (count supports - count contradicts, logistic transform)
- Question-type routing (lookup table from question type to track activation levels)
- Provider selection (keyword matching against domain provider map)
- Evidence deduplication (HDBSCAN clustering on embedding distances)
- Evidence dedup via shared similarity module (single-linkage clustering, Union-Find)
- Caveat dedup (group non-blocking uncertainties by embedding, keep medoid)
- Batched concern dedup (group buffered remaining_concerns, filter against existing)
- Saturation detection (verdict unchanged + no unresolved blocking uncertainties)
- Routing defaults (pre-marking skipped track flags on claim entities)

---

## Seven Verification Methods

Once a claim passes initial scrutiny, it faces up to seven independent verification tracks. Not all tracks fire for every question — the routing system activates tracks based on question type (see Question-Type Routing below).

### Adversarial Search

The system actively seeks evidence *against* the claim. An adversarial agent generates search queries designed to find counterexamples, contradictions, and competing explanations. Found counterarguments are evaluated by focused agents for relevance, specificity, and strength. The adversarial balance — the ratio of supporting to opposing evidence, weighted by quality — determines whether the claim can withstand challenge.

This is perhaps the most important verification method. Both human researchers and language models suffer from confirmation bias. Adversarial search directly counteracts this by making disconfirmation a first-class activity.

#### Pipeline

`AdversarialSearchOperation` (in `operations/verification.py`) runs per challenged claim in five steps:

1. **Template query generation** — deterministic domain-aware query templates based on the claim text.
2. **Agent query generation** — three parallel agent calls (`epistemic_generate_counterquery`) using independent framings (`contradicting_evidence`, `alternative_explanations`, `replication_failures`). Each framing runs without seeing prior outputs, preserving Kahneman's independence principle.
3. **Web search** — the combined query set (capped at 5) runs in parallel against the evidence gatherer with bounded concurrency (`asyncio.Semaphore(5)`).
4. **Counterargument evaluation** — each search hit is scored by a narrow agent (`epistemic_evaluate_counterargument`) for relevance, specificity, evidence-backing, and source credibility. Evaluations run in parallel with bounded concurrency (`asyncio.Semaphore(10)`). Failed evaluations fall back to a default counterargument with neutral quality scores so a single hiccup doesn't corrupt the balance calculation.
5. **Balance synthesis** — `synthesize_adversarial_result()` aggregates the weighted counterargument weight against the supporting evidence weight to produce the adversarial balance score, verdict (`SUPPORTED` / `CONTESTED` / `CHALLENGED` / `REFUTED`), and recommendation.

Parallelization in steps 2–4 is what keeps wall time bounded. A sequential implementation would incur 1 + 5 + N roundtrips (where N is typically 25–50 search hits per claim); the parallel implementation collapses this to roughly the worst single-call latency.

#### Persistence

Three distinct artifacts are written to the database after a successful adversarial search:

- **The `Claim` entity** gets its `adversarial_balance`, `adversarial_checked`, and (for strongly challenged claims) `needs_revalidation` fields updated in place.
- **Quality-passing counterarguments** are persisted as individual `Evidence` entities with `support_judgment="contradicts"`, deduplicated by source URL (the highest-quality counterargument per URL wins). The claim's `evidence_ids` list is extended with the new evidence IDs so TMS and the evidence bibliography can find them.
- **The full `AdversarialEvidence` wrapper** (with `epistemic_type="adversarial_evidence"`, containing the complete counterargument list, queries used, weights, verdict, explanation, and recommendation) is persisted via `EpistemicRepository.save_adversarial_evidence()`. This is the source of truth for report rendering — without it, the counterargument synthesis is lost between the verification operation and the report generator.

For strongly challenged claims (`balance_score < 0.3`), a non-blocking `Uncertainty` is also created summarizing the adversarial finding so the limitations section of the report flags the challenge.

#### Report rendering

Counter-evidence surfaces in the HTML report in two places:

1. **In the Findings section**, each challenged claim's Details block shows the adversarial summary line (`Counter-evidence search found strong opposition (balance: 0.16). This claim was demoted after adversarial challenge.`) followed by a nested list of the individual counterarguments, each with its text, weight, and source link. `ReportGenerator` builds `AdversarialSummary` objects (one per counterargument, tagged with the originating `claim_id`) from the persisted `AdversarialEvidence`; `html_report.py` groups them by claim and renders them inside the existing claim details toggle.
2. **In the Sources section**, contradicting evidence appears under its own "Contradicting" subheading in the evidence bibliography — populated from the `Evidence` entities written in the persistence step above.

The two renderings are complementary: the bibliography lists the raw sources, while the claim details show the system's synthesized interpretation of what each counterargument actually argues and how strongly it counts against the claim.

### Cross-Domain Convergence

Evidence is classified along four dimensions (methodology, data source, temporal approach, causal role). Evidence from different domains is clustered, and the system measures whether independent lines converge on the same conclusion. Convergence from methodologically independent sources is stronger than convergence from similar sources.

### Deductive Validation

The claim is tested against first principles, logical consistency, and known constraints. This catches claims that are empirically plausible but logically flawed — errors that evidence alone cannot reveal.

### Computational Verification

For claims with quantifiable predictions, computational verification independently verifies the numbers. Only applicable to claims with verifiable quantitative content, but when it applies, it provides the strongest evidence: reproducible calculation.

### Contrastive Evaluation

Pairwise comparison of competing claims. When the question involves choosing between explanations or comparing alternatives, the contrastive evaluator assesses each claim pair on explanatory power, evidential support, parsimony, and scope. Primary for explanatory and comparative question types. Skipped for verificatory and predictive questions where claims are evaluated independently rather than against each other.

### Cross-Claim Consistency

Pairwise conflict detection between claims within the same inquiry. Checks whether any two claims logically contradict each other, assert incompatible scope, or make conflicting predictions. Primary for exploratory, comparative, compositional, and normative question types — all cases where multiple independent claims are proposed and must cohere. When conflicts are found, uncertainties are created to flag the inconsistency.

### Argument Analysis

Formal analysis of a claim's argument structure: identification of premises, conclusions, logical validity, soundness, and common fallacies. Fires for claims that passed scrutiny. Primarily activated for explanatory and normative question types where the logical structure of the argument is as important as the empirical evidence.

---

## Question-Type Routing

Different research questions require different verification strategies. A verificatory question ("Does X cause Y?") needs strong adversarial search and convergence but not contrastive evaluation. An explanatory question ("Why does X happen?") needs contrastive evaluation between competing explanations but not necessarily adversarial search.

The routing system configures which verification tracks fire and what gate thresholds apply, based on the question type set by `ClassifyQuestionOperation` early in the pipeline.

### Seven Question Types

| Type | Core concern | Example |
|---|---|---|
| **Verificatory** | Is this true? | "Does spaced repetition improve long-term retention?" |
| **Explanatory** | Why does this happen? | "Why do neural networks generalize beyond training data?" |
| **Exploratory** | What is the landscape? | "What approaches exist for protein folding prediction?" |
| **Comparative** | Which is better? | "Is CRISPR more effective than TALENs for gene editing?" |
| **Predictive** | What will happen? | "Will mRNA vaccines work against future coronaviruses?" |
| **Compositional** | How do parts relate? | "How do sleep, exercise, and diet interact to affect cognition?" |
| **Normative** | What should we do? | "Should AI systems be required to explain their decisions?" |

### Routing Matrix

Each track is assigned one of four activation levels per question type:

| Track | Verificatory | Explanatory | Exploratory | Comparative | Predictive | Compositional | Normative |
|---|---|---|---|---|---|---|---|
| **Adversarial** | PRIMARY | SECONDARY | SKIP | SECONDARY | SECONDARY | SKIP | SECONDARY |
| **Convergence** | PRIMARY | SECONDARY | SECONDARY | SKIP | SKIP | PRIMARY | SKIP |
| **Deductive** | SECONDARY | PRIMARY | SKIP | SECONDARY | PRIMARY | SKIP | PRIMARY |
| **Computational** | IF_APPLICABLE | IF_APPLICABLE | SKIP | SKIP | PRIMARY | SKIP | SKIP |
| **Argument** | SECONDARY | PRIMARY | SKIP | SKIP | SKIP | SKIP | PRIMARY |
| **Contrastive** | SKIP | PRIMARY | SKIP | PRIMARY | SKIP | SKIP | SKIP |
| **Consistency** | SKIP | SKIP | PRIMARY | PRIMARY | SKIP | PRIMARY | PRIMARY |

### Activation Levels

- **PRIMARY**: Always fires for this question type.
- **SECONDARY**: Fires only when a deterministic condition on the claim is met. For example, adversarial search fires as SECONDARY only when the adversarial balance is below 0.6 (indicating conflicting evidence). Convergence fires as SECONDARY only when the claim has 3+ evidence items.
- **IF_APPLICABLE**: Like PRIMARY but semantically indicates the track may find nothing to do (e.g., computational verification on a non-quantitative claim).
- **SKIP**: Never fires. The `SetRoutingDefaultsOperation` pre-marks the track's boolean flag as `True` on the claim entity so that promotion gates are not blocked by a track that was intentionally skipped.

### SetRoutingDefaultsOperation

Before verification tracks begin, `SetRoutingDefaultsOperation` reads the objective's `question_type`, looks up the routing profile, and sets all SKIP track flags to `True` on every SUPPORTED claim. This is a deterministic operation — no LLM call. It runs at priority 4 (before verification at priority 5) and marks `routing_applied = True` on each claim to prevent re-execution.

### Gate Threshold Overrides

Each routing profile includes a `gate_thresholds` dict that can override the default stage gate requirements. For example:
- Exploratory questions lower the SUPPORTED evidence threshold to 0.5 (breadth over depth).
- Explanatory questions add a `requires_contrastive_superiority` check at PROVISIONAL.
- Predictive questions require `requires_falsification_criteria` at SUPPORTED.

The `validate_promotion()` function merges these overrides with default gate thresholds at runtime.

---

## Deduplication

When gathering evidence from multiple providers, the system frequently retrieves semantically similar content — the same finding reported by different sources, or overlapping coverage from web search results. Without deduplication, near-duplicates inflate counts and dilute the signal-to-noise ratio for downstream agents. The system uses two dedup mechanisms depending on the entity type.

### Evidence Deduplication (HDBSCAN)

Evidence is clustered using HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) on cosine distances between text embeddings. HDBSCAN discovers cluster count from data structure — no epsilon or k parameter to tune. Documents that are genuinely unique become noise singletons and are preserved as singleton clusters.

**Cluster-Ranked Top-K Selection**: After clustering, the system selects the top K=5 clusters ranked by the best `quality_score` of any member. Within each selected cluster, the **medoid** (most central document by embedding distance) is selected as the primary representative, and the **best-quality member** (highest `quality_score`) is also selected if different from the medoid.

**Evidence Status Tracking**: Each evidence entity is tagged with a `cluster_status`:

| Status | Meaning |
|---|---|
| `unclustered` | Not yet processed by deduplication |
| `representative` | Selected for downstream processing (medoid or best-quality member) |
| `corroborative` | Semantically similar to a representative; stored for provenance but filtered from pipeline |
| `deferred` | In a cluster outside top-K; stored but not processed this cycle |

Corroborative evidence is excluded from quality sums and agent context but its existence is recorded in the `corroboration_count` on the representative for provenance.

Evidence deduplication is applied at two points:
1. **Initial claim proposal** (`ProposeClaimsOperation`): clusters all extracted evidence before claims are proposed, ensuring claims are built from distinct findings.
2. **After investigation** (`ScrutiniseClaimOperation`): clusters newly fetched evidence before re-scrutiny, preventing investigation cycles from inflating evidence with near-duplicates.

### Unified Similarity Module

The shared `similarity.py` module provides deterministic threshold-based deduplication for assertions, uncertainties, and caveats. It uses cosine similarity with single-linkage clustering via Union-Find (items form transitive groups when similarity exceeds the threshold). A unified threshold `DEDUP_SIMILARITY_THRESHOLD = 0.7` applies across all uses.

Key components:
- **`embed_and_group()`** — Embed texts and group by cosine similarity using single-linkage clustering.
- **`medoid()`** — Select the most central item in each group as the representative.
- **`assess_clustering()`** — Silhouette diagnostics for cluster quality.
- **`validate_groups()`** — Optional LLM-assisted validation of large clusters (calls `epistemic_validate_group` agent).

### Caveat Dedup

During `FreezeSnapshotOperation`, non-blocking uncertainties are deduplicated before the snapshot is frozen. The similarity module groups semantically similar caveats and keeps only the medoid of each group, preventing the final report from listing near-identical caveats.

### Batched Concern Dedup

When `ResolveUncertaintyOperation` generates remaining concerns, they are buffered in the objective's `pending_concerns` list rather than immediately creating uncertainty entities. `DeduplicateConcernsOperation` fires when `pending_concerns_count > 0`, deduplicates the buffer using the similarity module, filters against existing uncertainties, and promotes distinct concerns to uncertainty entities.

---

## Saturation Control

Investigation cycling (Peirce) can enter an unproductive loop: scrutiny says "needs_resolution," investigation fetches more evidence, re-scrutiny still says "needs_resolution" with the same concerns. Without termination logic, this repeats until the investigation count limit (3) is reached — wasting LLM calls on a claim where additional evidence is not helping.

### Deterministic Saturation Check

After re-scrutiny (when `investigation_count > 0`), if the verdict is still `needs_resolution` and all blocking uncertainties for the claim have been resolved (including those marked as "Unresolvable"), the claim is marked `saturated = True`. This is a purely deterministic check — no LLM call.

Saturated claims are excluded from further investigation via a `saturated: False` filter on investigation patterns. The claim remains in its current state (not abandoned, not promoted) and is included in synthesis as-is. If the claim is demoted by a later event, `saturated` is reset to `False` alongside all other verification flags, allowing fresh investigation.

This prevents the infinite loop while preserving the claim and its evidence for the final report.

---

## Uncertainty Taxonomy

Uncertainties are classified into sixteen types in two categories.

### Blocking (7 types)

| Type | Meaning |
|---|---|
| **Unknown** | Genuinely missing critical information |
| **Contradiction** | Sources genuinely disagree on the core claim |
| **Computational Disagreement** | Dual execution results disagree |
| **Strong Counterevidence** | Adversarial search found strong counterarguments |
| **Logical Inconsistency** | Claim contradicts itself or established facts |
| **Physical Implausibility** | Claim violates conservation laws, causality |
| **Missing Premise** | Claim requires unstated assumptions |

### Non-Blocking (9 types)

| Type | Meaning |
|---|---|
| **Evidence Gap** | Insufficient evidence, but not fatal |
| **Assumption** | We assume X without proof |
| **Risk** | X could go wrong |
| **Weak Convergence** | Evidence sources show weak independence |
| **Definitional Variation** | Depends on how terms are defined |
| **Scope Difference** | Different sources apply to different contexts |
| **Methodological Variation** | Different methods yield different specifics |
| **Perspectival** | Valid different viewpoints on same fact |
| **Granularity Difference** | True at one level, nuanced at finer level |

A claim with 10 non-blocking uncertainties can still advance. A claim with 1 blocking uncertainty cannot. This prevents over-skepticism while ensuring genuine epistemic problems are addressed.

---

## The Inquiry Cycle (Peirce)

The system implements cycling inspired by Peirce's theory of inquiry: knowledge advances through hypothesis, test, revision, and re-test.

When a claim is demoted, its verification flags are reset — scrutiny verdict cleared, adversarial/convergence/deductive/computational marks removed. The claim returns to an earlier stage with a clean slate for re-evaluation against the updated evidence base. This forces genuine re-examination, not bookkeeping.

### Investigation Cycling

When scrutiny produces doubt but the claim is not clearly wrong (verdict: `needs_resolution`), the system enters an investigation cycle: analyze the scrutiny feedback, generate targeted search queries, create new evidence stubs, reset the scrutiny verdict. Investigation terminates in one of two ways: the claim is marked *saturated* when re-scrutiny produces the same verdict with no remaining blocking uncertainties (see Saturation Control), or the hard limit of 3 investigation attempts is reached.

Investigation cycling implements Peirce's insight more faithfully than demotion alone. Demotion says "this was wrong, try again." Investigation says "this might be right, but we need more evidence to tell."

---

## Degeneracy Detection (Lakatos)

Lakatos distinguished between *progressive* research programmes (generating novel predictions) and *degenerative* ones (surviving only through ad-hoc modifications). The system implements two detection rules:

| Rule | Trigger | Signal |
|---|---|---|
| **DEGEN_001** | modification_count > 3 | Suggests ad-hoc patching |
| **DEGEN_003** | ≥3 modifications in 24h | Rapid-fire patching |

Degeneracy warnings block promotion but are surfaced for human judgment about whether to continue investigation or abandon.

---

## Traceability (Doyle TMS)

Every element of the final output traces back through a justification chain:

```
Artefact → Snapshot → Claims → Evidence → Source
```

**Debugging**: When the output contains an error, the trace identifies which evidence was misinterpreted. The fix is targeted.

**Trust calibration**: A reader can inspect the justification chain for any statement.

**Revision**: When new evidence emerges, the trace identifies affected claims for selective revision (AGM minimal change principle).

---

## Belief Revision (AGM)

Revision follows the AGM framework: success (new information incorporated), consistency (no contradictions), minimal change (alter as little as possible). When a claim is demoted, evidence links remain intact, verification results are cleared, and the claim returns to an earlier stage. Demoting one claim does not cascade to unrelated claims.

---

## Pattern-Driven Architecture

The system has no central orchestrator. The workflow emerges from **entity state** and **declarative patterns**.

A pattern is a rule: "when an entity of type X is in state Y, perform operation Z." The scheduler scans all entities, matches against all patterns, and produces prioritized work. When an operation changes entity state, new patterns may match. The workflow terminates when no patterns match.

### Advantages

- **Extensible**: New verification method = new pattern + new operation. No orchestration changes.
- **Resilient**: Operation failures don't block unrelated work.
- **Transparent**: The pattern table is a readable specification of the system's behavior.
- **Naturally terminating**: No explicit "done" logic — absence of pending work is the termination condition.

### Pattern Table

| Phase | Priority | Entity | Condition | Operation |
|---|---|---|---|---|
| TMS | 1 | evidence | invalidated=True, invalidation_cascaded=False | `invalidate_evidence` |
| | 1 | claim | needs_revalidation=True, abandoned=False | `revalidate_claim` |
| Pre-planning | 1 | objective | phase=new | `clarify_question` |
| | 1 | objective | phase=clarified, question_type=None | `classify_question` |
| | 1 | objective | phase=clarified | `conceptual_analysis` |
| | 1 | objective | phase=analyzed | `plan_task` |
| Uncertainty | 2 | uncertainty | unresolved, blocking | `resolve_uncertainty` |
| Concern Dedup | 2 | objective | pending_concerns_count > 0 | `deduplicate_concerns` |
| Evidence | 3 | evidence | extracted=False | `extract_evidence` |
| Investigation | 4 | claim | verdict=needs_resolution, investigation_count<3, saturated=False | `investigate_claim` |
| | 4 | claim | verdict=fail, stage=hypothesis, investigation_count<3, saturated=False | `investigate_claim` |
| Claim Proposal | 4 | objective | phase=planned, claims_proposed=False | `propose_claims` |
| Routing | 4 | claim | stage=supported, scrutiny_verdict=pass, routing_applied=False | `set_routing_defaults` |
| Scrutiny | 5 | claim | scrutiny_verdict=None | `scrutinise_claim` |
| Verification | 5 | claim | stage=supported, adversarial_checked=False | `adversarial_search` |
| | 5 | claim | stage=supported, convergence_checked=False | `assess_convergence` |
| | 5 | claim | stage=supported, deductive_checked=False | `validate_deductively` |
| | 5 | claim | stage=supported, computational_checked=False | `verify_computationally` |
| | 5 | claim | stage=supported, contrastive_checked=False | `contrastive_evaluation` |
| | 5 | claim | stage=supported, consistency_checked=False | `cross_claim_consistency` |
| Analysis | 5 | claim | argument_analyzed=False, scrutiny=pass | `analyze_argument` |
| Promotion | 6 | claim | stage=hypothesis, scrutiny_verdict=pass | `promote_claim` |
| | 6 | claim | stage=supported, all 7 track flags=True | `promote_claim` |
| | 6 | claim | stage=provisional, evidence_count≥3 | `promote_claim` |
| | 6 | claim | stage=robust, evidence_count≥3 | `promote_claim` |
| Demotion | 6 | claim | scrutiny_verdict=fail, stage≠hypothesis | `demote_claim` |
| Prediction | 7 | claim | stage=robust, predictions_generated=False | `generate_prediction` |
| Synthesis | 7 | objective | phase=claims_done, snapshot_id=None | `freeze_snapshot` |
| | 8 | snapshot | final, artefact_id=None | `synthesize_report` |
| Decision | 9 | claim | stage=actionable, decision_recorded=False | `record_decision` |

Verification track patterns are subject to runtime routing filtering: the scheduler removes work items for SKIP tracks and checks SECONDARY conditions before enqueueing work (see Question-Type Routing above).

---

## Implementation Architecture

### Two Code Layers

| Layer | Location | Responsibility |
|---|---|---|
| **Layer 1: Library** | `packages/epistemic/src/epistemic/` | Framework-agnostic. Entities, operations, patterns, gates, adapters, agents (Python-native), routing, dedup, confidence, runner. Self-contained package (`pip install mosaic-epistemic[llm]`). |
| **Layer 4: Application** | `src/mosaic/epistemic/` | CLI integration with Mosaic SDK, display formatting. Thin wrapper that delegates to the package. |

### Agents

32 Python-native agent definitions, all narrow: each agent has a maximum of 7 output fields with no nested object lists. Agents are registered in `AGENT_REGISTRY` via `register_agent()` at import time, grouped by domain module:

- **Preplanning** (4): `epistemic_clarify_question`, `epistemic_classify_question`, `epistemic_conceptual_analysis`, `epistemic_formulate_query`
- **Evidence** (5): `epistemic_extract_evidence`, `epistemic_assess_evidence`, `epistemic_assess_evidence_quality`, `epistemic_extract_assertion`, `epistemic_screen_relevance`
- **Verification** (12): `epistemic_draft_claim`, `epistemic_identify_single_issue`, `epistemic_identify_testable_aspect`, `epistemic_investigate_claim`, `epistemic_deductive_validation`, `epistemic_verify_computationally`, `epistemic_analyze_argument`, `epistemic_contrastive_evaluation`, `epistemic_cross_claim_consistency`, `epistemic_generate_counterquery`, `epistemic_evaluate_counterargument`, `epistemic_check_pairwise_independence`
- **Synthesis** (5): `epistemic_validate_answer`, `epistemic_write_answer`, `epistemic_resolve_uncertainty`, `epistemic_record_decision`, `epistemic_classify_evidence_domain`
- **Prediction** (3): `epistemic_classify_prediction`, `epistemic_define_falsification`, `epistemic_specify_prediction`
- **Judge** (2): `epistemic_judge_evidence` (supports/contradicts/no_bearing), `epistemic_judge_independence` (binary independence)
- **Similarity** (1): `epistemic_validate_group` (generic group validation for dedup)
- **Output models** (1 module): Shared Pydantic output models in `agents/output_models.py`

### Operations

26 operation classes registered in `OPERATION_CLASSES`, each inheriting from `BaseOperation`:

| Operation | Entity | Effect |
|---|---|---|
| `ClarifyQuestionOperation` | objective | Disambiguate, set phase=clarified |
| `ClassifyQuestionOperation` | objective | Set question_type (7 types), deterministic routing follows |
| `ConceptualAnalysisOperation` | objective | Define terms, set phase=analyzed |
| `PlanTaskOperation` | objective | Create evidence stubs, set phase=planned |
| `ProposeClaimsOperation` | objective | Deduplicate evidence (HDBSCAN), propose claims from top-K representatives |
| `ExtractEvidenceOperation` | evidence | Fill extracted_content from source |
| `ScrutiniseClaimOperation` | claim | Set scrutiny_verdict, may create uncertainties; saturation check after re-scrutiny |
| `InvestigateClaimOperation` | claim | Create targeted evidence stubs, reset verdict |
| `SetRoutingDefaultsOperation` | claim | Pre-mark skipped track flags based on question type routing (deterministic, no LLM) |
| `AdversarialSearchOperation` | claim | Evaluate counterarguments, set adversarial_balance |
| `AssessConvergenceOperation` | claim | Assess cross-domain convergence |
| `ValidateDeductivelyOperation` | claim | Check logical soundness |
| `VerifyComputationallyOperation` | claim | Verify quantitative predictions |
| `ContrastiveEvaluationOperation` | claim | Pairwise comparison of competing claims |
| `CrossClaimConsistencyOperation` | claim | Pairwise conflict check between claims |
| `AnalyzeArgumentOperation` | claim | Analyze argument structure, premises, fallacies |
| `PromoteClaimOperation` | claim | Advance stage via deterministic gate check (routing-aware thresholds) |
| `DemoteClaimOperation` | claim | Lower stage, reset all 7 verification flags + routing_applied + saturated |
| `ResolveUncertaintyOperation` | uncertainty | Set resolution text |
| `DeduplicateConcernsOperation` | objective | Batch dedup buffered remaining concerns |
| `GeneratePredictionOperation` | claim | Create testable predictions |
| `RecordDecisionOperation` | claim | Create decision entity |
| `InvalidateEvidenceOperation` | evidence | TMS: cascade evidence invalidation to dependent claims |
| `RevalidateClaimOperation` | claim | TMS: re-validate claim stage gate after evidence invalidation |
| `FreezeSnapshotOperation` | objective | Create immutable state snapshot |
| `SynthesizeReportOperation` | snapshot | LLM writes prose; code assembles report + trace map |

### Key Implementation Details

**Adapters** normalize agent output. The runner validates output against the agent definition's `output_model`, returning typed Pydantic models. Adapters transform these into operation-specific dataclasses via direct field access — field mismatches surface as `AttributeError`, not silent wrong defaults.

**Code-driven synthesis**: `SynthesizeReportOperation` loads entities from the snapshot, calls the agent for prose only, and assembles the structured report and trace mapping deterministically in code. The LLM writes; the code structures.

**Cycling reset**: When `DemoteClaimOperation` fires, it clears `scrutiny_verdict` and resets all seven verification booleans, `routing_applied`, and `saturated` to False, forcing full re-evaluation from scratch.

**TMS operations**: `InvalidateEvidenceOperation` and `RevalidateClaimOperation` implement Doyle's truth maintenance. When evidence is invalidated (e.g., retracted source), cascading invalidation marks dependent claims for revalidation. `RevalidateClaimOperation` re-checks stage gates using `validate_current_stage()` and demotes claims that no longer meet their current stage requirements.

### Code Structure

```
packages/epistemic/src/epistemic/
├── entities/                      # Entity models (7 types)
├── agents/                        # Python agent definitions + output models
│   ├── __init__.py                # AGENT_REGISTRY, register_agent()
│   ├── output_models.py           # Shared Pydantic output models
│   ├── preplanning.py             # clarify, classify, conceptual, formulate
│   ├── evidence.py                # extract, assess, quality
│   ├── verification.py            # scrutiny, investigation, deductive, adversarial, etc.
│   ├── synthesis.py               # validate_answer, write_answer, domain classification
│   ├── uncertainty.py             # extract_assertion, screen_relevance
│   ├── judge.py                   # judge_evidence, judge_independence
│   └── similarity.py             # validate_group
├── validation/                    # Gate validators, output validators, traceability
├── operations/                    # 26 operation classes (split by pipeline phase)
│   ├── __init__.py                #   re-exports, OPERATION_CLASSES, create_operations()
│   ├── base.py                    #   protocols, BaseOperation, constants
│   ├── preplanning.py             #   clarify, classify, analyze, plan
│   ├── claims.py                  #   evidence selection + claim proposal + inline judge
│   ├── evidence.py                #   extract + quality scoring + inline judge
│   ├── scrutiny.py                #   scrutinise claim
│   ├── verification.py            #   adversarial, convergence, deductive, computational
│   ├── stage_management.py        #   promote/demote
│   ├── uncertainty.py             #   resolve uncertainty
│   ├── synthesis.py               #   freeze snapshot (with caveat dedup) + synthesize report
│   ├── analysis.py                #   argument, contrastive, consistency
│   ├── investigation.py           #   investigate, predict, decide
│   ├── belief_maintenance.py      #   invalidate, revalidate, routing defaults
│   └── concerns.py                #   batch dedup of remaining concerns
├── patterns.py                    # 30 pattern definitions + PatternScheduler with routing filter
├── gates.py                       # StageGate definitions + validate_promotion() + TMS validate_current_stage()
├── routing.py                     # Question-type routing table, provider selection
├── judge.py                       # Central judge module: judge_evidence(), judge_independence()
├── similarity.py                  # Shared embed-compare-group utility (cosine, Union-Find, medoid, validation)
├── dedup.py                       # HDBSCAN evidence deduplication
├── embeddings.py                  # Ollama embedding client for dedup clustering
├── confidence.py                  # Answer confidence (checklist) + posterior P(Y) scoring
├── adapters.py                    # Agent output normalization
├── repository.py                  # Entity CRUD interface
├── storage.py                     # StorageBackend protocol
├── runner.py                      # DefaultAgentRunner (PydanticAI agent execution)
├── operations_runner.py           # Scheduler loop + confidence computation
├── primitives.py                  # Shared enums and dataclasses
├── quality.py                     # Bibliometric quality scoring
├── adversarial_balance.py         # Balance calculation (deterministic)
├── adversarial_evaluator.py       # Counterargument evaluation
├── adversarial_query_generator.py # Search query generation
├── domain_classifier.py           # Evidence domain classification
├── domain_distance.py             # Inter-domain distance (deterministic)
├── convergence_detector.py        # Convergence detection (deterministic)
├── preflight.py                   # Preflight health checks (HealthCheckable protocol)
├── evidence_gathering.py          # Evidence provider routing
├── evidence_router.py             # Provider dispatch
├── config.py                      # ResearchConfig
├── result_models.py               # Typed result models
├── synthesis.py                   # Result synthesis
├── trace.py                       # Reasoning trace building
├── stats.py                       # Run statistics
├── console.py                     # Rich console output
├── cli_handlers.py                # Async handlers for CLI commands
├── cli.py                         # Standalone CLI entry point (mosaic-epistemic)
├── html_report.py                 # HTML report generation
├── report_generator.py            # Report formatting
└── trace_renderers.py             # Trace visualization
```

### Evidence Providers

| Provider | Type | Quality Signal |
|---|---|---|
| **Web Search** | General | Content assessment (focused agent) |
| **OpenAlex** | Academic | Bibliometric (deterministic) + content assessment |
| **Monarch** | Biomedical | Database authority (deterministic) + content assessment |
| **Knowledge Sources** | Biomedical DBs | Database authority (deterministic) |

Provider selection is deterministic: `select_providers()` in `routing.py` matches domain keywords from the question's key terms against a `DOMAIN_PROVIDER_MAP` (e.g., "gene" → pubmed + monarch + open_targets, "study" → openalex + pubmed). Web search is always included. Investigation stubs default to web_search. No LLM call is involved in provider selection.

### Extension Guide

**Add a new operation**: Create `BaseOperation` subclass → add Pattern to `WORK_PATTERNS` → register in `OPERATION_CLASSES` → create agent manifest → (optional) add adapter.

**Add an evidence provider**: Implement `EvidenceGatherer` protocol → wire into `DefaultEvidenceGatherer`. Implement `check_health()` → `async def check_health(self) -> CheckResult` so preflight discovers the provider automatically (see Preflight section below).

**Modify stage gates**: Edit `STAGE_GATES` dict in `gates.py`.

**Add an entity type**: Create entity class → register in `ENTITY_CLASSES` → add repository methods.

### Preflight Health Checks

Misconfiguration (wrong model ID, SearXNG not running, external API down) is not detected until mid-run without preflight validation. The preflight system provides fail-fast checks **before** an expensive research run starts.

**Design**: Provider-advertised health checks. Each component implements a `check_health()` method; the preflight system discovers and calls these. No hardcoded provider-specific logic in the preflight module.

**Protocol**:

```python
from epistemic.preflight import HealthCheckable, CheckResult

@runtime_checkable
class HealthCheckable(Protocol):
    async def check_health(self) -> CheckResult: ...
```

**Components that implement `check_health()`**:
- `DefaultAgentRunner` — tests LLM connectivity with a minimal inference call
- `WebSearchGatherer` — tests SearXNG reachability
- `MonarchProvider` — tests Monarch API search endpoint
- `OpenAlexProvider` — tests OpenAlex API works endpoint

**Adding preflight to a new provider**: Just implement `check_health()` on the class. The preflight function discovers it via `hasattr(provider, 'check_health')` — no registration needed.

```python
class NewDatabaseProvider:
    async def gather(self, query: str) -> list[GatheredEvidence]: ...

    async def check_health(self) -> CheckResult:
        from epistemic.preflight import CheckResult
        # Test own API endpoint, return pass/fail
        ...
```

**CLI**: `mosaic-epistemic preflight [--model MODEL] [--providers biomedical] [--verbose]`

**Python API**:

```python
from epistemic.preflight import preflight

result = await preflight(model="bedrock:claude-haiku-4-5", providers=providers)
if not result.ok:
    sys.exit(1)
```

---

## Observability

### Operation Profiling

After every run, the CLI prints an operation profiling table aggregated from execution step metadata stored during the run. The table shows each operation type, call count, total wall-clock time, and mean time per call. This enables identifying bottlenecks (e.g., adversarial search averaging 8 seconds vs. deductive validation averaging 2 seconds) without instrumenting individual operations.

### Confidence Scoring

Two scores run after the scheduler completes. Both are pure computations — no LLM calls, no trained weights, zero free parameters.

#### Answer Confidence (process completion)

`compute_answer_confidence()` produces an `AnswerConfidenceReport`. It evaluates a checklist of pass/fail checks and converts to a probability via the logistic function:

**Universal checks** (always evaluated):
- `evidence_basis` — at least one active claim has judged, non-invalidated evidence
- `scrutiny_complete` — all active claims have a scrutiny verdict
- `uncertainties_resolved` — no unresolved blocking uncertainties remain
- `belief_maintenance` — no active claims need revalidation

**Routing-dependent checks** (one per PRIMARY verification track for the question type):
- `track:adversarial`, `track:convergence`, `track:deductive`, `track:computational`, `track:contrastive`, `track:consistency` — each checks that all active claims completed the corresponding track

Each check contributes +1 (pass) or -1 (fail) to log-odds. The logistic transform converts to a probability: `confidence = 1 / (1 + exp(-log_odds))`. Classified as high (>=0.75), moderate (>=0.50), low (>=0.25), or insufficient (<0.25).

#### Posterior P(Y) (evidential direction)

`compute_posterior()` produces a `PosteriorReport` for yes/no-style research questions (verificatory, comparative, predictive). Returns None for other question types where a directional answer is not meaningful.

Aggregates evidence direction across all active claims: `log_odds = supporting_count - contradicting_count`, then `P(Y) = 1 / (1 + exp(-log_odds))`. Only representative (not corroborative/deferred), non-invalidated evidence with a directional judgment counts.

### Diagnostic Logging

Evidence selection during deduplication logs cluster counts, representative selections, and corroborative evidence counts at the INFO level. The scheduler logs each operation execution, pattern matching, and budget consumption. All diagnostic output uses standard Python logging, not print statements.

---

## Domain Agnosticism

The seven primitives, five stages, stage gates, verification methods, and uncertainty taxonomy are domain-independent. They apply equally to biomedical research, policy analysis, technology assessment, historical inquiry, and financial analysis. The domain-specific knowledge comes from evidence sources and language models, not from the epistemic framework. The framework provides scaffolding; content fills it.

---

## Usage

For CLI reference, common workflows, Python API, and troubleshooting, see [EPISTEMIC_USAGE.md](EPISTEMIC_USAGE.md).

Quick start:

```bash
# Ask a research question
uv run mosaic epistemic ask "What is spaced repetition and does it work?"

# With all trace visualizations
uv run mosaic epistemic ask "question" --trace all

# Generate HTML report
uv run mosaic epistemic ask "question" --output-path report.html
```

---

## References

1. **Alchourrón, C.E., Gärdenfors, P., and Makinson, D.** (1985). On the Logic of Theory Change: Partial Meet Contraction and Revision Functions. *The Journal of Symbolic Logic*, 50(2), 510-530.

2. **Doyle, J.** (1979). A Truth Maintenance System. *Artificial Intelligence*, 12(3), 231-272.

3. **Lakatos, I.** (1978). *The Methodology of Scientific Research Programmes: Philosophical Papers Volume 1*. Cambridge University Press.

4. **Peirce, C.S.** (1903). Pragmatism as a Principle and Method of Right Thinking: The 1903 Harvard Lectures on Pragmatism. Edited by P.A. Turrisi. SUNY Press, 1997.

---

*"The irritation of doubt causes a struggle to attain a state of belief. I shall term this struggle inquiry." — C.S. Peirce*
