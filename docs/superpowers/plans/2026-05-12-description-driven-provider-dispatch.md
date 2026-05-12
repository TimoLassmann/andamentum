# Description-driven provider dispatch — plan & PRD

**Date:** 2026-05-12  
**Pre-refactor restore point:** `git tag pre-dispatch-refactor` → commit `e30c314`  
**Status:** Planning. No code changes yet.

---

## TL;DR

Replace the current "one upstream formulator agent that knows every provider's query syntax" with **description-driven dispatch**: each provider self-describes in natural language plus a handful of example queries, and one generic agent reads the description at dispatch time to decide whether the provider applies and to construct a query in the provider's native syntax. Adding a new provider becomes a documentation task plus an HTTP wrapper — no agent design, no taxonomy update, no prompt engineering.

The architecture is the same shape that lets tool-using systems (MCP, function calling, agentic dispatch) work over many tools: providers carry schema, not intelligence, and one fixed agent reads schemas at runtime. Adding providers becomes O(provider docs + HTTP wrapper) rather than O(agent design + prompt engineering). The architecture *does not* limit catalogue size, but large-catalogue scaling (>~30 providers) has open second-order interactions — embedding pre-filter behaviour, `corroboration_count` denominator semantics, scrutiny-escalation loop bounds — that this PRD does not pressure-test. Those are explicitly left for a follow-up PR (see §10).

---

## 1. Motivation

### 1.1 What's broken

The current `epistemic.evidence_gathering` and provider system has three problems that compound as the number of providers grows:

1. **A single upstream "formulator" agent must know every provider's query syntax.** It is asked to generate queries in arXiv field-operator syntax, ClinicalTrials.gov `AREA[]` syntax, PubMed MeSH boolean, Open Targets GraphQL-ish, and natural language for web search — all in one agent context. Small LLMs do not reliably master 10 query languages simultaneously. The audit results show this concretely: arXiv had a 0% content-yield rate on dev30 v3, and ClinicalTrials.gov also returned 0% content, because the formulator over-applied fielded operators or asked clinically-irrelevant questions.

2. **Provider selection is currently a binary CLI flag (`--provider all | web_search`).** There is no granular control. To restrict the system to a subset of providers (as we did for dev30 v4 with the "4 working providers" subset), users have to edit the runner script. There is no per-claim provider triage at any layer — the system fans out to all configured providers regardless of relevance.

3. **Provider expertise is split across two locations.** The provider implementation (`gather()` HTTP code) lives in `providers/<name>.py`, while the natural-language description and `query_guidance` (which describe what the provider does and how it should be queried) live in `providers/__init__.py:register_provider()`. The formulator agent uses one; the HTTP layer uses the other. They drift independently.

### 1.2 What the dev30 sequence showed empirically

Three benchmark runs at `n=30`, `K=2`, single-rep:

| run | configuration | epistemic AUC | epistemic ECE |
|---|---|---|---|
| v3 | all 10 providers + web_search | 0.93 | 0.095 |
| v4 | 4 providers only (PubMed, Europe PMC, OpenAlex, web_search) | 0.79 | 0.176 |
| v5 | all 10 providers + web_search (replication of v3) | 0.88 | 0.230 |

The v4 → v5 swap (restoring all providers) recovers most of the AUC. This empirically confirms that the "low-yield" providers (chembl, open_targets, monarch, arxiv, clinicaltrials, cochrane) contribute non-trivial signal at the system level even when their per-evidence directional-judgment count is small — they help with convergence reasoning, IBE candidate enumeration, and cross-domain corroboration.

The implication: dropping providers that "don't pull weight" is the wrong tactic. The right tactic is making each provider work well *when invoked*, and being smart about *when* to invoke each. That requires per-provider query construction tuned to what each provider actually answers — exactly the thing the current upstream formulator does badly.

### 1.3 Why per-provider agents (the obvious fix) don't scale

A natural reaction is to give each problematic provider its own LLM agent that knows its syntax. At 3–5 problematic providers today that would work. At 50 providers it becomes a maintenance nightmare:

- N agents, each with bespoke prompts, schemas, evaluations
- N drift surfaces as APIs change
- N specialist authors required (someone has to know each provider intimately enough to write its agent)
- Cost-of-adding-a-provider includes an LLM engineering pass, not just a Python pass

This proposal advocates the opposite: **the agent count is independent of the provider count**. One generic agent reads provider descriptions at dispatch time. Adding a provider adds a documentation + HTTP wrapper, not an agent.

### 1.4 Why this matches the scaling pattern that works

The same pattern enables tool-using systems to scale to thousands of tools:

- **MCP servers** describe their tools in natural language; clients route at runtime by reading descriptions.
- **OpenAI function calling** uses tool descriptions; the model decides which to call.
- **Anthropic tool use** is the same shape; Claude reads schemas + descriptions and decides.

In none of these does each tool carry its own LLM. The agent is generic; the tools are described. The architectural commitment is that **provider knowledge is captured in prose**, not in a controlled vocabulary, and not in bespoke agents.

---

## 2. Goals and non-goals

### Goals

1. **Provider count scales without architectural cost.** Adding a new provider requires writing a description, example queries, and an HTTP wrapper. No agent design, no taxonomy update, no prompt engineering.

2. **Provider expertise is co-located.** Description, examples, syntax, and HTTP implementation live in one file (or one directory) per provider. Drift is eliminated by construction.

3. **Per-claim provider triage is automatic.** ClinicalTrials.gov should not be queried for "histone H2A.Z in yeast"; the dispatch agent decides this from the provider's description + the claim, with no manual configuration.

4. **Query construction matches each provider's native syntax.** arXiv gets queries that exercise its category and field operators correctly; ClinicalTrials.gov gets `AREA[Intervention]value` syntax when appropriate; PubMed gets MeSH-aware boolean. All from the same generic dispatch agent, just with different provider context.

5. **Domain-general.** The architecture works equally well for biomedical providers (current case) and hypothetical providers in other domains (economics, law, physics, code search, climate data, history) without modification.

6. **Calibration is preserved or improved on dev30.** A v6 run against this architecture should produce AUC ≥ 0.85, Brier ≤ 0.20, ECE ≤ 0.20 — comparable to or better than v5.

### Non-goals

1. **This is not the domain-modules refactor.** That's a separate, larger architectural change (grouping providers by domain and giving domains internal query strategies). Description-driven dispatch is upstream of that; domain modules could later be layered on top, but they're out of scope here.

2. **This is not a fix for the Category 2 problem (structured-reference providers).** ChEMBL, Open Targets, and Monarch return reference data (compound IDs, target-disease evidence rows, entity metadata) rather than assertion evidence. The judgment pipeline treats these as no_bearing because they're not assertions. This refactor improves how *queries reach* those providers but does not change how their output is *consumed*. A separate "context channel" for reference data is future work.

3. **No change to the IBE chain, K-agreement, or any other downstream architecture.** This refactor is scoped to evidence gathering. The seven primitives, 23-node graph, posterior computation, framing-tie cap, K-agreement check all remain identical.

4. **No code changes in this PRD.** This document is design only. Implementation is phased and follows the plan in Section 6.

---

## 3. Background — what exists today

### 3.1 Current dispatch flow

```
claim
  ↓
[PlanTaskOperation]
  ↓ generates a list of providers to query (via planner agent)
  ↓
[Per-provider query formulation]
  ↓ epistemic_formulate_search_queries agent
  ↓ produces 1 query string per provider, in that provider's syntax
  ↓
[Provider.gather(query)]
  ↓ each provider runs its own HTTP call
  ↓
GatheredEvidence list
```

The formulator agent is given the claim, the list of selected providers, and a "query guidance" string per provider (from `register_provider()`). It then emits one query string per provider, attempting to match each provider's syntax.

### 3.2 Where the current architecture concentrates knowledge

| Knowledge | Where it lives today |
|---|---|
| Provider's natural-language description | `providers/__init__.py:register_provider(description=...)` |
| Provider's query syntax guidance | `providers/__init__.py:register_provider(query_guidance=...)` |
| Provider's example queries | `providers/__init__.py:PROVIDER_EXAMPLES` |
| Provider's HTTP call | `providers/<name>.py:gather()` |
| Multi-syntax query formulation logic | The formulator agent prompt + LLM |

Five different places, all of which need to be updated together when a provider changes.

### 3.3 What works in the current design

- The `GatheredEvidence` return shape is good and shouldn't change.
- The `gather(query: str) -> list[GatheredEvidence]` interface is clean for providers that take string queries.
- The HTTP-call layer in `providers/<name>.py` is well-isolated and tested.
- The K=2 provider tournament for cross-domain convergence is sound.
- Embedding-based semantic routing exists in skeletal form (provider examples are embedded).

These all carry over to the new design.

---

## 4. Proposed architecture

### 4.1 The provider definition

Under the new architecture, a provider is a single Python file containing:

1. **A natural-language description** — what the provider covers, what it answers well, what it doesn't, written for a model to read.
2. **A list of example queries** — pairs of (natural-language claim or question, provider-native query or `None`). These serve as in-context examples for the dispatch agent, including negative examples where the provider should abstain.
3. **A `gather(query: str) -> list[GatheredEvidence]` async method** — the HTTP-call layer. Unchanged from today.
4. **An `output_kind` discriminator** — one of `"assertion_evidence"`, `"structured_record"`, `"trial_registration"`, `"compound_data"`. Read by the downstream judgement layer to decide how to score the returned items. `assertion_evidence` items go through the existing `supports / contradicts / no_bearing` judge agent. `structured_record` (ChEMBL, Open Targets, Monarch) and other non-assertion items are tagged with their kind and excluded from the supports/contradicts axis — they're available to the synthesis writer as context but don't drive the posterior on their own. This 5-line addition closes most of the Category-2 gap (§8.4) without expanding scope into the full context-channel refactor.
5. **A `provider_contract_version: int = 1` field** — versions the contract so future providers needing structured query arguments (e.g. typed filter dicts, date ranges, lists of compound IDs) can bump to version 2 and signal to the dispatch agent that a different prompt template applies. The shape of v2 is not designed in this PRD; what matters is that v1 doesn't paint v2 into a corner.
6. **Optional: an `independence_group` tag** — a small free-form string (e.g., `"biomedical_literature"`, `"clinical_registry"`, `"chemistry_structured"`) used by the convergence detector and the domain-diversity gate to know which providers are correlated sources. **The dispatch agent does NOT read this**; it's purely for downstream independence reasoning. May be omitted; defaults to the provider's own name (i.e., treated as a singleton independence class).

That's the entire provider contract. No taxonomies for dispatch. No schemas the dispatch agent has to parse. No per-provider agent.

Concretely:

```python
# providers/arxiv.py

class ArXivProvider:
    description = """
    arXiv preprint server. Indexes over 2 million scholarly preprints across
    physics, mathematics, computer science, quantitative biology, quantitative
    finance, statistics, electrical engineering, and economics.

    Strong for: theoretical work, mathematical formalism, machine learning
    methodology, computational methods, foundational quantitative results.

    Weak for: clinical trials, wet-lab experimental biology, epidemiology,
    qualitative social science. Most biomedical claims are not deposited here.

    Native search syntax: field operators (cat:, ti:, abs:, au:, all:),
    boolean (AND, OR, ANDNOT), and double-quoted phrases. Categories use
    the arXiv taxonomy: q-bio.GN (genomics), cs.LG (machine learning),
    stat.ML, math.ST, hep-ph, cond-mat, etc. Field-prefixed queries should
    not be wrapped in all: — the search_query parameter accepts them
    directly.
    """

    query_examples = [
        # (claim or question, native query)
        ("transformer attention mechanism scales with sequence length",
         "cat:cs.LG AND ti:transformer AND (ti:attention OR ti:scaling)"),
        ("does CRISPR Cas9 work in non-dividing cells",
         "cat:q-bio.GN AND ti:CRISPR AND (abs:non-dividing OR abs:G0)"),
        ("BERT pre-training objective comparison",
         '(cat:cs.CL OR cat:cs.LG) AND ti:BERT AND ti:"pre-training"'),
        # Negative example: claim arXiv won't help with
        ("does atorvastatin reduce mortality in heart failure", None),
    ]

    output_kind = "assertion_evidence"
    independence_group = "preprint_archive"
    provider_contract_version = 1

    async def gather(self, query: str) -> list[GatheredEvidence]:
        # Unchanged HTTP-call layer
        ...
```

The `query_examples` list demonstrates good queries AND shows what the provider can't help with (via `None` queries). This becomes the in-context training data for the dispatch agent.

For comparison, here's how ChEMBL would declare itself — same shape, but with `output_kind` reflecting that its records are structured reference data rather than literature assertions:

```python
class ChEMBLProvider:
    description = """
    ChEMBL bioactivity database. Returns structured records about
    compound–target interactions: IC50/EC50/Ki values, mechanism of
    action, target identifiers, compound identifiers (SMILES, ChEMBL ID),
    and physicochemical properties.

    Strong for: drug-target affinity claims, mechanism-of-action lookups,
    compound identification.

    Weak for: clinical outcomes (use ClinicalTrials.gov), literature
    assertions (use PubMed/Europe PMC), gene-disease associations (use
    Open Targets or Monarch).

    Returns structured reference data, NOT prose assertions. Items are
    consumed by the synthesis writer as context for compound-related
    claims but do not directly feed the supports/contradicts axis.
    """

    output_kind = "structured_record"
    independence_group = "chemistry_structured"
    provider_contract_version = 1

    query_examples = [
        ("what is the IC50 of imatinib against BCR-ABL?",
         "imatinib BCR-ABL"),
        ("which HDAC inhibitors have sub-nanomolar Ki?",
         "HDAC inhibitor"),
        # Out-of-domain — should abstain
        ("does atorvastatin reduce all-cause mortality?", None),
        ("are open access papers more cited?", None),
    ]
    ...
```

Two distinct providers, same contract shape. The judge reads `output_kind` to know how to score the returned items — it doesn't try to evaluate "compound MW=349.41" as a supports/contradicts assertion.

### 4.2 The dispatch agent

A single generic agent, invoked once per (claim, provider candidate) pair. Its interface:

```python
@dataclass
class DispatchResult:
    queries: list[str]      # 0+ native-syntax queries; empty list = abstain
    reasoning: str          # one-sentence justification (audit trail)
    confidence: float       # 0–1, how confident the agent is in the routing

async def formulate_provider_query(
    claim: str,
    provider_description: str,
    provider_examples: list[tuple[str, str | None]],
    *,
    model: str,
) -> DispatchResult:
    """Decide whether the provider can help and construct one or more queries."""
```

**Why `queries: list[str]` not `query: str`.** Some providers benefit from multiple queries against the same claim — PubMed in particular often needs both a MeSH-anchored boolean (`("Heart Failure"[MeSH] AND "Atorvastatin"[MeSH])`) and a free-text fallback (`heart failure atorvastatin mortality`) because MeSH coverage is incomplete for recent or non-MeSH-indexed papers. The legacy formulator could only produce one query per (claim, provider). The new contract allows 0+. An empty list is the unambiguous abstain signal; 1 query is the common case; 2+ is permitted with a per-provider budget cap (default 2, configurable). The orchestrator runs `gather(q)` for each `q` in `queries` and merges the results before deduplication.

**Edge case: providers with ID-resolution needs.** Open Targets really wants a target/disease ID, not a name; Monarch and ChEMBL similarly. Today the formulator emits a name and the provider does the lookup internally inside its `gather()`. Under the new architecture there are two viable approaches: (a) include resolved-ID examples in the provider's `query_examples` so the dispatch agent learns the right output shape from in-context examples; (b) add an optional `Provider.resolve_entities(claim) -> dict[str, str]` hook called before the dispatch agent runs, with the resolved IDs included in the dispatch agent's context. The PRD picks (a) — keeps the contract surface narrow, defers ID-resolution complexity until a provider genuinely can't be served by in-context examples. (b) is reserved for the contract-version-2 bump if needed.

**Edge case: decomposed claims.** Research-mode claims decompose into sub-claims via `DecomposeQuestionOperation`. Dispatch runs per (sub-claim, provider), not per (objective, provider). At K=10 candidates × 4 sub-claims = 40 dispatch calls per objective, parallelisable but not free. This is the expected cost; the PRD does not propose batching across sub-claims because cross-claim batching would weaken per-claim attribution. The cost is bounded by `K × n_sub_claims × small-model-call-cost`.

The agent's prompt template (sketched):

```
You are constructing a query for an evidence database. Read the provider's
description and example queries carefully, then either construct a query
that matches the provider's syntax, or return null if the provider cannot
help with this claim.

PROVIDER DESCRIPTION:
{description}

EXAMPLES (what good queries look like for this provider):
{examples_block}

CLAIM:
{claim}

Decide:
1. Does this provider's coverage plausibly include evidence relevant to
   this claim? Use the description's "weak for" / "strong for" guidance.
2. If yes, construct a query in the provider's native syntax that follows
   the patterns shown in the examples. Match the example style.
3. If no, return null and explain why in one sentence.

Output: {"query": "...", "reasoning": "...", "confidence": 0.0-1.0}
```

The agent is invoked **once per candidate provider**. It receives only that one provider's description and examples — never the full catalogue. This keeps context small and keeps the agent's task narrow: "construct one query for one provider, or admit you can't."

### 4.3 The candidate selection layer (embedding pre-filter)

At small catalogue sizes (~10 providers), calling the dispatch agent on every provider per claim is fine — and turns out to be the right default for two empirically-grounded reasons. At larger catalogue sizes (~100+ providers) a pre-filter becomes useful. The implementation supports both.

The pre-filter:

1. At provider registration, embed `provider.description` once (cached).
2. At dispatch time, embed the claim.
3. Compute cosine similarity between the claim and every registered provider's description embedding.
4. Take the top K by similarity, where K is configurable.
5. Invoke the dispatch agent only on these K candidates.

This is essentially RAG-over-tools — a well-understood pattern.

**Default K at current scale: `K = len(providers)` (no pre-filter).** The dev30 v3→v4 ablation showed empirically that "low-yield" providers contribute calibration signal *via their abstention pattern* (the convergence detector and IBE chain benefit from a provider correctly returning empty for an out-of-domain claim, not just from providers returning evidence). If the pre-filter silently prunes a provider before dispatch, the dispatch agent never gets to abstain — the system loses that calibration signal. At 10 providers, with parallel dispatch calls, the cost of running all 10 is comparable to running 8: ~1 LLM round trip wall-clock either way, and the marginal LLM-call cost is negligible against the calibration risk. So the Phase 4 default is no pre-filter.

**When to activate the pre-filter:** when the catalogue exceeds ~30 providers AND there's empirical evidence that running all of them per claim is wasteful. Both gates need to be true. Activation is a separate PR with its own validation — including an offline test that the chosen K achieves 100% recall of the "should be dispatched" providers on a fixed evaluation set. The architecture exposes `select_candidates_by_embedding(top_k=...)` for this future use; the Phase 1–5 work just sets `top_k = len(providers)` and treats the function as a pass-through.

**If the embedding service is unavailable** (Ollama down, network failure), the pre-filter falls back to "dispatch on all providers." Empty candidate sets are not acceptable — that would silently kill evidence gathering for the entire claim. This fallback is part of the contract regardless of catalogue size.

### 4.4 The orchestration layer

The current `PlanTaskOperation` + per-provider `extract_evidence` calls collapse into:

```python
async def gather_evidence(claim: str, deps: EpistemicDeps) -> list[Evidence]:
    # 1. Embed claim, embed all provider descriptions (cached).
    candidate_providers = await select_candidates_by_embedding(
        claim, deps.providers, top_k=deps.dispatch_k
    )
    
    # 2. Run dispatch agent on each candidate, in parallel.
    dispatch_results = await asyncio.gather(*[
        formulate_provider_query(claim, p.description, p.query_examples, model=deps.model)
        for p in candidate_providers
    ])
    
    # 3. Filter to providers that returned a query (i.e., triage said yes).
    actionable = [
        (p, r.query) for p, r in zip(candidate_providers, dispatch_results)
        if r.query is not None
    ]
    
    # 4. Run gather() on each, in parallel.
    evidence_batches = await asyncio.gather(*[
        p.gather(q) for p, q in actionable
    ])
    
    return [e for batch in evidence_batches for e in batch]
```

Four steps. Two LLM-call batches (embedding similarity is a cheap precomputed dot product). Linear in K, where K is bounded.

### 4.5 What goes away

- **Three legacy agents.** `epistemic_select_provider`, `epistemic_rank_providers` (its preplanning call site; the scrutiny-escalation call site gets a deterministic replacement), and `epistemic_formulate_query` — all subsumed by the new generic dispatch agent. Deleted in Phase 5.
- **`PROVIDER_EXAMPLES`** (the dead global in `providers/__init__.py:473`). Deleted in Phase 1.
- **The `--provider all | web_search` CLI flag.** Replaced by `--dispatch-k N` (default = catalogue size at Phase 4) and `--provider-allowlist a,b,c` / `--provider-blocklist a,b,c` for explicit overrides.
- **The `query_guidance` parameter to `register_provider`.** Folded into the provider's `description` and `query_examples`. (Kept in the shim during Phases 1–4 so legacy formulator quality doesn't degrade during migration.)

### 4.6 What stays — and what changes about `source_type` consumption

The core interface preserves:

- `gather(query: str) -> list[GatheredEvidence]` interface and all existing HTTP-call code.
- `GatheredEvidence` shape.
- The K=2 provider-distinctness concept (now expressed as a constraint on dispatch decisions for convergence-priority code paths).
- The downstream pipeline (judgment, scrutiny, IBE chain, K-agreement, synthesis) is untouched.

What changes is how `source_type` is consumed by independence-aware downstream code. Today, three call sites treat `source_type` as a proxy for source independence — *incorrectly*, since two providers in the same content domain (e.g., PubMed + Cochrane, both biomedical-literature) currently count as two independent sources. After the refactor, these call sites consume `provider.independence_group` instead:

| Call site | Today | Post-refactor |
|---|---|---|
| `gates.py:180` — domain-diversity gate | `len({e.source_type for e in evidence}) >= 2` | `len({provider.independence_group for e in evidence}) >= 2` |
| `dedupe_evidence.py:150` — `corroboration_count` | `len({distinct source_types in duplicate group})` | `len({distinct independence_groups in duplicate group})` |
| Reporters / synthesis traces | display raw `source_type` | display raw `source_type` (unchanged — the displayed identity matters for audit) |

**This is a behavioural correction, not just a code change.** Pre-refactor, PubMed + Cochrane returning the same paper counted as 2 independent corroborations. Post-refactor, they correctly count as 1 (both `biomedical_literature`). The independence-aware gates become stricter — a claim now needs evidence from *genuinely different* independence groups to pass the domain-diversity check.

**This is also a baseline shift between v5 and v6 that has nothing to do with dispatch quality.** Phase 4 acceptance must distinguish (a) calibration changes due to better dispatch and (b) calibration changes due to stricter independence accounting. The Phase 4 acceptance test (Section 6) is adjusted accordingly: AUC ≥ 0.85 (vs v5's 0.88) is the acceptance threshold, not "match or beat v5" — because some of v5's AUC was the system over-counting correlated sources as independent.

---

## 5. Architecture in action: existing + hypothetical providers

### 5.1 Existing 10 providers under the new design

| Provider | New description writes about | Independence group |
|---|---|---|
| **PubMed** | Biomedical literature, MeSH-indexed; strong for clinical, basic biology, drug studies; uses `[tag]` and boolean | `biomedical_literature` |
| **Europe PMC** | Biomedical full-text + abstracts, includes preprints; supports field queries | `biomedical_literature` |
| **OpenAlex** | Cross-disciplinary scholarly metadata; strong for citation graphs; uses filter syntax | `general_scholarly` |
| **bioRxiv** | Biology and medicine preprints; PubMed E-utilities-indexed | `preprint_archive` |
| **arXiv** | Physics, math, CS, q-bio preprints; field-prefixed queries (`cat:`, `ti:`) | `preprint_archive` |
| **ClinicalTrials.gov** | Trial registry; `AREA[Field]` syntax; only contains human trials | `clinical_registry` |
| **Cochrane (PubMed-filtered)** | Systematic reviews on specific clinical questions; narrow scope | `systematic_reviews` |
| **ChEMBL** | Compound bioactivity database; returns structured records (IC50, target IDs); query by compound or target name | `chemistry_structured` |
| **Open Targets** | Target-disease evidence; GraphQL-like; returns confidence-scored evidence rows | `genetics_structured` |
| **Monarch** | Gene-disease and gene-phenotype associations from curated databases | `genetics_structured` |
| **Web search** | General web search; natural-language queries; broad recall, variable quality | `general_web` |

Each gets a self-contained description file under the new architecture. None requires a controlled vocabulary or schema definition.

### 5.2 Hypothetical providers from other domains (to demonstrate generality)

To show the architecture is domain-general, here are providers we could add tomorrow without changing any system code:

**FRED (economics) — Federal Reserve Economic Data:**
> Time-series economic indicators from the St. Louis Federal Reserve. Strong for: US economic claims (GDP, unemployment, inflation, interest rates), historical economic time series since the 1940s, monetary policy. Weak for: causal claims, micro-economic studies, non-US economies before 1990. Native syntax: search by series ID (e.g., GDPC1) or natural language for series discovery.

**CourtListener (law) — Free Law Project case law:**
> US federal and state court opinions; over 9 million cases. Strong for: precedent claims, US constitutional/statutory interpretation, federal regulatory cases. Weak for: international law, scholarly legal theory, non-US cases. Native syntax: boolean keyword search with field operators for court, judge, date range, citation.

**NASA ADS (astronomy/physics) — Astrophysics Data System:**
> Astrophysics and physics literature; ~17 million records. Strong for: astronomy observations, cosmology theory, planetary science, instrumentation. Weak for: ground-based experimental physics, biomedical physics applications. Native syntax: ADS-specific field operators (`author:`, `bibstem:`, `year:`, `keyword:`).

**OpenFEMA (disasters) — FEMA Open API:**
> US disaster declarations, individual assistance, public assistance records since 1953. Strong for: disaster timing/scope claims, federal emergency response, regional impact. Weak for: forecasting, root-cause analysis. Native syntax: OData-style query parameters.

**HathiTrust (humanities) — Digital library of scanned books:**
> 17 million scanned volumes, mostly pre-1950 books and US government documents. Strong for: historical claims, primary-source citations, biographical research. Weak for: contemporary topics, peer-reviewed scholarship. Native syntax: catalog search with Solr-like operators.

Each of these would be a single new file in `providers/`. The dispatch agent would learn what they cover by reading the descriptions — no system change required.

This is the test of generality: a researcher with a question about disaster timing, presidential precedents, or 19th-century literary criticism could plug in the relevant provider and have it work within the same pipeline that today serves biomedical claims.

### 5.3 What this changes about provider routing

A claim about "atorvastatin reduces mortality in heart failure":

- Embedding similarity: PubMed, Europe PMC, ClinicalTrials.gov, Cochrane all rank high; arXiv ranks low; FRED, CourtListener, NASA ADS all rank very low.
- Top-K=8 selected: PubMed, Europe PMC, OpenAlex, ClinicalTrials.gov, Cochrane, bioRxiv, Open Targets, web search.
- Dispatch agent invokes per provider:
  - PubMed: "yes, build MeSH-aware query: `(atorvastatin[MeSH] OR atorvastatin) AND (heart failure[MeSH] OR HF) AND mortality`"
  - ClinicalTrials.gov: "yes, build `AREA[Intervention]atorvastatin AND AREA[Condition]heart failure AND AREA[OutcomeMeasure]mortality`"
  - Cochrane: "yes, build clinical-question-shaped query"
  - bioRxiv: "no, this is established clinical pharmacology; preprints won't be the primary source"
  - Open Targets: "no, this is a clinical outcome question, not a target-disease association question"
  - web search: "yes, plain natural language"
- Six providers get queries; two abstain. Each query is in the correct native syntax. The system has natural per-claim provider routing without any hand-coded routing rules.

Compare to the current system, where all 10 providers + web search receive the same fielded query (or a poorly-translated approximation) from the upstream formulator. The new architecture is strictly more correct and strictly more efficient.

---

## 6. Migration plan

The refactor lands in five phases, each independently shippable. The current system continues to work throughout — the new dispatch path is built alongside, validated, and only then swapped in.

### Phase 1 — Provider self-description (no behavioural change)

Move each provider's `description`, `query_guidance`, and example queries from `providers/__init__.py:register_provider(...)` calls into the provider's own file as class attributes (`description`, `query_examples`, `output_kind`, `independence_group`, `provider_contract_version`).

**Three existing agents and their fates during the refactor.** This PRD's "the legacy formulator" shorthand glosses over the fact that the current evidence-gathering path uses *three* distinct LLM agents, not one. Each has a different scope and a different post-refactor disposition:

| Agent | Current role | Disposition under refactor |
|---|---|---|
| `epistemic_select_provider` | Per-provider relevance check — "is this provider relevant to the claim?" Called from `PlanTaskOperation` in `preplanning.py`. | **Subsumed by the new dispatch agent's abstain capability.** Deleted in Phase 5. |
| `epistemic_rank_providers` | K=2 provider tournament — picks distinct top providers from a shrinking candidate pool. Called from `PlanTaskOperation` AND from `InvestigateClaimOperation` (scrutiny escalation, `investigation.py:298–335`). | **Subsumed in the planner call site.** The dispatch agent's per-provider triage replaces the tournament logic. **For the scrutiny-escalation second call site, replaced by an "embedding-distance to claim + has-not-been-queried" deterministic picker, OR by re-running dispatch with provider-allowlist for unused providers.** Decided in Phase 2. |
| `epistemic_formulate_query` | Per-(sub-claim, provider) query construction — emits one query string per pair. | **Replaced by the new dispatch agent.** Deleted in Phase 5. |

The PRD's earlier "the legacy formulator" was an oversimplification. The dispatch agent replaces all three responsibilities in Phase 2; the legacy agents survive Phase 1–3 in parallel and are deleted in Phase 5.

**`PROVIDER_EXAMPLES` is dead code.** The global dict in `providers/__init__.py:473` has no live consumer — the file that imported it (`provider_routing.py`) was deleted in an earlier refactor; only the `.pyc` remains in `__pycache__`. Phase 1 *deletes* this global rather than migrating it. The "examples" that *do* matter are the per-provider `query_examples` class attribute introduced in this refactor.

**The Phase 1 shim.** Forwards `description`, `query_examples`, AND `query_guidance` to the legacy registry so the legacy formulator's query quality is preserved during migration:

```python
# providers/__init__.py — shim during Phases 1–4
def register_provider(name, cls, **kwargs):
    # Prefer class attributes (post-refactor); fall back to kwargs (legacy).
    description = kwargs.get("description") or cls.description
    query_examples = kwargs.get("examples") or cls.query_examples
    # query_guidance kept in the shim explicitly — legacy formulator
    # still uses it during Phases 1–4. Removed in Phase 5.
    query_guidance = kwargs.get("query_guidance") or getattr(cls, "query_guidance", "")
    # output_kind etc. live only on the class (not in the legacy registry shape).
    ...
```

Dropping `query_guidance` from the shim would silently degrade legacy formulator query quality during migration. The shim explicitly keeps it.

**Files changed:** 10 provider files (one each) + `providers/__init__.py` shim + delete `PROVIDER_EXAMPLES`.

**Tests:** existing provider tests should pass unchanged. Add tests verifying each provider exposes `description`, `query_examples`, `output_kind`, and `independence_group` as class attributes (a structural test that catches accidental regressions).

**Acceptance:** all 10 providers have self-contained class attributes. The existing upstream formulator (`epistemic_formulate_query`) still works against the shim. `PROVIDER_EXAMPLES` is deleted. No behavioural change observable in dev30 runs.

### Phase 2 — Dispatch agent (new path, not yet wired in)

Implement `formulate_provider_query` agent and `select_candidates_by_embedding` helper. These live in a new module `epistemic/dispatch.py`. Build a switchable orchestrator: `gather_evidence_new` runs the description-driven path; the existing pipeline keeps calling the old one.

A new CLI flag `--dispatch-mode legacy|new` selects which path is used for a given run (default: `legacy`).

**Tests:** unit tests on the dispatch agent (mock providers, check it correctly triages and constructs queries). Integration test: run a single claim through `gather_evidence_new` and confirm output shape matches `gather_evidence_legacy`.

**Acceptance:** can run `--dispatch-mode new` on a single claim end-to-end without errors. Outputs are inspectable.

### Phase 3 — Retrieval-quality benchmark (two-tier validation)

The thing we're trying to fix is *retrieval relevance*: providers returning evidence that's actually on-topic for the claim. Full-pipeline validation (running dev30 cases end-to-end and watching the final posterior move) is the wrong shape for iterating on this — most per-claim variance comes from downstream stages (judgment, scrutiny, IBE chain, K-agreement), which dilutes the retrieval-quality signal and makes per-provider attribution hard.

The sharper experiment is a **retrieval-quality benchmark** that scores each provider in isolation, plus a smaller end-to-end pipeline test at the end as acceptance.

**Tier 1 — Per-provider retrieval-quality benchmark (iteration loop)**

For each provider:

1. **Curate ~10 claims** spanning two categories:
   - **In-domain claims** (5–7) where the provider should return relevant evidence (e.g., for PubMed: clinical-research claims; for arXiv: ML / physics / q-bio claims).
   - **Out-of-domain claims** (3–5) where the provider should *abstain* (e.g., for ClinicalTrials.gov: pure molecular-biology questions; for arXiv: clinical-pharmacology questions).
   
   The out-of-domain set is essential — it's how the abstention-decision quality gets measured.

2. **For each (claim, provider) pair, run both dispatch modes** (legacy and new).

3. **Score each returned evidence piece** by invoking the existing per-evidence judgment agent (`epistemic_judge_evidence` or equivalent) against the claim. Output: one of `supports / contradicts / no_bearing` per evidence piece.

4. **Compute three per-provider metrics:**
   
   | Metric | Definition | Direction |
   |---|---|---|
   | Relevance rate | `(supports + contradicts) / total_returned` on in-domain claims | ↑ better |
   | Hit rate | `fraction of in-domain claims where provider returned ≥ 1 record` | ↑ better |
   | Abstention accuracy | `fraction of out-of-domain claims where new dispatch correctly returned empty` | ↑ better (new dispatch only — legacy always returns something) |

5. **Compare new vs legacy** on relevance rate and hit rate. Abstention accuracy is reported only for the new path.

**Cost estimate:** 10 providers × 10 claims × 2 modes × ~10 evidence pieces × 1 judge LLM call ≈ 2,000 LLM calls, on the order of $1–5 depending on model. Wall-clock ~30–60 minutes. **An order of magnitude cheaper than running full-pipeline dev30 (30 cases × ~6 min/case ≈ 3 hours per mode).**

**Iteration on this benchmark is fast.** If arXiv's relevance rate is bad in the new dispatch, you iterate on arXiv's description + example queries + the dispatch agent prompt, then re-run *only the arXiv portion* of the benchmark (~10 claims × 1 provider, minutes). No need to re-run the whole catalogue.

**Tier 1 acceptance:** for each provider, the new dispatch matches or beats legacy on relevance rate and hit rate on in-domain claims. Abstention accuracy on out-of-domain claims is ≥ 80%. If a provider fails on relevance rate, fix the description/prompt and re-run that provider's benchmark slice.

**Tier 1 has a known blind spot.** Per-provider relevance rate and hit rate by construction reward providers that return on-topic evidence. They cannot measure the calibration contribution of a provider that *correctly abstains* — the v3→v4 dev30 evidence showed that "low-yield" providers added AUC via their abstention pattern, not via their evidence. If the new dispatch agent makes ChEMBL or arXiv abstain more often than the legacy formulator did (entirely correct behaviour at the per-claim level), Tier 1 says "great, abstention accuracy is high" and Tier 2's 5-claim end-to-end test is too small to detect a 0.05–0.10 AUC drop from systematic over-abstention. Tier 1.5 closes this gap.

**Tier 1.5 — Abstention-pattern stability against the dev30 corpus (no judge calls)**

For each provider, run the new dispatch on every claim in the dev30 corpus and compare against the legacy formulator's "returned nothing" pattern on the same corpus (we already have v5 data for this).

- **Metric:** per-provider, `agreement_rate = (matching_abstain_decisions + matching_nonabstain_decisions) / 30`. A dispatch decision counts as "abstain" if `queries == []` for the new path, or if legacy returned zero records.
- **Acceptance:** for each provider, `agreement_rate ≥ 0.80` against legacy. That is, the new dispatch's pattern of "this provider can/can't help" matches legacy on at least 80% of dev30 claims.
- **Where deviations are allowed:** if a deviation reflects the new dispatch being *correct* where legacy was wrong (e.g., the new path correctly abstains for arXiv on biomedical claims where legacy issued a useless query), flag it as an "intentional improvement" with one-sentence justification per case. Intentional improvements don't count against agreement_rate.
- **Cost:** ~10 providers × 30 claims × 1 dispatch call = 300 LLM calls (no judge needed; just measuring abstain-vs-not for each pair).

This catches the calibration-via-abstention regression *before* Phase 4 deployment, gives per-provider attribution, and runs in minutes.

**Tier 1.5 acceptance:** every provider's abstention pattern is within 80% agreement with legacy on dev30, OR each deviating case is documented as an intentional improvement. If a provider drifts below 80% with unintentional deviations, that's a routing-quality regression to fix before Tier 2.

**Tier 2 — Small end-to-end pipeline sanity check (acceptance gate)**

Once every provider passes Tier 1, run 5 hand-picked claims (one per major category: clinical, molecular biology, drug discovery, definitional, and one cross-domain claim) through the full pipeline under both dispatch modes. Confirm:

- The pipeline runs to completion without errors under the new dispatch.
- Per-claim posteriors are within ±0.10 of the legacy values, OR the new dispatch's directional verdict matches the legacy's where they differ.
- Wall-clock per claim is comparable (within 20%).

**Tier 2 acceptance:** clean end-to-end runs on the 5-claim fixture, no behavioural regressions vs legacy on this fixture.

**Why three tiers:** Tier 1 is the *iteration loop* — cheap, fast, per-provider-attributable, run repeatedly during prompt and description tuning. Tier 1.5 is the *abstention-pattern check* — catches the calibration-via-abstention regression Tier 1 can't see. Tier 2 is the *acceptance gate* — confirms retrieval-quality gains translate into pipeline-level behaviour and don't break end-to-end orchestration. Tier 1 runs many times; Tier 1.5 and Tier 2 each run once at the end of Phase 3.

This two-tier structure also means a regression caught in Tier 2 (but not in Tier 1) is informative: it indicates the issue isn't in retrieval relevance but in some pipeline-level interaction. That kind of issue probably belongs in a separate fix, not in this PRD's scope.

### Phase 4 — Full dev30 validation + swap

Run the full dev30 corpus (n=30) under `--dispatch-mode new`. Compare against v5 baseline (commit `e30c314`, legacy dispatch, AUC 0.88).

**Acceptance is not "match or beat v5" — it must account for the baseline shift introduced by independence-group accounting (§4.6).** v5 over-counted correlated providers as independent sources. v6 will correct this, which is expected to compress some posteriors toward less-confident-but-better-calibrated values. The acceptance criteria:

- **AUC ≥ 0.85** (vs v5's 0.88). The 0.03 drop is the expected baseline shift, not a regression. AUC below 0.85 is a regression.
- **ECE ≤ 0.20** (vs v5's 0.105). Slightly looser tolerance to allow for the baseline shift; calibration-via-abstention quality is checked by Tier 1.5 in Phase 3.
- **Brier ≤ 0.20** (vs v5's 0.156). Same reasoning.
- **Per-case audit:** for the 5 dev30 cases where v5 most confidently committed (posterior ≤ 0.10 or ≥ 0.90), verify v6 either matches or has a documented reason for a direction change.

If acceptance is met, flip the default to `--dispatch-mode new`. Mark the three legacy agents (`epistemic_select_provider`, `epistemic_rank_providers`, `epistemic_formulate_query`) as deprecated. The actual deletion happens in Phase 5.

If acceptance is *not* met, the analysis branches:
- If the gap is on AUC/Brier/ECE: re-examine Tier 1.5's abstention agreement — likely the dispatch agent is over- or under-abstaining on some provider class.
- If the gap is on per-case posteriors: re-examine the `independence_group` assignments — possibly two providers were lumped that shouldn't have been.
- If neither: the architectural change has a deeper issue and rollback is the right call.

### Phase 5 — Cleanup

- Delete the three legacy agents and their prompt files: `epistemic_select_provider`, `epistemic_rank_providers`, `epistemic_formulate_query`.
- Replace `epistemic_rank_providers`'s scrutiny-escalation call site in `investigation.py:298–335` with the deterministic replacement decided in Phase 2 (embedding-distance + "has not been queried" filter on the candidate pool).
- Remove the shim in `providers/__init__.py`.
- Remove `query_guidance` parameter from `register_provider`.
- Remove the `--provider all|web_search` CLI flag, replace with `--dispatch-k N` and `--provider-allowlist` / `--provider-blocklist`.
- Update docs: `epistemic_flow.html`, `overview.md`, `CONTRIBUTING.md` (provider-author guide).

**Acceptance:** clean architecture, all tests pass, docs accurate.

### Phase scheduling

| Phase | Estimate | Notes |
|---|---|---|
| 1 | 1–2 days | Mostly mechanical; touches 10 files |
| 2 | 2 days | Real new code: agent prompt + dispatch logic + embedding pre-filter |
| 3 | 1–2 days | Tier 1 per-provider retrieval-quality benchmark (fast iteration) + Tier 2 5-claim end-to-end sanity check |
| 4 | 1 day | Run the benchmark + analyse results |
| 5 | 0.5 days | Pure cleanup |
| **Total** | **5–7 working days** | |

This assumes no major surprises. If Phase 3 reveals systematic issues with the dispatch agent (e.g., it consistently fails to triage some category of claim), Phase 3 could extend by another 2–3 days for prompt engineering.

---

## 7. Testing strategy

### 7.1 Unit tests

- **Dispatch agent triage:** mock provider descriptions, fixed claims, check the agent returns `queries=[]` for clearly-irrelevant providers and at least one non-empty query for relevant ones.
- **Dispatch agent construction:** for each existing provider, give the agent a representative claim and verify the constructed query parses correctly under that provider's syntax (e.g., arXiv queries are well-formed field expressions; ClinicalTrials.gov queries are valid AREA[] expressions).
- **Embedding pre-filter:** given a fixed catalogue and a claim, check that the top-K candidates are stable and semantically appropriate. At Phase 4 default (K = catalogue size), verify the function returns all providers unchanged (pass-through behaviour).
- **Provider self-description:** each provider's `description`, `query_examples`, `output_kind`, `independence_group`, and `provider_contract_version` are present and well-formed. Structural test that fails CI if a new provider is added without filling in the required fields.

### 7.2 Integration tests

- **End-to-end on a small fixture corpus** (~5 hand-picked claims spanning biomedical, clinical, and structured-data domains). Run through the full new path, verify evidence is gathered and the IBE chain runs to completion.
- **Cost test:** verify the number of LLM calls per claim is bounded. With K = catalogue size (default at Phase 4), expect `len(providers)` dispatch calls + 1 embedding call (or 0 if pre-filter is pass-through) + downstream pipeline. Measure and bound.

### 7.3 Benchmark validation

- **dev30 v6 run** under the new architecture (Phase 4 acceptance criterion). Compare AUC / Brier / ECE / judge F1 against v5 baseline.
- **Per-claim trace audit:** for 5 randomly-selected dev30 cases, examine which providers were dispatched-to vs which were dispatched-against, and confirm the routing decisions look reasonable.

### 7.4 Generality test (optional but recommended)

Add one hypothetical-domain provider (e.g., FRED for economics) with a description and 2–3 example queries. Verify the dispatch agent correctly routes an economics claim to FRED and a biomedical claim away from it. This proves the architecture isn't quietly hard-coded to biomedical providers.

---

## 8. Risks and mitigations

### 8.1 Concentration of risk in one agent

The dispatch agent is now a single point: if it triages badly or constructs bad queries, every provider downstream suffers. Mitigations:

- **Per-provider validation in Phase 3.** If the new path under-performs on any provider, we know which provider's description needs work.
- **The dispatch agent's prompt is testable in isolation.** Unlike the current upstream formulator (which is entangled with claim decomposition and topic-routing), the new agent's input is small and structured.
- **Cheap rollback.** If the dispatch agent regresses behaviour after deployment, revert the `--dispatch-mode` default to `legacy`. No data loss.

### 8.2 Embedding pre-filter false negatives

If a provider's description doesn't embed close to a claim it could actually help with, the dispatch agent never sees that provider. Mitigations:

- **K is conservatively large.** At default K=8 (current catalogue of 10 providers), we miss at most 2. Tunable up.
- **Manual allowlist.** `--provider-allowlist` forces specific providers into the candidate set regardless of embedding score. Useful for cases where a researcher knows a non-obvious provider is relevant.
- **Periodic re-embedding.** As providers' descriptions evolve, re-embed and re-cache.

### 8.3 Agent cost per claim

K=8 dispatch calls per claim is more than the current 1 formulator call. Mitigations:

- **Calls are parallel.** Wall-clock is bounded by one LLM round trip, not K.
- **Cost is bounded in dollars.** With small-model dispatch (e.g., gpt-5.4-nano), 8 calls × ~500 tokens each is fractional cents per claim.
- **K is tunable.** At 10 providers, K=5 might be sufficient. At 1000, K=20 with embedding pre-filter is still O(K), not O(N).

### 8.4 The structured-reference providers — partially addressed

ChEMBL/Open Targets/Monarch return reference data, not assertions. The `output_kind` discriminator added in §4.1 lets the judgement layer skip the supports/contradicts axis for these items and tag them as structured context instead. This closes most of the immediate problem without expanding scope.

What's still future work: building a "context channel" for structured-reference items so the synthesis writer can incorporate them properly (e.g., a synthesis paragraph that says "the claim references HDAC1, which has known inhibitors imatinib, vorinostat, ..." using ChEMBL records that didn't directly support/contradict the claim but provide useful context). That work is orthogonal to this refactor and can land later.

### 8.5 Independence-group baseline shift

The `independence_group` mechanism (§4.6) is *cleaner* than the current `source_type`-based grouping in `gates.py` and `dedupe_evidence.py`, where PubMed + Cochrane are currently counted as 2 independent sources despite both being biomedical literature. Post-refactor they correctly count as 1.

**This is a baseline shift, not a regression.** Phase 4 acceptance is explicitly adjusted to allow AUC ≥ 0.85 (vs v5's 0.88) because some of v5's headline number came from over-counting correlated sources. Phase 4 acceptance criteria (§6 Phase 4) name this explicitly so the dispatch-quality change isn't conflated with the independence-accounting change.

### 8.6 Ollama availability on the new critical path

The embedding pre-filter adds an Ollama dependency to the dispatch path that doesn't exist today. If the Ollama service is unhealthy or slow, naive implementations would degrade to "claim never gets evidence" rather than "claim gets all-provider fan-out via legacy."

**Mitigation:** the pre-filter implementation falls back to "dispatch on all providers" when embedding fails (HTTP error, timeout, malformed response). Empty candidate sets are never returned. This is part of the `select_candidates_by_embedding` contract regardless of catalogue size. The fallback is tested explicitly in Phase 2's unit tests.

### 8.7 Provider description quality varies

A poorly-written description will produce poor dispatch decisions. Mitigations:

- **Provider description is the contract.** A new section in `CONTRIBUTING.md` for the post-refactor era describing how to write a good description (strong-for/weak-for, native syntax, 3–6 examples including 1 negative example, `output_kind`, `independence_group`).
- **Description quality is testable.** During Phase 3, if a provider's dispatch performance is bad, the first hypothesis is "description is unclear" — and we iterate on the description, not the agent.

### 8.8 Migration regressions

Phases 1–2 build the new path without disrupting the existing one. Phase 4 swaps the default. If Phase 4 reveals a regression we didn't catch in Phase 3, we can flip back to legacy with a CLI flag. Phase 4 acceptance criteria (§6 Phase 4) are explicit about the baseline shift vs v5; rollback is to `pre-dispatch-refactor` tag if needed.

---

## 9. Success criteria

The refactor is successful if and only if:

1. **All 10 current providers run under the new dispatch architecture** and produce equivalent or better per-provider evidence yield (Tier 1 of Phase 3) than the legacy path on the dev30 corpus.
2. **Per-provider abstention pattern stability (Tier 1.5) is achieved**: every provider's abstain-vs-not pattern on the dev30 corpus matches legacy within 80% agreement, or each deviation is documented as an intentional improvement.
3. **dev30 v6 (under new dispatch) achieves AUC ≥ 0.85, ECE ≤ 0.20, Brier ≤ 0.20.** The thresholds account for the baseline shift from `independence_group` adoption (§4.6) and are explicitly *not* "match v5"; the goal is honest calibration with corrected independence accounting, not chasing a v5 number that contained correlated-source double-counting.
4. **At least one hypothetical-domain provider** (e.g., FRED, NASA ADS) is added in test infrastructure to prove the architecture is genuinely domain-general — it's routed correctly for an out-of-biomedical claim, and ignored for biomedical claims.
5. **`output_kind` correctly excludes structured-record providers from the supports/contradicts axis.** ChEMBL/Open Targets/Monarch records are tagged `structured_record` and the judge no longer scores them — verified by inspecting judgement outputs on 5 sample claims.
6. **The total provider-system code shrinks net of additions.** The dispatch agent + helpers add ~400 lines; the three legacy agents + their prompts + the multi-syntax formulator machinery delete ~700 lines. Net cleaner.
7. **Adding a new provider takes ≤ 1 hour** of provider-author work, demonstrably (test by adding one new provider end-to-end during Phase 5 cleanup).
8. **All 1054 tests still pass** (no regressions); new dispatch-specific tests are added.
9. **Pyright + ruff stay clean.**

---

## 10. Open questions to resolve during implementation

These are not blockers but should be answered before completing Phase 5:

1. **Per-provider dispatch in parallel vs sequential?** Parallel is faster (one LLM round trip) but bursts the rate-limit. Sequential is slower but safer. Default: parallel with a small concurrency cap.

2. **At what catalogue size does the embedding pre-filter become useful?** Phase 4 default is `K = catalogue size` (pass-through). The architecture supports activation at larger scales. Open: precise threshold, what offline test validates the chosen K, and how the pre-filter interacts with `corroboration_count` semantics at large scale.

3. **`corroboration_count` denominator at large catalogue.** Currently it's the raw count of distinct independence groups in a duplicate group. At 1000 providers spanning many independence groups, the raw count grows; should it be normalised by "fraction of dispatched-not-abstaining providers" or kept as raw? Decision deferred to follow-up PR; relevant only when catalogue exceeds ~30 providers.

4. **Scrutiny-escalation loop bound at large catalogue.** `InvestigateClaimOperation` currently picks "next unused provider" up to `MAX_INVESTIGATION_ROUNDS` ~ 3. At 10 providers this nearly exhausts the catalogue; at 1000 it doesn't. The current safety is the per-claim cycle cap. Open: at large catalogue, does the escalation also need a per-claim "max distinct providers tried" cap? Deferred to follow-up PR.

5. **Should `dispatch_k` be per-mode (research vs verify) or fixed?** Research mode might want broader coverage; verify mode might want narrower. Investigate during Phase 3.

3. **What happens when the dispatch agent itself fails?** (LLM timeout, malformed output, etc.) Options: skip the provider, fall back to a sensible default query (e.g., plain claim text), abort the claim. Default: skip, log, continue. Behaviour mirrors how individual provider HTTP failures are handled today.

4. **Should we cache dispatch decisions?** If the same claim is investigated multiple times (e.g., during scrutiny cycles), the dispatch decision is cacheable. Probably yes; design the cache key as `hash(claim, provider_description_version)`.

5. **Is the `independence_group` tag enough for convergence reasoning?** Or do we need a small structured cross-provider similarity matrix? Evaluate during Phase 4 by looking at convergence verdicts and whether they meaningfully changed vs v5.

6. **CLI flag naming.** `--dispatch-k`, `--dispatch-mode`, `--provider-allowlist`, `--provider-blocklist`. Bikeshed during Phase 5.

7. **Does the dispatch agent need its own model, separate from the main `ANDAMENTUM_MAIN_LLM_MODEL`?** Probably should support `--dispatch-model` separately, with a sensible default of the main model. Useful for cost optimisation (cheap dispatch + expensive synthesis).

---

## 11. Why this is the right architecture, in one sentence

Provider knowledge belongs in prose, not in agents or taxonomies, because prose is what the agent can read and the human can write — and the same one agent reading many prose descriptions is what scales when the human can no longer write many agents.

---

## Appendix A — File layout after refactor

```
src/andamentum/epistemic/
├── dispatch.py                 # NEW: formulate_provider_query + select_candidates_by_embedding
├── evidence_gathering.py       # Reorchestrated to use dispatch.py
├── operations/
│   └── evidence.py             # Updated: calls dispatch path
├── providers/
│   ├── __init__.py             # Slim registry, no descriptions or examples
│   ├── arxiv.py                # description + query_examples + gather
│   ├── biorxiv.py              # same shape
│   ├── chembl.py               # same shape
│   ├── clinicaltrials.py       # same shape
│   ├── cochrane.py             # same shape
│   ├── europepmc.py            # same shape
│   ├── monarch.py              # same shape
│   ├── open_targets.py         # same shape
│   ├── openalex.py             # same shape
│   ├── pubmed.py               # same shape
│   ├── web_search.py           # same shape
│   └── CONTRIBUTING.md         # Updated for new provider-author contract
└── operations/preplanning.py   # Old multi-syntax formulator REMOVED
```

## Appendix B — Provider self-description schema (informational)

Each provider class declares:

```python
class XProvider:
    description: str              # natural-language, multi-paragraph
    query_examples: list[tuple[str, str | None]]
                                  # (claim or question, native query or None)
    output_kind: str              # one of: "assertion_evidence", "structured_record",
                                  #         "trial_registration", "compound_data"
    independence_group: str       # short tag for convergence + dedup;
                                  # NOT read by the dispatch agent
    provider_contract_version: int = 1
                                  # bump when the contract shape changes
    max_results: int              # already exists; preserved

    async def gather(self, query: str) -> list[GatheredEvidence]:
        ...
    
    async def check_health(self) -> CheckResult:
        ...
```

No taxonomies for dispatch. No schemas the dispatch agent parses. Just prose + examples + HTTP code + two short tags consumed by downstream code (`output_kind` by the judge, `independence_group` by `gates.py` and `dedupe_evidence.py`). The dispatch agent reads only `description` and `query_examples`.

---

## Appendix C — Example dispatch agent invocation (full trace)

**Claim:** "The DdrB protein from Deinococcus radiodurans is an alternative SSB."

**Embedding pre-filter at Phase 4 default (K = catalogue size):** all 11 providers (10 + web_search) are candidates. No pre-filter pruning. Pre-filter activation deferred to follow-up work at larger catalogue sizes.

**Dispatch agent invocations (parallel, one per provider, returning `queries: list[str]`):**

```
PROVIDER: PubMed                         output_kind=assertion_evidence
  → queries:
      '("DNA-Binding Proteins"[MeSH] OR "single-stranded DNA binding") AND 
       ("Deinococcus radiodurans"[Organism] OR "DdrB")'
      'DdrB Deinococcus radiodurans SSB'
  → reasoning: "Biomedical literature with strong coverage of bacterial 
                proteins and DNA repair; running both MeSH and free-text 
                queries since MeSH coverage on recent bacterial-protein 
                literature can be patchy."
  → confidence: 0.85

PROVIDER: Europe PMC                     output_kind=assertion_evidence
  → queries: ['TITLE:"DdrB" OR TITLE:"DdrB Deinococcus" OR ABSTRACT:"alternative SSB"']
  → reasoning: "Full-text fielded search complements PubMed's abstract-only 
                index."
  → confidence: 0.80

PROVIDER: OpenAlex                       output_kind=assertion_evidence
  → queries: ['DdrB Deinococcus radiodurans SSB']
  → reasoning: "Cross-disciplinary scholarly metadata; accepts 
                natural-language."
  → confidence: 0.70

PROVIDER: bioRxiv                        output_kind=assertion_evidence
  → queries: ['DdrB AND Deinococcus radiodurans']
  → reasoning: "Preprint archive may contain recent biology work on this 
                protein not yet in PubMed."
  → confidence: 0.60

PROVIDER: arXiv                          output_kind=assertion_evidence
  → queries: []  (abstain)
  → reasoning: "arXiv has limited coverage of microbiology and bacterial 
                protein function; this claim is squarely biomedical."
  → confidence: 0.88

PROVIDER: ClinicalTrials.gov             output_kind=trial_registration
  → queries: []  (abstain)
  → reasoning: "ClinicalTrials.gov registers human trials; this claim is 
                about a bacterial protein's molecular function."
  → confidence: 0.95

PROVIDER: Cochrane                       output_kind=assertion_evidence
  → queries: []  (abstain)
  → reasoning: "Cochrane reviews are clinical systematic reviews; bacterial 
                protein function is out of scope."
  → confidence: 0.92

PROVIDER: ChEMBL                         output_kind=structured_record
  → queries: []  (abstain)
  → reasoning: "ChEMBL is a compound bioactivity database; DdrB is a protein, 
                not a small-molecule drug or target with bioactivity data."
  → confidence: 0.95

PROVIDER: Open Targets                   output_kind=structured_record
  → queries: []  (abstain)
  → reasoning: "Open Targets is a target-disease evidence platform; this 
                claim is about a bacterial protein's function, not a 
                target-disease association."
  → confidence: 0.90

PROVIDER: Monarch                        output_kind=structured_record
  → queries: []  (abstain)
  → reasoning: "Monarch focuses on gene-disease associations in human 
                disease genetics; bacterial protein function is out of scope."
  → confidence: 0.92

PROVIDER: web_search                     output_kind=assertion_evidence
  → queries: ['DdrB protein Deinococcus radiodurans alternative SSB']
  → reasoning: "Natural-language; web search is a broad fallback."
  → confidence: 0.55
```

**Result:** 5 providers get queries (PubMed with 2, Europe PMC, OpenAlex, bioRxiv, web_search); 6 correctly abstain (arXiv, ClinicalTrials.gov, Cochrane, ChEMBL, Open Targets, Monarch). The abstaining structured-record providers don't burn an HTTP call. Compare to the current system where all 11 providers would have received the same upstream-formulator query regardless of relevance, and arXiv + ClinicalTrials.gov would have correctly returned 0 results from wasted API calls.

---

*End of document.*
