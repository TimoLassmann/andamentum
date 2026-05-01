# Epistemic Efficiency — Make the Pipeline Usable

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Each phase is independently committable; stop at the gate, run the benchmark, verify, then continue.**

**Goal:** Reduce both LLM call count and wall-clock runtime of the epistemic pipeline so a single decomposed run finishes in 1-2 minutes (currently 5-7) and costs roughly half its current cost (~$5/run → ~$2). Without losing data the system would currently surface.

**Constraint:** Each investigation round stays equally thorough — no round-narrowing, no provider-narrowing across rounds, no embedding-based pre-filtering of evidence relevance. The cost reduction has to come from bounding per-step work, parallelizing independent calls, and widening Path 1 in quality scoring.

**Tech stack:** Python 3.13, asyncio, pydantic-graph, the existing `andamentum.epistemic` package.

**Non-goals:** Architectural refactor of how evidence flows through the pipeline. Cross-run caching (deferred to a follow-up plan). Quality optimization of any specific operation. Reducing the number of investigation rounds (the user wants the outer loop to remain thorough).

---

## Where the cost lives (verified, not guessed)

A typical decomposed run with 4 sub-investigations makes ~500 LLM calls. The dominant phase is the **inquiry loop** (3 rounds of Scrutinize → Investigate → ExtractNewEvidence), which produces ~70% of the calls. Within that:

- **Per-evidence work** is the multiplier: each round produces ~12 query stubs, each stub returns ~5 "extras" from the gatherer, and each extra goes through extract → quality-score → support-judge ≈ 2-3 LLM calls. A single round can hit 150+ LLM calls just on evidence processing.
- **Adversarial search** is the other big spender: 8 queries per claim × ~2 supported claims = 16 searches, each with extract+score+judge.
- **IBE chain** is moderate: 5 candidates × 3 stages × 2 claims = 30 calls.
- **Plan task with retries** is moderate: 4 subs × 6 providers × ~3 retries = 72 calls.

Almost everything runs **sequentially** today even when the operations are independent (no shared state). That's the second lever.

---

## Architecture

Three independent levers that compose:

### 1. Bound per-step work

When the operation has a discretionary depth parameter, cap it to a sensible default:

| Parameter | Today | Proposed | Effect |
|---|---|---|---|
| Evidence extras per stub (in `extract_evidence` + dedup) | unbounded (gatherer-decided, often 5-10) | 3 | -30% inquiry-loop calls |
| Adversarial queries per claim (`AdversarialSearchOperation`) | 8 | 5 | -15% verification calls |
| IBE candidates per claim (`EnumerateCandidates`) | 5 | 3 | -8% IBE calls |
| Plan-task retries per slot (`MAX_SLOT_RETRIES`) | 3 | 2 | -10% plan-task calls |

These are bounded per-step caps, not round-narrowing. Round 1, 2, and 3 each get the same caps — what's reduced is the depth *within* a step, not breadth across rounds.

The depth being cut is the place where marginal information per LLM call is lowest:
- Items 4-10 in a single query's gatherer return are typically near-duplicates of items 1-3
- Adversarial queries beyond 5 hit diminishing returns (the search-quality of the 8th query is well below the 1st)
- IBE candidates beyond 3 mostly produce minor variations on already-explored options

### 2. Parallelize independent LLM calls

Many loops over (claim × evidence × provider) currently run sequentially under `for ... await`. The calls are independent (no shared mutable state during the LLM call itself; they all just write a single field on a single entity). They can be `asyncio.gather`'d.

Concrete sites:

| Site | Current shape | Concurrent shape | Wall-clock effect |
|---|---|---|---|
| `MultiSeedClaim` per-sub judges | 4 subs × ~6 evidence each, serial | per-sub: gather all evidence judges concurrently | ~6× speedup on this step |
| `ExtractNewEvidence` per-extra judges | 12 stubs × ~5 extras serial = 60 sequential calls | gather all extras per stub concurrently | ~5× speedup on this step |
| `RunVerification` per-claim tracks | 2 claims × 5 tracks each, serial | gather all (claim, track) pairs concurrently | ~10× speedup on this step |
| IBE `score_loveliness` / `score_likeliness` | 2 claims × 5 candidates serial = 10 sequential calls | per-claim: gather candidates concurrently | ~5× speedup on these steps |
| `AdversarialSearchOperation` query generation | 8 queries serial | gather all 8 concurrently | ~8× speedup on this step |

Same number of LLM calls, much less wall-clock time. Limited by:
- Per-API-key rate limits (we'd need to check OpenAI's tier)
- Pydantic-AI agent execution model (does it support concurrent runs? — yes, individual `agent.run()` calls are async-safe)

### 3. Widen Path 1 in quality scoring

`_score_evidence` has three paths in priority order:
1. OpenAlex DOI/PMID lookup (free, real bibliometric data)
2. LLM agent quality assessment (slow, paid)
3. Gatherer-supplied score (free, gatherer-trusted)

Currently Path 1 only fires when the gatherer-supplied `source_ref` already contains a clean DOI/PMID. Many sources have these in their *content* (e.g. PDFs with `doi:10.1234/...` in the first 500 chars; web pages with DOI in the URL but in non-standard form).

The change: add a general identifier extractor that runs against `source_ref` AND `extracted_content[:1000]`. Identifiers it recognizes:
- DOIs: `10.\d{4,9}/[-._;()/:A-Z0-9]+` (case-insensitive, broad pattern)
- PMIDs: `PMID:\s*\d+`, `pmid:\s*\d+`, `pubmed/\d+`
- arXiv IDs: `arXiv:\d{4}.\d{4,5}`

This is **not** OpenAlex-specific. It's identifier extraction from arbitrary text. The existing `quality_scorer` abstraction (Protocol in `operations/base.py:161`) consumes the identifier; OpenAlex is just the current implementation. A future Crossref or Semantic Scholar quality scorer would benefit from the same extractor.

Effect: ~30-40% of items currently hitting Path 2 (LLM fallback) move to Path 1 (free).

---

## Phases

### Phase 1 — Bound per-step work (1 commit, low risk)

Adds default caps as constants with clear names. Behavior change only for runs that previously exceeded the caps (most runs).

- [ ] In `operations/evidence.py` `ExtractEvidenceOperation`: add `MAX_EXTRAS_PER_STUB = 3` constant; truncate `gathered[1:]` to this bound when populating new evidence. Log when truncation fires (info level) so we can see how often the cap was binding.
- [ ] In `operations/verification.py` `AdversarialSearchOperation`: add `MAX_ADVERSARIAL_QUERIES = 5` constant; cap the adversarial query generation to this count.
- [ ] In `graph/nodes.py` `EnumerateCandidates`: change the K=5 default to `MAX_IBE_CANDIDATES = 3` (declared in `operations/integration.py` near `EnumerateCandidatesOperation`).
- [ ] In `deep_research` query loop: lower `MAX_SLOT_RETRIES` from 3 to 2.
- [ ] Add unit tests for the new caps:
  - `test_max_extras_per_stub_caps_extraction` — feed a fake gatherer that returns 10 results; assert only 3 are processed.
  - `test_max_adversarial_queries_caps_generation` — assert only 5 queries are produced.
  - `test_max_ibe_candidates_caps_enumeration` — assert only 3 candidates after EnumerateCandidates.
- [ ] Run benchmark. **Acceptance:** posterior + verdict shape consistent with baseline (`d280573`-class output: `n_no_verdict==0`, IBE fired, posterior in [0,1], coherent verdict). Operation count meaningfully reduced (target: ~30% fewer LLM calls).

### Phase 2 — Parallelize independent LLM calls (2-3 commits, medium risk)

Wraps independent loops in `asyncio.gather`. Splits per-site so each is independently testable.

- [ ] Phase 2a: `MultiSeedClaim` and `ExtractNewEvidence` judges (these are the highest-multiplicative). Convert each `for ev in claim_evidence: judgment = await _judge(...)` into `judgments = await asyncio.gather(*[_judge(...) for ev in claim_evidence])`.
- [ ] Phase 2b: `RunVerification` per-claim tracks. Convert the nested `for claim: for track:` into a single `gather` over all `(claim, track)` pairs (keeping track-internal sequencing).
- [ ] Phase 2c: IBE chain (`score_loveliness`, `score_likeliness`) per-claim candidate scoring. The existing operations already do per-candidate work internally; lift that to use `asyncio.gather`.
- [ ] Phase 2d: `AdversarialSearchOperation` adversarial query generation — gather the 5 (post-Phase-1 cap) queries concurrently.
- [ ] Add a "concurrent calls" smoke test that monkey-patches the LLM agent with a sleep(0.5) and asserts that wall-clock time for the parallelized site is much less than serial would predict (proves the gather is actually firing).
- [ ] Run benchmark. **Acceptance:** posterior + verdict shape consistent with baseline. Wall-clock time meaningfully reduced (target: ~3-4× faster than Phase 1).

**Risk note:** parallel agent execution may hit OpenAI rate limits on free-tier API keys. If we see 429s in the benchmark, add a `Semaphore(max_concurrent=10)` per agent to bound concurrency, leaving the gather pattern but capping in-flight calls.

### Phase 3 — Widen Path 1 (1 commit, low risk)

Add general identifier extraction so more evidence items hit OpenAlex (free) instead of LLM Path 2.

- [ ] Create `operations/identifier_extraction.py` with a single function `extract_identifiers(source_ref: str, content: str | None) -> dict[str, str | None]` that returns `{"doi": ..., "pmid": ..., "arxiv": ...}` (or None for absent fields). Use compiled regex patterns. No dependencies on any provider.
- [ ] In `operations/evidence.py` `_score_evidence`: before Path 1's `quality_scorer.score(...)` call, run `extract_identifiers` against both `source_ref` and `extracted_content[:1000]` and pass the identifiers to the scorer.
- [ ] Update `OpenAlexQualityScorer.score` signature to accept identifiers directly (currently re-extracts DOI/PMID from `source_ref` itself — this duplication can go away). The `quality_scorer` Protocol in `operations/base.py:161` gains the identifiers parameter.
- [ ] Unit tests for `extract_identifiers`:
  - DOI in URL form: `https://doi.org/10.1234/abc` → `{"doi": "10.1234/abc"}`
  - DOI in PDF text: `"... published. doi:10.1234/abc ..."` → `{"doi": "10.1234/abc"}`
  - PMID: `"PMID: 12345678"` and `"pubmed.gov/12345678"` → `{"pmid": "12345678"}`
  - arXiv: `"arXiv:2401.12345"` → `{"arxiv": "2401.12345"}`
  - Multiple identifiers: returns all of them
  - No identifiers: returns all-None dict
- [ ] Run benchmark. **Acceptance:** posterior + verdict shape consistent with baseline. The Path 2 LLM-quality-scoring call count visibly drops (we should be able to see this in the operation profile, since `_score_evidence` Path 2 hits an agent call for "needs_assessment" sources).

### Phase 4 — Closeout

- [ ] Update `CLAUDE.md` to mention the per-step caps as constants in the "Known quirks" section so future code touching these limits sees them.
- [ ] Update `feedback_run_benchmark_before_done.md` memory with the post-efficiency baseline shape (so future sessions know what "healthy" looks like with the caps in place).
- [ ] Final benchmark run, reported alongside the pre-efficiency baseline for comparison. Document: target wall-clock, target LLM call count, target cost.

---

## Open decisions

### 1. Cap on `MAX_EXTRAS_PER_STUB` — 3 or 5?

3 is more aggressive (more cost reduction, slightly higher chance of missing a useful result from positions 4-5 in a query). 5 is safer.

**Recommendation:** start at 3. The system has 3 inquiry rounds, so an item missed at position 4 in round 1 will likely surface from a different query in round 2 or 3 (because the system queries different angles each round).

### 2. Per-claim concurrency cap

If we parallelize fully, we might issue 30+ concurrent LLM calls during peak. OpenAI's nano tier may rate-limit this.

**Recommendation:** set a `Semaphore(20)` global, applied via a decorator on `AgentRunner.run`. This caps in-flight calls without changing call count. 20 is a typical sustainable number for paid tier; can tune.

### 3. Should Phase 2 separate per-site sub-commits?

Phase 2 has 4 sites. Doing them all in one commit is faster but risk-stacked. Separate commits make bisection cheaper if one site breaks.

**Recommendation:** separate commits per site (Phase 2a, 2b, 2c, 2d). Benchmark gate after Phase 2a (the highest-impact site) to verify the pattern works; then 2b-2d quickly.

### 4. What if Phase 1's cap on adversarial queries (8 → 5) measurably drops verdict quality?

Adversarial search is the system's main mechanism for surfacing counter-evidence. Cutting from 8 to 5 might reduce the diversity of counter-arguments tried.

**Recommendation:** include in the Phase 1 benchmark check: does `adversarial_balance` distribution shift meaningfully? If yes, revert to 8 and accept the cost.

### 5. The "extras per stub" cap interacts with cross-provider dedup

Today: stub returns 10 extras, dedup may invalidate 3 as duplicates → 7 unique. With cap at 3: stub returns 3 extras, dedup invalidates 1 → 2 unique. We're losing breadth.

**Recommendation:** don't double-count. Apply the cap *after* dedup, not before. So we'd extract all 10, dedup down to ~7, then take the top 3 by initial gatherer score.

This makes Phase 1's first task slightly more involved than just truncating early. Worth the correctness gain.

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Caps drop verdict quality measurably | Medium | High | Phase 1 benchmark explicitly checks `adversarial_balance` distribution; revert specific caps that fail |
| Concurrency hits API rate limits | Medium | Medium | Add `Semaphore(20)` global; surface 429s in benchmark log |
| Parallel agent execution exposes shared mutable state we missed | Low | High | Audit each parallelized site for state writes; entities-as-data discipline (P3) means most ops shouldn't have shared state, but verify |
| Wall-clock improves but cost doesn't (we just paid faster) | High | Low | Phases 1 and 3 are explicit cost reductions; Phase 2 is wall-clock only — that's by design, the goal is *both* |
| Identifier extraction false positives | Low | Low | Cost is one wasted OpenAlex API call (free) per false positive; benign |
| Phase 6 typed Decomposition + Phase 1 caps interact unexpectedly | Low | Medium | Existing test_topology + reachability tests still apply; rerun them between phases |
| Plan takes longer than estimated | Medium | Low | Each phase is independently committable |

---

## Estimated effort

- **Phase 1 (caps)**: half day
- **Phase 2 (parallelize)**: full day, split across 4 sub-commits
- **Phase 3 (Path 1 widening)**: half day
- **Phase 4 (closeout)**: half day

**Total: ~2.5 working days, spread across multiple sessions with benchmark gates between phases.**

---

## Acceptance criteria for the whole effort

When all phases complete, the following must be true:

1. A single decomposed benchmark run (`andamentum-epistemic ask --decompose --model openai:gpt-5.4-nano "Does intermittent fasting reduce all-cause mortality?"`) finishes in **≤ 2 minutes wall-clock** (down from ~5-7).
2. Total LLM call count per run is **≤ 350** (down from ~500-550).
3. The cost per run drops to roughly **half** of pre-effort cost.
4. Posterior + verdict consistent with `d280573`-class output: `n_no_verdict==0`, IBE fired, posterior valid float in [0,1], verdict in {supports, contradicts, insufficient, no_data, union}, headline-prose alignment.
5. `adversarial_balance` distribution across claims hasn't drifted meaningfully (we still find counter-evidence at the same rate).
6. All existing tests (1736+) pass.
7. Pyright + ruff clean.
8. The system is *usable* — running it for normal investigation feels like waiting for a slow query, not waiting for a build.

---

## What this plan does NOT cover

- **Cross-run caching** (e.g. memoizing extract+judge results by `(claim_hash, evidence_hash)`). Genuinely big win when the same question runs multiple times, but a separate architectural concern. Follow-up plan.
- **Per-provider quality reputation** (e.g. "if Cochrane says no mortality data, that's load-bearing — boost confidence in that finding"). Domain rule, off-limits per the user's general-mechanisms preference.
- **Reducing inquiry rounds** (3 → 2 or 2 → 1). User explicitly preferred each round equally thorough; we're not narrowing or shortening.
- **Embedding-based pre-filtering of evidence relevance.** Discussed and rejected as too lossy in scientific domains.
- **Switching models** (e.g. nano → mini for some operations). The user's `feedback_test_model.md` memory says nano is the cheap-test default; switching mid-pipeline is a separate decision.

If any of these become necessary, write a follow-up plan rather than expanding this one.
