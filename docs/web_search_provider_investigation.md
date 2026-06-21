# Investigation: web search in the epistemic evidence layer — loss and restoration

Triggered by a dev10 SciFact run returning `retrieval_failed` 10/10 under
`--provider web_search`, and the observation that web search — once the system's
default evidence fallback — appeared to be gone. A 10-agent code+history sweep
established what follows.

## How it used to work (before 2026-05-12)

Evidence gathering was a **three-agent chain**: `epistemic_select_provider` →
`epistemic_rank_providers` → `epistemic_formulate_query`, driven by
`PlanTaskOperation`. **web_search was the universal fallback, hard-wired in:**

```python
# Always include web_search as universal fallback.
if "web_search" not in providers:
    providers.append("web_search")
```

and `CompositeGatherer` silently fell back to web search when a specialist
provider returned empty. The source catalogue described web_search as *"always
available as fallback."* Every run had web search as a guaranteed net.

## The pivot — deliberate, benchmark-justified

Commit **`837f941` "description-driven dispatch is the only gather path"**
(2026-05-12) removed the legacy chain in favour of the description-driven
dispatch over a `PROVIDER_REGISTRY` of **10 biomedical APIs only**. The commit
message cites dev30 **v6**: the new path was at-or-better on all calibration
metrics, **2.4× faster, 40% less evidence, 12pp lower invalidation**. The
removal of the legacy chain also removed web_search as a primary/fallback
evidence source — a side effect of an intentional, well-motivated refactor.

Earlier commit `e9b79cd` had already removed `CompositeGatherer`'s *silent*
provider→web fallback.

## How it worked at the start of this investigation

- `PROVIDER_REGISTRY` = 10 biomedical APIs; **no web_search registered**.
- Single gather path = description-driven dispatch (`DispatchGatherOperation` /
  `InvestigateClaimOperation`), which **requires a non-empty providers dict** and
  auto-loads only for `provider="all"` → biomedical.
- `--provider web_search` was a **dead CLI flag**: accepted, but mapped to no
  provider and no auto-load → empty dict → `retrieval_failed`. (This is exactly
  what the dev10 run hit.)
- web search was **not entirely gone**: `WebSearchGatherer` (deep_research +
  SearXNG) was still wired into **adversarial counter-evidence search**
  (`verification.py`) and preflight's health check — just not primary gathering.

## What the dev30 benchmark actually used

Only `dev30_v9` has a committed results file (biomedical-only, AUC 0.89);
earlier versions are reconstructed from the PRD/git (medium confidence):

| dev30 | Evidence source | AUC |
|---|---|---|
| v3 | biomedical **+ web_search** | ~0.93 |
| v5 | biomedical **+ web_search** | ~0.88 |
| v6+ | biomedical only (web removed) | ~0.86 |
| **v9 (frozen)** | biomedical only | **0.89** |

So the strong early runs *did* include web search; the frozen snapshot dropped
it. For **biomedical** claims, biomedical-only is the validated-best config —
what was lost is **general-domain** capability.

## What we did (restoration)

Re-added web search as a **first-class dispatch provider** — not the old
hard-wired fallback:

- New `providers/web_search.py::WebSearchProvider` — a pure-retrieval dispatch
  provider (contract-compliant: `quality_score=None`, returns
  `list[GatheredEvidence]`, never raises). It composes the **model-free**
  deep_research backend (`HttpxSearchBackend.search` + `fetch_page`), inheriting
  SSRF protection, robots/paywall gating, safe redirects, and the circuit
  breaker. No LLM synthesis (the epistemic judge does the reasoning).
- Registered as `web_search` in `PROVIDER_REGISTRY`, with full dispatch metadata
  (`description`, `query_guidance`, `query_examples` incl. out-of-domain abstain
  cases, `output_kind`, `independence_group="general_web"`,
  `provider_contract_version=1`). The dispatch agent commits on general claims
  and abstains on specialist biomedical ones.
- `get_all_providers()` now includes web_search; **`get_biomedical_providers()`
  deliberately excludes it** (preserves dev30 biomedical-only semantics).
- Graph auto-load: `provider="all"` → `get_all_providers()` (web included);
  `provider="web_search"` → web-only. The **dead flag now works**.
- Tests: `TestWebSearchProvider` (gather/fallback/empty/error/registration) +
  the existing self-description contract tests now cover web_search.

**Validated:** full epistemic suite 1180 passed (the only failures are live
OpenAlex smoke tests returning HTTP 429 from rate-limiting after our heavy
biomedical run — environmental, unrelated); pyright 0 errors in src; ruff clean;
and a **live** `gather()` returned real extracted evidence from Forbes / APA etc.

## Trade-off + open notes

- Including web_search in `"all"` restores generality but reintroduces a
  general-evidence source to biomedical runs. The dispatch agent's per-provider
  commit/abstain (and the description steering it away from specialist
  biomedical claims) is the mechanism that keeps it from re-adding the noise
  that v6 removed. If a future biomedical benchmark regresses, the lever is the
  description/examples, or routing biomedical runs through
  `get_biomedical_providers()` explicitly.
- Preflight's "WebSearch" check now matches reality again (web search is a real
  provider once more).
- This is an **evidence-layer change, independent of the Tier 0/1 confidence
  work** committed earlier on this branch.
