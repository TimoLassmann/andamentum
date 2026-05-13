# Architecture summary

A one-page orientation to the `andamentum` package. For deeper detail
see `doc/epistemic/overview.md` and the per-sub-module README files.

## What it is

`andamentum` is a single Python distribution of tightly-scoped
sub-modules for building agentic reasoning pipelines. The headline
sub-module is `andamentum.epistemic` — a formal-epistemology pipeline
that verifies research claims with calibrated confidence by gathering
evidence, judging it against the claim, scrutinising it, optionally
investigating gaps, and synthesising a report with an honest posterior.

The other sub-modules are supporting infrastructure: document
authoring (`scribe`), figure generation (`figures`), document review
(`whetstone`), web research (`deep_research`), personal knowledge base
(`document_store`), markdown chunking (`chunker`), source extraction
(`harvest`), readability analysis (`proofread`), vision critique
(`vision_critique`), HTML/PDF rendering (`typeset`), and shared model
infrastructure (`core`).

## The epistemic pipeline at one glance

```
question / claim
      │
      ▼
  Preplanning  ──── clarify, classify, optionally decompose into sub-investigations
      │
      ▼
  Initial gather  ── description-driven dispatch (per-provider, parallel)
      │            ── each provider self-describes; dispatch agent commits or abstains per query
      │            ── persisted Evidence is judged against the originating claim
      ▼
  Scrutinise  ──── identify uncertainties (evidence gaps, contradictions, ambiguities)
      │
      ▼
  Investigate? ─── if scrutiny says "needs resolution":
      │           ── gap-analysis agent proposes methodological intents (angles) with memory
      │              of prior rounds; angles must shift a named dimension (method, population,
      │              temporal frame, control, level of analysis), not paraphrase
      │           ── each intent routes through the same dispatch agent: claim + angle →
      │              per-provider native queries → new Evidence → judged on arrival
      ▼
  Verification ── deductive, computational, contrastive, convergence, cross-claim consistency
      │
      ▼
  Integration  ── IBE chain per claim (enumerate → loveliness → likeliness → select);
      │           multi-claim combination per decomposition rule (AND / OR / WEIGHTED_AND / UNION)
      ▼
  Synthesise   ── posterior-aware report with directional verdict, or rational suspension
                  ("honest insufficient") when evidence cannot resolve the claim
```

## Three architectural commitments

1. **Operations are pure transforms.** Each operation reads entities,
   does work (LLM calls, computations), writes the result. The graph
   (a pydantic-graph DAG) is the sole flow controller. Cross-entity
   effects are the graph's job, not an operation's.

2. **Entity fields are data, not signals.** Every field on `Claim`,
   `Evidence`, `Objective`, etc. represents something real (a verdict,
   a score, a stage). No field exists solely to tell the scheduler
   what to do next; that's what graph state is for.

3. **Stage invariants enforce implicit contracts at boundaries.**
   `graph/stages.py` defines per-stage exit invariants. The most
   recent one — `_all_active_claim_evidence_judged` — refuses to exit
   the scrutiny-and-investigation stage if any non-abandoned claim
   has content-bearing Evidence with `support_judgment = None`. Bugs
   that used to silently degrade calibration (Evidence persisted but
   never judged) now fail loudly at the boundary.

## Three architectural choices worth knowing

1. **Description-driven dispatch.** Each evidence provider class
   self-describes (`description`, `query_guidance`, `query_examples`,
   `output_kind`). A single generic dispatch agent reads those
   attributes at runtime and either commits one or two native-syntax
   queries or abstains. Adding a provider is a class-attribute plus
   HTTP-wrapper task — no agent design or prompt engineering.

2. **Investigation is intent + routing, separated.** When scrutiny
   demands more evidence, the `epistemic_investigate_claim` agent
   names a methodological *angle* (intent) — but does not pick
   providers or write native queries. Routing is the dispatch agent's
   job. The investigation prompt enforces dimensional shifts so
   successive rounds explore different aspects rather than reshuffling
   the same lexicon. Prior intents are passed back with their yield
   counts as memory; the agent can rationally return zero intents to
   suspend judgment when the search space looks exhausted.

3. **The judging contract is source-agnostic.** Every content-bearing
   Evidence linked to a non-abandoned claim must have a
   `support_judgment` before terminal nodes run. This is enforced as
   a stage invariant, not assumed via implementation-path coincidence.

## What's where

```
src/andamentum/
├── core/             — shared model resolution + AgentRunner
├── epistemic/        — the pipeline (entities, operations, graph, agents)
│   ├── entities/     — Objective, Claim, Evidence, Uncertainty, Decision, Snapshot, Artefact, IntentRecord
│   ├── operations/   — pure transforms (one module per family)
│   ├── graph/        — pydantic-graph DAG (nodes, state, stages, topology)
│   ├── agents/       — agent prompts + output models
│   ├── providers/    — evidence providers (one class per provider, self-describing)
│   └── ...           — supporting modules (judge, confidence, gates, dedupe, ...)
├── deep_research/    — web research pipeline (search → fetch → extract → verify → synthesise)
├── document_store/   — SQLite + FTS5 + sqlite-vec personal knowledge base
├── whetstone/        — multi-lens document review over user drafts
├── scribe/           — block-based document authoring → .docx
├── figures/          — publication-quality figure generation
├── chunker/          — structural-first semantic markdown chunking
├── harvest/          — universal source → markdown extraction
├── vision_critique/  — bounded LLM critique of rendered figures
├── proofread/        — deterministic readability + style checking
└── typeset/          — typesetting system (HTML + PDF output)
```

## Where to read next

- `CLAUDE.md` — full project conventions, sub-module dependencies,
  command reference, and the canonical green-state baseline.
- `doc/epistemic/overview.md` — long-form epistemic-module
  walkthrough, including the philosophical foundations (Peirce,
  Popper, Lakatos, IBE).
- `doc/epistemic/epistemic_flow.html` — the same content as an
  illustrated HTML page.
- `src/andamentum/epistemic/providers/CONTRIBUTING.md` — how to add a
  new evidence provider.
- `docs/superpowers/plans/` — historical design plans for major
  refactors. The most recent (description-driven dispatch) is the
  current architecture.

## Pre-release checkpoint

This codebase has been frozen for a pre-release version
(`0.X.0-rc1`). Behaviour is locked. Pre-release artifacts:

- `docs/results/dev30_v9.md` — benchmark results pinned to this
  release tag.
- `CHANGELOG.md` — narrative of the development arc that led here.
- Test baseline: pyright 23 errors, ruff clean, pytest 2075 passing.
