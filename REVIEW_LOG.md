# REVIEW_LOG — `andamentum.forge`

Iterative review-and-fix loop, scoped to the forge meta-system (`src/andamentum/forge/`).
Severity bar: **P0** blocker · **P1** real problem · **P2** polish (logged, not fixed unless
trivial + zero-risk).

---

## Cycle 1

**Lenses used:** Security · Error-handling & edge-cases · State management & data-flow ·
API / contracts · Data model & integrity · Performance · Config / build / deploy · Tests &
correctness · Import / ingestion.
**Lenses N/A for a CLI/library (not reached, by design):** UX flows · UI / visual ·
Accessibility.

**Method:** five independent review agents across distinct lenses, each verifying against
the real code; findings then adjudicated (deduped, dropped false positives, re-verified the
load-bearing security claims by running proof-of-concept against the actual functions).

### Core-reliability verdict (the primary ask)
The self-correction loop's **termination, best-selection, regression guard, and settle
re-materialisation are correct** — traced across all branches (works-on-first-pass, empty
attribution, regression, cap-exhaustion, multi-round chains): no infinite loop, no off-by-one,
no wrong-package-on-disk. The findings were elsewhere.

### P0 fixed
- **[Security] Silent sandbox downgrade → unisolated execution.** `PodmanSandbox.run` silently
  fell back to the non-host-isolated `SubprocessSandbox` (a `_log.warning`, no raise) when
  `podman` was missing on a *pure* run — so LLM-authored code could execute un-contained with
  no hard signal when the caller relied on the isolated default. *Fixed* (`sandbox.py`): podman
  missing now **fails loud** (`SandboxUnavailableError`) for every run; the un-isolated
  subprocess is only ever used when the caller *explicitly* passes `--sandbox subprocess`.
  Removed the `_fallback`/`_log` machinery. Regression tests added.

### P1 fixed
- **[Security] Purity gate bypassable.** Verified by PoC that `check_purity` passed bodies
  using `importlib.import_module(...)`, the `().__class__.__bases__[0].__subclasses__()`
  dunder-escape, and builtin aliasing (`bad = eval`). *Fixed* (`astcheck.py`): banned
  `importlib`/`builtins`/`gc`/`posix`/`pty`/`code`/`codeop`; flag interpreter-internals attrs
  (`__subclasses__`/`__bases__`/`__mro__`/`__globals__`/`__builtins__`/…); flag any *load* of a
  banned builtin (catches aliasing), not just calls. Adjudication note: the purity gate is a
  *quality/defense-in-depth* gate, **not** the containment boundary (that is the sandbox) —
  AST analysis cannot soundly prove capability absence in Python; the overstated "build-time
  guarantee" docstring was corrected to say so.
- **[Security] Podman missing hardening flags.** *Fixed* (`sandbox.py`): added `--cap-drop=all`,
  `--security-opt=no-new-privileges`, `--cpus=2` to the `podman run` profile for untrusted code.
- **[Error-handling] Attribution not crash-aware for import failures.** A pytest
  collection/import error (package won't import) fell through the fallback ladder to
  "rebuild all spine nodes" — burning a bounded rebuild round on a failure re-authoring can't
  fix. *Fixed* (`audit.py` + `attribute.py`): parse pytest's distinct `error` count
  (`tests_errored`) and route an import/collection error to a **loud terminal**; a behavioural
  `failed` assertion still flows to the normal rebuild path (the error/failure distinction is
  the precise, safe signal — not a 0/0 inference, which would misclassify behavioural failures).
- **[Data model] `AuditRound.rebuild_targets` / `reauthored` provably identical + inverted
  docs.** *Fixed* (`schemas.py`, `graph.py`, `cli.py`, tests): collapsed to a single
  `rebuild_targets` field with a corrected description; dropped the redundant `reauthored`.
- **[API] `ForgeResult` field types absent from `__all__`.** Types reachable from the exported
  result (`AuditRound`, `NodeFinding`, `Fitness`, `DesignReport`, `PlanVerdict`,
  `CriticVerdict`, `RequirementsVerdict`, `BuildConcern`, `AuditIssue`) weren't public,
  violating the project's public-API convention. *Fixed* (`__init__.py`): exported them.
- **[Tests] `cli.main()` (incl. `--json`) had zero coverage.** *Fixed*: added
  `test_review_hardening.py` covering the `--json`, human-summary, and error-exit paths.
- **[Tests] Generated `validate_input` never exercised.** *Fixed*: added a test asserting the
  generated fail-loud input door.

### P2 fixed (trivial + zero-risk only)
- Stale `Assess` docstring (said `stateful_function` is refused; it is buildable) — corrected.
- `schemas.py` module docstring didn't mention the new `.runtime` import — corrected.
- Settle re-materialisation emitted phantom reporter build-events after Build was "done" —
  now runs with no reporter (deterministic disk fix-up, not a build stage).

### P2 logged, NOT actioned (see backlog below)
- **[Perf] Sequential per-node LLM typing in Decompose** (Stage-2 + repair rounds), and
  **Review→Frame re-runs Decompose from scratch**, and the **component-manager call ~doubles
  build-stage calls**. These explain the observed ~631s local-model Decompose. Deferred
  deliberately: (a) the ollama runner caps concurrency to 1, so parallelising doesn't reduce
  wall-clock — the lever is *fewer* calls (batching node-typing / memoising across redesign),
  a real design change; (b) the design loop's per-node sequential typing is what makes it
  *converge on small models* — the project's own history warns that efficiency changes at
  convergence sites regress quality silently. This needs a dedicated, benchmarked effort, not
  a review-loop edit.
- **[Perf] Per-hole sequential build** — load-bearing (all bodies share `nodes.py`; concurrent
  authoring would race), not a bug.
- **[Config] `Containerfile` deps unpinned** (only `pydantic-graph` ranged; base image a
  mutable tag). Real supply-chain hygiene gap; pinning needs an image build to validate (no
  podman here), so deferred rather than done blind.
- **[Security] `HOME`/`PATH`/`VIRTUAL_ENV` in the scrubbed subprocess env** — minor info
  disclosure; the subprocess path is the explicit opt-in un-isolated tier.
- **[Ingestion] No brief length cap / no control-char / prompt-injection tests** — low risk:
  the brief only reaches the design heads; everything downstream is deterministic + sandboxed,
  so a hostile brief can at worst fail the fitness gate or a red audit, not escape.

### Outstanding P1 — deferred with rationale (needs a human/design decision)
- **[Security] Network isolation is per-test-process, not per-node.** `_run_tests` passes
  `allow_network=spec.has_network` once for the whole pytest run, so a system with *any*
  network node runs *every* node's code in a network-enabled container. A proper fix needs
  per-node test isolation (the smoke runs the whole package's pytest in one container, and a
  network node's body genuinely needs network in its smoke) — a re-architecture whose rushed
  version risks breaking the audit. Flagged for a design decision rather than a hasty fix.
  Mitigation in place: still fully containerised (host-isolated) under podman; this is a
  defense-in-depth gap, not a host escape.

### Changed files
`sandbox.py` (no silent fallback + hardening flags) · `astcheck.py` (purity hardening + docs) ·
`attribute.py` (import-error loud terminal) · `audit.py` (`tests_errored` parse) · `schemas.py`
(`AuditRound` collapse, `CheckResult.tests_errored`, docs) · `graph.py` (AuditRound ctor,
settle reporter, Assess docs) · `cli.py` (AuditRound field) · `__init__.py` (public exports) ·
tests: `test_review_hardening.py` (new, 10) · `test_attribute.py` (+2) · plus the earlier
self-correction / reporting / envelope work in this branch.

**Verification:** `pytest src/andamentum/forge/tests` → **164 passed**; `pyright` → 0 errors;
`ruff check` → clean; `andamentum-agentic-dialect check` → passes.

---

## Cycle 2

**Lenses re-run on the changed code:** Security · Error-handling · API — an adversarial pass
over exactly the Cycle-1 diff, hunting for problems the fixes themselves introduced (not
re-reporting the fixed originals). Lens list otherwise exhausted (UX/UI/accessibility N/A).

**Result:** the Cycle-1 fixes hold (sandbox no-fallback + hardening, `AuditRound` collapse,
`__all__` exports, purity dunder/import closure all clean), but **two fixes over-reached and
introduced new P1 false-positives** — caught and fixed here.

### P1 fixed (introduced by Cycle 1)
- **`_parse_counts` matched "error" anywhere in pytest output, not the summary line**
  (`audit.py`). A behavioural failure whose assertion prose mentioned "errors" (common for
  forge's target domain) set `tests_errored>0` → misrouted to a loud terminal instead of a
  rebuild. *Fixed*: scope the parse to pytest's summary line, identified by its trailing
  duration (`in 0.10s`) or `===` wrapper — assertion prose has neither. Regression tests
  added (`test_review_hardening.py`).
- **The purity Name-load ban flagged legit locals named `input`/`open`/`vars`**
  (`astcheck.py`). Broadening the aliasing check to any Load of a banned builtin caught
  ordinary variables. *Fixed*: exclude names the body binds locally (params + Store/Del
  targets) from the Load ban — a shadowing local is fine, while unbound aliasing
  (`bad = eval`) is still caught. Regression tests added.

### Verification
`pytest src/andamentum/forge/tests` → **168 passed**; `pyright` → 0 errors; `ruff check` →
clean; `andamentum-agentic-dialect check` → passes. Full repo suite green except the 4
pre-existing OpenAlex live-API smoke failures (HTTP 503, external, unrelated).

---

## Cycle 3 (verification pass) — TERMINATE

Re-ran the changed-code lenses over the two Cycle-2 fixes (regression suite + targeted edge
analysis: `no tests ran` / mixed-count summaries; a shadowed-local `eval` stays safe while
the `__mro__` reach-around is still caught). **No new P0/P1 surfaced.** The loop terminates
on a clean pass (3 of a max 5 cycles).

---

## Final summary

1. **Cycles run:** 3. **Stopped because:** a clean verification pass (Cycle 3) surfaced no new
   P0/P1 — not the 5-cycle cap.
2. **Lens coverage:** *used* — Security, Error-handling & edge-cases, State management &
   data-flow, API / contracts, Data model & integrity, Performance, Config / build / deploy,
   Tests & correctness, Import / ingestion. *Never reached (N/A for a CLI/library)* — UX
   flows, UI / visual, Accessibility.
3. **P0/P1 fixed:** Cycle 1 — 1 P0 (silent unisolated sandbox downgrade) + 7 P1 (purity-gate
   bypass, podman hardening flags, import-error loud terminal, `AuditRound` field collapse,
   `__all__` exports, `cli.main`/`--json` coverage, generated `validate_input` coverage);
   Cycle 2 — 2 P1 regressions from the Cycle-1 fixes (`_parse_counts` over-match, purity
   local-name false-positive). Details above.
4. **Outstanding P2 backlog (not actioned):** decompose sequential-typing performance
   (~631s on small models — deferred, benchmark-gated, quality-regression risk); per-hole
   sequential build (load-bearing, not a bug); unpinned `Containerfile` deps; `HOME`/`PATH`
   in the scrubbed subprocess env; no brief length-cap / injection tests.
5. **For a human to confirm:** the one **deferred P1** — network isolation is per-test-process,
   not per-node (a system with any network node runs every node's code in a network-enabled
   container). Fixing it properly needs a per-node test-isolation design decision; a rushed
   fix risks breaking the audit. Still fully host-isolated under podman — a defense-in-depth
   gap, not an escape. Flagged for your call rather than a hasty change.
