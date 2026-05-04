# Entry-point consolidation — one mode parameter, two modes

**Goal.** Replace the current entry-point surface (`decompose: bool` argument
on the function + a separate hidden mode signalled by `Objective.claim_to_verify`)
with a single explicit `mode: Literal["verify", "research"]` parameter that
constructs the Objective internally. Eliminates the silent-precedence bug
the SciFact harness hit, and gives the CLI a symmetric way to invoke
single-claim verification.

**Non-goals.** No graph-topology changes. No changes to scrutiny / IBE /
verification / synthesis. The downstream pipeline is untouched.

---

## Final shape

```python
# Programmatic
await run_research_question(
    question_or_claim: str,
    mode: Literal["verify", "research"] = "research",
    ...,  # other kwargs unchanged
)

# CLI — symmetric
andamentum-epistemic ask "<question>"             # mode=research (default)
andamentum-epistemic verify "<claim>"             # mode=verify
```

In `research` mode, the graph always tries `Decompose`. If the decomposer
produces meaningful sub-investigations, `MultiSeedClaim` mints N claims.
If it produces nothing usable, the existing fallback in `CreateClaims`
(graph/nodes.py:635-649) routes to `ProposeClaims` — the open-research path
emerges naturally, no separate flag needed.

In `verify` mode, the graph skips `Decompose` and goes straight to
`SeedClaim` from the user-provided text.

---

## What changes (~100-150 LOC across 5 files)

| File | Edit |
|---|---|
| `entities/objective.py` | `model_validator` rejecting `claim_to_verify + decomposition` co-existence (defense in depth, ~10 LOC) |
| `graph/__init__.py:run_epistemic_graph` | Replace `decompose: bool` with `mode`; construct `Objective` from mode + claim/question (~40 LOC) |
| `graph/nodes.py:Decompose` | Gate flips: run unless `obj.claim_to_verify` is set (today gated by `state.decompose`). Drop `state.decompose` reads. (~5 LOC) |
| `graph/state.py` | Remove `decompose: bool` from `EpistemicGraphState` (or keep for one release with deprecation note) |
| `operations_runner.py:run_research_question` | Same signature change as graph entry point (~10 LOC) |
| `cli.py` + `cli_handlers.py` | Drop `--decompose`; add `verify` subcommand (~30 LOC) |

Test sweep: ~15-20 call sites across `test_stage_runners.py`,
`test_decompose_node.py`, `test_question_validator.py`,
`test_no_silent_fallbacks.py` migrate from `decompose=...` to `mode=...`.
Entity-level tests touching `claim_to_verify` directly (the bulk of the 58
references) don't change.

---

## Phases

### Phase 1 — Entity invariant (safety, no behaviour change)

- [ ] Add Pydantic `model_validator` on `Objective` rejecting both
      `claim_to_verify` and `decomposition` set simultaneously.
- [ ] Test: bad construction raises with a message that names the precedence
      rule. Good constructions pass.
- [ ] **Acceptance:** ruff/pyright clean, all existing tests pass (the
      validator should fire on no existing path).

### Phase 2 — `Decompose` runs by default in research mode

- [ ] Change `Decompose.run()` gate from `state.decompose` to "skip if
      `obj.claim_to_verify` is set".
- [ ] Keep `state.decompose` field for one phase (read-only, unused) so
      Phase 3 can remove it cleanly.
- [ ] Verify the existing `MultiSeedClaim → ProposeClaims` fallback still
      catches degenerate decompositions (no code change; just a test that
      asserts the path).
- [ ] **Acceptance:** existing test that asserts open-research-mode behaviour
      now reaches `ProposeClaims` via the fallback path, not via the gate
      skip. All tests pass.

### Phase 3 — New entry-point signature

- [ ] Add `mode: Literal["verify", "research"]` to `run_epistemic_graph` and
      `run_research_question`.
- [ ] Internal construction: `verify` → seed `Objective(claim_to_verify=...)`;
      `research` → seed `Objective(description=...)`.
- [ ] Drop `decompose: bool` parameter and `state.decompose` field.
- [ ] **Acceptance:** all entry-point tests rewritten to use `mode=...`;
      pyright clean (the type signature change catches every stale call site).

### Phase 4 — CLI

- [ ] Add `verify` subcommand mirroring `ask`'s flags but taking a claim
      and routing to `mode="verify"`.
- [ ] Drop `--decompose` from `ask` (now no-op behaviour-wise; hidden by
      research-mode default).
- [ ] Update `--help` text and CLAUDE.md's CLI table.
- [ ] **Acceptance:** `andamentum-epistemic verify "<claim>"` runs end-to-end
      against a saved DB with one minted claim. `andamentum-epistemic ask`
      runs the research path with decomposition attempted by default.

### Phase 5 — Closeout

- [ ] CLAUDE.md: rewrite the entry-point description; remove the
      `--decompose` mention.
- [ ] Memory: write a feedback entry capturing the lesson — "mode lives in
      one parameter at the entry point; never split it across a function arg
      + entity field again."
- [ ] One probe in `verify` mode against a SciFact claim to confirm the
      benchmark harness call shape matches the new API.

---

## Risks

| Risk | Mitigation |
|---|---|
| Wasted `DecomposeQuestion` LLM call on questions that don't decompose | Existing `MultiSeedClaim → ProposeClaims` fallback absorbs it; the optimisation (decomposer signals "non-decomposable" and `Decompose` short-circuits) is a follow-up, not part of this change |
| Breaking change for external consumers passing `decompose=True/False` | The function signature change is loud (TypeError) rather than silent; pyright catches in-tree call sites; out-of-tree (the SciFact harness) gets a clean error message |
| Test churn | ~15-20 call sites; mechanical rename. Phase 3 is the choke point — the type change forces the migration in one commit |
| Behaviour drift between old `decompose=False` (skip decompose entirely) and new `research` mode (try then fall back) | Document in CLAUDE.md; one extra LLM call per non-decomposable run is the only observable difference, and the fallback shape is what users actually want |

---

## Acceptance criteria for the whole effort

1. `andamentum-epistemic ask "<question>"` runs research mode with
   decomposition attempted by default; falls back to ProposeClaims when
   the question doesn't decompose.
2. `andamentum-epistemic verify "<claim>"` runs single-seed verification.
3. `run_research_question(question, mode="verify"|"research", ...)` is the
   one programmatic entry point.
4. `Objective` cannot be constructed with both `claim_to_verify` and
   `decomposition` set — Pydantic refuses loudly.
5. All existing tests pass; pyright + ruff clean.
6. CLAUDE.md's entry-point description fits in ~5 lines, not the current
   half-page.
7. The SciFact harness's call shape becomes a single line:
   `await run_research_question(claim_text, mode="verify", model=...)`.

---

## What this plan does NOT cover

- The "decomposer signals non-decomposable" optimisation (cuts one LLM call
  for open-style questions). Useful but separable. Follow-up.
- Stage-runner CLI changes — the `stage` subcommand still takes
  `--decompose` today; updating it is part of Phase 4 if scope allows,
  otherwise a small follow-up.
- The benchmark harness itself. Once the new API is in place, the harness
  becomes a one-liner.

---

*Written 2026-05-04 to scope the entry-point consolidation surfaced by the
SciFact harness's silent claim_to_verify + decompose=True bug.*
