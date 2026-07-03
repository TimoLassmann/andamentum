# D. PRD — Self-Correction (closing forge's dynamic verification loop)

*A product requirements document, a sibling to document C. C made rung-2 statefulness real. This
document specifies a different, orthogonal capability: making forge **self-correct** when the
system it built does not actually run. It adds one bounded loop — an `Audit → Rebuild` back-edge —
plus the single field that makes each rebuild surgical, and nothing else.*

*Code examples are illustrative of intent, not final implementations. The end state and the
behavioural contract are normative; the exact line-by-line code is the coding agent's to
finalise against the real source.*

*Unlike documents A/B/C, this work needs **no dialect change** (see §8). It is entirely internal
to forge; the systems forge generates are byte-identical before and after. There is no document
"B-for-self-correction" because the decision never crosses the boundary into generated code.*

*(This is revision 4. Review resolutions are logged in §13.)*

---

## 1. The end state, in one paragraph

After this work, when a built system fails its own sandboxed audit — a generated test crashes, an
import is missing, `check_code` flags a dialect violation in a node body, a hole was left unfilled,
or the critic catches a hardcoded stand-in — forge does not hand back a broken system with
`works=False` and stop. Instead it **attributes** the failure to the specific node(s) responsible,
routes back through Render (which re-emits the pristine package deterministically), **re-applies the
node bodies that were fine and re-authors only the attributed ones** with the concrete failure fed in
as repair context, and **re-audits** — up to a small bounded number of rounds, and never past a
round that makes things worse. If it converges, it returns a working system. If it exhausts the
rounds, it returns the **best** build it reached — deterministically re-materialised onto disk so the
package the user gets always *is* the best one — with a loud, structured, round-by-round account of
what failed and what it tried. The rebuild is surgical because each successfully authored body is now
carried verbatim on its `FilledNode`, so the pristine re-render can be repopulated with the good work
and only the culpable nodes re-authored. The whole change lives inside forge's own graph; the render
output and the generated package are unchanged.

That is the whole deliverable. Everything below specifies it precisely.

---

## 2. What this explicitly is, and is not

**Is:** the closing of forge's *dynamic* verification surface. Static gates already verify a body
before it runs; this closes the loop on what only execution reveals. A bounded back-edge that routes
`Audit → Render → Verify → Build`, a deterministic attribution of failures to nodes, a targeted
re-author, a best-so-far/regression guard, and one new field (`FilledNode.body`) that makes the
targeted re-author possible.

**Is not:** a new architecture. It is **not** a runtime orchestrator, **not** a dynamic task
ledger, **not** a conductor that plans work at runtime. The hole-list is still fully known after
render; this loop *repairs* known holes with runtime feedback, it does not *discover* work. It does
not raise forge's rung ceiling, add tools to generated agents, or change what forge builds — only
how reliably it finishes building it.

**Is not** a new `WorkerReport` type. An earlier draft proposed one; review showed the only datum the
loop actually consumes is the authored body, and everything else it needs is already on `BuildReport`
or re-derivable from `spec`. So the "report" collapses to a single field (§5). The same YAGNI
discipline the PRD applies to speculative LLM output applies to speculative state.

**Is not** a replacement for the static gates or the component manager. Those remain the first line
and stay exactly as they are. This loop is the *last* resort, reached only when a body that passed
every static check still fails on execution. A design goal is that it fires **rarely** (§6.3).

---

## 3. The two verification surfaces (the framing)

Forge verifies generated code on two surfaces. It closes one today and drops the other.

- **Static surface — closed.** The gate suite (`astcheck.py`: contract, coverage, purity, deps,
  fail-loud) verifies a body *without running it*. The per-hole draft→repair loop (`build.py`,
  `attempt_cap`) closes this surface tightly.

- **Dynamic surface — open.** Whether the code *behaves* when executed is checked once in
  `audit.py` (sandboxed `pytest` + `check_code` + requirements/critic heads) and then **discarded**:
  `works=False` is handed straight to `Finish`. A system whose generated tests fail is returned as-is.

The loop in this document closes the dynamic surface using the same machinery that closes the
static one. Static catches what is visible without running; dynamic catches what is only visible on
execution. Together they are complete up to the reach of the generated tests (§6, the wall).

A structural consequence to hold onto (it recurs below): the inner per-hole `attempt_cap` loop
verifies **only static gates** — it cannot confirm a *dynamic* failure is fixed. So a re-authored
body can pass every gate and still re-fail identically at audit. That is precisely why the outer loop
needs a monotonic guard (§4.5), not just a cap: the inner loop guarantees *static* validity each
round, the outer guard guarantees the rounds *make progress*.

---

## 4. The loop — `Audit → Render → Verify → Build`

### 4.1 Where it lives: a graph back-edge, deterministically routed

The loop is a back-edge in forge's own graph. Critically, it routes to **`Render`, not `Build`** —
because re-authoring an already-filled node requires the pristine package back, and `render()` is the
only thing that produces it (`build.py`/`extract.py` only ever *fill* `NotImplementedError` holes;
after round 1 a filled node has no hole left for Build to touch). Routing through Render re-emits the
pristine holes deterministically, then the existing forward path `Render → Verify → Build → Audit`
carries the rebuild home. Render and Verify are unchanged; only Build (consumes the rebuild inputs,
§4.3) and Audit (computes attribution + routes, §4.4) change.

The routing decision is a **pure predicate**, never a model call:

```
works = tests.passed and dialect.passed and not remaining_holes    # unchanged: audit.py:193
route:  works                          → Finish
        (not works) and improvable     → Render   (rebuild; improvable defined in 4.5)
        otherwise                      → Finish    (best-so-far, works=False, full history; 4.6)
```

`improvable` folds together "under the round cap", "attribution found a fixable target", and "the
last round strictly improved" — all pure, all §4.5. The load-bearing conformance point, stated
first: **the LLM never decides whether to loop.** `works` is a deterministic function of sandboxed
test results; the critic and requirements heads produce *findings* (data); attribution turns findings
into *targets* (data); the graph routes on the predicate. This is the dialect's own rule — "an LLM
call never also picks the path" — the same rule behind `compile_spec._split_multi_successor_heads`.

### 4.2 State, result, and cap

Mirror the existing `plan_review_rounds` / `MAX_PLAN_REVIEW_ROUNDS` pair (`graph.py:59, 102, 219`):

- `ForgeState.audit_rounds: int` — the **number of rebuilds performed so far**. Starts at `0`;
  incremented by one each time Audit routes back to Render. This counts *rebuilds*, not audit passes
  — pinned to remove the off-by-one ambiguity (the guard reads cleanly as "rebuilds so far <
  max rebuilds"; see §4.5). Total audit passes = `audit_rounds + 1`.
- `ForgeState.rebuild_targets: list[str]` — node names Audit attributed this round, **sorted** for
  determinism (data: the nodes audit found culpable). Empty on the first Build ⇒ author all holes,
  today's behaviour. Consumed by Build, then cleared.
- `ForgeState.best_build: BuildReport | None`, `ForgeState.best_audit: AuditReport | None` — the best
  result reached so far by the total order in §4.5.
- `ForgeState.audit_history: list[AuditRound]` — one entry per audit pass (`AuditRound` = pass index,
  `rebuild_targets` that produced it, the audit's failing checks, and the set of node names
  re-authored). New small Pydantic model in `schemas.py`.
- `ForgeDeps.max_audit_rounds: int` — the **maximum number of rebuilds**, default **2** (so up to
  3 builds total: the initial build plus two repairs). Tune against the benchmark; do not guess.

`AuditReport.rounds` — the field the current code hardcodes to `1` (`audit.py:196`) and reads nowhere
— becomes the real audit-pass count (`audit_rounds + 1`). That vestigial field is the seam this loop
was left half-built around.

`ForgeResult` (`schemas.py:400`) gains `audit_history: list[AuditRound]` and, on a non-converging
run, carries the **best** build/audit (§4.6). `Finish` (`graph.py:294-322`) must be changed to read
`state.best_build`/`state.best_audit` (falling back to `state.build`/`state.audit` when the loop never
ran) rather than unconditionally the last.

### 4.3 The targeted rebuild mechanism (deterministic, minimal)

On a red audit, forge re-authors **only** the implicated nodes and preserves the good work:

1. **Render re-emits the pristine package** (deterministic and idempotent — the back-edge routes here
   for exactly this reason, §4.1). This overwrites `nodes.py` back to holes; the good bodies survive
   in `best_build`, on state, not on disk.
2. **Build re-applies every non-target body verbatim** — read from `best_build.filled[node].body`
   (the new field, §5). These nodes are not re-authored; their known-good source is patched straight
   back in via the existing `apply_body`.
3. **Build re-authors only the attributed targets** via the existing `_build_one`, with the audit
   failure added to the repair context (a new feedback channel alongside the gate-violation feedback
   `_repair_context` already threads).

So `build_system` grows two optional parameters — `prior_bodies: dict[str, str]` and
`targets: set[str] | None` — with an explicit **tri-state** on `targets` (spell it out, because "empty
list" and "empty set" mean opposite things here):

- `targets=None` — author **every** hole. Today's behaviour; the first Build.
- `targets={a, b}` — author **only** those holes, re-apply the rest from `prior_bodies`. A rebuild.
- `targets=∅` (empty set) — author **nothing**, re-apply everything from `prior_bodies`. The §4.6
  deterministic re-materialisation.

The `Build` node therefore maps an **empty `rebuild_targets` list → `None`** (author-all on the first
pass), never to `∅`; the empty *set* is reserved for the `Finish` re-materialisation call. No re-render
inside Build (that stays the Render node's job, so the render/build boundary the module CLAUDE.md
protects is intact); no partial-file surgery. Every rebuild starts from the deterministic pristine
render and re-applies known-good bodies. This is the simplest correct design.

*Edge case (unfillable target):* a node that stayed `unfillable` last round has no `body` to re-apply
and left a `NotImplementedError` hole, which is *why* `works=False` (`not remaining_holes`). Such
holes are **unconditionally** eligible targets (§4.4) — Build re-attempts them as normal holes. They
get `attempt_cap` static-gate tries with no new dynamic information, so they may fail identically —
and the §4.5 regression guard then stops the loop after that single non-improving round. This
terminates correctly but costs `attempt_cap` calls; that is the bounded price of trying.

### 4.4 Attribution: deterministic first, LLM only as a supplement

A red test names no culprit. Compute `rebuild_targets` with a pure function
`attribute_failures(audit, build, spec, pkg) -> list[str]`, ranked so deterministic signals lead and
the model is only ever a supplement. Return the result **`sorted()`** for reproducibility. The `pkg`
argument is the package path: signals 1 and 2 both need to AST-map a `line` to its owning node class,
which requires the **filled** `nodes.py` source, so **attribution must run inside the `Audit` node,
before the back-edge re-enters `Render` and overwrites `nodes.py` back to holes.** (Equivalently,
retain the filled `nodes.py` source on the audit result; passing `pkg` and reading it in `Audit` is
simpler.)

1. **`check_code` violations, scoped to `nodes.py`.** `_run_dialect` runs `check_code` over the whole
   package (`audit.py:116`), so it also flags render-owned files (`deps.py`, `graph.py`, `models.py`).
   A violation **outside `nodes.py` is a forge render bug, not something re-authoring a node body can
   fix** — filter signal 1 to violations whose file is the generated `nodes.py`, map each to its
   owning node by line (the AST line-map of signal 2), and treat any non-`nodes.py` violation as a
   **loud terminal** (a forge defect surfaced honestly), never a rebuild target.
   **This requires a data-source change, symmetric to signal 2's:** `_run_dialect` collapses
   `check_code`'s structured `list[Violation]` (each carries `.file`/`.line`/`.law`, `checks.py:25-32`)
   into a **truncated formatted string** on `CheckResult.detail` (`audit.py:121-124`) — nothing retains
   the untruncated structured list. So this PRD must surface the raw `list[Violation]` on the dialect
   result (`CheckResult`/`AuditReport`), and `attribute_failures` must consume *that*, never re-parse
   the `detail` string. Without this, signal 1 has no structured input and acceptance §9.2 is
   unbuildable.
2. **Failing-test tracebacks → node, by AST line-mapping.** Do **not** rely on a class name appearing
   in a traceback frame — generated node methods are `async def run(...)`, so the frame header reads
   `in run`, and the class only surfaces in pytest's verbosity-dependent local-repr line (audit runs
   `-q`, `audit.py:85-93`). Instead, parse the `nodes.py:line` frames from the traceback (always present
   in the frame headers) and map each line to its owning node class via a small AST pass over
   `nodes.py` (top-level class → line span; the generated classes are top-level `@dataclass`
   `BaseNode` subclasses, `render.py:180-292`, so the spans are clean). Deterministic and robust. A
   frame landing in a render-owned top-level helper (e.g. a `_user_prompt_*` function, `render.py:385-390`)
   maps to no class span and is correctly dropped — it is not an authored body, so it falls through to
   the fallback ladder / loud terminal, never mis-attributed.
   **This requires a data-source change:** today `_run_tests` collapses the sandbox result to a
   summary, and `_pytest_summary` (`audit.py:57-71`) keeps only `FAILED`/`ERROR` lines and the count
   line — it **discards the traceback body where the `nodes.py:line` frames live.** So this PRD must
   have `_run_tests` retain the **raw pytest stdout** (or pre-extracted `(file, line)` frames) on the
   audit result (`CheckResult`/`AuditReport`), alongside the count parsing of §4.5. `attribute_failures`
   then extracts the frames from that retained output. Also **pin the traceback style** — add
   `--tb=short` to the pytest invocation (`audit.py:85-93` runs `-q` only, relying on the default
   `--tb=auto`) so `nodes.py:line` frames are emitted for every failure regardless of count or a future
   config change. Without this, signal 2 has no input and the crash case (§9.1) is unattributable.
3. **`BuildConcern` nodes** — the component manager already flagged these as suspect bodies during
   build (`BuildReport.concerns`, `schemas.py:331`); a behavioural failure makes them prime suspects.
   Already computed, zero new work.
4. **Critic-named nodes** — the *one* LLM signal. Tighten the `CRITIC` head's output to include a
   `node: str` per finding (`list[NodeFinding]`). Because the critic's output model is a fixed
   Pydantic class and node names are per-spec, `node` cannot be a `Literal`; it is a free string
   **reconciled to a real node name via rapidfuzz** — the exact tiered-match pattern `diagnose.py`
   already uses. An unreconcilable name is dropped (not a target).

`targets = ((1 ∪ 2 ∪ 3 ∪ 4) ∩ authored_nodes) ∪ remaining_holes`, where `authored_nodes` is
`{n.node for n in BuildReport.filled} ∪ {n.node for n in BuildReport.unfillable}` (both own a
body-or-hole; no stored field needed) and `remaining_holes` are the still-unfilled nodes
(`BuildReport.unfillable`) — **unconditionally** eligible because a hole is *definitionally* work that
needs doing and is why `works=False`. **Fallback ladder** if the attributed set is empty but the
audit is red: all `BuildConcern` nodes → all authored spine nodes. Never target heads on a
behavioural failure unless a traceback (signal 2) implicates one — heads produce structured output and
rarely carry the behavioural bug. If attribution is **still** empty (audit red, nothing attributable —
e.g. a package-level import error or a non-`nodes.py` dialect violation), that is a **loud terminal**
(`improvable=False`), because re-authoring cannot fix it.

Attribution is **pure** — it consumes already-produced audit findings and makes no model call of its
own. The model contributes at most one of four ranked signals, and only ever as data.

### 4.5 Termination: cap *and* monotonic progress

The cap alone is insufficient — the inner loop only guarantees static validity, so a re-authored body
can pass gates and re-fail identically at audit (§3), and a naive loop would thrash (fix one node,
break another). Two guards, both deterministic:

- **Cap.** `audit_rounds < max_audit_rounds`, where `audit_rounds` is the number of rebuilds already
  performed (§4.2). With the default `max_audit_rounds = 2`, the loop performs at most two rebuilds
  (three builds, three audit passes). Bounds total LLM work; same shape as every other forge cap.
- **Regression guard.** Rank every audit result by the total order
  `(works, tests_passed, −tests_failed, dialect_clean)`. After a round's audit completes, compare it to
  the best reached *before this round* (`best_before`, which is `None`/−∞ on the first pass, so the
  first round always "improves"): `improved = current ≻ best_before`. If it improved, it becomes the
  new best; if it did not (strictly-better fails — a tie counts as non-improving), the loop stops and
  keeps `best_before`. This is the load-bearing safety, the audit-loop analogue of `PEIRCE_CYCLE_CAP`:
  the cap bounds effort, the monotonic guard bounds *thrash*. (A tie is treated as non-improving on
  purpose — no new information was gained, so spending another rebuild is thrash.)

  **This requires pass/fail counts the audit does not produce today.** `_run_tests` returns a single
  `CheckResult(passed: bool, detail: str)` (`audit.py:74-112`) and `_pytest_summary` (`audit.py:57`)
  keeps the `=== N passed, M failed ===` line only as *text*. So this PRD must add: parse that summary
  line into `tests_passed: int` / `tests_failed: int` and surface them (on `CheckResult` or
  `AuditReport`), *together with* retaining the raw stdout for signal 2 (§4.4). Without counts the
  order degrades to `(works, dialect_clean)` — near-boolean, much weaker thrash detection. The count
  parsing is in scope and specced here, not assumed.

  *Degenerate case:* a pytest collection/import failure prints `=== 1 error in ... ===` with no
  `passed`/`failed`, so both counts parse to `0` — indistinguishable from "0 collected" by the order
  alone. That is acceptable and deliberate: such failures carry no `nodes.py` frame, so they attribute
  to nothing and route to a loud terminal (`improvable=False`) rather than relying on the order's
  tie-break. Document the `0/0` tie as intentional, not accidental.

`improvable` (§4.1), evaluated after the just-completed round's audit = `audit_rounds < max_audit_rounds`
**and** attribution found a fixable target **and** `improved` (the round just completed strictly
improved on `best_before`, per the regression guard; always true on the first audit). Route to Render
iff `improvable`; else settle.

### 4.6 Settle as a *result*, not a raise

A deliberate departure from forge's other caps. `Assess` / `Decompose` / `Review` **raise** at their
caps because those are *design* failures with nothing to hand back. At the audit cap the system **is
built** — it just does not pass. That is a *result*, not a design failure. So:

- When `improvable` is false (cap hit, or non-improving round, or nothing attributable), the loop
  settles on `best_build`/`best_audit` and `Finish` returns `ForgeResult(works=False, ...)` carrying
  them plus the full `audit_history` (§4.2). Loud and honest — but it does not throw away a partial
  system the user may still want.

**The on-disk trap — settle must re-materialise best.** The final round's files are what sit in
`dest/` on disk (Render overwrites the package each pass — the `Render` node calls `render()`
(`graph.py:239`), which unconditionally rewrites every file including `nodes.py`, `render.py:697,709`;
Build then writes the round's bodies). When the loop stops on a **regression**, `best_build` is an
*earlier* round, so `dest/` holds the **worse** build while the result reports the best — a
silently-shipped worse package, violating §1 and §11. Therefore, at settle, **if the last round is
not the best, deterministically re-materialise `best_build` onto `dest/`**: a pure render of the
pristine package plus a re-apply of every `best_build` body (`build_system(prior_bodies=best_bodies,
targets=∅)` — **no authoring, no LLM, no re-audit**; render is deterministic, so this reproduces
exactly the files that were audited as best).

This is a deterministic finalisation. Host it in `Finish` (`graph.py:294-322`) or a dedicated
deterministic `Settle` step before it; it writes no new logic, only re-applies already-audited bodies,
so the Assembly law holds. `Finish` is reached on several paths (design/render/build-only, converged
audit, settled audit) — **guard the re-materialisation to fire only when `dest is not None` and
`state.best_audit is not None` and `state.best_audit is not state.audit`** (i.e. the last audit was
not the best), so it is a no-op on every path except a regression settle. On convergence the last
round *is* the best, so it is skipped. The invariant to guarantee: **the package in `dest/` always
equals the `best_build` the `ForgeResult` reports.**

*(Implementation note: `audit_system` is an engine-free worker with no graph-state access today
(`AuditReport.rounds` is a hardcoded `1`, `audit.py:196`). To set `rounds = audit_rounds + 1` it must
be passed the current `audit_rounds` from the `Audit` node — a one-argument thread, not a state
dependency.)*

This obeys "capped at inquiry, not at output": the cap stops the *rebuilding*; it never zeroes out
what was *built*. Forge's contract remains "either a working system or a loud, specific account of
why not" — the loop only widens the "working" outcome.

---

## 5. The record that makes the rebuild surgical — one field, not a type

The user's question was "what should a per-node WorkerReport contain, and can we fill / pre-fill it?"
The honest answer, after checking what the loop actually consumes, is: **so completely that it
collapses to a single field.** Here is the full provenance analysis that yields that conclusion.

**What the rebuild loop actually reads:** to re-apply a non-target node (§4.3 step 2) it needs the
node's authored **body**. To pick targets (§4.4) it needs `check_code` violations, tracebacks,
`BuildConcern`s, and critic findings — *none of which is per-node build metadata*; they come from the
audit and from `BuildReport.concerns`. To know which nodes are authored it needs `authored_nodes` —
already derivable from `BuildReport.filled`.

**So every candidate "report" field is already available or re-derivable, except the body:**

| Candidate field | Where it already lives / how it's re-derived |
|---|---|
| `job`, `contract`, `role_brief` (context) | re-derivable from `spec` + `node_contract(spec, name)` at any time; no need to store |
| `outcome`, `attempts_used` | already implied by `BuildReport.filled` vs `unfillable`; can live on `FilledNode` if wanted, but the loop doesn't route on them |
| `gate_history`, `contract_coverage`, `compile_ok` | **not consumed by the loop.** (Also: `check_node_body` returns only `list[str]` violations, discarding the coverage map — harvesting it would be a refactor for a field with no reader.) |
| `manager_verdict` | already on `BuildReport.concerns` as `BuildConcern` |
| **`body`** | **not stored today — this is the one field to add.** |

**The deliverable, therefore: add `body: str` to `FilledNode`** (`schemas.py`), populated with the
authored source `build.py` already produces. **It must be the exact source that was applied to disk
and audited** — populate it from the value passed to `apply_body` at *both* build return points: the
`body` at the success return (`build.py:305`) and `last_valid_body` at the keep-on-budget return
(`build.py:309-315`, where the trailing `body` is a manager-rejected draft, not what was written).
Storing the wrong variable would make the re-apply diverge from what was audited. **Make `body` a
required field (no default)**: `patch.py:_normalise` turns an empty body into `pass` (`patch.py:54-55`),
so a defaulted-empty `body` re-applied at settle would silently replace a node with `pass` — no logic,
no `NotImplementedError`, a fail-loud violation. No code constructs `FilledNode` with a default today
(grep-confirmed), so a required field is safe and pyright-enforced. No `WorkerReport` type, no new
model output, no speculative metadata. This is the minimal design the standard demands, and it is
strictly what §4.3 step 2 requires.

(The two audit-level inputs attribution also needs — retained raw pytest stdout for signal 2 and the
pass/fail counts for the guard — are *not* per-node report fields; they live on the audit result, per
§4.4/§4.5. This section is only about per-node build data.)

**Deferred (explicitly not v1): a soft `needed_capability` annotation.** The one genuinely
model-only datum — "I cannot implement this job without a capability I don't have" (§7) — would be a
future advisory field on `FilledNode`, and even it has a deterministic shadow (a worker repeatedly
failing the purity gate on the *same* banned import is that signal, visible without asking). Defer it
until attribution proves insufficient; do not add speculative model output.

---

## 6. The reach of the loop, and the wall (normative)

This is the section to be honest against. The audit loop is only as strong as what the audit can
*see*, and today's generated tests are weaker than "shape and type."

### 6.1 What the loop can catch (in scope)

Failures today's audit already surfaces, now *repaired* instead of reported:

- **Crashes / exceptions** on the smoke path (traceback → node by AST line-mapping, §4.4 signal 2).
- **Import errors** for a library not in the sandbox image. Note these usually attribute to no fixable
  node → loud terminal, and may signal a capability the brief needs that forge does not provide (§7).
- **Dialect violations *inside a node body*** (`nodes.py`-scoped `check_code`, §4.4 signal 1). A
  violation in a render-owned file is a forge bug → loud terminal, not a rebuild target.
- **Remaining holes** an earlier build left `unfillable` (deterministic).
- **Hardcoded stand-ins, dropped inputs, silent fallbacks** — the critic's remit (the one LLM signal).

### 6.2 What the loop cannot catch (the wall)

The generated smoke test asserts only that the run reaches `End` without raising (`render.py:665`
asserts `out is not None`); `test_graph.py`/`test_recipe.py` assert assembly and recipe, not
behaviour. So the audit's real behavioural reach is **"runs to completion without crashing"** —
*weaker even than type-correctness*. Therefore the loop **cannot** catch:

- **Any wrong-but-non-crashing logic** — an off-by-one, a wrong formula, a mis-sorted list, a body
  that returns a plausible wrong value. It runs, it returns, the smoke test passes, the loop never
  fires. The critic is the *only* thing standing between such a body and shipping, and the critic
  judges plausibility, not correctness.

Catching this needs forge to generate **behavioural** assertions (expected inputs → expected outputs),
which means deriving ground truth from a brief — a genuinely harder problem and **out of scope for
this PRD.** State the boundary sharply (as C §8 does): the loop makes forge *finish reliably against
the crash-level tests it generates today*; stronger generated tests are a separate, later PRD. Do not
quietly scope behavioural test generation into this work.

### 6.3 The loop must fire rarely

A design constraint, not just an observation: the audit loop is the last resort. The first line stays
the static gates and the component manager, and the *right* place to reduce bugs is to strengthen
those and to ground drafting better (the `role_brief` the build prompt already feeds), not to lean on
the loop. If the loop fires on most builds, that is a signal the static surface has a gap to close
first. The benchmark (§9) reports loop-fire rate for exactly this reason.

---

## 7. The tool question: can forge build a system that uses a new tool (= a python function)?

A user asks: "build me a system that *uses a tool* — a python function that does X." In forge's
model, deterministic work lives in **spine-node bodies**, so "a tool" usually maps to "a spine node
body that `build.py` authors behind the purity gate." The answer is case-by-case, and it connects to
the deferred `needed_capability` annotation (§5).

| Case | Buildable today? | Why |
|---|---|---|
| **Pure deterministic computation** (count, sort, parse, format, apply a formula) | **Yes** — the single thing forge's build stage is *most* designed to do. The hole *is* the function body. | authored by `build.py`, kept clean by the purity gate |
| **Tool that calls the network** (hit an API) | **Yes** | the node declares `network=True`; `check_purity` admits `_NETWORK_MODULES` for it; the sandbox grants network to exactly that node. Control flow stays fixed ⇒ still rung 1/2 |
| **Tool needing filesystem / subprocess / clock / random** | **No — refused by the purity gate**, correctly (the determinism law) | reshape or refuse at `Assess`; do not build |
| **Tool from a library not in the sandbox image** | **No** unless the `Containerfile` is extended (today: pydantic, pydantic-graph, numpy, httpx, rich, pytest) | body authors fine, audit fails at import — exactly the failure the new loop surfaces honestly (§6.1) |
| **A user's *own* existing python function injected as a dependency** | **No mechanism today** | `Deps` has no user-tools port. A real gap — see below. |
| **An LLM *agent* that calls tools at runtime** | **No — and correctly out of scope** | `run_head` produces structured output, not a tool-calling loop; that is rung 4 (agent), refused by `Assess` |

**How this ties back.** The deferred `needed_capability` annotation (§5) is the honest channel for the
three "no" rows. Instead of a worker hallucinating a stub for a job it cannot purely implement, it
would report "this needs a capability I don't have," and the loop could **escalate to an
`Assess`-style refuse-and-reshape** rather than burning audit rounds on an unfixable body. A far
better failure mode than today's silent stub — but, per §5, deferred until proven necessary.

**On the user-supplied-tool gap (case 5): explicitly out of scope for this PRD**, noted because it is
structurally symmetric to C's Store and belongs in the backlog. A `tools: ToolBox` Port on `Deps`
(injected user functions behind a typed Protocol; spine nodes call `ctx.deps.tools.foo(...)`; the deps
gate admits `ctx.deps.tools.*` the same way it already admits `ctx.deps.store`) would make it real. If
the injected tools are pure and control flow stays fixed, it does not raise the rung — a rung-1.5
capability port, the same move as rung-2 memory. It is a *separate* PRD ("E — the tool port"), not
part of self-correction. Do not fold it in.

---

## 8. Dialect conformance (why this is clean, and why there is no document B here)

### 8.1 Generated systems are unchanged

The loop lives in forge's own graph. `render` is untouched; the emitted package is byte-identical
before and after. Therefore **generated-system conformance is not affected at all** — no new law for
generated code to obey, which is exactly why this work needs no dialect edit and no "B" sibling. The
decision never crosses the boundary into generated code (contrast B §1: the Store *did* cross it, so
it needed a law).

### 8.2 forge's own graph stays conformant — law by law

forge is itself a dialect-conforming pipeline (`graph.py` is its only engine-aware file;
`test_conformance.py` holds it to `check_code`). The loop must not break that:

- **Graph is the sole flow controller (P2).** The loop is a graph back-edge; routing is the pure
  `works`/`improvable` predicate. ✓
- **An LLM call never also picks the path.** Critic/requirements produce data; attribution is a pure
  function producing `rebuild_targets` as data; the graph acts on the predicate. ✓
- **Bounded loops (L5 / I6).** `audit_rounds` counter in State, `max_audit_rounds` cap in Deps, plus
  the monotonic regression guard. Same shape as `plan_review_rounds` / `MAX_PLAN_REVIEW_ROUNDS`. ✓
- **Fields are data, not signals (P3).** `rebuild_targets`, `audit_rounds`, `audit_history`,
  `best_build`, `FilledNode.body` all represent real things (audit's culpability finding; passes run;
  the round record; the best result; the authored source), exactly like the already-accepted
  `plan_feedback`. ✓
- **`graph.py` stays the only engine-aware file.** Attribution, the count parsing, the AST
  line-mapping, the regression guard are pure Python in ordinary modules. ✓
- **Fail loud / no silent failures.** At the cap: `works=False` + best build + full history. On
  regression / non-attributable red: loud terminal. Nothing swallowed. ✓

### 8.3 The topology change

forge's graph gains a **second back-edge**: `Review → Frame` (before Render) and now
`Audit → Render` (after Build; the forward path `Render → Verify → Build → Audit` carries the rebuild).
The two cycles are sequential and independently capped, so they cannot interact pathologically.
Worst-case LLM work is finite and bounded by the caps: one full initial build, plus per rebuild
`|targets| × (attempt_cap + manager)` authoring calls (a rebuild re-authors only its targets, not the
whole system), plus per audit pass the two head calls (requirements + critic) — all under
`max_plan_review_rounds` and `max_audit_rounds`. `test_topology.py` currently asserts
`branchy == {"Review", "Verify", "Build"}` with only Review looping back; it must be updated to add
`Audit` as branchy (successors `{Render, Finish}`). The **cap** assertions belong in
`test_conformance.py` or a dedicated loop test, **not** the topology test — `test_topology.py` reflects
only `run`-return successor sets, and the cap integers are not visible there. This is the only
structural change to forge's own shape.

---

## 9. Acceptance criteria

forge passes this PRD when, on the benchmark's build cases (all runnable offline with a fake sandbox
and stub agents — the loop must be fully exercisable with scripted stub responses):

1. **A crashing body is repaired** across ≤ `max_audit_rounds` and the system ends `works=True`.
   (Deterministically testable: script the stub sink to emit a crashing body on the first authoring of
   a node and a correct one on re-author; assert the loop fires, attributes that node via traceback
   line-mapping, and converges.)
2. **A node-body dialect violation is repaired** (attributed via `nodes.py`-scoped `check_code`,
   re-authored, passes) — and a *render-owned* violation instead settles as a loud terminal, not a
   rebuild.
3. **A hardcoded stand-in the critic catches is repaired** (attributed via the critic's `NodeFinding`,
   reconciled by rapidfuzz, re-authored).
4. **A genuinely unfixable failure settles as a loud result**, not a hang or a raise: `works=False`,
   **best** build returned, full `audit_history` present, within the cap.
5. **A regression is caught, and best is on disk**: a scripted stub that degrades on the second
   attempt causes the loop to keep round *n*'s build and stop (verifying the count-based total order
   and the strict-improvement rule) — **and** the test asserts the on-disk `nodes.py` in `dest/`
   matches `best_build` (round *n*), not the worse final attempt, confirming the settle
   re-materialisation of §4.6.
6. **Loop-fire rate is reported** by the benchmark and is low on clean build cases (no fire when the
   first build already passes; no duplicate audit rows in the reporter across rounds).

---

## 10. Build order

1. **`FilledNode.body`** (§5) — carry the authored source on every filled node, from the value passed
   to `apply_body` at *both* return points (`body` at success `build.py:305`; `last_valid_body` at the
   keep-path `build.py:309-315`). The one new datum; everything else keys off it or `BuildReport`/`spec`.
2. **`build_system(prior_bodies, targets)`** (§4.3) — re-apply known-good bodies, re-author only
   targets, thread audit feedback into `_repair_context`. Prove offline that a rebuild re-applying
   known-good bodies and re-authoring one target yields an identical package plus the changed body.
3. **Surface structured failure detail** (§4.4 signals 1 & 2, §4.5) — three retentions the audit
   currently discards, all required by attribution/guard: (a) `_run_tests` keeps the **raw pytest
   stdout** and runs `--tb=short` (frames for signal 2); (b) `_run_tests` parses
   `tests_passed`/`tests_failed` off the summary line (the guard's order); (c) `_run_dialect` surfaces
   the **untruncated `list[Violation]`** (signal 1), not just the capped `detail` string. Surface all
   on `CheckResult`/`AuditReport`.
4. **`CRITIC` → `list[NodeFinding]`** with rapidfuzz node-name reconciliation (§4.4 signal 4).
5. **`attribute_failures(audit, build, spec, pkg)`** — the pure ranked-signal function, run **inside
   `Audit`** (before the back-edge re-renders, so the filled `nodes.py` is still on disk for the AST
   line-map): `nodes.py`-scoped structured `Violation`s + AST line-mapping, traceback frame
   line-mapping, `BuildConcern`, reconciled critic; `((signals) ∩ authored_nodes) ∪ remaining_holes`
   (`authored_nodes` = filled ∪ unfillable); fallback ladder; `sorted()`; non-attributable-red ⇒ loud
   terminal (§4.4). Unit-test each signal.
6. **The back-edge + result plumbing** — `Audit → Render` routing, `audit_rounds` (rebuild count) /
   `rebuild_targets` / `best_*` / `audit_history` state, the `improved`/`improvable` predicates +
   regression guard, `AuditRound` and `ForgeResult.audit_history` schemas, the `targets` tri-state on
   `build_system` (`None`=all / set=those / `∅`=none), `audit_system` threaded the round number to set
   `AuditReport.rounds`, `Finish` selecting *best* not *last* + the **guarded settle re-materialisation**
   of `best_build` onto disk (§4.6). Update `test_topology.py` (branchy `+ "Audit"` only); cap
   assertions go in `test_conformance.py` / a loop test.
7. **Reporter round-handling** (`reporter.py`) — the `RichReporter` special-cases stage re-entry only
   for `Frame` (`reporter.py:202`); extend it to bump a round indicator on a second `Render`/`Verify`/
   `Build`/`Audit` pass, and make `audit_check` **replace** rather than **append** results across
   rounds (`reporter.py:253`) so a second audit pass does not duplicate rows.
8. **Benchmark cases** (§9), including the scripted-degradation regression case (asserting best is on
   disk); report loop-fire rate.

Steps 1–2 and 5–6 are the load-bearing, hardest-to-get-right parts; budget for them. Steps 3, 4, 7
are mechanical once the shape is fixed.

---

## 11. Invariants this work must not regress

- **Deterministic does the heavy lifting.** Routing is a pure predicate; attribution is a pure
  function; the only new stored datum (`FilledNode.body`) is code-produced. The model contributes at
  most one attribution signal (a reconciled node name) and never a routing decision.
- **Fail loud, no silent fallbacks.** At the cap / on regression / on a non-attributable red audit, a
  loud `works=False` result with best build + full history; never a hang, a swallowed failure, or a
  silently-shipped worse build. A render-owned dialect violation is a loud terminal, never faked as a
  rebuild.
- **Capped at inquiry, not at output.** The cap stops rebuilding; it never discards the best build.
- **The package on disk equals the reported best.** When the loop settles after a regression, `dest/`
  is deterministically re-materialised to `best_build` (§4.6), so the artifact the user receives is
  never worse than what the `ForgeResult` claims.
- **One code path.** The first Build and every rebuild are the *same* `build_system` call; the first
  is just `prior_bodies={}, targets=None`. Re-render is always the Render node's job, never Build's.
- **Generated systems are untouched.** `render` output is byte-identical; no dialect law changes; no
  new obligation on generated code.
- **`graph.py` stays the only engine-aware file** in forge; the loop is a typed back-edge; attribution,
  count parsing, AST line-mapping, and guards are pure modules.
- **The loop fires rarely.** It is the last resort behind the static gates and the component manager.
- **The wall is honest.** The loop repairs crash-level failures (crashes, imports, node-body dialect
  violations, holes, critic-caught stubs); it does not claim to catch non-crashing wrong logic that
  today's completion-level generated tests miss (§6.2).

---

## 12. The six things most likely to be got wrong

1. **Routing the back-edge to `Build` instead of `Render`.** Build cannot re-author a filled node —
   there is no hole left and Build never re-renders. The loop *must* route through Render to get the
   pristine package back (§4.1). This is the single structural mistake that makes the loop a no-op.
2. **Building the regression guard without pass/fail counts.** The audit produces a bool + text, not
   ints. The total order needs `tests_passed`/`tests_failed`; parsing them is in scope (§4.5), not
   assumed. Skip it and the guard degrades to near-boolean and cannot detect thrash.
3. **Inventing a `WorkerReport`.** The loop consumes exactly one new datum: the authored body. Add
   `FilledNode.body`; derive everything else from `BuildReport`/`spec` (§5). A metadata type is
   speculative state the same way speculative LLM output is.
4. **Attributing by class-name-in-traceback, or by whole-package `check_code`.** Frames read `in run`,
   not the class; and `check_code` flags render-owned files too. Use AST `file:line` → owning-class
   mapping, and scope dialect violations to `nodes.py`, treating render-owned violations as forge bugs
   (§4.4). Get this wrong and attribution targets the wrong node — or a node that cannot be the cause.
5. **Overclaiming the reach.** The generated smoke test only checks the run *completes*; the loop
   catches crashes, not wrong values. Pretending otherwise — or quietly scoping in behavioural test
   generation — is the trap (§6.2). That is a separate, harder PRD.
6. **Reporting best while disk holds worse.** The `Render` node (`graph.py:239` → `render.py:697,709`)
   overwrites `dest/` every pass, so after a regression stop the on-disk package is the *worse* final
   attempt while `ForgeResult` reports the best. Settle must deterministically re-materialise
   `best_build` onto disk first, guarded to fire only on the regression-settle path (§4.6); the
   benchmark asserts the on-disk `nodes.py` matches best (§9.5). Skip it and forge ships a package that
   contradicts its own result.

---

## 13. Resolved review findings

For traceability of the review loop. Each maps to the section that now resolves it.

### Round 1

- **B1 (blocker) — back-edge had no re-render, so it could not re-author filled nodes.** Resolved:
  the back-edge routes `Audit → Render` (§4.1, §4.3, §8.3, §12.1), reusing the deterministic pristine
  re-render; Build re-applies known-good bodies and re-authors targets only.
- **M1 — regression guard referenced pytest counts the audit doesn't produce.** Resolved: count
  parsing specced explicitly (§4.5, build order §10.3).
- **M2 — `WorkerReport` over-built.** Resolved: collapsed to `FilledNode.body` (§5), with the full
  provenance analysis showing every other field is already available or re-derivable.
- **M3 — traceback→node by class name is fragile.** Resolved: AST `file:line` → owning-class mapping
  (§4.4 signal 2).
- **M4 — `check_code` runs over the whole package.** Resolved: scope to `nodes.py`; render-owned
  violations are loud terminals (§4.4 signal 1, §6.1).
- **M5 — `ForgeResult` had no history; `Finish` returned last not best.** Resolved: `audit_history` +
  `AuditRound` schemas, `Finish` selects best (§4.2, §4.6).
- **m1 — `contract_coverage` source discards the map.** Resolved: field dropped with `WorkerReport`
  (§5 table).
- **m2 — critic can't be a `Literal` enum.** Resolved: free `node: str` + rapidfuzz reconciliation
  (§4.4 signal 4).
- **m3 — nondeterministic target order.** Resolved: `sorted()` (§4.2, §4.4).
- **m4 — reporter needs round-handling.** Resolved: added to build order (§10.7).
- **m5 — §6.2 over-claimed test strength.** Resolved: reach corrected to "runs to completion" (§6.2).
- **m6 — inner `attempt_cap` × outer loop, and unfillable targets, unstated.** Resolved: the
  static-only inner loop is why the outer needs a monotonic guard (§3, §4.5); unfillable-target cost
  is bounded and stated (§4.3 edge case).

### Round 2

- **B-1 (blocker) — attribution signal 2 had no data source; the audit discards the traceback.**
  Resolved: `_run_tests` retains raw pytest stdout for frame extraction (§4.4 signal 2, §4.5, §10.3).
- **B-2 (blocker) — a regression stop left the worse build on disk while reporting best.** Resolved:
  deterministic settle re-materialisation of `best_build` onto `dest/` (§4.6), with an on-disk
  assertion in the regression acceptance case (§9.5) and a new invariant (§11).
- **MA-1 — `targets ∩ filled` dropped unfillable holes, contradicting §6.1.** Resolved: `authored_nodes`
  = filled ∪ unfillable, and remaining holes are unconditionally eligible targets (§4.4, §4.3).
- **MA-2 — `audit_rounds` off-by-one on the cap.** Resolved: `audit_rounds` pinned to *rebuilds
  performed* (starts 0), `AuditReport.rounds = audit_rounds + 1`, cap math stated (§4.2, §4.5).
- **MI-1 — which `body` variable to store.** Resolved: from the value passed to `apply_body` at both
  return points (§5, §10.1).
- **MI-2 — cost formula imprecise.** Resolved: restated per-rebuild/per-pass (§8.3).
- **MI-3 — cap assertions don't belong in the topology test.** Resolved: topology test gets only the
  branchy-set update; caps move to conformance/loop test (§8.3, §10.6).
- **MI-4 — count parsing degrades on collection/import errors.** Resolved: the `0/0` tie documented as
  intentional; such failures route to loud terminal, not the order's tie-break (§4.5).

### Round 3

Round-2 fixes were verified holding against source (settle re-materialisation is faithful and
authoring-free; body-variable choice correct; counter semantics consistent; targets formula
consistent; signal-2 stdout feasible; loop termination sound). New findings:

- **MAJOR — signal 1 had no structured data source** (symmetric to round-2's signal-2 blocker):
  `_run_dialect` truncates `check_code`'s `Violation` list into a string. Resolved: surface the
  untruncated `list[Violation]`; attribution consumes it, not `detail` (§4.4 signal 1, §10.3).
- **MINOR — `attribute_failures` signature insufficient** (needs the filled `nodes.py` source before
  Render overwrites it). Resolved: `attribute_failures(audit, build, spec, pkg)`, run inside `Audit`
  (§4.4, §10.5).
- **MINOR — `targets` empty-list vs empty-set conflation.** Resolved: explicit tri-state — `None`=all,
  set=those, `∅`=none; Build maps empty `rebuild_targets`→`None` (§4.3).
- **MINOR — `FilledNode.body` could default-empty → silent `pass`.** Resolved: required field, no
  default (§5).
- **MINOR — "previous rebuild round strictly improved" imprecise.** Resolved: restated as
  `improved = current ≻ best_before_this_round`, tie = non-improving (§4.5, §4.1).
- **MINOR — `Finish` re-materialisation needs a path/`dest` guard; `audit_system` needs the round
  number.** Resolved: fire only when `dest` set and `best_audit is not audit`; thread `audit_rounds`
  into `audit_system` (§4.6, §10.6).
- **MINOR — wrong citation (`graph.py:279`) + unpinned traceback style.** Resolved: corrected to the
  `render()` call site (`graph.py:239` → `render.py:697,709`); pin `--tb=short` (§4.6, §12.6, §4.4).
