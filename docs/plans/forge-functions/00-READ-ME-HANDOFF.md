# 00. Read-Me / Handoff — forge scope + rung-2 store

*Orientation for a coding agent picking up this work. Three documents, one decision, a
specific order to read and act in. Read this page first.*

---

## The one decision these documents serve

**forge builds *functions*: one input, one output, one run, with control owned inside the
run. It builds two flavours (stateless, and stateful-with-durable-memory) and refuses
everything above them at the door with a concrete reshape suggestion.**

The single test for any brief: *does anything outside the system have to decide what happens
next — a user choosing an operation, a session continuing, an event firing?* No → it is a
function, build it. Yes → name the external driver (app / agent / service), refuse, reshape.

Everything in these documents exists to make that line sharp and to make the "build it"
branch reliable on small local models.

---

## The three documents, and why there are three

They form a hierarchy. Each is one level more concrete than the last, and each inherits its
authority from the one above.

| Doc | Level | Question it answers | Code? |
|---|---|---|---|
| **A — Why Functions** | principle | *What* do we build and *why* is the line here? What were the alternatives? | none |
| **B — Dialect Alignment** | contract | *Where* does this decision get written so it survives into generated systems, not just forge? | hints only |
| **C — Store PRD** | build | *How* do we make "stateful function" real: the store, the touch points, the queries enabled, the functions forge must create? | yes, illustrative |

**Why B exists at all** (the non-obvious one): the agentic dialect document defines *both*
forge itself *and* every system forge generates (forge is built in the dialect and audits
its output against the dialect). So a decision written only into forge's source reaches
forge but **dies at the boundary of forge's own tree** — the generated functions are checked
against the dialect, not against forge's planning docs. B moves the decision into the
dialect so the generated functions inherit it. Skip B and the conclusion does not "live on"
into the functions forge produces.

---

## Read in this order

1. **A** — to understand the decision and believe it (or challenge it on the merits before
   building).
2. **B** — to see what must change in the dialect documents, and why the change has to live
   there rather than only in forge.
3. **C** — to build the store and turn on rung 2.

## Act in this order (from C §10, surfaced here)

1. Encode the **scenario corpus** (C §9) as fixtures — the acceptance test for everything.
2. Apply the **dialect edits** (B §6): law L9, the L1 cross-run clause, the two function
   shapes. Re-run the dialect conformance check.
3. Ship the **`Store` component** (C §3) in `forge.runtime`.
4. **Wire it through render** (C §5): Deps field, run-entry path resolution, CLI flag,
   in-memory smoke test, deps-gate allowed-set.
5. Build the **entity-classification round-trip check** (C §7) — *this is the part that
   actually fails; budget for it.*
6. **Turn on rung-2** in the front fitness gate (C §10 step 5).

---

## The five things most likely to be got wrong

1. **Faking the boundary as a vocabulary check.** The fitness gate (and dialect law L9)
   judge *shape* — is there an external driver? — never words. No lists of "save"/"track"/
   "manage". Keyword-matching is the hard-coded-domain trap the project forbids. (A §6, B §4,
   C §7.)
2. **Reaching for `document_store`.** It is forbidden by layering (forge stays a near-leaf),
   it is the wrong shape (content+metadata semantic search, not key-to-blob), and its only
   create path needs an embedding model so the offline audit can't save a record. The store
   is forge-owned stdlib sqlite. (A §5, B §5, C §2.)
3. **Letting the store leak into the middle of the run.** Load at start, save at end, pure
   spine in between. The store is touched only at the graph's edges, or the smoke test stops
   being meaningful. (B §3.2, C §4.)
4. **Under-budgeting entity classification.** The store is easy; deciding *durable entity vs
   run-scoped signal* is the part that breaks. Solve it with the deterministic round-trip
   detector (read-at-entry + written-at-exit under one identity), not a prompt. (C §7.)
5. **Growing the store to fit a hard brief.** A sixth operation (a `query`, a `where`) is
   never the answer — it is a signal the brief left rung 2. The closed five-operation surface
   is itself an enforcement mechanism. (C §8.)

---

## Vocabulary (shared across all three)

- **rung** — position on the system-class ladder. Rungs 1–2 are functions (in scope); 3–5
  (app / agent / service) are out.
- **function** — one input, one output, one run, control owned inside.
- **stateless function (rung 1)** — nothing remembered between runs. The default.
- **stateful function (rung 2)** — plus a durable store loaded at start, saved at end.
- **external driver** — the thing that puts a brief above rung 2: an operation-chooser
  (app), a session (agent), or an event source (service).
- **signal** — a run-scoped State value; forgotten when the run ends.
- **entity** — a durable record that lives in the store; survives between runs.
- **Port** — an injected capability on Deps behind a Protocol (agent sink, sandbox,
  reporter, and now the store).
- **gate** — a pure static check that turns a build-prompt suggestion into a guarantee
  (model proposes, gate disposes).
- **the dialect** — the house style for agentic graph systems; forge is built in it and
  holds its output to it. The reason B exists.
