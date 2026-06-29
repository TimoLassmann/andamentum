# B. Carrying the Function Boundary Into the Dialect

*A design note, one level beneath document A. Document A fixed the decision: forge builds
functions (rung 1 and 2) and refuses everything above. This note answers a narrower
question: does that decision belong in the **agentic dialect** documents, and if so, where
and how? It proposes specific, surgical additions. It contains almost no code, only "hints"
in the dialect's existing voice.*

---

## 1. Why this note has to exist at all

The scope decision lives, today, only in `forge`'s planning document. That is not enough,
and the reason is structural.

**The dialect document does double duty.** It defines two different things at once:

1. the house style that `forge` *itself* is built in (forge is a dialect-conforming
   pipeline), and
2. the house style that every system forge *generates* is held to (forge audits its output
   against the same dialect).

So a principle that lives only in forge's own planning doc reaches forge, but **does not
reach the systems forge produces.** The generated functions are conformance-checked against
the *dialect*, not against forge's SCOPE.md. If "this artefact is a function: one input, one
output, one run, control owned internally" is a law of forge's *output* and not merely a
design choice of forge's *implementation*, then it has to be written where the output is
judged. That place is the dialect.

Put bluntly: change only forge, and the conclusion we reached in document A dies at the
boundary of forge's own source tree. The functions it emits would carry no record of the
contract they were built to satisfy, and a future maintainer extending the generated-code
recipe would have no law telling them that an external control loop is out of bounds. The
decision has to become part of the worldview, not just part of one program that holds the
worldview.

**The dialect is also the natural home for it.** The dialect already is the thing that says
"here are the few pieces, the fixed grammar, the handful of laws; if what you are about to
write is not obviously one of these pieces, it is in the wrong place." The function boundary
is exactly that kind of statement. It is a *shape law*: it says what shape a conforming
system has at its outermost edge. The dialect already legislates the inside of a system
(surfaces, roles, the one agent, the disciplines, the eight laws). It is currently silent on
the system's *outer shape*: what it is as a whole, seen from the caller. That silence is the
gap document A's decision needs to fill.

---

## 2. The precise gap in the current dialect

Read the dialect as it stands and notice what it does and does not constrain.

It constrains the **interior** thoroughly. The tape-machine model fixes three data surfaces
(Deps, State, Inputs) and one moving head. The pieces fix two code roles (orchestrator,
worker) plus the one agent and two disciplines (Port, Schema). The eight laws govern
placement, thin-orchestrator/fat-worker separation, the single LLM seam, bounded loops, and
so on. All of this describes how work flows *within a run*.

It says almost nothing about the **boundary**: the system seen as a unit. There is an
implicit one-entry/one-exit assumption (a graph has a start and an end), but it is nowhere
stated as a law, and crucially nothing says:

- that the system is a **function** (one input at the door, one output at the end, one run);
- that **control is owned internally** (no external driver decides what runs next: no
  command surface, no session, no event trigger);
- that **memory across runs lives in a Port-backed store**, loaded at the start and saved
  at the end, and nowhere else (in particular, not faked in run-scoped State).

These three are the dialect-level expression of document A. The first two are the boundary
itself. The third is what makes rung 2 expressible without breaking the boundary. None of
them is currently a law, so a generated system that violated them could still pass dialect
conformance, which is the gap.

---

## 3. What to add: one new law, one extension, one shape entry

The fix is deliberately small. The dialect's strength is that it is short; the wrong
response is to bolt on a section. The right response is three surgical additions that read
as if they were always there.

### 3.1 A new boundary law: "The system is a function"

Add one law to the eight (it becomes L9, or slots wherever the placement reads most
naturally). In the dialect's existing terse, imperative voice:

> **L9 · The system is a function.**
> A conforming system is a function: one input at the door, one output at the end, one run.
> Control is owned inside the run, never handed out. If something *outside* the system would
> decide what happens next (a user choosing among operations, a session continuing across
> turns, an event firing on its own), the thing you are describing is not a conforming
> system. It is an app, an agent, or a service, and it belongs outside the dialect, built by
> hand on top of conforming functions.
>
> *Review:* one entry, one exit, one run? Yes. Does anything external drive what runs next?
> No. A bounded internal loop (generate, check, retry within the run) is control flow, not
> an external driver, and is allowed.

This is the load-bearing addition. It states the outer shape as a law, names the three ways
a system can stop being a function (the three external drivers), and pre-empts the most
common confusion (an internal loop is fine). Everything in document A compresses into this
law; the law is how the decision survives into every generated artefact, because every
generated artefact is checked against the laws.

### 3.2 An extension to the surface-placement law: memory is a Port

The dialect's first law already sorts data across the three surfaces: given to the run goes
in Deps, produced-and-read-widely goes in State, produced-for-the-next-step goes in Inputs.
There is a fourth case it does not yet name: **produced in one run and needed in a later
run.** That case has no home among the three surfaces, because all three are run-scoped:
they are reborn every call. So it needs the one durable thing the dialect already has a word
for: a **Port**.

Extend the surface-placement law with one clause:

> Data that must outlive the run (needed by a *later* run, not merely a later step) lives in
> neither State nor Inputs. It lives in a durable store reached through a **Port** on Deps:
> loaded into State at the start of the run, saved back to the store at the end. State holds
> the working copy *during* the run; the store holds the truth *between* runs. Never simulate
> cross-run memory in run-scoped State: State forgets when the run ends, so memory kept there
> is a fiction.

This is what makes rung 2 a first-class dialect citizen rather than an exception. It tells
the placement law where memory goes (a Port-backed store, the same mechanism already used
for the agent sink, the sandbox, and the reporter), and it forbids the failure mode we
actually observed (faking persistence in a State string). It also keeps the dialect's
internal consistency: persistence is not a new kind of thing, it is the existing Port
discipline applied to a durable capability.

### 3.3 A shape entry: the two function flavours

The dialect's "shapes" section (the handful of canonical system shapes) should gain the
distinction that document A draws, so an author choosing a shape sees both flavours and the
boundary between them:

> **Stateless function** · one input, one output, one run, nothing remembered between calls.
> The default. Same input, same output, every time.
>
> **Stateful function** · one input, one output, one run, *plus* a Port-backed store loaded
> at the start and saved at the end, so the output can depend on earlier runs. Still one in,
> one out: part of the input is loaded, part of the output is saved. The store path is an
> explicit argument; absent means an ephemeral in-memory store, which means the function
> behaves statelessly.
>
> Above these two flavours there is no conforming shape. A system with several caller-chosen
> operations, a multi-turn session, or an event trigger is out of dialect (see L9).

This gives the shapes section a clean top edge that matches the law: two flavours of
function, and an explicit "nothing above this is a shape."

---

## 4. What *not* to change

Three temptations to resist, because the dialect's value is its smallness.

**Do not add a "persistence" piece to the vocabulary.** The store is not a new surface or a
new role. It is a Port. Adding it as a fourth surface would break the tape-machine model
(three surfaces, one head) for no gain. The whole point of 3.2 is that persistence reuses
the Port discipline. Keep it that way.

**Do not write storage mechanics into the dialect.** The dialect is engine- and
library-neutral and stays that way. *How* the store is implemented (a small sqlite table,
keyed CRUD, an explicit path) is a `forge` concern and belongs in document C, not here. The
dialect says only: cross-run memory is a Port-backed store, loaded at start, saved at end,
never faked in State. The shape of that store's API is downstream.

**Do not turn L9 into a vocabulary checklist.** L9 judges shape, not words. It must not list
forbidden terms ("save", "track", "manage", "watch"). A system is out of dialect because an
external driver owns its loop, not because the brief used a particular verb. Writing
keywords into the law would re-introduce the hard-coded-domain trap at the level of the
worldview, which is the worst place for it. The law names the three *structural* drivers
(operation chooser, session, event source) and stops there.

---

## 5. How the dialect change and the forge change relate

These dialect additions are the *contract*. forge is one *enforcer* of the contract, in two
places, both detailed in document C:

- forge's **front fitness gate** is L9 applied at the door, to the brief, before building.
  It is how forge refuses to *start* building a non-function.
- forge's **store provisioning and deps gate** are 3.2 applied to the output, during
  building. They are how forge makes cross-run memory real (a provisioned Port) and how it
  forbids the fake (a static check that the only persistence door is the declared store
  handle).

The relationship is deliberate and worth stating plainly: **the dialect declares the law;
forge is one program that obeys and enforces it; the generated functions inherit the law by
being checked against the dialect.** Because the law now lives in the dialect, a *second*
generator, or a hand-written conforming system, or a future maintainer extending the recipe,
all meet the same boundary. That is the entire reason this note exists. The decision in
document A is only as durable as the place we write it down, and the dialect is the only
place that both forge and forge's output can see.

---

## 6. Checklist of edits to the dialect documents

For the coding agent applying this note to the dialect source (the canonical
`DIALECT.md` and its rendered references, plus the structured laws in
`andamentum.agentic_dialect` that the prose is drift-tested against):

- [ ] **Add L9 "The system is a function"** to the laws (prose + the structured law table,
      so the conformance checker and the prose stay in sync). Include the one-entry/one-exit/
      one-run review and the internal-loop exception.
- [ ] **Extend the surface-placement law (L1)** with the cross-run clause: durable memory is
      a Port-backed store, loaded at start, saved at end, never simulated in State.
- [ ] **Add the two function flavours** (stateless / stateful) to the shapes section, with
      the explicit "nothing above these is a shape" top edge.
- [ ] **Confirm the `role` slices** (the prompt-slices the dialect CLI emits per job, e.g.
      `agentic-dialect role worker`) still read correctly with L9 present; a worker's
      slice should not need to change, but the orchestrator's framing should reflect that the
      whole assembly is a function.
- [ ] **Do not** add a persistence piece, storage mechanics, or any keyword list. Keep the
      additions engine-neutral and shape-based.
- [ ] **Re-run the dialect's own conformance check** (`agentic-dialect check`) after the
      edits, so the structured laws and the prose remain drift-tested against each other.

The test of a good edit here is the dialect's own standard: after the change, if a thing you
are about to build is not obviously one of the pieces, it is in the wrong place, and now
"the whole thing is a function with control owned inside" is one of the laws that decides
what "the wrong place" means.
