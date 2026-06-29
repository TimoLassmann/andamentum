# A. Why forge Builds Functions, and Where the Line Is Drawn

*A conceptual assay. No code. This document fixes the single decision that governs every
other decision in forge: what class of system forge will and will not build. Read it before
the dialect-alignment note (B) or the store PRD (C); both inherit their authority from the
boundary argued here.*

---

## 0. The decision, in one sentence

**forge turns a natural-language brief into a runnable agentic system, and that system is
always a *function*: one input, one output, computed in a single run. It builds two
flavours of function (stateless, and stateful-with-memory) and refuses everything above
them at the door with a concrete reshaping suggestion.**

Everything that follows is the argument for why this is the right line, what the
alternatives were, and why "function" and "stateful function" are the right place to stop.

---

## 1. The thing we are really deciding

It is tempting to frame forge's scope as a feature list: "it supports loops, branching,
fan-out, human gates, persistence." That framing is a trap. A feature list has no natural
boundary, so it grows until forge is trying to generate arbitrary software from a sentence,
which no system can do reliably, least of all one built to run on small local models.

The real decision is about a single structural property: **who owns the control loop.**

Strip any agentic system to its skeleton and ask one question: when one unit of work
finishes, *what decides what happens next?*

- If the answer is *the system itself, following a path fixed before the run began*, then
  the system is a **function**. It has one entrance and one exit. Everything between them,
  however elaborate, is internal machinery the caller never sees. You call it; it runs; it
  returns. This is what the wider field now calls a **workflow**: a system where the model
  and its tools are orchestrated through predefined code paths.

- If the answer is *something outside the system, deciding live* (a user choosing the next
  command, a session continuing across turns, the world emitting an event), then the system
  is **not a function**. Control has left the box. This is what the field calls an
  **agent** or a **service**: a system where an LLM, a user, or an event stream
  dynamically directs the process at run time.

This is not a distinction we invented. It is the same architectural line Anthropic draws in
*Building Effective Agents*: workflows orchestrate through predefined code paths and offer
predictability and consistency; agents let the model drive and offer flexibility at the
cost of that predictability. The field has since converged on an even terser definition of
the agent end: an LLM autonomously using tools in a loop. The crucial word is *loop*, and
the crucial question is *whose loop*.

**forge builds the workflow end of that line, exhaustively and reliably, and refuses the
agent end.** Not because the agent end is bad. Because it is a different machine, and a
tool that does one machine superbly and says "no, but here is the version of your request I
can build" is worth more than one that fakes both.

---

## 2. The ladder (a diagnostic, not a menu)

Every brief a user could write sits somewhere on a ladder of system classes. The rung is
set by how much external control the brief demands.

| Rung | Class | Who owns the loop | Example brief |
|---|---|---|---|
| 1 | **Stateless function** | the system, fixed path | "summarise this", "classify this ticket and route it", "research X, looping until evidence is sufficient" |
| 2 | **Stateful function** | the system, fixed path, plus durable memory | "given my list and a new message, return the updated list", "append this note and return the new count" |
| 3 | **App** | the **caller** chooses among many operations | "manage my reading list" (add / show / remove / mark-read) |
| 4 | **Agent** | the **caller** drives a multi-turn session | "a chatbot about my documents" |
| 5 | **Service** | the **world** emits triggering events | "watch my inbox and file incoming mail" |

**forge's scope is the line under rung 2.** Rungs 1 and 2 are functions: the system owns
the loop. Rungs 3, 4, and 5 each hand the loop to a different external driver: a command
chooser, a session, an event source. That is the whole boundary, and it is a single fact
expressed three ways, not three separate facts.

Two clarifications that prevent the most common misreadings:

**A loop inside the run is not a higher rung.** "Research a question, searching until the
evidence is enough" *has a loop* but the loop is bounded and internal: generate, check,
retry, all within one call, terminating on a declared cap. It produces one output from one
input. It is a rung-1 function. Do not confuse "the system loops internally" with "an
external driver loops the system." The first is control flow. The second is a change of
system class. forge already supports the first and must always refuse the second.

**The disqualifying axis is external control, and it can arrive on its own.** "Manage my
reading list" is often described as failing because it needs memory *and* multiple
operations. But memory is a red herring there. A purely read-only "show my list, filtered
three ways" needs no persistence at all and is *still* rung 3, because the caller still
chooses which of several operations to invoke. Multiplicity of caller-chosen operations is
disqualifying by itself. The ladder is a diagnostic for locating the external driver, not a
points system where you tally axes.

---

## 3. Why a function is a *good* thing to be, not a limitation

The instinct that "only functions" is a weakness gets the engineering backwards. The
function boundary is exactly what buys forge its reliability, and reliability is the entire
point of a generator aimed at small local models.

A function has the properties that make generated code trustworthy:

- **It is testable in isolation.** One input, one output, no hidden context to reconstruct.
  The generated smoke test can drive the whole system end to end with stub model calls,
  offline and deterministic, because there is no session to simulate and no event source to
  mock. A reproduced bug starts from a fresh, fully specified state every time.
- **It has one place to be correct.** A workflow's behaviour is the composition of its
  fixed nodes. You can gate each node statically, check that the path covers the goal, and
  validate the whole spec against a recipe *before any code runs*. An agent's behaviour is
  whatever the model decides at run time, which can be inspected but not guaranteed in
  advance.
- **It composes.** Because a function is closed (input in, output out, nothing leaking
  sideways), a larger system can call it without inheriting its internals. The agent and
  service layers that forge refuses to build can themselves be built *by hand on top of*
  forge's functions: an event loop that calls a forge function per event, a command
  dispatcher that calls a different forge function per command. forge builds the reliable
  unit; the unreliable orchestration stays where a human can own it.

This is the standard, hard-won lesson of stateless architecture, imported into agent
design: default to the pure function, keep each unit independent, and push state and
control to the edges where they can be managed deliberately. The field's own advice for
agents echoes it: start simple, use a workflow when the task decomposes cleanly into fixed
steps, and reach for a full agent only when you genuinely need model-driven flexibility at
run time. forge institutionalises that advice as a hard boundary rather than a
recommendation, because a *generator* cannot afford to let an under-specified brief quietly
escalate it into building software it cannot make correct.

---

## 4. The two flavours, and why memory is the natural top

Within "function," there is exactly one meaningful internal split, and it is the difference
between rung 1 and rung 2.

**A stateless function (rung 1)** computes its output purely from the input it was handed.
Same input, same output, every time, with nothing remembered between calls. This is the
pure function in the classical sense, and it is the safe default. Most useful briefs are
here: summarise, classify, route, extract, translate, research-until-enough. forge already
builds these.

**A stateful function (rung 2)** is still one input to one output per call, but part of its
input is loaded from a durable store at the start of the run, and part of its output is
saved back to that store at the end. The output can therefore depend on what earlier runs
produced. "Append a note and return the new total" is the canonical case: the *count* is
the accumulated history of every prior run, and a purely stateless function would count to
one forever, because each previous run left nothing behind.

Memory is the natural top of the ladder for functions because of *what state is, and what a
function does to it.* A function, by construction, is born with fresh state and dies when it
returns. State (the run-scoped tape) is reborn on every call. So the only way for a later
run to depend on an earlier one is for something to survive *outside* the run. That
something is a durable store. Adding it is the single largest capability a function can gain
while *remaining a function*: it changes where some of the input comes from and where some
of the output goes, but it does not change the shape (still one in, one out, one run) and
it does not hand the loop to anyone. The system still owns its control flow start to finish.

The rung above (an app) is not "a stateful function plus more memory." It is a category
change: the caller now chooses among operations. No amount of storage turns a function into
an app, because the new thing an app has is not storage, it is an *external chooser*. That
is precisely why the line sits where it does. Rung 2 is the last rung you can reach by
adding capability to a function. Rung 3 is the first rung you can only reach by surrendering
the loop.

---

## 5. The alternatives we considered, and why we rejected them

Three other boundaries were genuinely on the table. Each is defensible; each is wrong for
this tool.

**Alternative 1: stateless only (line under rung 1).** Simplest possible scope: every
generated system is a pure function, no store, no persistence machinery at all. State that
needs to survive is pushed entirely to the caller, who passes prior state in as ordinary
input and stores the returned state themselves. This is a real and honest design (it is the
"caller persists" half of rung 2), and for many workflows it is the *better* design,
because it keeps the generated function pure and shifts the burden of remembering to a
caller who can choose any storage they like.

We rejected it as the *ceiling* (while keeping it as a *default*) because there is a narrow
but real class of briefs where pushing all state to the caller is unreasonable: where the
accumulated history is large enough that threading it through every call, in and back out,
is clumsy and error-prone, or where you want the *system itself* to guarantee its own
continuity rather than trusting every caller to save and replay state correctly, forever.
For that class, an owned store is the right tool. Stopping at rung 1 would force every such
brief into a worse shape. So rung 1 remains what forge does when no store path is supplied,
and rung 2 exists for when one is.

**Alternative 2: go up to apps (line under rung 3).** Tempting, because "manage my list" is
a request users actually make. We rejected it because an app is a *dispatcher over several
functions*, and the dispatcher is exactly the part that needs an external chooser: a
command surface, an argument parser, a notion of "which operation did the caller ask for
this time." That machinery is not a bigger function; it is a different worldview, and it is
the part a small model is worst at generating reliably. The honest response to "manage my
list" is to refuse and offer the function hiding inside it: "given my current list and a
new message, return the updated list" (rung 2), or "extract the recommendations from this
message" (rung 1). Both are buildable and reliable. The app is neither, from a sentence.

**Alternative 3: build the agent or service layer too (line under rung 4 or 5).** This is
the "do everything" option, and it fails for the same reason, more sharply. An agent owns a
multi-turn session; a service reacts to events on its own schedule. Both put the control
loop firmly outside any single graph, and both need infrastructure (session state,
schedulers, event subscriptions) that has nothing to do with computing one output from one
input. A graph engine has one entry and one exit by construction. The moment what-runs-next
depends on a live external decision, you no longer have a graph; you have an external loop
wrapping a graph. forge's job is to build the thing inside the loop correctly. The loop
itself is a hand-built wrapper, and a generator that pretends to produce it from a sentence
will produce a plausible fiction, which is worse than a refusal.

**The common thread.** Every rejected upward alternative fails on the same point: above
rung 2, the new capability is not something you add *to* a function, it is an external
controller you wrap *around* one. forge generates functions. Controllers are the caller's
to write, by hand, on top of those functions. Drawing the line under rung 2 is what keeps
forge honest about the difference between "I can build this" and "I can fake this."

---

## 6. What "enforcing the line" has to mean

A boundary that is only described is not a boundary. Today an ill-shaped brief sails
through, because the early stage that names a plausible input and output succeeds at naming
*some* input and output for almost any sentence, including a rung-3 one. The result is a
system that compiles, runs, and silently does the wrong thing: it pretends to manage a list
it never stored. That is the worst possible failure, because it looks like success.

Enforcement therefore has two parts, developed in the documents that follow this one:

1. **A front fitness gate** that judges, before any expensive design work, whether the
   brief is realisable as a function (rung 1 or 2). If yes, proceed. If it is an app, agent,
   or service, fail loud and return a concrete reshape: a rung-1/2 brief that captures the
   user's intent, phrased so the user can resubmit it. The gate judges *shape* (is there an
   external driver?), never vocabulary (it must not key off words like "save" or "list" or
   "track"; that is the hard-coded-domain trap the project forbids). When genuinely
   ambiguous, it proceeds but *declares the interpretation*, rather than silently building a
   fiction.

2. **Real statefulness for rung 2**, so that "stateful function" is a thing forge can
   actually build rather than fake: a durable store the generated function loads from at the
   start and saves to at the end, provisioned by the generator, reachable only through a
   declared handle that a static gate enforces, and stubbed in tests so the audit stays
   offline. Until this exists, the fitness gate treats rung-2 briefs as not-yet-buildable
   and reshapes them down to rung 1.

The principle uniting both: **fail loud, never fake.** A generated system that stops and
explains itself is always better than one that runs and lies. The fitness gate makes forge
refuse the briefs it cannot honour. The store makes forge honour the briefs it accepts.

---

## 7. The one-line test to apply to any future brief

When a new brief arrives and you are unsure where it sits, do not count axes and do not
match keywords. Ask the single question this entire document is built on:

> **To do what this brief asks, does anything *outside the system* have to decide what
> happens next: a user choosing an operation, a session continuing a conversation, or an
> event firing on its own?**

- **No** → it is a function. If its output depends on earlier runs, it is rung 2; otherwise
  rung 1. Build it.
- **Yes** → name the external driver (operation-chooser → app; turn-driver → agent;
  event-source → service), refuse, and return the function hiding inside the request.

That question is the line. Everything else in forge, the dialect it conforms to, the gates
it runs, the store it is about to grow, exists to keep that line sharp and to make the
"build it" branch reliable enough to trust.
