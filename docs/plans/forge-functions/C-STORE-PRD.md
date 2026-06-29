# C. PRD — The Light Store (rung-2 statefulness for forge)

*A product requirements document, one level beneath documents A and B. A fixed the decision
(forge builds functions, up to and including stateful functions). B carried it into the
dialect (cross-run memory is a Port-backed store, loaded at start, saved at end). This
document specifies the concrete capability that makes "stateful function" real: a
lightweight durable store, forge-owned, that generated functions load from and save to.*

*Code examples are illustrative of intent, not final implementations. The end state and the
behavioural contract are normative; the exact line-by-line code is the coding agent's to
finalise against the real source.*

---

## 1. The end state, in one paragraph

After this work, a brief whose output depends on earlier runs (rung 2) is **built for real
instead of faked.** The generated function takes an optional store path. When a path is
supplied, the function loads its prior records from a durable store at the start of the run
and saves its updated records at the end, so accumulated history actually persists across
calls. When no path is supplied, the same code runs against an ephemeral in-memory store and
behaves statelessly (rung 1). The store is a single small component owned by forge,
dependency-light, reached only through a declared handle on Deps that a static gate enforces,
and stubbed (by the in-memory mode itself) in the generated smoke test so the audit stays
offline and deterministic. The per-system schema lives entirely in the generated entity
model; the store itself is schema-agnostic.

That is the whole deliverable. Everything below specifies it precisely.

---

## 2. What this explicitly is, and is not

**Is:** a place to put the slice of input that comes from the past and take the slice of
output that should outlive the present. Keyed CRUD over JSON records. Load at start, save at
end. Nothing in between touches it.

**Is not:** a database the generated agent designs. Not a query engine. Not a
semantic-search store. Not a session store. Not memory in any grand sense. If a brief seems
to need any of those, that is a signal it is the wrong rung (see §8), and the answer is to
refuse and reshape, not to grow the store.

**Explicitly rejected: reusing `document_store`.** Three reasons, all standing:
1. **Layering.** forge must stay a near-leaf so it remains independently extractable, and it
   must not depend on `document_store` (or any heavy sibling module). This alone is
   disqualifying, independent of API quality.
2. **Wrong shape.** `document_store` stores `content + metadata` and is built for find-by-
   meaning (FTS5, vectors, rank fusion). forge needs key-to-blob record-keeping, not
   semantic search. Holding an arbitrary record there means stuffing real data into the
   metadata sidecar while the content field becomes dead weight that still gets embedded.
3. **No offline create.** Its only public create path (`ingest`) mandates an embedding model
   and refuses to run without one. forge's offline, stub-agent smoke test could not save a
   record, so the audit could not exercise persistence at all. (Its read/update/delete side
   is genuinely clean and embedding-free, but create is the half a state store exists for.)

`document_store` remains the right tool for systems that genuinely need "find documents by
meaning," and a generated system can opt into it for that. It is the wrong tool as forge's
general persistence.

---

## 3. The store component

A small `Store` in `forge.runtime` (which generated packages already import), backed by
stdlib `sqlite3`. One table, five operations, synchronous, no new dependency.

### 3.1 The shape: `(collection, key, value)`

The store is schema-agnostic by design. It holds one table where `value` is JSON. The
per-system schema is the generated Pydantic entity model; the store never knows what a
"note" or a "reading-list item" is.

- **`collection`** — the record kind. Maps directly onto the existing
  `EntitySpec.record_type` discriminator.
- **`key`** — the record's identity within its collection (see §6 for how the key is chosen).
- **`value`** — the entity serialised to JSON (`model_dump_json` in, `model_validate_json`
  out).

### 3.2 Illustrative implementation

```python
# forge/runtime/store.py
import json, sqlite3
from typing import Any

class Store:
    """One table: (collection, key, value-as-JSON). Keyed CRUD, nothing else.

    path=None  -> in-memory (":memory:"), forgets at process exit (rung-1 behaviour).
    path="..." -> durable file at exactly that path (rung-2 behaviour).
    """

    def __init__(self, path: str | None = None) -> None:
        self._db = sqlite3.connect(path or ":memory:")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS records ("
            "  collection TEXT NOT NULL,"
            "  key        TEXT NOT NULL,"
            "  value      TEXT NOT NULL,"   # JSON
            "  PRIMARY KEY (collection, key))"
        )
        self._db.commit()

    def add(self, collection: str, key: str, value: dict[str, Any]) -> None:
        # INSERT OR REPLACE collapses create and update into one verb.
        self._db.execute(
            "INSERT OR REPLACE INTO records (collection, key, value) VALUES (?, ?, ?)",
            (collection, key, json.dumps(value)),
        )
        self._db.commit()

    def get(self, collection: str, key: str) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT value FROM records WHERE collection = ? AND key = ?",
            (collection, key),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, collection: str) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT value FROM records WHERE collection = ? ORDER BY key",
            (collection,),
        ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def remove(self, collection: str, key: str) -> None:
        self._db.execute(
            "DELETE FROM records WHERE collection = ? AND key = ?",
            (collection, key),
        )
        self._db.commit()
```

### 3.3 The five operations are the closed surface

`add`, `get`, `list`, `remove`. Note `add` is create-or-update (INSERT OR REPLACE), so there
is no separate `update` verb to track. There is deliberately **no `query` / `where` /
`filter`**. The surface is closed: a sixth operation is not a feature request, it is a
signal that the brief is the wrong rung (§8).

---

## 4. The behavioural contract: load at start, save at end

This is the heart of rung 2, and it is the dialect law from document B (§3.2) made concrete.
The generated function is morally:

```
run(text, store) = save(store, f(text, load(store)))
```

where `f` is the pure graph forge already generates. **Statefulness is "wrap the pure
function in a load before and a save after."** Keep it that literal.

- An **entity-reading node** at (or near) the start of the run loads the prior record(s)
  from the store into State. This is the "part of the input that comes from the past."
- The **pure spine** runs in the middle, exactly as today, touching only run-scoped State.
  It does not touch the store. This keeps the interior pure and testable.
- An **entity-writing node** at (or near) the end of the run saves the updated record(s)
  back to the store. This is the "part of the output that should outlive the present."

The store is touched only at the edges of the graph. Nothing in the middle reads or writes
it. This is what keeps the spine a pure function and the smoke test meaningful.

Contrast the current fiction this replaces: the "existing list" was *invented by an LLM*,
and the result was returned as the string `"Success"` and discarded. After this work, the
prior records are loaded from a real store and the updated records are saved to it.

---

## 5. Touch points: where this integrates

The change threads through a small number of named seams. Each mirrors a pattern already in
the codebase, which is what keeps it safe for small models to author against.

### 5.1 `Deps` gains one field (the store is a Port)

`store` is **always a live `Store`**, never `None` on Deps. The None-vs-path decision is
resolved once, at the door (§5.2).

```python
@dataclass(frozen=True)
class Deps:
    model: str
    agent_overrides: dict[str, object]
    loop_cap: int
    store: Store          # NEW. Always present. Never None.
```

This is the existing Port discipline (the agent sink, sandbox, and reporter are already
Ports on Deps); the store is one more.

### 5.2 The run entry resolves path-or-None, once

In the generated `graph.py`, the run entry gains an optional `store` path and constructs the
`Store` exactly once, above every node:

```python
async def run_<system>(text: str, *, model: str, store: str | None = None) -> RunEndT:
    deps = Deps(
        model=model,
        agent_overrides={},
        loop_cap=LOOP_CAP,
        store=Store(store),     # None -> in-memory; path -> durable. Decided here, once.
    )
    ...
```

Consequence: nodes never see a path and never see a `None`. They see a live five-method
`Store` whose durability was decided above them. **The same node code runs statelessly
(in-memory) or statefully (file-backed) with no branching.** This is the one-code-path rule
paying off inside the generated function.

### 5.3 The CLI forwards the path string, nothing more

```python
# generated __main__.py
parser.add_argument(
    "--store",
    default=None,
    help="Path to a database file. Omitted means in-memory (the function forgets).",
)
...
run_<system>(args.text, model=args.model, store=args.store)
```

No hidden `~/.local` default, no env var. Persistence is opt-in via an explicit path
(standing rule: explicit paths, no hidden defaults).

### 5.4 `render` provisions the store for entity-bearing systems

For a system with at least one entity, `render`:
- adds `store: Store` to the emitted `Deps` (replacing today's TODO comment),
- constructs `Store(store)` in the run entry as in §5.2,
- adds `--store` to `__main__.py`,
- tells the entity-reading node body to load via `ctx.deps.store.get` / `.list`,
- tells the entity-writing node body to save via `ctx.deps.store.add`.

For a purely stateless system (no entity), `render` may still wire `Store(None)` uniformly
(one code path) or omit the field; **prefer wiring it uniformly** so there is a single
generated shape and the deps gate's allowed-set is consistent. The store simply goes unused.

### 5.5 The deps gate admits `store` and still shuts the fakes

The existing deps gate (`check_deps_access`) reads the allowed `ctx.deps.*` attribute set off
the generated `deps.py`, so gate and renderer cannot drift. Adding `store` to `Deps` means
the gate now **permits** `ctx.deps.store` and still **forbids** any undeclared handle
(`ctx.deps.repo_url`, a throwaway, an invented HTTP client). This is the mechanism that makes
persistence *real instead of hopeful*: the legitimate door is open and the fake ones are shut,
statically, at build time, never as a runtime `AttributeError`. **This gate is the
enforcement half of document B's anti-faking clause.**

### 5.6 The smoke test is offline by construction

The generated `tests/test_smoke.py` injects an in-memory store, which is simply `Store(None)`
— the *same object* production uses with no path. There is no stub class and no mock; the
in-memory mode **is** the test mode.

```python
# generated tests/test_smoke.py (illustrative)
deps = Deps(model="stub", agent_overrides=STUBS, loop_cap=1, store=Store(None))
# Persist-then-read-back assertions run fully offline, no Ollama, deterministic:
#   run the entity-writing path, then assert the entity-reading path sees it.
```

Because `Store(None)` exercises the real load and save code paths (just against RAM), the
audit verifies persistence behaviour without any external service. This is the single
property that `document_store` could not provide.

---

## 6. The one genuine per-system decision: the key

Everything above is identical for every system forge builds. The only thing that varies is
how `key` is chosen, and it is a one-line decision the design front-end makes, not a schema
design.

- **Single-record entity** (one reading list, one running counter): the key is a **constant**
  — the `record_type` itself, or `"_"`. There is exactly one row; you always
  `get(collection, "_")` and `add(collection, "_", ...)`. Identity is trivial because there
  is only one record.
- **Multi-record entity** (many notes, many reviews): the key is an **id field on the
  entity**, a `uuid4()` string assigned at creation. `list(collection)` returns all of them;
  `get(collection, id)` fetches one.

That is the entire variation: a constant key, or a uuid. Not a database design — a single
fork the front-end resolves from whether the entity is "the X" (one) or "a X" (many). It
connects to the entity-classification check (§7): a datum read-at-entry and written-at-exit
under one identity is the single-record case; a datum accumulated across runs is the
multi-record case.

---

## 7. The hard problem this PRD must not under-weight: entity classification

The store is the easy part. The part that actually breaks is **the design front-end
reliably deciding that a piece of data is a durable entity (lives in the store) rather than a
run-scoped signal (lives in State and is forgotten).** We have already seen this fail: a
durable queue was mis-modelled as a run-scoped signal (a `produces_kind` mistake). A perfect
store still yields broken systems if the front-end calls the wrong kind.

The existing R1 validator enforces the *type* discipline (State holds signals and IDs, not
entities) but not the *semantic* judgment of "this datum outlives the run."

**Required approach — deterministic, not prompt-hope, and not keyword-matching:**

The strongest signal is structural and already available. forge extracts input/output
boundaries at the Understand stage. A datum that is **both read at the start of the run and
written at the end under the same identity** is, by its data-flow signature, durable: it
flows in, gets mutated, and flows back out as the same thing. That round-trip *is* the entity
signature.

So add a deterministic check at compile: **if a boundary datum round-trips (read-at-entry and
written-at-exit under one name) but no entity was declared for it, fail loud** with "this
looks like persistent state; declare it as an entity or reshape the brief." This is the
rung-2 analogue of the existing puzzle-fit: just as producer/consumer matching *derives*
fan-in/out rather than asking the model, the round-trip *derives* entity-ness rather than
asking the model.

Do **not** implement this as keyword-matching on "save" / "list" / "track" / "remember".
That is the hard-coded-domain trap the project forbids. Judge the data-flow shape, not the
vocabulary.

---

## 8. What queries this enables, and where the wall is (normative)

This is the section to design against. After this work, a generated function can do exactly
these things with its store, and must refuse anything beyond them.

**Enabled (these are the rung-2 capabilities forge must be able to generate):**

| Capability | Store calls | Example brief it unlocks |
|---|---|---|
| Read the one durable record | `get(coll, "_")` | "given my reading list and a new message, return the updated list" (load the list) |
| Save the one durable record | `add(coll, "_", v)` | (…and save the updated list) |
| Read all records of a kind | `list(coll)` | "summarise everything in my notebook" (load all notes) |
| Read one record by id | `get(coll, id)` | "update note #id and return it" |
| Append / create a record | `add(coll, uuid, v)` | "append a note and return the new total count" (add note, then `len(list(coll))`) |
| Update a record (same verb) | `add(coll, id, v)` | "mark this record updated" (overwrite by id) |
| Remove a record | `remove(coll, id)` | "drop this item and return the remaining list" |
| Count records | `len(list(coll))` | "return how many notes I have now" |

The unifying pattern for all of them: **load at start (`get`/`list`), compute in the pure
spine, save at end (`add`/`remove`).** The counter case is the clearest demonstration of why
the store is needed at all: the count *is* the accumulated history, which a stateless
function throws away.

**The wall (these must be refused or escalated, never built by growing the store):**

- **"Find the records where `<field>` `<predicate>`"** (filter/query by content). The store
  has no `where`. This is either a rung-3 app (the caller is choosing a query operation) or a
  `document_store` job (find-by-meaning). Refuse and reshape, or point at `document_store`.
  Do **not** add a query method.
- **"Several operations on the same data"** (add / show / remove / mark, chosen by the
  caller). This is rung 3 (an app): the caller owns the loop. Refuse and reshape to a single
  rung-2 function (e.g. "given my list and a message, return the updated list").
- **"Find by meaning / semantic search."** That is `document_store`, not this store. A
  generated system can opt into `document_store` for that specific need; it is not forge's
  general persistence.
- **"Remember across a conversation / session."** That is rung 4 (an agent). Refuse.

The discipline: a sixth store operation is never the answer. The closed five-operation
surface is itself an enforcement mechanism for the rung boundary. When a brief wants more
than load-at-start / save-at-end keyed access, the brief has left rung 2.

---

## 9. The functions forge must be able to create (acceptance criteria)

forge passes this PRD when it can generate, build, and pass-the-offline-audit-for each of
these, and refuse the out-of-scope ones with a sensible reshape. (These extend the scenario
corpus; encode them as fixtures.)

**Must build (rung 2):**

1. *"Given my current reading list and a new chat message, return the updated list."*
   Single-record entity, constant key. Load the list, apply the message, save the list,
   return it. (Also buildable as caller-persists rung 1; the store-backed version is the
   rung-2 acceptance case.)
2. *"Append a note to my notebook and return the new total count."* Multi-record entity,
   uuid key. `add` the note, then return `len(list(coll))`. The count must actually increase
   across runs against a file-backed store, and must reset across runs against in-memory.
3. *"Record this outcome, then report how many outcomes have been recorded."* Same shape as
   #2; confirms the pattern generalises.
4. *"Update the saved record for `<id>` with this change and return it."* Multi-record,
   get-by-id then add-by-id (overwrite).

**Must refuse + reshape (rung 3+):**

5. *"Manage my personal reading list."* Rung 3 (caller-chosen operations). Reshape → #1 or a
   rung-1 extraction.
6. *"A chatbot that answers questions about my documents."* Rung 4 (session). Reshape →
   "answer ONE question about a document set" (rung 1).
7. *"Watch my inbox and file incoming mail."* Rung 5 (event trigger). Reshape → "classify ONE
   email into a folder" (rung 1).
8. *"Find all my notes mentioning `<topic>`."* Beyond the closed surface (content query).
   Reshape → rung-1 "given these notes and a topic, return the matching ones" (caller passes
   the notes), or point at `document_store` if find-by-meaning is genuinely wanted.

**Offline-audit requirement for every "must build":** the generated smoke test must persist
and read back through the real load/save paths using `Store(None)`, with stub agents, no
model, deterministically.

---

## 10. Build order

1. **Encode the scenario corpus first** (§9), as the acceptance test for everything else.
2. **Ship the `Store` component** (§3) in `forge.runtime`. Trivially unit-testable in
   isolation.
3. **Wire it through render** (§5): Deps field, run-entry path resolution, CLI flag,
   in-memory smoke test, deps-gate allowed-set update.
4. **Build the entity-classification check** (§7) — the round-trip detector. This is the part
   that actually fails; budget for it accordingly. Without it, the store is correct but fed
   wrong decisions.
5. **Turn on rung-2 in the front fitness gate.** Until steps 2 to 4 land, the fitness gate
   treats `stateful_function` as not-yet-buildable and reshapes down to rung 1. Once they
   land, the gate admits `stateful_function`, and the deps gate plus store provisioning make
   it real instead of faked.

---

## 11. Invariants this work must not regress

- **Fail loud, no silent fallbacks** (in forge and in generated code). A node told to persist
  with no declared store handle fails the deps gate at build, never fakes it at runtime.
- **Deterministic does the heavy lifting.** Entity-vs-signal is decided by a structural
  round-trip check (§7), not by a prompt and not by keywords.
- **One code path.** The same generated function runs stateless (in-memory) or stateful
  (file-backed) on the presence of a path; no small-vs-large or stateful-vs-stateless
  branching in the body.
- **No env vars, no hidden defaults.** The store path is an explicit argument; absent means
  in-memory.
- **forge stays a near-leaf.** The store is forge-owned stdlib sqlite, not `document_store`,
  not any heavy sibling. This is *why* the store is forge-owned (document A, §5; document B,
  §5).
- **`graph.py` stays the only engine-aware file** in the generated package; the store lives
  in engine-free `runtime`, reached through a Port on Deps.
- **The store surface stays closed at five operations.** A sixth operation is a rung signal,
  not a feature (§8).
