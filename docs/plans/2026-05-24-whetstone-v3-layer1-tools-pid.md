# Whetstone v3 — Layer 1 review tools (PID)

Status: design, not yet implemented
Date: 2026-05-24
Branch: `whetstone-iterative-review`

## 1. Context

v3 currently reviews a document by handing each criterion-stage agent a
**compact digest** (verbatim claims + per-section gists + citation markers)
and — when the source fits within `raw_text_budget` (currently 80,000 chars)
— the **full markdown** as well. The agent then reasons over that prompt
input and emits findings.

The v2-vs-v3 benchmark on five seminal arXiv papers, judged blinded by
gpt-5.4-mini after the SPECS-style cascade landed, found v3 winning two
papers, losing one (Adam), and inconclusive on two. Reading the adjudication
detail showed v3's losses come from a structural pattern: when an issue
threads across sections (proof in §4 depending on assumption in §3, etc.)
v3's digest doesn't preserve enough to surface it, and the reviewer can't
recover what the digest lost. Adam's 12 critical-cross-section findings
that v3 missed are the canonical case.

The AAAI-26 SPECS paper closes this gap by giving its Correctness/Evaluations
stages a code-interpreter tool and its Significance stage a web-search tool;
the stage can investigate the paper instead of just synthesising what it was
handed. This PID brings that pattern into v3 in a deliberately minimal form
— pure-Python tools that complement the digest, scoped uniformly across all
criteria, before any external-network or code-execution capability.

## 2. Goals and non-goals

### Goals

1. Give every criterion-stage agent the ability to **read more of the source
   than the digest gave it**, on demand, via a small set of pure-Python tools.
2. Establish the pydantic-ai pattern (`tools=[...]`, `deps_type=...`,
   `RunContext[Deps]`, `UsageLimits`) so that layer 2 (novelty search) and
   layer 3 (code interpreter, deferred) can slot in without re-architecting.
3. Resolve the redundancy between **three current paths to the same content**
   (digest in prompt, full markdown in prompt, eventual tools). Pick one
   on-demand path; drop the always-in-prompt full-markdown branch.
4. Update the codebase's pydantic-ai usage to the **current, non-deprecated**
   API surface (`Agent(retries={'tools': N, 'output': M})` style — see §7).

### Non-goals

- **No layer 2** in this PID. `search_prior_work` / `deep_research` integration
  is the immediately-following work, but is gated on layer 1 actually moving
  the benchmark. Designing it now would be premature.
- **No layer 3** (`python_exec`, code interpreter). Out of scope for this
  iteration of the project.
- **No graph-level changes** beyond what `review_criterion` needs. The cascade,
  consolidate, gate, synth, critique-revise nodes all stay exactly as they are.
- **No changes to how the digest is built.** `ExtractClaims`, `BuildModel`,
  the `Claim` / `SectionGist` / `Citation` shapes — untouched. The digest's
  job stays "orient the reviewer"; tools take over "let the reviewer read."
- **No changes to other criterion sets** beyond SPECS. The mechanism is
  general (criterion-as-data with a `tools` field), but the only set in the
  tree is SPECS.

## 3. The architectural cleanup

Today's per-criterion prompt has three overlapping content paths:

| Path | Always present? | Cost | Granularity |
|---|---|---|---|
| Digest (claims + gists + citations) | Yes | ~1.5k tokens | per-section gist, per-paragraph claims |
| Full markdown | Iff source ≤ 80k chars | ~20k tokens | full prose |
| Tools (proposed) | Yes, but opt-in per-call by the agent | ~0 tokens base | section / substring |

This is too many ways to reach the same content. Once tools exist, the full
markdown in the prompt is redundant **and** introduces a fits-or-doesn't-fit
branch that makes per-call behaviour non-uniform.

**Decision: drop "full markdown if it fits" from the prompt.**

After this PID:

- Digest is the **always-present orientation** layer: a structured index of
  what's in the paper.
- Tools are the **on-demand source-of-truth** layer: read or search.
- No "raw text in the prompt" any more. `V3Deps.raw_text_budget` is removed.
  The just-merged 80k tuning guide gets a single-line eulogy in the commit
  message; the comment block is replaced with the criterion-tools field.

### Why this is the right cleanup, not just an aesthetic one

- **Uniform behaviour across paper sizes.** No fits/doesn't-fit branching;
  every paper goes through the same shape.
- **Lower per-call prompt cost.** Significance — last in the cascade and the
  biggest prompt — drops from ~24k tokens of input down to ~5–8k.
- **Forces the reviewer to be selective.** The model has to decide what it
  needs to read, rather than being handed everything and pretending to read.
  This is closer to how a real reviewer works.
- **Tool cost is bounded by pydantic-ai's existing limit machinery** (§7), so
  this isn't a wall-clock or LLM-cost regression — it's a redistribution.

### What we lose, honestly

- **Lazy strong models** that would have just held the whole paper in context
  and reasoned over it now have to issue `read_section` / `search_paper` calls.
  On capable models (gpt-5+, opus-4+) the tool-loop is reliable; on
  weaker/local models it's a behaviour change.
- The "full text in prompt" path was a fallback that papered over some weak
  digesting. If the digest misses something, the agent now has to ask, not
  just notice in the full text. This puts more pressure on tool-use quality.

Both are net-positive in the long term (uniform behaviour, smaller prompts)
but worth naming.

## 4. Tool surface

Two tools, both pure-Python, both universally available to every criterion.

### `read_section(section_id: str) -> str`

Returns the full text of the section identified by `section_id`. Section
ids match those already in the digest's gists (e.g. `"3.2"`, `"abstract"`,
or whatever `sectionize.py` emits).

- Pure function over `DocumentModel.sections`.
- Returns the section's `text` field. If the section_id doesn't match,
  returns a brief error message (string) that the model can read and react to
  — not a Python exception (we don't want pydantic-ai to burn retries on
  invalid section ids).
- No length cap. The longest single section in the corpus is ~20k chars.
  Significance reading a couple of sections back-to-back stays well under
  the agent's per-call token budget.

### `search_paper(query, *, max_results=5, regex=False) -> list[Match] | str`

Search across `DocumentModel.source`. Returns up to `max_results` matches,
each as a `Match` value:

```
Match = {
  section_id: str,          # which section the hit is in
  snippet: str,             # ~200 chars centred on the match
  position: int,            # absolute char offset (for traceability)
}
```

Two modes:

- **`regex=False` (default)** — case-insensitive substring search. Always
  succeeds, can't fail to compile, easy for any model to call. This is the
  workhorse path: "does the paper mention 'limitation'?" /
  "where is 'baseline' discussed?" / "find this verbatim phrase."

- **`regex=True`** — treat `query` as a Python `re` pattern with
  `re.IGNORECASE`. Useful for alternation, character classes, and word
  boundaries: `(limitation|caveat|weakness)`, `Theorem [0-9]+`, `\bAdam\b`.
  A real reviewer uses this kind of pattern constantly in cmd-F; giving the
  agent one regex call instead of three substring calls is cheaper against
  the `UsageLimits.request_limit` budget.

Other semantics:

- **Library: Python stdlib `re`.** No new dependency; fast enough at our
  document sizes (≤100k chars). If catastrophic-backtracking patterns
  become a problem in practice, swap to the third-party `regex` module
  (real `timeout` parameter, better Unicode) — single-line change. We're not
  pre-emptively adopting it.
- **Pure function.** Stateless; same query → same results.
- **`max_results` defaulted at 5** to keep the response tractable. Agent
  can raise it if it explicitly wants more (e.g. to count occurrences).
- **Zero matches → empty list.** A clear signal that the paper doesn't
  discuss the term. (Half the value of this tool is confirming *absence* —
  that's a real reviewer move.)

### Safety on the regex path

Three guards keep `regex=True` from causing trouble:

1. **Query length cap.** Patterns over ~200 chars are rejected at the door
   with a returned string error. Reasonable patterns are short; long ones
   from an LLM almost always indicate confusion (e.g. trying to encode
   logic in regex that doesn't belong there).
2. **`re.error` is caught.** Compile failure returns a string the agent
   can read and react to, e.g.
   `"invalid regex 'foo[': missing close bracket. Try simpler or set regex=False."`
   The agent corrects on its next turn instead of pydantic-ai burning a
   retry.
3. **Asyncio wall-clock guard.** The `finditer` call is wrapped with
   `asyncio.wait_for(..., timeout=2.0)`. Python `re` has no native timeout
   parameter, and `signal.alarm` isn't async-safe; the wrapper is the
   cleanest path. Two seconds is generous — well-formed patterns finish
   in microseconds on our document sizes — so it only trips on genuine
   pathological backtracking. On timeout, returns a string error.

All three errors land as string returns rather than exceptions, so the
model gets a chance to correct without spending one of the `retries={'tools': 1}`
attempts on what is effectively user error.

### Why exactly these two tools, with regex in `search_paper`

The two tools compose to cover almost every "I want to look this up"
question a reviewer asks:

- "What does §4.2 actually say?" → `read_section("4.2")`.
- "Does the paper address limitations anywhere?" →
  `search_paper("limitation")` → if empty, finding; if hits, follow up with
  `read_section(s.section_id)` on the top hit.
- "Find limitation, caveat, or weakness as variants" →
  `search_paper("(limitation|caveat|weakness)", regex=True)`.
- "Find every Theorem N cross-reference" →
  `search_paper(r"Theorem [0-9]+", regex=True)`.
- "What surrounds this claim?" → the claim contains a verbatim quote;
  `search_paper(claim.quote, max_results=1)` gives `section_id` →
  `read_section(...)` reads the full section.
- "How many times is X mentioned?" → `search_paper("X", max_results=20)`.

What we are deliberately not adding:

- `find_table(N)` / `find_figure(N)` — domain-specific. Reviewer can
  achieve it with `search_paper("Table 3")` or
  `search_paper(r"Figure [0-9]+", regex=True)`.
- `get_quote_context` — subsumed by `read_section`.
- `list_claims_by_section` — the digest already shows claims grouped by
  section (see §5). No tool needed.
- `find_numbers_near` — too narrow; criterion-specific tools belong in
  their own layer if they belong anywhere.
- `case_sensitive` flag — academic papers use inconsistent casing; the
  reviewer almost never wants case sensitivity. Use regex with explicit
  `(?-i:...)` group if the rare case arises.

Two tools, both general, both pure-Python. That's the entire layer 1 surface.

## 5. Digest enrichment for tool-readiness

The current digest, rendered by `_project()` in `v3/review.py`, has two
real gaps for tool use — both blockers, not nice-to-haves:

1. **No `section_id` anywhere in the prompt.** The gists block emits
   `- {title}: {gist}` per section. The agent has no way to know whether
   the right id to pass to `read_section` is `"4.2"` or `"sec_004"` or
   `"Convergence Analysis"`. Without this, the read tool is unusable.

2. **No size signal.** All sections look equally substantial in the
   prompt — one-sentence gist each. A reviewer needs to know that §4
   "Convergence Analysis" is 9,000 chars and §7 "Conclusion" is 900 chars,
   because that's where the meat is.

A third gap is less critical but worth fixing in the same change:

3. **Claims aren't grouped by section.** They're emitted as a flat list of
   verbatim quotes with no context. Grouping by section turns them into a
   "what each section asserts" view — far more useful for navigation
   decisions.

### What the enriched digest looks like

The data needed is already in the DocumentModel (`Section` has `id`,
`title`, `text`; `Claim` has `span.section_id`). This is purely a prompt
formatting change in `_project()`, not a model change.

```
SECTIONS (id | title | size | gist):
  - [abstract] Abstract (823 chars) — We propose Adam, an algorithm...
  - [1] Introduction (3,421 chars) — Stochastic gradient-based methods...
  - [2] Algorithm (2,103 chars) — Algorithm 1 describes Adam updates...
  - [3] Initialization Bias Correction (1,890 chars) — Bias correction...
  - [4] Convergence Analysis (8,991 chars) — Under bounded gradients...
  - [5] Experiments (5,634 chars) — We compare on three benchmarks...
  - [6] Related Work (2,891 chars) — Prior optimizers including AdaGrad...
  - [7] Conclusion (945 chars) — We have introduced Adam...

CLAIMS BY SECTION:
  [1]:
    - "We propose Adam, a method for efficient stochastic optimization."
    - "Adam combines the advantages of two recently popular methods..."
  [4]:
    - "Under bounded gradients, the regret bound is O(√T)."
    - "Our convergence guarantees match the best known online learning rates."
  ...

CITATIONS PRESENT: [1], [12, 13], [@vaswani2017], ...
```

### Why this is enough — not more

This enrichment gives the agent:

- **A complete table of contents.** Every section visible with its id,
  title, and size. Nothing is hidden; the question "what's in the paper"
  is answered without any tool calls.
- **Size hints.** Reviewer knows where the substantial content lives.
- **Per-section semantic hint** (the gist) for navigation decisions.
- **Per-section claims** so the agent knows what each section asserts
  without having to read it.
- **The ids it needs to invoke `read_section`.**

What we are deliberately *not* adding to the digest:

- **Per-section key-term extraction** ("§3 discusses momentum, learning
  rate, ..."). Tempting but expensive — needs an extra LLM pass at digest
  build time, and small models extract terms unreliably. The agent can get
  this on demand via `search_paper`.
- **Subsection hierarchy as nested structure.** If the underlying section
  ids encode order (`4`, `4.1`, `4.2`) the natural flat ordering already
  conveys it. Adding indentation/structure is visual noise.
- **Cross-reference graph** (which sections cite which figures/tables).
  Domain-specific.

The remaining failure mode — gist misses something subtle, title is
generic ("Background"), section is short so size doesn't flag it, no
claim from that section made the digest — has a clean recourse:
`search_paper`. If a prior-stage finding flags "the contribution claim
seems narrow," Significance can `search_paper("novel")` and follow hits;
no gist mention required.

### Where the change lives

`_project()` in `v3/review.py` is rewritten to emit the new
SECTIONS / CLAIMS-BY-SECTION blocks. This is **part of commit 3** in §8
(the `review_criterion` wiring commit) since the prompt change and the
tool wiring have to land together — the tools rely on the prompt
exposing `section_id`s the agent can call them with.

### Why exactly these two, and not more

These two compose to cover almost every "I want to look this up" question
a reviewer asks:

- "What does §4.2 actually say?" → `read_section("4.2")`.
- "Does the paper address limitations anywhere?" →
  `search_paper("limitation")` → if empty, finding; if hits, follow up with
  `read_section(s.section_id)` on the top hit.
- "What surrounds this claim?" → the claim contains a verbatim quote;
  `search_paper(claim.quote, max_results=1)` gives `section_id` →
  `read_section(...)` reads the full section.
- "How many times is X mentioned?" → `search_paper("X", max_results=20)` gives
  a count.

What we are deliberately not adding:

- `find_table(N)` / `find_figure(N)`: domain-specific (papers have these;
  user docs don't; legal docs have other structures). Reviewer can achieve
  it with `search_paper("Table 3")` if needed.
- `get_quote_context`: subsumed by `read_section`.
- `list_claims_by_section`: the digest already groups claims by section in
  the prompt — no tool needed.
- `find_numbers_near`: too narrow; criterion-specific tools belong in their
  own layer if they belong anywhere.

Two tools, both general, both pure-Python. That's the entire layer 1 surface.

## 6. Behaviour examples

Concrete picture of a Correctness review on Adam, after layer 1 lands:

```
[review.Correctness]
   prompt: digest + prior findings (Story flagged "convergence claim overstated")
   agent: starts with one model call, sees the digest, sees Story's flag.
   agent: "Story flagged the convergence claim. Let me check §4.2 in full."
     tool: read_section("4.2") -> full proof text
   agent: "Proof assumes bounded gradients. Is that formalised anywhere?"
     tool: search_paper("bounded gradient", max_results=5) -> 0 matches
     tool: search_paper("Assumption", max_results=5) -> 2 hits, both in §3
     tool: read_section("3") -> full text of §3
   agent: forms finding "Theorem 4.1 depends on bounded gradients but §3
          discusses it informally; no formal assumption."
   total: 5 model requests (1 initial + 3 tool turns + 1 final).
```

This is the qualitative shift. Same 5 model requests as today's parallel
review per criterion, but the reviewer is *grounded* in the source for the
specific thing it's checking.

Story (which mostly reasons over high-level claims and gists) will probably
issue zero tool calls on most papers. Correctness and Evaluations will issue
the most. Significance will issue some now and will issue many more when
layer 2 lands.

## 7. Criterion configuration

`Criterion` (currently `name: str`, `questions: list[str]`, `facets: list[Facet]`)
gains one optional field:

```
tools: list[str] = []
```

Names map to a registry of tool builders (e.g. `"novelty_search"` will map
to layer 2's deep_research wrapper later). **Layer 1 tools are not declared
in this field** — they are always available to every criterion. The field
is for *opt-in extras*: layer 2 and layer 3 capabilities a particular
criterion needs.

For SPECS today, every criterion gets `tools=[]` (no opt-ins yet, since
layer 2/3 don't exist). When layer 2 lands, Significance gets
`tools=["novelty_search"]`.

This separation keeps the universal/cheap tools (layer 1) implicit, and
makes the costly/specialised tools (layer 2/3) explicit and auditable in
the criterion definition.

## 8. Pydantic-AI integration

### Construction

```
agent = Agent(
    model=resolve_model(agent_model),
    instructions=_PROMPT,
    output_type=_CriterionFindings,
    deps_type=DocDeps,
    tools=[read_section_tool, search_paper_tool],
    retries={"tools": 1, "output": 2},  # modern dict form
)
```

Notes:

- `tools=[...]` accepts callable plain functions OR `Tool` objects. For layer
  1 we use plain functions decorated with the right signature; pydantic-ai
  infers the schema from type hints + docstring.
- `deps_type=DocDeps` declares the typed dependency object. Tools receive
  `ctx: RunContext[DocDeps]` as their first arg and reach the DocumentModel
  via `ctx.deps.document_model`.
- `retries={"tools": 1, "output": 2}` is the dict form. The older
  `Agent(retries=N, output_retries=M)` shape with separate positional ints
  is being normalised — the dict form is the documented current API.
  **Follow-up: migrate `core/agents.py:build_pydantic_ai_agent` to the same
  shape in a separate small commit.** It currently passes
  `retries=defn.retries, output_retries=defn.output_retries` as separate
  kwargs.

### Execution and limits

```
result = await agent.run(
    prompt,
    deps=DocDeps(document_model=model),
    usage_limits=UsageLimits(
        request_limit=8,
        total_tokens_limit=80_000,
    ),
)
```

`UsageLimits.request_limit` is the load-bearing cap: total model requests in
this single `agent.run(...)` loop, including the initial call, each
tool-call iteration (a tool call is itself a model request), and any retries
within the loop. `request_limit=8` allows roughly: 1 initial reasoning call +
up to 5–6 tool-call iterations + 1 final answer. Past that the agent is
force-finished by pydantic-ai with whatever it has.

`total_tokens_limit` is the secondary backstop — a runaway reviewer that
keeps calling `read_section` on long sections could otherwise eat through
the context window.

### Tool definition shape

```
async def read_section(ctx: RunContext[DocDeps], section_id: str) -> str:
    """Return the full text of the section identified by section_id."""
    section = ctx.deps.document_model.section_by_id(section_id)
    if section is None:
        return f"no section with id {section_id!r}; check the digest for valid ids"
    return section.text

async def search_paper(
    ctx: RunContext[DocDeps],
    query: str,
    max_results: int = 5,
) -> list[dict]:
    """Substring-search the paper. Returns up to max_results matches."""
    # implementation: case-insensitive find over ctx.deps.document_model.source
    ...
```

Pydantic-ai introspects the type hints + docstring to build the tool schema
the model sees. Returning structured data (`list[dict]` or a pydantic model)
is fine; pydantic-ai serialises it for the model.

### Deps object

```
@dataclass
class DocDeps:
    document_model: DocumentModel
    # layer 2 will add: novelty_calls_used, novelty_calls_max, deep_research
```

Lives in `whetstone/v3/tools.py` (new file) so the deps type can be imported
by both the tool definitions and `review_criterion`.

### Tool-use hint in the criterion system prompt

Pydantic-ai automatically exposes the tools' typed signatures + docstrings
to the model, so a capable model can infer when to call them. We additionally
include a short hint at the end of `_PROMPT` to give models an explicit
activation signal — particularly important for capable-local-tier models
that may need a nudge to reach for tools when the digest doesn't tell them
enough. The same hint is shown to every model regardless of size; it costs
~80 prompt tokens and is helpful redundancy for frontier models that would
have figured it out anyway.

The hint reads:

> You have two tools available to investigate the source beyond the digest:
> `read_section(section_id)` to read a section in full, and
> `search_paper(query)` to find where a term appears (with `regex=True` for
> patterns like `(limitation|caveat|weakness)`).
> Use them when:
>   - the digest doesn't tell you enough to answer a criterion question
>     (e.g. you want to confirm whether the paper mentions a concept);
>   - prior-stage findings draw attention to a section worth reading in full;
>   - you're considering flagging an absence — verify it with a search first.
> The section ids appear in the SECTIONS list (e.g. `4.2`, `abstract`).

This sits in the system prompt (the `instructions=` field on the Agent),
not in the per-call user message, so it's stable across the cascade.

## 9. Implementation plan

The changes localise to v3, plus a small migration in `core/agents.py`.

### Files affected

1. `src/andamentum/whetstone/v3/tools.py` (new). Defines `DocDeps`,
   `read_section`, `search_paper`. ~80 lines.
2. `src/andamentum/whetstone/v3/review.py`. Two changes land together
   in commit 3: (a) `review_criterion` switches to constructing the Agent
   with `tools=`, `deps_type=`, and `usage_limits=` on the `run` call; the
   "full text if it fits" code path is deleted. (b) `_project()` rewrites
   the digest's section/claim blocks to expose `section_id`, char count,
   and claims-grouped-by-section per §5. The two changes have to land
   together because the tool wiring depends on the agent seeing
   `section_id`s in the prompt. ~60 lines changed.
3. `src/andamentum/whetstone/v3/graph.py`. `V3Deps.raw_text_budget` is
   removed (along with the just-merged tuning-guide comment block). The
   `ReviewCriteria` node no longer computes `full_text` — passes None or
   nothing onward. ~10 lines changed.
4. `src/andamentum/whetstone/v3/criteria.py`. `Criterion` gains
   `tools: list[str] = []`. ~3 lines changed. SPECS preset unchanged
   (every criterion's `tools` defaults to `[]`).
5. `src/andamentum/core/agents.py` (small migration). Switch from
   `retries=, output_retries=` separate-int kwargs to
   `retries={"tools": N, "output": M}` dict form. ~5 lines changed. Cleanly
   isolates the pydantic-ai API normalisation.
6. Tests: a focused new test file `v3/tests/test_tools.py` covering
   `read_section` and `search_paper` on a small fixture DocumentModel
   (deterministic, no LLM, no agent — pure-Python tools tested directly).
   The existing `test_review.py` is updated to stub the tool-using Agent
   the same way it currently stubs the tool-less Agent (mock
   `build_pydantic_ai_agent`).

### Order

1. Land the `core/agents.py` migration first as its own commit (so the
   pydantic-ai API normalisation is reviewable separately).
2. Add `v3/tools.py` with the two tools, plus `test_tools.py`. Standalone,
   doesn't touch the review pipeline yet. Commit 2.
3. Wire `review_criterion` to use the tools + `UsageLimits`, drop full-text
   branch, drop `raw_text_budget`. Update `test_review.py`. Commit 3.

Three commits, each independently revertable.

## 10. Testing and success criteria

### Unit tests (deterministic, no LLM)

- `read_section` returns the right text for known ids, and a clear error
  string for unknown ids.
- `search_paper` (substring mode): finds known substrings, is
  case-insensitive, returns the right section_id per match, caps at
  `max_results`, returns empty list when the term is absent.
- `search_paper` (regex mode): finds known patterns; rejects over-long
  queries with a clear error string; returns a clear error string on
  `re.error`; honours `asyncio.wait_for(timeout=2.0)` (test with a
  deliberately pathological pattern on a synthetic source).
- The agent construction path is exercised in `test_review.py` with a
  mocked agent (no actual tool calls executed).

### Capable-local-model validation pass

Before declaring layer 1 done, run one paper from the benchmark corpus
through `review_document_v3` on each of the three canonical local
models the project targets:

- `ollama:gemma4:31b-nvfp4`
- `ollama:gemma4:26b-nvfp4`
- `ollama:gpt-oss:20b`

Inspect the agent's behaviour on each. The bar is **the same code path
works for all three as for frontier**; we're verifying the unified path
holds, not validating a small-model-specific code path (which we
explicitly don't have).

What to look for:

- Does the model call any tools at all on its own? (Expected: yes on at
  least one criterion, most likely Correctness or Significance.)
- When it calls `search_paper`, does it use substring or regex? Does
  regex compile correctly when used?
- When errors come back (bad section_id, malformed regex), does the
  model recover on the next turn?
- Are findings produced for every criterion, even ones that didn't use
  tools? (The enriched digest should carry them.)

What does NOT trigger a separate code path:

- If the local model never calls tools at all: that's the baseline
  (today's v3 quality from the digest alone). Acceptable. The fix, if
  desired, is to strengthen the prompt hint — the same hint visible to
  every model.
- If the local model writes bad regex repeatedly: the error-return path
  should let it recover or fall back to substring. The fix, if needed, is
  to soften the regex docstring's encouragement — again, visible to
  every model.

Either result is informative and addressable with prompt-level changes
that affect every caller uniformly.

### Benchmark expectation

The hypothesis this PID rests on: **layer 1 tools close the
v3-on-Adam gap**, because the Adam loss traces to the digest dropping
content the reviewer needed.

Falsification: re-run the v2-vs-v3 benchmark on `gpt-5.4-mini` judge after
layer 1 lands. Expected outcomes:

- **If v3 now wins or ties on Adam**, the digest-compression hypothesis is
  vindicated and layer 2 (novelty search) is the right next bet.
- **If Adam still goes to v2**, the loss isn't about digest compression; it's
  about reasoning capacity or criterion phrasing. We look elsewhere
  (criterion-question wording, cross-criterion threading depth, etc.).

The benchmark is the falsifier. No code in this PID is justified by anything
other than "does it move Adam."

### Cost expectations

- Per-criterion review: 1 initial call + 0–5 tool calls + 1 final = 2–7
  model requests, up from 1 today.
- Per paper (5 criteria, sequential cascade): 10–35 model requests, up from
  ~5 today.
- On gpt-5.4-nano at current pricing, a 5-paper benchmark goes from ~$0.07
  to ~$0.30. On mini, ~$0.16 to ~$0.70. Both still trivial.
- Wall clock per paper: roughly doubles. With concurrency=2 holding,
  acceptable.

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Agent gets stuck in tool loops | `UsageLimits(request_limit=8)` forces termination |
| Tool errors compound | `retries={"tools": 1}` caps per-tool retries; tools return string errors instead of raising |
| Cost explosion on stronger models | `request_limit=8` + `total_tokens_limit=80_000` bound it deterministically |
| Reproducibility drift (same paper → different tool-call sequences) | `temperature=0` everywhere; pure-Python tools are deterministic; only the model's *choice* of tool calls varies, and that's already true for finding selection |
| Tool argument-shape errors | Pydantic-ai handles via tool schema; retries=1 caps the blast radius |
| pydantic-ai API normalisation breaks something else in the codebase | Migrate `core/agents.py` in its own commit; existing tests catch regressions |

### Local-model capability: one path that works across the range

The bar is **one unified code path that works for the canonical
local-model targets**: `ollama:gemma4:31b-nvfp4`,
`ollama:gemma4:26b-nvfp4`, `ollama:gpt-oss:20b`, scaling up to frontier.
No conditional dispatch based on detected capability, no
"small_model_mode" flag, no two-code-paths-with-perpetual-drift.

This is achievable because the design has four properties that hold for
any model that can call a tool at all:

1. **`regex=False` is the default.** The agent doesn't opt into the
   error-prone path; substring search is foolproof. Frontier models opt in
   to regex when they want alternation/character classes; local models
   that don't can ignore it entirely.

2. **Errors return as strings, not exceptions.** Bad `section_id`, malformed
   regex, timeout — all surface as plain readable strings the model
   processes on its next turn. No pydantic-ai retries burned, no crashes.
   The recovery path is the same whether the caller is opus-4 or gpt-oss.

3. **The enriched digest is reasoning-complete.** Section list with ids and
   sizes + claims grouped by section + citations + cascade's prior
   findings — a model that never calls a tool can still produce a coherent
   review from this substrate alone. Today's v3 already does. Tools are
   strictly additive: they let capable callers go deeper, they don't make
   the tool-less path worse for anyone.

4. **Pydantic-ai's typed tool schemas teach the model the API.** Every
   model that supports tool calls sees the signature and docstring; that
   plus the prompt hint (§8) is the same instruction set regardless of who's
   asking.

The realistic spread we expect:

- Frontier (gpt-5+, opus-4+): tool calls used freely, both modes, with
  good judgment about when to read versus search.
- Canonical local (`gemma4:31b-nvfp4`, `gemma4:26b-nvfp4`,
  `gpt-oss:20b`, mini cloud): tool calls used for substring search and
  section reads; regex used occasionally and sometimes incorrectly;
  error-return path keeps the loop alive.
- Smaller/older local (sub-10B, pre-2026): rarely calls tools; the
  enriched digest carries them. They produce today's v3 quality, which is
  the baseline we're starting from — not a regression. They are
  explicitly NOT the target tier; the validation pass does not include
  them.

There is no separate code path serving any of these tiers. The same
review_criterion call runs the same agent with the same tools and the same
limits; what differs is only how the model itself uses what it's given.

## 12. Open questions / future work

- **Layer 2 (novelty search).** Gated on layer 1 actually moving the
  benchmark. Will reuse the `Criterion.tools` mechanism added here. Design
  already drafted at a high level (deep_research wrapper, budgeted per
  criterion via `DocDeps`).
- **Layer 3 (code interpreter for Correctness).** Deferred entirely.
  Possibly a future PID if SPECS-parity becomes a goal.
- **Tool result caching.** Currently tools are stateless; a `search_paper`
  with the same query is computed twice. Cheap enough we don't care today;
  worth revisiting if budgets get tighter.
- **Surfacing tool calls in the rendered review.** "Verified by reading
  §4.2 and searching for 'bounded gradient'." Would add real provenance.
  Likely a small change in the renderers once finding metadata carries
  tool-call records.
- **Other criterion sets.** When non-academic criterion sets land (legal,
  technical reports, drafts), the layer-1 tools should still apply
  unchanged — that's their generality test. If a new set needs a new tool,
  it's a candidate for layer 2.

## 13. Decisions, summarised

- **Architecture: drop full-text-in-prompt; tools are the only path to source
  beyond the digest.** Digest = orientation; tools = source-of-truth access.
- **Tools: two, universal.** `read_section(section_id)`,
  `search_paper(query, max_results)`. Pure-Python, free, available to every
  criterion implicitly.
- **Criterion config: add `tools: list[str] = []`** for opt-in layer 2/3
  capabilities. Empty for all SPECS criteria today.
- **Limits: `UsageLimits(request_limit=8, total_tokens_limit=80_000)` per
  criterion call. `retries={"tools": 1, "output": 2}` on the Agent.**
- **pydantic-ai API: migrate `core/agents.py` to the modern dict-form
  retries** as a separate small commit before the v3 changes.
- **Test plan: pure-Python unit tests for tools, mocked-Agent test for
  review_criterion, benchmark re-run for the hypothesis (does this move
  Adam).**
