# PID — whetstone v3 self-correction via pydantic-ai's native surface

*Status: draft for approval. Author: Claude. Date: 2026-05-24.*

## 1. Context

The layer-1 tools (`docs/plans/2026-05-24-whetstone-v3-layer1-tools-pid.md`)
landed at commit `80a63e1`. The three smoke runs against
`arxiv_1412.6980_v1.md` (gpt-oss / gemma26 / gemma31) surfaced two real
behavioural gaps and one infra observation:

1. **Some quotes don't anchor.** `verify_findings` silently drops findings
   whose verbatim quote can't be located in the source. The smoke reports
   show the model producing quotes that look right but contain small
   normalisation drift (line breaks, expanded glyphs, paraphrase). The model
   is given no in-loop signal that the quote failed; we lose the finding.

2. **Tool errors leak as plain strings.** `read_section` and `search_paper`
   return `"no section with id ..."` / `"invalid regex ..."` / `"too long
   ..."` / `"timed out ..."` as ordinary tool results. The model sees the
   text but the framework has no idea anything went wrong — no per-tool
   retry budget, no `UnexpectedModelBehavior` when a model loops on bad
   ids, no clean signal in logs.

3. **An Ollama HTTP 400 ("invalid message content type: <nil>") hit two
   of the three smoke runs.** The criterion's run was caught by
   `run_criteria`'s bare `except Exception`, logged as `"<name> crashed:
   <msg>"`, and skipped. The exception body — which contains the provider's
   actual error payload — was lost.

Pydantic-AI already provides the three things we need:

- `raise ModelRetry(msg)` inside a tool → framework synthesises a
  `RetryPromptPart`, sends `msg` back to the model on its next turn,
  increments the per-tool retry counter, raises
  `UnexpectedModelBehavior("Tool 'X' exceeded max retries count of N")`
  when the per-tool cap is hit.
- `@agent.output_validator` — runs after the model produces structured
  output. Raises `ModelRetry` to send the model back to fix the output
  with the validator's message attached. Consumes the
  `output_retries=N` budget (separate from tool retries).
- Typed `AgentRunError` subclasses (`UnexpectedModelBehavior`,
  `UsageLimitExceeded`, `ContentFilterError`, `IncompleteToolCall`,
  `ModelHTTPError`) carry `message`, `body`, `status_code` — diagnostic
  fields the bare-Exception catch is throwing away.

Prior art in this codebase: `src/andamentum/document_store/extraction.py`
uses `@agent.output_validator` with `raise ModelRetry(...)` exactly the way
we'd use it in whetstone v3. We are not inventing a pattern.

## 2. Scope and non-goals

**In scope:**

1. (Stage 1) Convert the four error-return paths in
   `whetstone/v3/tools.py` from `return "<error string>"` to `raise
   ModelRetry("<error string with suggested fix>")`. The agent sees the
   same text via the framework's structured `RetryPromptPart` instead of
   as a tool result.
2. (Stage 2) Add an `@agent.output_validator` to the per-criterion agent
   in `whetstone/v3/review.py` that runs `locate()` over each
   `_RawFinding.quote` against `model.source`. On any miss, raise
   `ModelRetry` listing the offending quotes and asking for re-quotes.
   Falls back to anchored-only on retry exhaustion — preserves today's
   "silent drop" final behaviour as the deterministic floor.
3. (Stage 3) Replace the bare `except Exception` in
   `whetstone/v3/review.py:run_criteria` with a typed cascade —
   `UnexpectedModelBehavior` first (logging the `body` attribute when
   present), `UsageLimitExceeded` next, then `Exception` as the
   defence-in-depth catch.

**Out of scope:**

- Retry-with-backoff loops around the Ollama HTTP 400. That's an
  upstream pydantic-ai/Ollama interop bug; the right fix is upstream.
  Stage 3 only ensures we *see* the body the next time it fires.
- `args_validator` on `read_section` / `search_paper`. Duplicates the
  tool body's existing check; adds a hop with no observable benefit
  over Stage 1.
- Output-validator-enforced finding count floors/ceilings (e.g. "at
  least one finding per criterion"). Risks correction loops with no
  convergence; defer until we have evidence that it pays.
- Migrating `retries=2, output_retries=2` to the dict form
  (`retries={"tools": N, "output": M}`). pydantic-ai 1.84.1 still
  accepts the int form; no value in churn.
- Layer-2 tools (novelty search). Separate PID.

## 3. The "no breaking changes" frame

The user's explicit constraint is: don't break anything. We satisfy it by
making each stage **strictly preserve current end-to-end behaviour on the
unhappy path** while improving it on the happy path:

| Stage | Before (today) | After (with this PID) | End-to-end caller sees |
|---|---|---|---|
| 1 (tools) | Tool returns error string; model reads it and (hopefully) corrects on its next turn | Tool raises `ModelRetry`; framework sends same text to model as `RetryPromptPart`; model corrects on its next turn | Same model behaviour. If model loops 3+ times on bad ids → `UnexpectedModelBehavior` raised → caught by Stage 3's typed catch → criterion logged + skipped, exactly as today |
| 2 (output validator) | Bad-quote findings silently dropped post-run by `verify_findings` | Validator asks model to re-quote ONCE on the first attempt; on retry or if model still produces bad quotes, returns only the anchored findings — `verify_findings` does the same drop it does today | At worst: same set of findings as today. At best: recovers findings that today are silently dropped |
| 3 (typed exceptions) | `except Exception → log "crashed" → continue` | `except UnexpectedModelBehavior` (with `body`) / `except UsageLimitExceeded` / `except Exception` — all three log + continue | Same outer behaviour: a failed criterion is logged and skipped. Logs are more informative |

The public API surface (`review_document`, `review_criterion`,
`run_criteria`, `Finding`, the three renderers) is unchanged. No imports
move, no type signatures change for any exported symbol.

## 4. Stage 1 — Convert tool error returns to `ModelRetry`

### 4.1 The four error sites in `whetstone/v3/tools.py`

| Site | Today | After |
|---|---|---|
| `read_section`: unknown id (line 86-91) | `return f"no section with id {section_id!r}; check the SECTIONS block in your prompt for valid ids"` | `raise ModelRetry(f"no section with id {section_id!r}; valid ids are listed in the SECTIONS block of your prompt. Re-issue read_section with one of them.")` |
| `search_paper`: regex too long (133-139) | `return f"regex pattern too long ..."` | `raise ModelRetry(f"regex pattern too long ({len(query)} chars; max {_MAX_REGEX_LENGTH}). Use a simpler pattern, or call search_paper with regex=False for plain substring search.")` |
| `search_paper`: regex compile error (142-149) | `return f"invalid regex ..."` | `raise ModelRetry(f"invalid regex {query!r}: {e}. Try a simpler pattern, or call search_paper with regex=False for plain substring search.")` |
| `search_paper`: regex timeout (155-161) | `return f"regex ... timed out ..."` | `raise ModelRetry(f"regex {query!r} timed out (>{_REGEX_TIMEOUT_S}s) — likely catastrophic backtracking. Use a simpler pattern, or call search_paper with regex=False for plain substring search.")` |

### 4.2 Type-signature consequence

- `read_section(ctx, section_id) -> str` — unchanged.
- `search_paper(ctx, query, *, max_results, regex) -> list[dict] | str` →
  `search_paper(...) -> list[dict]`. The `str` return arm goes away
  because every error path now raises. **Internal-only signature
  change** — `search_paper` is not in any `__init__.py` export list;
  only the agent's tool harness and our tests call it.

### 4.3 Imports

`tools.py` adds `from pydantic_ai import ModelRetry` (it already imports
`RunContext` from the same module).

### 4.4 Logging

The existing `logger.info("[v3.tool] read_section(%r) → no such section",
section_id)` and the three regex-error log lines all stay. Raising
`ModelRetry` doesn't replace logging; the log is for *us* (the operator),
the `ModelRetry` is for the model.

### 4.5 Tests that change

`whetstone/v3/tests/test_tools.py` has three tests that assert on the
string-return contract:

| Test | Current assertion | New assertion |
|---|---|---|
| `test_read_section_returns_error_string_for_unknown_id` (line 80) | `assert isinstance(result, str); assert "no section with id" in result; assert "999" in result` | Wrap call in `pytest.raises(ModelRetry) as exc; assert "no section with id" in str(exc.value); assert "999" in str(exc.value)`. Rename to `test_read_section_raises_modelretry_for_unknown_id` |
| `test_search_paper_regex_compile_error_returns_string` (171) | substring-checks the returned string for `"invalid regex"` | `pytest.raises(ModelRetry)` with same substring on `exc.value` message. Rename to `..._raises_modelretry_on_compile_error` |
| `test_search_paper_regex_rejects_overlong_pattern` (177) | substring-checks `"too long"` | `pytest.raises(ModelRetry)` with same substring on `exc.value` message. Rename to `..._raises_modelretry_on_overlong_pattern` |

A new test should be added for the timeout path. Today there's no test
covering the timeout branch (it's hard to construct a fast deterministic
catastrophic-backtracking test, but we can fake it by
monkeypatching `_REGEX_TIMEOUT_S = 0.0` for one test and giving any
non-trivial pattern):

```python
async def test_search_paper_regex_timeout_raises_modelretry(monkeypatch):
    monkeypatch.setattr("andamentum.whetstone.v3.tools._REGEX_TIMEOUT_S", 0.0)
    with pytest.raises(ModelRetry) as exc:
        await search_paper(_ctx(_model()), r"a+b", regex=True)
    assert "timed out" in str(exc.value).lower()
```

The other 14 tests in the file are unaffected (substring mode, character
classes, alternation, word-boundary, snippet padding, section tagging,
empty-query, helper functions).

### 4.6 Per-tool retry budget

The current `_build_agent` in `review.py` sets `retries=2,
output_retries=2`. With Stage 1, `retries=2` becomes the **per-tool**
cap — a tool that raises `ModelRetry` twice in one run will raise
`UnexpectedModelBehavior` on the third bad call.

Today, a model that loops on bad ids 5 times just keeps "succeeding"
with error strings — costing tool-call budget and time but never
crashing. With Stage 1, the third bad call surfaces the loop. **That's
the behaviour we want** — and Stage 3 catches the resulting
`UnexpectedModelBehavior` and degrades gracefully (log + skip
criterion) instead of taking the run down.

Net: same "criterion failed → log and continue" envelope, less wasted
budget on broken tool calls inside it.

### 4.7 Why this is not a breaking change

- Tool surface: internal. No public re-export.
- Test surface: three test names change, one new test. No external
  callers of the tool functions.
- End-to-end: `run_criteria` already catches `Exception`; with Stage 3
  it catches `UnexpectedModelBehavior` more specifically. Either way,
  the user-visible outcome of "model loops on bad ids" is the same: log
  the criterion, move on.

## 5. Stage 2 — Output validator for verbatim quote anchoring

### 5.1 The pattern

Inside `_build_agent` in `whetstone/v3/review.py`, register an
`output_validator` that closes over a small piece of context (the
`DocumentModel` is reachable via `ctx.deps`, so the closure only needs
the `locate` symbol — already imported at module top):

```python
def _build_agent(criterion, agent_model):
    agent = Agent(
        resolve_model(agent_model),
        instructions=_PROMPT,
        output_type=_CriterionFindings,
        deps_type=DocDeps,
        tools=[read_section, search_paper],
        retries=2,
        output_retries=2,
    )

    @agent.output_validator
    async def _validate_quotes_anchor(
        ctx: RunContext[DocDeps], output: _CriterionFindings
    ) -> _CriterionFindings:
        if ctx.partial_output:
            return output  # we don't stream, but defensive
        source = ctx.deps.document_model.source
        anchored: list[_RawFinding] = []
        unanchored_quotes: list[str] = []
        for f in output.findings:
            if locate(f.quote, source) is not None:
                anchored.append(f)
            else:
                unanchored_quotes.append(f.quote)
        if unanchored_quotes and ctx.retry < 2:
            # Up to two re-quote attempts (ctx.retry == 0 and == 1).
            # On the third call (ctx.retry == 2) we accept anchored-only.
            preview = "\n".join(f"  - {q!r}" for q in unanchored_quotes[:5])
            raise ModelRetry(
                f"{len(unanchored_quotes)} quote(s) are not present verbatim "
                f"in the source. Re-quote each from the document exactly "
                f"(copy-paste; do not paraphrase, expand abbreviations, "
                f"or fix line breaks) — or remove the finding. Offending "
                f"quotes:\n{preview}"
            )
        # ctx.retry >= 2 or no bad quotes: keep what anchored.
        return _CriterionFindings(findings=anchored)

    return agent
```

### 5.2 Why `ctx.retry < 2` (two re-quote attempts)

- **First attempt** (`ctx.retry == 0`): model produces output, validator
  finds bad quotes, raises `ModelRetry`. Model gets the message and
  tries again.
- **Second attempt** (`ctx.retry == 1`): validator runs again. If still
  bad quotes, raise `ModelRetry` one more time.
- **Third attempt** (`ctx.retry == 2`): validator returns whatever
  anchored. Caps the wall-clock cost at ≤2 extra round-trips per
  criterion in the worst case.

`output_retries` is bumped from `2` to `3` in `_build_agent` so the
validator's two re-quote attempts don't collide with pydantic-ai's own
structured-output coercion budget. Two retries for our validator + one
reserve for the framework = `output_retries=3`.

### 5.3 `verify_findings` stays

`verify_findings` (line 255-274) keeps its current job: locate every
quote, set the `Span`, drop any that still don't anchor. With Stage 2:

- Happy path: validator passes on first try, `verify_findings` does the
  `Span` enrichment only.
- One-strike path: validator forced a retry, model fixed all quotes,
  `verify_findings` enriches.
- Pathological path: model failed twice, validator returned only
  anchored findings, `verify_findings` is a no-op drop because they
  all anchor.
- Retry-exhausted path (validator raised twice and pydantic-ai's
  `output_retries` is exhausted before the validator's `ctx.retry==0`
  guard returns): `UnexpectedModelBehavior` raised → Stage 3 catches
  it → criterion logged + skipped, exactly as today.

`verify_findings` is the deterministic floor. The validator is an
optional pre-emption layer. They don't conflict.

### 5.4 Per-criterion budgets

No change to `_REQUEST_LIMIT=18` / `_TOOL_CALLS_LIMIT=10` /
`_TOTAL_TOKENS_LIMIT=80_000`. The single validator-driven retry consumes
1-2 extra model requests in the worst case; well under the 18-request
limit.

### 5.5 Tests for Stage 2

Three new unit tests in a new file
`whetstone/v3/tests/test_review_output_validator.py`:

1. `test_validator_passes_when_all_quotes_anchor` — build a
   `DocumentModel` with known source, construct a `_CriterionFindings`
   where every quote is a verbatim slice of the source, pass to the
   validator directly, assert it returns `output` unchanged.

2. `test_validator_raises_modelretry_on_first_bad_quote` — same model,
   include one `_RawFinding` whose quote is not in the source.
   Construct a `RunContext` stub with `retry=0`. Assert
   `pytest.raises(ModelRetry)` with the bad quote substring in the
   message.

3. `test_validator_raises_modelretry_on_second_attempt_too` — same
   setup as #2 but with `retry=1`. Assert `pytest.raises(ModelRetry)`
   — second re-quote attempt still pushes the model.

4. `test_validator_returns_anchored_only_on_third_attempt` — same
   setup as #2 but with `retry=2`. Assert no raise; returned
   `_CriterionFindings.findings` is the anchored subset.

These tests construct the validator function directly (it's a closure
inside `_build_agent` — Stage 2 will pull it out into a module-level
helper `_make_quote_validator(model: DocumentModel) -> Callable[...]`
or, simpler, the validator can be a module-level function that reads
`ctx.deps.document_model` and is registered inside `_build_agent`). The
module-level form is more testable.

**Design choice**: lift the validator to a module-level coroutine
`async def _validate_quotes_anchor(ctx, output)` and register it inside
`_build_agent` with `agent.output_validator(_validate_quotes_anchor)`.
This matches `core/agents.py:180` style.

### 5.6 Tests that may need updating

Three existing tests touch the agent shape:

- `test_review.py` — patches `_build_agent`. Already returns a stub
  agent whose `.run()` is mocked. The stub doesn't go through the
  validator; the validator is registered on the real agent inside the
  real `_build_agent`. So `test_review.py` tests don't exercise the
  validator at all — they don't need to change.
- `test_graph.py` — same shape. Unchanged.
- `test_tools.py` — covered in Stage 1.

No existing test asserts on "this bad quote gets dropped silently", so
the silent-drop → in-loop-retry transition doesn't break any existing
test expectation.

### 5.7 Why this is not a breaking change

- Public API: `review_document`, `review_criterion`, `run_criteria`,
  `Finding` — all unchanged.
- Renderers: consume `Finding`s — unchanged shape.
- Failure floor: bad quotes still get dropped after retry-exhaustion,
  exactly as today. Only the happy path improves.
- Wall-clock: at most one extra model round-trip per criterion with bad
  quotes (≤ 1.5x the latency on those criteria; unchanged on clean
  ones). User has accepted longer runs in exchange for richer output
  before — this is the same trade.

## 6. Stage 3 — Typed exception handling in `run_criteria`

### 6.1 The change

`whetstone/v3/review.py:run_criteria` (line 219-252) — replace:

```python
except Exception as exc:
    logger.warning("[v3.review] %s crashed: %s", c.name, exc)
    continue
```

with:

```python
except UnexpectedModelBehavior as exc:
    body = getattr(exc, "body", None)
    body_part = f" — body: {str(body)[:500]!r}" if body else ""
    logger.warning(
        "[v3.review] %s: model behaviour error (%s)%s",
        c.name, exc, body_part,
    )
    continue
except UsageLimitExceeded as exc:
    logger.warning("[v3.review] %s: usage limit hit (%s)", c.name, exc)
    continue
except Exception as exc:
    logger.warning("[v3.review] %s crashed: %s", c.name, exc)
    continue
```

### 6.2 Imports

`from pydantic_ai.exceptions import UnexpectedModelBehavior,
UsageLimitExceeded` at the top of `review.py`.

### 6.3 What this catches

| Exception | Source | What we learn from the log |
|---|---|---|
| `UnexpectedModelBehavior` | Pydantic-AI itself when (a) per-tool retries exhausted ("Tool 'X' exceeded max retries"), (b) output validator retries exhausted, (c) content filter, (d) `IncompleteToolCall`, (e) provider HTTP errors that propagate as unexpected behaviour | The exception body (when the provider sets it). For the Ollama null-content bug this captures the provider's error payload that today is lost. |
| `UsageLimitExceeded` | Our `UsageLimits(request_limit=18, tool_calls_limit=10, total_tokens_limit=80_000)` | Which limit fired |
| `Exception` (last resort) | Everything else (network errors, ValueError, asyncio cancellation, ...) | Same as today — defensive |

### 6.4 Why this is not a breaking change

It's strictly additive — every previously-caught exception is still
caught by the trailing `except Exception`. Only the log format for two
specific exception classes gains diagnostic detail. The
"failed criterion → log + continue" envelope is identical.

### 6.5 Tests for Stage 3

One new unit test in `whetstone/v3/tests/test_review.py`:

`test_run_criteria_logs_unexpected_model_behaviour_body` — mock
`review_criterion` to raise
`UnexpectedModelBehavior("test failure", body="upstream payload")`,
call `run_criteria` with a one-criterion list, assert (via `caplog`)
that the log line contains `"upstream payload"`.

The existing
`test_run_criteria_skips_failed_criteria_and_returns_remainder`
already covers the "log + continue on Exception" path — unchanged.

## 7. End-to-end behaviour, before vs after

Concrete scenarios drawn from the three smoke runs:

| Scenario | Today | Stage 1+2+3 |
|---|---|---|
| Model produces a finding with a paraphrased quote | Finding silently dropped by `verify_findings` after run | Model asked once to re-quote; if it succeeds, finding kept; if it fails twice, finding silently dropped — same as today |
| Model calls `read_section("999")` (unknown id) | Sees `"no section with id '999'; check the SECTIONS block..."` as tool result. Pays a tool-call slot. May or may not correct | Sees same text via `RetryPromptPart`. Per-tool counter advances. If model keeps doing it: `UnexpectedModelBehavior` after 3 attempts → Stage 3 catches → criterion logged with `"Tool 'read_section' exceeded max retries count of 2"` → next criterion runs |
| Ollama returns HTTP 400 (null content bug) mid-run | `Exception` caught with generic message `"<criterion> crashed: <opaque error>"` | `UnexpectedModelBehavior` caught with `body` attribute — log shows the Ollama payload that today is lost |
| `UsageLimits.request_limit` hit (model loops on tool calls) | Surfaces as `UsageLimitExceeded` → currently caught by generic `Exception` with no diagnostic | Caught explicitly with `"usage limit hit (<which limit>)"` log |
| Model produces 0 findings | Run completes, criterion contributes nothing | Same — out of scope for this PID |

The single user-visible behaviour change is **fewer findings get
silently dropped**. That's the goal.

## 8. Implementation order

Land in this order, each as its own commit:

1. **Stage 3 first** (most defensive, smallest blast radius). It changes
   only the log format and adds two imports. If anything goes wrong it
   degrades to today's behaviour via the trailing `except Exception`.
   Ship: 1 file edit + 1 test.

2. **Stage 1 second** (tools). Rename three tests, add one. The agent
   prompt's mention of "tools" is unchanged (it already says "the agent
   reads the error the same way it reads any other tool result and
   corrects on its next turn" — that's still true; the surface just
   changed). Ship: 1 file edit + 4 test diffs.

3. **Stage 2 third** (output validator). Builds on Stage 3's catch (so
   that retry exhaustion degrades gracefully if it ever fires). Lift
   validator to module-level helper. Ship: 1 file edit + 1 new test
   file with 3 tests.

Each commit must keep the canonical-green invariants:

- `uv run pyright` — 23 pre-existing test-only errors, no new ones.
- `uv run ruff check` — clean.
- `uv run pytest src/andamentum/whetstone` — all 567 whetstone tests
  pass (53 v3 + 514 v2/legacy/docx).

Skip the full 2075-test sweep per commit (slow, and these changes are
local to whetstone/v3); run it once at the end of the series.

## 9. Verification — what "done" looks like

1. **Tests green** for the three suites above on every commit.
2. **One smoke re-run** at the end of the series against
   `benchmarks/whetstone/corpus/arxiv_1412.6980_v1.md` with
   `ollama:gemma4:31b-nvfp4` (the slowest of the three canonical local
   models, and the one with most findings to validate against). Compare:
   - findings count vs. `smoke_gemma31.md` (expect ≥ today; possibly +1
     or +2 from validator-recovered quotes)
   - tool-call-log lines (expect ModelRetry-driven retries to be
     visible as `"[v3.tool] ... → no such section"` lines followed by
     the same model attempting a valid id)
   - log lines for any criterion crash (expect richer body / typed
     error class)
3. **Re-run the two-model split** (`smoke_gemma26`, `smoke_gptoss`) only
   if the gemma31 result is clean — the goal is to confirm the unified
   code path still works across model tiers, not to re-benchmark.
4. **Open question to revisit** post-merge: is `output_retries=2` still
   the right ceiling now that the validator burns one retry on bad
   quotes? Today it's `2` for pydantic-ai's own structured-output
   coercion; the validator uses up to `1`. If we see retry exhaustion
   in practice we'd bump to `3`. Don't pre-emptively change it.

## 10. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ModelRetry inside a tool consumes the per-tool budget faster than expected; some real criteria crash from "Tool exceeded retries" where today they wouldn't | Medium on weak models | Criterion lost (logged + skipped) | Stage 3 catches it gracefully; we keep `retries=2` (3 attempts per tool per run); the per-tool counter resets between criteria; observable in logs so we can tune |
| Output validator triggers excessive retries on a noisy paper where many findings have edge-case quotes | Low-medium | One extra round-trip per criterion (≤ 18-request budget) | `ctx.retry == 0` guard caps validator retries at 1 |
| `ctx.retry` semantics differ from what I read in docs (e.g. tools and validator share a counter — they don't, but worth verifying empirically) | Low | Validator might fail-open instead of asking for fix | First smoke run will surface this; behaviour matches today's "silent drop" floor either way |
| Lifting the validator to a module-level helper subtly changes the test infrastructure expectations | Low | Test churn | Mirror the `core/agents.py:180` pattern exactly |
| Streaming partial-output gotcha (validator fires on partial outputs) | None — we use `agent.run`, not `agent.run_stream` | n/a | Defensive `if ctx.partial_output: return output` guard anyway, costs one line |

## 11. Decisions (resolved 2026-05-24)

1. **Validator retry policy**: two re-quote attempts then accept
   (`ctx.retry < 2`). `output_retries` bumped from 2 → 3 in
   `_build_agent` to make room for the two validator retries plus one
   reserve for structured-output coercion.
2. **Bad-quote preview**: full quotes (no per-quote truncation), capped
   at the first 5 bad quotes in any one retry prompt.
3. **Log body cap**: first 500 chars of `exc.body` in Stage 3's
   `UnexpectedModelBehavior` log line.
4. **Commit strategy**: three separate commits directly to `main`,
   landed in order Stage 3 → Stage 1 → Stage 2, each passing
   canonical-green before the next. Mirrors the layer-1-tools PID
   workflow.

---

Proceeding: Stage 3 → Stage 1 → Stage 2.
