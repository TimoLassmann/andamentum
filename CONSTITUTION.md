# Coding Constitution

Rules for writing Python. Reference from any repository's `CLAUDE.md`.

These rules bias toward caution over speed. For trivial one-liners, use judgment.

---

## Process

How to approach a task. Most coding mistakes happen here, not in the code itself.

### 1. Simplicity first
Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No configurability that wasn't requested.
- No error handling for impossible scenarios.

If you wrote 200 lines and it could be 50, rewrite it. Ask: would a senior engineer call this overcomplicated?

### 2. Think before coding
Before writing anything:

- State assumptions. If uncertain, ask.
- If multiple interpretations exist, list them. Don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 3. Surgical changes
Touch only what you must.

- Before writing new code, check whether similar code already exists. Grep for the pattern. Read the relevant module. Don't reinvent a utility that lives three directories away.
- Don't improve adjacent code, comments, or formatting.
- Don't refactor what isn't broken.
- Match existing style even if you'd write it differently.
- If you notice unrelated dead code, mention it. Don't delete it.

When your changes create orphans (imports, variables, functions made dead by your edit), remove them. Don't remove pre-existing dead code unless asked.

The test: every changed line traces back to the request.

### 4. Goal-driven execution
Transform tasks into verifiable goals.

- "Add validation" becomes "Write tests for invalid inputs. Make them pass."
- "Fix the bug" becomes "Write a test that reproduces it. Make it pass."
- "Refactor X" becomes "Tests pass before. Tests pass after."

For multi-step work, state the plan:

```
1. [step] → verify: [check]
2. [step] → verify: [check]
3. [step] → verify: [check]
```

Strong success criteria let the work loop run without constant clarification.

### 5. Don't fabricate
When you don't know whether a function exists, what a signature looks like, or how an API behaves, check. Grep the codebase. Read the docs. Run a probe. Ask.

"Pretty sure this library has this method" is not a green light. It's a signal to verify.

---

## Principles

What good code looks like.

### 1. Structured over ad-hoc
Dataclasses, Pydantic models, or enums at system boundaries. No raw dicts passed between modules.

### 2. Fail fast, fail loud
Errors surface immediately. Silent failures are bugs. Swallowed exceptions are crimes.

```python
# YES
if not path.exists():
    raise FileNotFoundError(f"Document not found: {path}")

# NO
if not path.exists():
    return None  # caller has no idea why
```

### 3. Single source of truth
One authoritative location per concept. Duplicating constants, schemas, or logic across files is a bug.

### 4. Observable
Log what the code did and why. When something breaks, the logs should tell you what happened. If the only way to debug is to add print statements and re-run, the logging was inadequate.

### 5. Dependencies flow one way
Imports go in one direction. If `A` imports `B`, `B` must not import `A`. Circular imports are a design bug, not a Python quirk to work around.

Core utilities (pure functions, simple types, data structures) stay free of heavy framework imports like web frameworks, ORMs, LLM clients, or UI libraries. This keeps them reusable anywhere in the project.

---

## Rules

Hard rules. Non-negotiable.

### 1. Test before claim
Never claim something works without evidence.

1. Write the code.
2. Run the tests.
3. Run the type checker.
4. Manually verify.
5. Then claim it works.

"I think it should work" is not evidence. Command output is evidence.

### 2. No placeholder code
No code returning fake data, hardcoded results, or TODOs presented as working features. Either done or not started.

### 3. No zombie code
Code is either in active use, explicitly marked as experimental (a dedicated folder like `experiments/`, or a module docstring saying so), or deleted. Commented-out blocks, "just in case" branches, and utilities kept around for someday all count as zombies. Delete them. Git remembers.

### 4. Deterministic and intelligent code are different
Deterministic code handles precision: validation, enum enforcement, exact-match queries, computed fields.

Intelligent code (LLM calls) handles fuzziness: natural language, extraction from unstructured text, composing responses.

Never use an LLM for what deterministic code can do. Never use deterministic code for what needs language understanding.

### 5. No silent fallbacks
If a required service (database, model, API) is unavailable, raise. Don't silently degrade to a simpler mode. The user must know when something is broken so they can fix it.

The only acceptable soft failure: the service was reachable but returned bad data for one item after retries. Log a warning, use defaults for that item, continue.

### 6. Don't silently drop data
If you computed a result, return it whole. Don't slice lists (`results[:5]` when the caller asked for results). Don't truncate strings with an ellipsis. Don't return a summary instead of the full output.

When a real constraint forces truncation (UI space, API payload limit), make it visible: return the total count alongside the slice, flag the truncation in the response, or expose a way to fetch the rest.

---

## Done

Before claiming work complete:

1. Automated checks pass (tests, type checker, linter).
2. Manual verification done (ran the feature, tested error cases).
3. Documentation updated where the change affects it.
4. Evidence saved (command output).

Only then claim done.

When code falls short of these rules, fix the code. Don't lower the standard.

---

## How to use this

Put this file at the root of any Python repository as `CONSTITUTION.md`. In `CLAUDE.md`:

```
Follow the rules in CONSTITUTION.md. Project-specific rules below override
or extend them where explicitly stated.
```

Keep project-specific rules in `CLAUDE.md`. Don't repeat the constitution there.
