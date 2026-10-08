# CLAUDE.md — `andamentum.agentic_dialect`

Operating notes for working **in** this module. The module *is* the house style for building
agentic graph systems; treat changes to it with the same care as changes to a language spec.

## Two facets, one truth

- **`DIALECT.md`** is authoritative for **prose** — rationale, examples, the narrative.
- **`_laws.py` / `_roles.py`** are authoritative for the **enforceable surface** — law ids,
  one-line statements, tiers, role→law slices, checklist items.
- **`tests/test_agentic_dialect_drift.py` binds them.** Every law id has a tagged section in the doc;
  every `Law.statement` appears in the doc verbatim (modulo whitespace/emphasis); the checklist
  items and the skeleton are present. Change one facet without the other and the drift test
  fails — by design.

## Changing a law

1. Edit the statement in `_laws.py` **and** the matching `**Lx — Name.**` line in `DIALECT.md`
   so the statement is a verbatim substring (the drift test normalizes `*` / backticks /
   whitespace, nothing else).
2. Re-run `uv run pytest src/andamentum/agentic_dialect/tests`.
3. Don't add or retire a law lightly. The dialect was scoped deliberately (see the
   out-of-scope list in `DIALECT.md`); it is **not** a superset of every system's rules, and
   that is intentional.

## File map

`_laws.py` laws · `_roles.py` role briefs + `for_role` · `checks.py` portable AST gates +
`check_code` · `doc.py` canon access + `skeleton`/`normalize` · `cli.py` adapter (exit 0 / 1 `check` found violations / 2 unknown law or role / 3 path unreadable; `--json` on `check` and the listings) · `errors.py` `DialectError`, `UnknownLawError`, `UnknownRoleError` (also `KeyError`s) ·
`__init__.py` public API. (`_laws`/`_roles` are underscore-private to avoid colliding with the
`laws()`/`roles()` functions.)

## Hard constraints

- **Leaf + extraction-ready:** `pydantic` + stdlib only; import no other andamentum sub-module;
  relative imports only. The subtree must stay liftable into a standalone package.
- **`check_code` gates are conservative** (they gate builds — a false positive is worse than a
  miss). Add a gate as a small AST check with a test fixture; don't pull in a lint framework.
- The rendered HTML (`docs/agentic-systems/agentic-dialect.html`) regenerates from `DIALECT.md` via
  `andamentum-typeset`; the visual one-pager (`agentic-dialect-reference.html`) is hand-authored —
  keep it to law statements + glyphs so it can't drift into a second spec.

## From the root CLAUDE.md

Moved from the root `CLAUDE.md`, word for word. Claude Code loads this file when it reads files in this directory.

### Overview

- `andamentum.agentic_dialect` — the house style for building agentic graph systems (pydantic-graph + pydantic-ai), as data + portable conformance checks. The prose canon is `src/andamentum/agentic_dialect/DIALECT.md` (3 surfaces / 2 roles / 1 agent / 9 laws); the structured laws, role briefs (`for_role`), checklist, and skeleton are the enforceable kernel, drift-tested against the doc. `check_code(path)` runs the portable static gates (no engine import in workers, no `dict[str, Any]`, no nondeterminism/client/literal-bound in node bodies). A leaf module: depends only on `pydantic` + stdlib, imports no other sub-module, relative imports only — kept independently extractable.

### Layering

- `agentic_dialect` depends only on `pydantic` + stdlib. MUST NOT import any other andamentum sub-module, and uses only relative imports internally, so it stays independently extractable (a later spin-out to a standalone package is a `git filter-repo`). Other modules MAY depend on `agentic_dialect`. It is governance/knowledge, not a runtime pipeline — nothing in the data pipelines imports it.
