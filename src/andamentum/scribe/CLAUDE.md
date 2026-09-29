# andamentum.scribe — notes for Claude

Moved from the root `CLAUDE.md`, word for word. Claude Code loads this file when it reads files in this directory.

## Overview

- `andamentum.scribe` — structured document drafting: block-based authoring (paragraph, heading, figure, table), section abstraction, built-in `article` scaffold (no grant scaffold — funder AI-disclosure rules are tightening and we don't want to make AI-assisted grant drafting more frictionless), SQLite-backed source of truth, one-way render to `.docx`. Replaces the standalone `document-tools:doc-draft` plugin.

## Layering

- `scribe` depends only on `typeset` (for HTML/PDF rendering) and stdlib `sqlite3`. MUST NOT depend on `epistemic`, `deep_research`, `document_store`, `whetstone`, `figures`, or `core`.

## Design

**Scribe module** (`andamentum.scribe`) — block-based document authoring. Documents live in SQLite at `~/.local/share/scribe/<name>.db` (override with `SCRIBE_DIR`). Public entry point: `Document.create(title=..., database=..., scaffold="article" | None)`; mutate with `append`/`replace`/`replace_section`; render with `render(path, format="docx")`. Section operations (`list_sections`, `section`, `replace_section`) are derived from heading blocks — there is no separate sections table. Each block has an integer revision counter; `replace()` enforces optimistic locking via `BEGIN IMMEDIATE` and writes an audit row to `scribe_revisions`. Citations are Pandoc-flavoured `[@key]` spans extracted by regex; references live in their own table; `[verify]` and `[citation needed]` markers are recognised and reported by `validate()`. Inline markdown (bold/italic/code) renders as styled runs in `.docx`. `.docx` is the only format wired into `render()`; an internal block→atom mapping toward typeset HTML/PDF lives in `render_typeset.py` but is not yet exposed through the public API. Scribe replaces the standalone `document-tools:doc-draft` plugin for Word file authoring; `.pptx` stays out of scope.
