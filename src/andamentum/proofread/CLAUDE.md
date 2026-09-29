# andamentum.proofread — notes for Claude

Moved from the root `CLAUDE.md`, word for word. Claude Code loads this file when it reads files in this directory.

## Overview

- `andamentum.proofread` — deterministic, dependency-light proofreading. Single sync API: `analyze(text: str) -> ProofreadResult`. Combines five readability scores (SMOG, Flesch–Kincaid grade, Flesch reading ease, Gunning Fog, Coleman–Liau, ARI) with classic style checks (weasel words à la Matt Might, passive voice via be-verb + past-participle heuristic, duplicate words, weak sentence openers, -ly adverb density). No LLM, no I/O, pure function. Designed to complement `whetstone`'s LLM-driven review, not replace it. Findings carry char offsets and `sentence_index` so callers can map back into the source. The CLI (`andamentum-proofread`) accepts URLs and document files via `harvest.extract` — the only point where the module touches another sub-module.

## Layering

- `proofread` library depends only on `pydantic` and `textstat`. The optional `proofread.cli` submodule is the ONLY place where `harvest` is imported — so the CLI can accept PDF / HTML / DOCX / PPTX / Markdown sources — and the rest of `proofread` MUST NOT touch `harvest` or any other andamentum sub-module. Other modules MAY depend on `proofread` (e.g. `whetstone` to attach deterministic readability scores to a review, `scribe` for pre-render style flags).
