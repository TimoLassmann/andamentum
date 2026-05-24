# PID — string-match consolidation + whetstone v3 release-blockers

*Status: approved 2026-05-24, in execution.*

## Why

The whetstone v3 release audit surfaced five distinct string-matching /
text-anchoring implementations across the codebase, with materially
divergent normalization rules. The user wants ONE unified approach to
"given a target string and a source document, find the target as a
verbatim span in the source." Consolidating is release-blocking for v3
because:

1. **Correctness bug**: a whetstone-v2 node and a whetstone-v3 node
   disagree on whether the same LLM-emitted quote was hallucinated
   (v3 strips markdown via path B; v2 routes through chunker's
   `find_anchor` (path C) which doesn't).
2. **Robustness gap**: smart quotes (`"` `'` `'` `"`) are only stripped
   by epistemic's matcher (E). LLMs frequently emit curly quotes; v3's
   hallucination gate silently rejects them.
3. **Dead code**: `whetstone/docx/text_processor.py` (~700 LOC) is
   labelled "Unified text processing utilities" but isn't used by any
   of the actual matchers — it's dressed-up dead weight.

Plus two whetstone-v3 release-blockers stack on top:

- **Lock-and-refine validator** (audit's #2): today's validator asks
  the model to regenerate the entire output when one quote misses.
  Smoke logs show the model can reintroduce errors in previously-good
  quotes during retry. The fix is to lock anchored findings across
  retries and ask the model to refine only the unanchored ones.
- **Non-academic criterion sets** (audit's #1): `criterion_set_for()`
  only knows `"academic"`; everything else silently falls back to
  SPECS. Need real alternates for `essay`, `memo`, `generic`.

## Scope

**In scope (this PID):**

1. New module `andamentum/core/text_match.py` — single canonical
   `find_span` + `normalize_for_match` + `Match` named-tuple.
2. Migrate all five matchers to use it:
   - **A** `whetstone/docx/anchor.py:normalize_with_map` (keep DocIndex
     for its docx-specific paragraph-boundary handling; share the
     char-folding rules)
   - **B** `whetstone/v3/locate.py:locate` (thin shim)
   - **C** `chunker/validation.py:find_anchor` (preserves case-sensitive
     exact tier; default fuzzy=rapidfuzz at 0.85)
   - **E** `epistemic/passage_extraction.py:_find_pointer_in_chunks`
     (explicit `fuzzy_threshold=0.30` to preserve its loose contract)
3. Delete **D** `whetstone/docx/text_processor.py` — replace its single
   live caller in `patch_editor.py:265` with `find_span(..., fuzzy="sequence")`
   + a small suggestion formatter.
4. Lock-and-refine validator (`review.py` + `gaps.py`) using new helper.
5. Three non-academic criterion sets (`essay`, `memo`, `generic`) +
   routing fix in `criterion_set_for()`.

**Out of scope:**

- URL/ID normalizers (`dedupe_evidence.py:normalize_source_ref`,
  `deep_research/verification.py:normalize_url`). Different concern.
- Refactoring `DocIndex` past sharing its char-folding (the
  cross-paragraph synthetic-separator behaviour is genuinely
  docx-specific).
- Adding new fuzzy backends beyond `rapidfuzz` / `difflib`.

## Unified API (the contract)

```python
# andamentum/core/text_match.py

from typing import Literal, NamedTuple

class Match(NamedTuple):
    start: int                                     # source offset
    end: int                                       # source offset (exclusive)
    method: Literal["exact", "normalized", "fuzzy"]
    score: float                                   # 1.0 for exact/normalized

def normalize_for_match(
    text: str,
    *,
    strip_markdown: bool = True,
    strip_quotes: bool = True,
    fold_case: bool = True,
    collapse_whitespace: bool = True,
) -> tuple[str, list[int]]:
    """Return (normalized, index_map) where idx_map[i] == original offset
    of normalized[i]. All toggles default-on for the canonical contract."""

def find_span(
    target: str,
    source: str,
    *,
    within: tuple[int, int] | None = None,
    fuzzy: Literal["off", "rapidfuzz", "sequence"] = "off",
    fuzzy_threshold: float = 0.85,
    strip_markdown: bool = True,
    strip_quotes: bool = True,
    fold_case: bool = True,
    collapse_whitespace: bool = True,
) -> Match | None:
    """Find target in source. Tries exact-in-normalized-space first; if
    fuzzy != "off" and no exact match, falls back to fuzzy with the
    specified backend and threshold."""
```

**Canonical defaults** (markdown-strip + smart-quote-strip + case-fold +
ws-collapse, fuzzy off) become the answer to "is this verbatim in the
source." Every divergence from canonical must be explicit at the
callsite.

## Per-caller migration

| Path | Callsite | Migration |
|---|---|---|
| A | `docx/anchor.py:normalize_with_map` | Move char-folding into `normalize_for_match`; `DocIndex` keeps its paragraph-boundary indexing but shares the folding rules. `closest()` → `find_span(..., fuzzy="sequence")`. |
| B | `v3/locate.py:locate` | One-liner: `find_span(quote, source, within=within)` and return `(start, end)` if `Match`. Gains smart-quote stripping (improvement). |
| C | `chunker/validation.py:find_anchor` | Thin shim returning the existing `AnchorMatch` dataclass. `fold_case=False` at exact tier to preserve byte-identical contract. `fuzzy="rapidfuzz", fuzzy_threshold=0.85`. Gains markdown + smart-quote stripping at whitespace/fuzzy tiers (bench-pin Strunk lens nodes first). |
| D | `docx/text_processor.py` | Delete the whole module. Replace `patch_editor.py:265` with `find_span(..., fuzzy="sequence")` + ~20 line suggestion formatter. |
| E | `epistemic/passage_extraction.py:_find_pointer_in_chunks` | `find_span(pointer, chunk, fuzzy="sequence", fuzzy_threshold=0.30)`. Explicit threshold. |

## Risks (from the audit)

| Risk | Mitigation |
|---|---|
| Strunk lens nodes pass raw markdown both sides; switching to default-on markdown-strip might affect their output | Run the relevant tests before flipping; only flip C's default markdown-strip behaviour for the FUZZY tier, keep exact tier verbatim-only by passing `strip_markdown=False` at exact |
| Chunker's case-sensitive exact tier is load-bearing (byte-identical FTS5 contract) | Pass `fold_case=False` to the exact tier inside `find_anchor`; preserve unchanged |
| E's 0.30 threshold is unusual and must stay explicit | Don't make 0.30 a default anywhere; require explicit kwarg |
| Cross-paragraph synthetic separator in `DocIndex` is docx-specific | Stays as a `DocIndex` method; not hoisted into flat `find_span` |
| 5+ migrations in one branch — large blast radius | One commit per migration; full project test suite green between commits |

## Commit order

1. Write this PID
2. `core/text_match.py` + tests (foundational; no callers yet)
3. Migrate B (locate.py) — smallest blast radius, validates the API
4. Migrate A (docx/anchor.py) — second-smallest, DocIndex preserved
5. Migrate C (chunker/validation.py) — most behaviour change risk; bench-pin Strunk lens first
6. Migrate E (epistemic/passage_extraction.py)
7. Delete D (text_processor.py) + replace patch_editor caller
8. Issue 2 — lock-and-refine validator on top of unified helper
9. Issue 1 — non-academic criterion sets (independent)

Each commit: full project pytest green + pyright canonical-green +
ruff clean.

## Verification

1. Full project test suite passes throughout (currently 2075 tests).
2. Pyright canonical-green (23 pre-existing test-only errors).
3. Smoke re-run of arxiv_1412.6980_v1.md with both `openai:gpt-5.4-mini`
   and `ollama:gpt-oss:20b` — finding counts and severity distribution
   in the same ballpark as the post-Phase-3 smokes (i.e. no quality
   regression).
4. Spot-check: feed the system a non-academic essay (≤1000 words) with
   `--document-type essay` and confirm the essay criterion set runs
   instead of forcing SPECS on it.
