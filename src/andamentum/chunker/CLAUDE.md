# andamentum.chunker — notes for Claude

## Overview

`andamentum.chunker` splits markdown into chunks. It is deterministic and uses no model: same input, same chunks. Public surface: `Chunk`, `chunk` and `ChunkerError` (a `ValueError` subclass; `__all__`), nothing else.

```python
chunk(text: str, *, max_chars: int) -> list[Chunk]
Chunk(text, start, end, heading_path)   # text == source[start:end]
```

## Layering

Standard library only. `chunker` MUST NOT import `core` or any other sub-module; a test pins this. Other modules MAY depend on it. Whetstone uses the structural layer (`structural.py`: `Heading`, `find_headings`, `build_section_tree`, `split_section_recursively`) for its sectionizer, so keep those signatures stable.

## Algorithm (`splitter.py`)

1. Split at ATX headings. A section runs from its heading to the next heading of any level. `heading_path` is the stack of enclosing titles (a new heading pops deeper-or-equal levels; closing `#`s are stripped). Text before the first heading has path `()`.
2. A section of at most `max_chars` is one chunk. Otherwise pack consecutive paragraphs (blank-line separated) greedily.
3. A paragraph over `max_chars` is split at sentence ends (`(?<=[.!?])\s+(?=[A-Z(\["'])`); a sentence over `max_chars` is cut at the last whitespace; a token with no whitespace is cut hard.
4. Atomic blocks are never cut: fenced code (closed only by the same character, at least as long as the opener; an unclosed fence runs to the end of the document), tables (runs of lines starting with `|`) and YAML front matter. An atomic block over `max_chars` is its own oversize chunk.
5. A heading never stands alone: it joins the next block. A heading-only section followed by another section is carried into that section's first chunk, whose path is the later section's path. Only a heading at the very end of the document stands alone.

## Invariants (tested on seeded random markdown in `tests/test_chunk.py`)

Chunks are in source order and do not overlap; `text == source[start:end]`; chunk text has no leading or trailing blank lines; every non-whitespace character lies in exactly one chunk; no chunk exceeds `max_chars` unless it holds an atomic block or an unbreakable token. Empty or whitespace-only input gives `[]`; `max_chars < 1` raises `ChunkerError`.

## Pitfalls

- `find_headings` skips headings inside fenced code (same fence rule as `chunk`). It does not know about front matter; `chunk` filters those itself.
- Setext headings (underlines) are not recognised; harvest output is ATX.
- `max_chars` is in characters, not tokens.

## CLI

`andamentum-chunker INPUT --max-chars N [-o FILE] [--text-only] [--version]`. `INPUT` is a path or `-` for stdin; `--max-chars` is required. Exit codes: 0 success, 2 usage error, 3 input unreadable (missing file, not UTF-8). Errors are one `error:` line on stderr. JSON output: `{"chunks": [{text, start, end, heading_path}], "total_chars": n}`.
