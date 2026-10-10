# andamentum.docx_edit — notes for Claude

Tracked-change editing of a .docx. One implementation, shared by `whetstone` (review
annotation, footnotes, the review report) and `compose` (the Word writer and reader).

## Layering

- A leaf. It imports no other andamentum module except `andamentum.core.text_match`
  (a shared utility). `tests/test_leaf.py` enforces this, and that nothing still
  imports the old `whetstone.docx.atoms / surgery / textmap / xml_builder` paths.
- Whetstone and compose import it. It never imports either.

## What lives where

- `atoms.py` — the one character stream of a paragraph (`paragraph_atoms`,
  `document_atoms`, `projection`, `Atom`, `AtomKind`, `EXCLUDED_SUBTREES`).
- `body.py` — which `<w:p>` elements are body paragraphs (`body_paragraphs`),
  the main document part (`main_document_part`), the review-report bookmark prefix.
- `textmap.py` — `TextMap` and `AtomSpan`: offsets into the projection.
- `surgery.py` — `apply_edits` and the run-level primitives: tracked insertions and
  deletions, `tracked_paragraph`, `next_change_id`, `split_at_boundaries`,
  `run_bounds`, `translate_span`. `Edit` is the span edit (whetstone's review
  schema also has an `Edit`; compose imports this one as `SpanEdit`).
- `xml_builder.py` — `XMLElementBuilder` for comment markers and `w:comment` elements.

## Behaviour is pinned elsewhere

The characterisation tests in `src/andamentum/tests/docx_characterisation/` pin what
both engines do on the synthetic corpus in `src/andamentum/tests/docx_corpus/`.
A change here must keep those tests green, or change them deliberately in the same commit.
