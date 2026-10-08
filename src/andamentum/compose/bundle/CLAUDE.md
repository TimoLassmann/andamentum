# andamentum.compose.bundle — notes for Claude

Claude Code loads this file when it reads files in this directory.

## Overview

- `andamentum.compose.bundle` — a document as a plain folder bundle: one markdown file per section with YAML frontmatter, derived renders to `.docx` / `.md` / `.html` / `.txt`. No LLM, no database. Built-in `article` scaffold only (no grant scaffold; the compose `grant` genre is a separate, generic thing that does not use a scaffold).

## Layering

- `compose.bundle` depends only on `typeset` (HTML output), `python-docx`, `pyyaml` and the standard library. Nothing under `bundle/` may import `pydantic_ai`, `pydantic_graph`, `andamentum.core`, `andamentum.whetstone`, `andamentum.llm_judge`, or any compose module outside `bundle/`; `compose/tests/test_bundle_boundary.py` enforces this with `ast`. It also MUST NOT depend on `epistemic`, `deep_research`, `document_store` or `figures`. `figures.bundle_glue` is the ONLY place in `figures` that imports it. The rest of compose imports `compose.bundle`; the bundle never imports the rest of compose.
- The bundle↔whetstone contract is tested from the compose side (`compose/tests/test_bundle_whetstone_roundtrip.py`) so the bundle stays free of a whetstone import.

## Design

**Bundle layout.** `Bundle.create(path, title=..., scaffold=None)` makes:

```
<bundle>/bundle.yaml        title, created, format_version: 1, metadata (free dict)
<bundle>/sections/sNN.md    frontmatter (id, title, order, revision, status) + markdown body
<bundle>/figures/           images referenced by bodies
<bundle>/references.yaml    list of {key, text, url?}
<bundle>/out/               render outputs, written only when asked
```

It refuses an existing non-empty directory. `Bundle.open` refuses a directory with no `bundle.yaml` or a `format_version` other than 1 (hard switch from the old SQLite store; no migration code exists).

**Disk is the state.** `Bundle` holds only the path, the title and the metadata; `sections()` / `section()` read the files every call. Every write goes through `_atomic_write` (temp file in the same directory, `os.replace`), and identical input yields identical bytes, so a re-run never churns the files. `content_hash` is computed from the body on read, never stored.

**Revisions.** `write_section`, `rename_section` and `add_figure` take `expected_revision`; a mismatch raises `StaleRevisionError` and writes nothing; success bumps `revision` by one. Ids are `sNN` = highest existing number + 1, so a removed id is reused only if it was the highest. Orders are spaced by 10; `add_section(after=...)` takes the midpoint of the gap, and respaces every section when there is no gap.

**Body markdown subset** (`markdown.py`): paragraphs, `- ` bullets, `1. ` numbered lists, pipe tables (first row is the header), whole-line images `![caption](path){#fig:label}`, blank-line separation; inline bold / italic / code via `parser.inline_runs`. Everything else is a plain paragraph. A `#` heading line in a body is a validation **error** and is never rendered as a heading, because the section title is the only heading a section has.

**Rendering.** `Bundle.render(out, format=..., text_filter=...)`; `text_filter` is applied to each section body before parsing (compose uses it to turn evidence markers into citations). `image_root=` sets the folder bundle-relative image paths resolve against (default: the bundle itself); pass it when rendering a copy of a bundle so md / html links point at the real figures. The docx renderer stamps the core-properties keyword `andamentum:contains-ai-markers` when a rendered body carries an `[ai-drafted]` / `[ai-edited]` marker (`parser.find_ai_markers`).
- docx: the title is Word `Title`, each section title is exactly `Heading 1`, then `List Bullet` / `List Number`, `Table Grid`, `Caption`. A `References` Heading 1 closes the document when `references.yaml` is non-empty.
- md: `# title`, `## section`, body; image paths are rewritten relative to the output file. html: through `typeset` (`render_typeset.py`). txt: plain lines, no references, inline markup stripped.

**Pitfall: whetstone bands sections by size.** whetstone sectionises a rendered `.docx` at its heading styles, but only splits a parent that exceeds its 6000-character band. Measured on synthetic bundles: a document under the band is ONE whetstone section named after the document title; a longer one yields a leading piece named after the title, then one piece per `Heading 1` section that fits the band. Anything that maps whetstone sections back to bundle sections by title has to handle both cases.

**Validation** (`Bundle.validate`): duplicate section titles (error), heading lines in a body (error), image files missing (error), `[@key]` citations with no `references.yaml` entry (warning), `[verify]` / `[citation needed]` markers (warning), empty bodies (info). Nothing is swallowed; the CLI exits 1 on any error.

**CLI** (`andamentum-compose`, no model): `new -o DIR --title T [--scaffold article]`, `sections DIR`, `show DIR ID`, `edit DIR ID --from FILE|- --revision N`, `add DIR --title T [--after ID]`, `validate DIR`, and `render DIR --plain --format F -o FILE` for the plain render (the plain-less `render DIR` is compose's own, with citations, sources list and provenance). The commands live in `compose/cli.py`; the user-facing word for a bundle is "document folder". A missing folder or a folder without `bundle.yaml` exits 3; a `BundleError` (stale revision, existing non-empty target) prints `error: ...` and exits 2; `validate` exits 1 when it finds an error.
