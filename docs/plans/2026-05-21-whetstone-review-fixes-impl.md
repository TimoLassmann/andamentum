# Whetstone reviewed-document failures — Implementation plan (Phase 3)

**Status:** Draft. User sign-off required before Phase 4 (writing code).
**Companion to:** the PRD and design-options docs (same date).
**Branch:** `whetstone-review-fixes`

Decisions locked in Phase 2:

| Problem | Chosen approach |
|---|---|
| P1 — comments invisible in Word | Fix the label + route all "build a new tagged file" spots through one shared piece of code so the mistake can't recur + regression test |
| P2 — lens emits false "no claims / missing section" findings | Prompt change (permission to find nothing + don't flag absences it can't see) + a post-filter that drops self-admitted non-findings |
| P3 — counted findings vs shown comments | List un-anchorable findings in the report instead of "See detailed reviews below" |
| Q4 — patch visibility | CLI logs applied/failed comment counts (separate commit) |

Four commits, in dependency order. Each is independently testable and
must leave the canonical green state (pytest passing, pyright 23, ruff
clean). Each gets the user's go-ahead before I start it, and I show the
diff before committing.

---

## Commit 1 — P1: make Word comments visible (the critical fix)

**Why first:** every other improvement is invisible until comments
render. This unblocks everything.

**What changes:**
1. In the Word-writing code (`whetstone/docx/`), add one small shared
   function that creates a new tagged-file root with the correct `w`
   label always attached.
2. Route the comments-file creation (`low_level.py:479`) through it.
3. Audit the ~10 other spots that create tagged-file roots; route any
   that create *independent* files through the same function. (Most of
   the 96 element-creation spots are sub-elements that inherit the
   label automatically — those need no change. The audit confirms
   which are true roots.)

**Test (the regression guard):**
- A new test builds a `.docx` with at least one comment, opens the
  resulting zip, parses each internal XML file, and asserts every
  WordprocessingML tag uses the `w` label — no `ns0`, `ns1`, etc.
- This test would fail on today's code and pass after the fix; it
  prevents the bug from silently returning.

**Files touched:** `whetstone/docx/low_level.py` (+ possibly
`xml_builder.py`), one new test file under `whetstone/tests/`.

**Acceptance:**
- The regression test passes.
- pytest / pyright / ruff at canonical green state.
- **User verification:** re-run whetstone on a real manuscript, copy
  the output `.docx` into the repo, and I confirm every comment is
  under the `w` label (and you confirm they're visible in Word).

**Not in this commit:** anything about which findings get emitted, the
report text, or the CLI output.

---

## Commit 2 — P2: stop the lens flagging absences it can't see

**Why second:** with comments now visible (Commit 1), the false
"no claims here / no Discussion section" findings become the most
visible remaining noise.

**What changes:**

*Part A — prompt change* in the per-section lens prompt
(`critical_read.py:_run_lens`). Two additions, both domain-agnostic
(no mention of references / funding / any section type):
1. Explicit permission to return nothing: an empty list is the correct
   answer when a section has no substantive content to assess; do not
   invent issues to fill the list.
2. Scope discipline: you are reviewing ONE section of a larger
   document; comment only on what is present here; do not flag things
   as missing from the document as a whole — you cannot see the other
   sections.

*Part B — post-filter safety net.* A small deterministic function that
drops a lens finding when the finding's own text admits there is
nothing to assess (e.g. "there is nothing to assess", "this section
contains only references", "no claims are made here"). This matches the
*model's own admission*, not a section type — it stays domain-agnostic.
Applied to lens (LLM) findings only, not to the deterministic /
proofread findings.

**Tests:**
- Deterministic unit tests for the post-filter: absence-admitting
  findings are dropped; normal findings pass through; edge cases
  (a real finding that merely mentions "references" in passing is
  NOT dropped).
- A test asserting the prompt contains the two new instructions (so a
  future edit can't silently remove them).
- Note: we cannot unit-test that a real small LLM actually complies
  with the prompt — that requires a live run. Verification of the
  prompt's *effect* is by the user re-running on a manuscript.

**Files touched:** `critical_read.py` (prompt + filter call), a new
filter function (likely in `whetstone/structural/` or alongside the
lens code), tests.

**Acceptance:**
- Post-filter unit tests pass.
- pytest / pyright / ruff green.
- **User verification:** re-run on a manuscript; confirm the
  References / bookkeeping sections (and others) no longer carry
  "no claims here" type comments.

**Not in this commit:** any section-type label list, any input-side
skipping of sections, the document-map preamble (all deferred /
rejected per PRD).

---

## Commit 3 — P3: no finding vanishes

**Why third:** once Commits 1–2 land, most findings render as comments
and the absence-noise is gone. What remains: findings the agent made
that have no quotable anchor (e.g. "the paper never states its sample
size") are still silently dropped, and the report still says "See
detailed reviews below" even when nothing is below.

**What changes:**
- Findings that *can* be anchored continue to become Word comments
  (unchanged).
- Findings that *cannot* be anchored (no quote) are collected and
  listed as text under the report's "Critical Issues" heading.
- The "See detailed reviews below" placeholder is only used when there
  genuinely are anchored comments to see; otherwise the un-anchored
  findings are shown inline.

**Files touched:** `renderers/docx.py` (collect the un-anchored
findings, pass them through), `docx/finalization.py`
(`_format_critical_issues` to render the list).

**Tests:**
- A finding with no quote appears in the report text.
- A finding with a quote does NOT get duplicated into the report text
  (it's a comment instead).
- The "See detailed reviews below" string only appears when there are
  anchored comments.

**Acceptance:**
- Tests pass; green state.
- **User verification:** the report header's count matches what's
  visible (comments + listed findings); nothing is silently lost.

---

## Commit 4 — Q4: show whether comments anchored (CLI visibility)

**Why last:** pure logging; depends on nothing; smallest.

**What changes:**
- The `.docx` output path in the CLI logs the applied/failed comment
  counts that `render_docx` already returns but currently discards.
  Example line:
  `[output] wrote reviewed.docx — 137/137 comment(s) applied, 0 could not be anchored`

**Files touched:** `whetstone/cli.py` only.

**Tests:** light — assert the log line is emitted with the right
counts (or just rely on the existing CLI test coverage if adding a
dedicated assertion is awkward).

**Acceptance:**
- Green state.
- The CLI prints the applied/failed counts on a `.docx` run.

---

## Sequencing & checkpoints

1. I implement **Commit 1**, show you the diff, run the green-state
   checks, and (if you want) wait for your manuscript re-run before
   committing.
2. Repeat for **Commit 2**, **3**, **4** — one at a time, your
   go-ahead between each. No bundling.
3. When all four land and you're satisfied, you merge
   `whetstone-review-fixes` into `main` manually.

## Definition of done (from the PRD, restated)

- Every anchorable finding renders as a visible Word comment.
- The lens emits no absence-findings; bookkeeping sections carry no
  "nothing here" noise; works for any document type (no section-type
  labels).
- A regression test enforces the `.docx` label correctness.
- Un-anchorable findings are visible in the report, not silently
  dropped.
- The CLI shows applied/failed comment counts.
- No env vars added; no domain-specific label lists; no surprise scope
  expansion.
- Canonical green state throughout.

## Sign-off

Approve this plan (or adjust the commit boundaries / ordering) and I'll
start **Commit 1 only**, show you the diff, and stop for your review
before committing or proceeding.
