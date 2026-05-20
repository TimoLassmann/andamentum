# Responsible-release hardening

**Date:** 2026-05-16
**Status:** Open questions resolved 2026-05-16 — ready to execute on approval
**Scope:** In-code protections + minimum documentation for first public release of `andamentum`.

> **DO NOT PUSH.** The user retains exclusive control over publishing to the public GitHub repository. This plan never includes `git push`. All "release" references mean "preparing the commit on a local branch for the user to push manually."

## Decisions locked in 2026-05-16

| Question | Answer |
|---|---|
| Repository URL | Personal GitHub. Use `<your-github-handle>` placeholder throughout code and docs; user will fill in before release. Never guess the handle. |
| Visible watermark default on standalone review reports (.docx / .md / .html) | **ON.** Top banner + per-page footer. The `--apply-patches` modified-manuscript output stays banner-OFF (invisible metadata only). |
| Audit log default behaviour | **Pure opt-in.** No file written by default. Activates only when user sets `ANDAMENTUM_AUDIT_LOG=/explicit/path`. No XDG-default-path; no `~/.local/share/andamentum/`. |
| Cloud-inference signal for research / epistemic | **TTY-aware hybrid.** One-time stderr warning in TTY; silent in non-TTY (pipes, CI, scripts). |
| Whetstone cloud-gate | Tier A — interactive y/N prompt in TTY; refuse in non-TTY unless `--yes-external-inference` flag or `ANDAMENTUM_ALLOW_EXTERNAL_INFERENCE=1`. |
| Chunker / vision_critique cloud-gate | Tier C — silent. With audit log opt-in by default, these are completely silent by default. |

## Goal

Before public release, add proactive in-code measures that make accidental misuse hard and deliberate misuse visible. Documentation alone is insufficient — readers ignore READMEs. The bias is toward small, automation-friendly friction at high-risk choke points, not blanket gating.

The package will be MIT-licensed and open-source. We can't prevent determined misuse; we can prevent inattentive misuse and make the provenance recoverable when an integrity question is asked later.

## Out of scope (deliberate)

- **Philosophy-of-science citations** in `epistemic/` (Popper, Lakatos, Lipton, Peirce, Reichenbach, Mill, Quine, Duhem, Doyle, Kahneman). The epistemic system is still being refined; the citation set will stabilise when the system stabilises. We will add them later.
- **Provider-level compliance hardening** (PubMed `tool=`/`email=`, arXiv 3-s throttle, OpenAlex polite-pool UA, deep_research robots.txt). Separate workstream — important but doesn't block release.
- **Verdict-vocabulary rename** in epistemic audit reports (`Confirmed` → `Evidence supports`). Defer until epistemic stabilises.
- **Schneider (2025) full citation.** Defer with the other epistemic citations.

## Workstreams

### A — Whetstone hardening (highest risk surface)

#### A1. Locked AI-author attribution in docx
**Files:** `src/andamentum/whetstone/renderers/docx.py:54`, `src/andamentum/whetstone/cli.py:324`.
**Change:** Default `author` parameter and `--patch-author` value to `"andamentum-whetstone (AI)"`. Remove the current `"Whetstone Review"` and `"Reviewer"` defaults — both can be mistaken for human reviewers in tracked-changes metadata.
**Override path:** `--allow-author-override "Name"` accepts a custom string. Prints a stderr warning when used: `"Attributing AI-generated edits to a human name may constitute research misconduct. Continuing..."`.
**Tests:** Default-attribution test; override-flag-warning test; refuse-without-override test.

#### A2. Tiered provenance watermark

Two layers, different defaults per artifact:

**Tier 1 — invisible metadata (always on):**
- For .docx: write to `core.xml` document properties — `dc:contributor`, `dc:description`, `cp:lastModifiedBy`, custom property `andamentum:produced-by`. Survives "Save As".
- For PDF: write XMP metadata via the typeset rendering pipeline. Values: `andamentum-whetstone v<ver>`, `model=<id>`, `produced=<ISO date>`, `mode=<review|panel|guidelines|custom>`.
- For HTML: `<meta name="generator" content="andamentum-whetstone v<ver> (model=<id>)">` + `<meta name="andamentum:produced">`.
- For markdown: a comment line at top `<!-- andamentum-whetstone v<ver> (model=<id>) on <date> -->`. Easy to strip but standard practice.

**Tier 2 — visible banner (default depends on artifact):**
- Standalone review reports (output of full review pipeline): banner ON by default. These are internal scratch, not submissions.
- `--apply-patches` modified manuscript: banner OFF by default. The user's manuscript shouldn't be polluted.
- `--visible-watermark` / `--no-visible-watermark` overrides.

**Stderr reminder at end of every run** (independent of watermark):
```
Note: Disclose AI assistance in your methods/acknowledgements section
per your target journal's policy. See whetstone/RESPONSIBLE_USE.md for
suggested wording.
```

**Files:** `src/andamentum/whetstone/renderers/{docx,html,markdown}.py`; the docx-finalization machinery in `src/andamentum/whetstone/docx/`; CLI hook in `cli.py`.
**Tests:** metadata-present test per renderer; banner-on by default for review reports; banner-off by default for `--apply-patches`; override flags work; stderr reminder fires.

#### A3. Confidentiality-marker tripwire
**Files:** `src/andamentum/whetstone/nodes/harvest_source.py` (after extraction, before pipeline starts).
**Change:** Scan extracted text for any of:
- `"Manuscript ID:"`, `"MS#"`, `"Submission ID:"`
- `"Confidential — do not distribute"`, `"CONFIDENTIAL"`
- `"Reviewer Instructions"`, `"Editorial Office"`, `"Decision Letter"`
- `"This manuscript is being considered"`
- common publisher boilerplate

If any marker is found, refuse to proceed unless `--confirm-own-draft` is passed. Error message names which marker fired so the user understands why.
**Tests:** each marker triggers refusal; `--confirm-own-draft` bypasses; clean drafts proceed unaffected.

#### A4. Novelty-cache hygiene
**Files:** `src/andamentum/whetstone/nodes/novelty_check.py:94-98`.
**Change:** Remove the on-disk cache at `~/.cache/whetstone/novelty/` by default. Cache is in-memory per run only. Opt-in via `--persist-novelty-cache`. Reason: hashed digests of unpublished claims sit on disk indefinitely with the current implementation.
**Tests:** no `~/.cache/whetstone/novelty/` files created in default run; `--persist-novelty-cache` enables disk persistence.

#### A5. Panel-mode author affirmation
**Files:** `src/andamentum/whetstone/cli.py` (mode dispatch).
**Change:** `--mode panel` requires `--i-am-the-author` flag (or `ANDAMENTUM_PANEL_OWN_AUTHOR=1`). Reason: panel mode output is shaped exactly like a real peer review (3–5 fictional reviewer biosketches + Accept/Reject recommendation), highest laundering risk.
**Tests:** panel mode without flag refuses with clear error; with flag proceeds; env var works.

#### A6. CLI banner + `review_document()` docstring
**Files:** `src/andamentum/whetstone/cli.py:33-34`; `src/andamentum/whetstone/api.py:47`; `src/andamentum/whetstone/__init__.py:1`.
**Change:** Add explicit "not a peer-review tool" first line to argparse description and module docstrings. Print a short one-line note on every CLI invocation:
```
Whetstone is for your own drafts. Do not use on confidentially-shared
manuscripts. See whetstone/RESPONSIBLE_USE.md.
```

#### A7. `whetstone/RESPONSIBLE_USE.md`
**Files:** new `src/andamentum/whetstone/RESPONSIBLE_USE.md`.
**Content:**
1. Who whetstone is for / not for
2. Confidentiality and AI peer review — policy-landscape pointers (COPE, ICMJE, NHMRC, NIH, individual publishers)
3. Data classification — when to use local-only models
4. `--check-novelty` warning (query-leak via SearXNG)
5. `--apply-patches` warning (provenance preservation)
6. Suggested ICMJE-style disclosure wording

### B — Scribe + figures

#### B1. Remove `grant` scaffold from scribe
**Files:** `src/andamentum/scribe/scaffolds.py:39-61` (the `_GRANT_SCAFFOLD` definition); `src/andamentum/scribe/cli.py` (the `--scaffold` choices); `src/andamentum/scribe/api.py` (`Document.create` scaffold parameter); existing tests that reference the grant scaffold.
**Change:** Delete entirely. Keep `article` scaffold. Update `--scaffold` choices to `["article"]` (or `None`). Update README, docs/index.md, CLAUDE.md to drop grant references.
**Tests:** existing grant-scaffold tests deleted; `--scaffold grant` errors with clear message; `--scaffold article` unaffected.

#### B2. `[ai-drafted]` / `[ai-edited]` first-class markers
**Files:** `src/andamentum/scribe/parser.py:22` (`_UNRESOLVED_MARKERS`); `scribe/api.py` `Document.validate()`; `scribe/render_docx.py` (or wherever .docx properties are written).
**Change:** Add `"[ai-drafted]"` and `"[ai-edited]"` to `_UNRESOLVED_MARKERS`. `validate()` reports them as warnings (same shape as `[verify]` and `[citation needed]`). The .docx renderer writes a custom document property `andamentum:contains-ai-markers=true` when any marker is present.
**Tests:** markers detected by validate(); document property written when present.

#### B3. Figures: "generation" → "rendering" terminology
**Files:** `README.md:13,57`; `docs/index.md:14`; `src/andamentum/figures/__init__.py:1`; `src/andamentum/figures/render.py:64` (and any other docstrings); `src/andamentum/figures/style.py:25,171`; `src/andamentum/figures/cli.py:14`.
**Change:** Replace `"publication-quality scientific figure generation"` with `"publication-quality scientific figure rendering — deterministic plotting of your data with journal-matched sizing"`. Reason: "generation" reads as generative-AI in 2026.
**Tests:** none needed — documentation/string change only.

#### B4. Figures: caption-template guidance in `figure()` docstring
**Files:** `src/andamentum/figures/render.py` `figure()` docstring.
**Change:** Add a "Caption guidance" section pointing out that `FigureResult.advisor_notes` lists auto-applied visual changes (orientation, log scale, sort order) and should be mirrored in the figure caption. Plus a one-line stderr reminder when the CLI saves a figure with non-empty advisor notes.
**Tests:** stderr-reminder test when advisor_notes is non-empty.

### C — Cloud-inference gates (tiered)

Centralised in `src/andamentum/core/cloud_gate.py` (new file). Every CLI imports a single `check_cloud_inference(cli_name, model, *, payload_bytes=None)` function.

#### C1. Tier classification

| CLI | Tier | Behaviour |
|---|---|---|
| `andamentum-whetstone` | A — interactive prompt | TTY: y/N prompt. Non-TTY: refuse unless env var set. |
| `andamentum-research` | B — one-time stderr warning | Per-process; first cloud call prints; subsequent calls silent. |
| `andamentum-epistemic` | B — one-time stderr warning | Same. |
| `andamentum-chunker` | C — audit log only | No prompt, no stderr. |
| `andamentum-vision-critique` | C — audit log only | No prompt, no stderr. |
| `andamentum-scribe`, `andamentum-figures`, `andamentum-harvest`, `andamentum-proofread` | n/a | No LLM. |

#### C2. Cloud-provider detection
**Files:** `src/andamentum/core/cloud_gate.py`.
**Change:** Function `is_cloud_model(model: str) -> bool`. Returns True for `openai:*`, `anthropic:*`, `bedrock:*`, `gemini:*`, `mistral:*`, `groq:*`, `cohere:*`, `passthrough:*` (passthrough is unknown — treat as cloud to be safe). Returns False for `ollama:*` and anything obviously local. Unknown providers default to True with a stderr note.
**Tests:** standard provider strings classified correctly; unknown handled safely.

#### C3. Tier A — interactive prompt (whetstone)
**Files:** `src/andamentum/whetstone/cli.py` (entry point, before any LLM call).
**Behaviour:**
```
WARNING: <openai:gpt-5.4-nano> is an external inference provider.
Your document (~6 KB, 1,247 words) will be sent to OpenAI.

Do NOT proceed if this document was shared with you confidentially
(e.g. as a peer reviewer, examiner, or grant panel member).
Do NOT proceed if it contains patient data, embargoed sequences,
or anything covered by NDA / MTA / DUA.

Proceed? [y/N]:
```
- TTY: prompt; default N; "y" or "yes" proceeds.
- Non-TTY: refuse with the message above + `"Set ANDAMENTUM_ALLOW_EXTERNAL_INFERENCE=1 or pass --yes-external-inference to proceed in non-interactive mode."`
- `--yes-external-inference` flag: skips prompt.
- `ANDAMENTUM_ALLOW_EXTERNAL_INFERENCE=1`: skips prompt.
- If the user has opted into the audit log (`ANDAMENTUM_AUDIT_LOG=/path` set), all paths including bypass write a line. If not, no I/O.

**Tests:** TTY-prompt simulation (input="y" proceeds; "n" exits); non-TTY refuses; flag bypasses; env-var bypasses; audit-log line written only when opted in.

#### C4. Tier B — TTY-aware one-time stderr warning (research, epistemic)
**Files:** CLI entry points for research and epistemic.
**Behaviour:** On first cloud call in the process:
- If stderr is a TTY (interactive use): print warning once.
- If stderr is not a TTY (CI, pipe, script): silent.

Warning text:
```
Note: using external inference provider <openai:gpt-5.4-nano>.
Content will be sent to OpenAI. Suppress with ANDAMENTUM_QUIET_CLOUD=1.
```
Subsequent calls in the same process: always silent (in-process flag).
**Tests:** TTY-simulated stderr fires once; non-TTY silent; env var suppresses TTY case; second call always silent.

#### C5. Tier C — silent (chunker, vision_critique)
No stderr, no prompt. By default the audit log is also off (see C6), so these operations are completely silent. If the user has opted into the audit log, a line is written.
**Tests:** no stderr output by default; audit log entry written only when `ANDAMENTUM_AUDIT_LOG` is set.

#### C6. Audit log (pure opt-in)
**Files:** `src/andamentum/core/audit_log.py` (new file).
**Behaviour:** No file is written unless the user explicitly sets `ANDAMENTUM_AUDIT_LOG=/path/they/choose` to a writable path. When unset (the default), audit-log functions are pure no-ops with no I/O side effects.

When opted in, every cloud LLM call (regardless of tier, CLI vs library) appends one line to the user-chosen path:
```
2026-05-16T14:23:01Z whetstone --mode panel anthropic:claude-haiku-4-5 sha256:abc123def 6234B
```
Format: ISO-8601 timestamp, CLI/caller name, mode/op, model id, document SHA-256 (or "n/a" for query-only ops), payload byte count.

- The file is created at the user-chosen path with `0o600` permissions on first write.
- No XDG fallback. No `~/.local/share/andamentum/`. No default path of any kind.
- Failures to write (path unwritable, disk full) emit a stderr warning but never block the operation.
- Documented in `RESPONSIBLE_USE.md` as available for institutional / compliance scenarios where a local paper trail is wanted.

**Tests:** unset env var → no file written, no I/O; explicit path → file at exact path with correct fields and 0o600 perms; write-failure → stderr warning, operation continues; tests configure a tmpdir path via `monkeypatch.setenv` in `conftest.py`.

### D — Harvest

#### D1. User-Agent with contact URL
**Files:** `src/andamentum/harvest/fetch.py:123`.
**Change:** Change UA from `"andamentum-harvest/0.1"` to `f"andamentum-harvest/{__version__} (+https://github.com/<your-github-handle>/andamentum)"`. Use the version from `andamentum.__version__`. The `<your-github-handle>` placeholder is a literal string in the code, with a `# TODO: replace before release` comment — user fills in before pushing.
**Tests:** UA contains version and the placeholder (or, once replaced, the real URL).

#### D2. robots.txt check (default on)
**Files:** new `src/andamentum/harvest/robots.py`; integration in `harvest/fetch.py`.
**Change:** Before any HTTP fetch, retrieve and cache `<scheme>://<host>/robots.txt` for the target host. If the path is disallowed for our User-Agent (or `*`), refuse to fetch with a clear error. `--ignore-robots` flag and `ANDAMENTUM_IGNORE_ROBOTS=1` env var bypass with a stderr warning.
**Cache:** in-memory per-process; not persistent.
**Tests:** disallowed path refused; allowed proceeds; bypass flag works; malformed robots.txt treated as allow-all; robots.txt fetch failure treated as allow-all.

#### D3. Paywalled-publisher tripwire
**Files:** `src/andamentum/harvest/fetch.py`.
**Change:** Maintain a small list of paywalled-academic-publisher netlocs (symbolic, not exhaustive):
```python
_PAYWALLED_PUBLISHERS = frozenset({
    "sciencedirect.com", "linkinghub.elsevier.com",
    "link.springer.com", "rd.springer.com",
    "onlinelibrary.wiley.com",
    "ieeexplore.ieee.org",
    "dl.acm.org",
    "nejm.org",
    "jamanetwork.com",
    "cell.com",
    "nature.com",
    "science.org",
})
```
If a target URL matches (or is a subdomain of) any entry, refuse without `--accept-tdm-responsibility` flag or `ANDAMENTUM_ACCEPT_TDM=1` env var. Error message:
```
<sciencedirect.com> is a paywalled academic publisher. Bulk extraction
typically requires a TDM (text and data mining) licence with the
publisher. Pass --accept-tdm-responsibility to confirm you have the
right to extract from this URL.
```
**Tests:** listed netlocs refused without flag; bypass works; non-listed hosts unaffected; subdomain matching.

#### D4. CLI description update
**Files:** `src/andamentum/harvest/cli.py:32-37`.
**Change:** Add to description:
```
You are responsible for respecting robots.txt (checked by default;
opt out with --ignore-robots), publisher Terms of Service, and any
applicable TDM (text and data mining) licensing.
```

### E — Cross-cutting documentation

#### E1. Top-level `RESPONSIBLE_USE.md`
**Files:** new `RESPONSIBLE_USE.md` at repo root.
**Content:** Sections covering intended use, AI-disclosure expectations per major standard (ICMJE, NIH NOT-OD-25-122, NHMRC, COPE — pointers not full text), source-access expectations, data-fabrication line for figures, out-of-scope uses, reporting misuse.

#### E2. README "Intended use and limits" section
**Files:** `README.md`.
**Change:** New section between "Documentation" and "License". Short — points readers at `RESPONSIBLE_USE.md` and the module-level `RESPONSIBLE_USE.md`s. Includes:
- Whetstone is for your own drafts
- AI disclosure is your responsibility
- Harvest doesn't enforce ToS / robots.txt is checked by default
- Figures plots data, doesn't generate it
- Research-stage software, MIT-licensed, no warranty

#### E3. README "Acknowledgements" section
**Files:** `README.md`.
**Change:** New section at end of README. Lists software acknowledgements only (deferred: the philosophy-of-science citations stay out until epistemic stabilises):
- SearXNG, trafilatura (Barbaresi 2021), Docling (IBM Research), pydantic-ai, pydantic-graph, sqlite-vec (Alex Garcia), Ollama, EmbeddingGemma (Google)
- Algorithms with code citations already in place: RRF (Cormack et al. 2009), DHP (Du et al. 2015) — surface here too
- Data sources used by epistemic providers: PubMed/E-utilities (NCBI), arXiv, Europe PMC, bioRxiv, ClinicalTrials.gov, ChEMBL, Monarch, Open Targets, OpenAlex — each with a one-liner

#### E4. CITATION.cff
**Files:** new `CITATION.cff` at repo root.
**Content:** Standard cff-version 1.2.0 file with author (Timo Lassmann), affiliation (Telethon Kids Institute), version (from pyproject.toml), license (MIT), repository URL placeholder (`https://github.com/<your-github-handle>/andamentum` with TODO comment). ORCID field left as `<TODO>` for user to fill. Feeds GitHub's "Cite this repository" widget.

## Sequencing

1. **First:** Documentation (E1, E2, E3, E4) — non-code, lowest risk to land. Pure additions.
2. **Then:** Core infrastructure (C2, C6) — cloud-gate detection + audit log. Foundation for C3, C4, C5.
3. **Then:** Tiered cloud gates (C3, C4, C5).
4. **Then:** Whetstone hardening (A1, A2, A3, A4, A5, A6, A7).
5. **Then:** Scribe + figures (B1, B2, B3, B4).
6. **Then:** Harvest (D1, D2, D3, D4).

Workstreams A, B, D are independent of each other; only C is a shared dependency (everything that has LLM calls uses it).

## Acceptance criteria

Definition of done for this plan:

- All tests pass under `uv run pytest`.
- Pyright clean at current baseline.
- Ruff clean.
- Manual end-to-end run of:
  - `andamentum-whetstone draft.md --model openai:gpt-5.4-nano` → prompts in TTY, refuses in pipe.
  - `andamentum-whetstone draft.md --model ollama:llama3` → no prompt, runs.
  - `andamentum-whetstone --mode panel draft.md` → refuses without `--i-am-the-author`.
  - `andamentum-whetstone --apply-patches patches.json draft.docx --out out.docx` → no visible banner in `out.docx`; XMP metadata present; stderr reminder printed.
  - `andamentum-scribe init --scaffold grant` → errors with clear message.
  - `andamentum-harvest https://sciencedirect.com/article/...` → refuses without `--accept-tdm-responsibility`.
  - `andamentum-harvest https://example.com/robots-disallowed-path` → refuses; `--ignore-robots` bypasses.
  - `andamentum-chunker bigfile.md --model openai:gpt-5.4-nano` → no prompt, audit log entry written.
- Audit log writes nothing by default; when `ANDAMENTUM_AUDIT_LOG=/path` is set, entries are written to the user-chosen path with `0o600` perms.
- `RESPONSIBLE_USE.md`, `whetstone/RESPONSIBLE_USE.md`, `CITATION.cff` present.
- README has "Intended use and limits" and "Acknowledgements" sections.

## Implementation-time decisions

(Resolved during implementation without further user input. Documented here so the choices are visible.)

1. **`andamentum-whetstone --no-llm` interaction with Tier A.** When `--no-llm` is in effect, no model is resolved, so the cloud gate is skipped entirely. The gate is a function of "is a cloud model about to be called", not "is the user running whetstone".
2. **Existing flag conflicts for `--yes-external-inference`.** Read each CLI's argparse setup before adding the flag; pick a non-colliding short form (or none) if the long form is taken.
3. **Test isolation of audit log.** Since the default is no I/O, tests are unaffected unless they explicitly test the audit log. Those tests use `monkeypatch.setenv("ANDAMENTUM_AUDIT_LOG", str(tmp_path / "audit.log"))`.
4. **Confidentiality-marker list (A3).** Ship with the initial strict list; tune if real false-positives appear. The list is centralised in one constant for easy editing.
5. **Paywalled-publisher netloc list (D3).** Ship with the initial list of 10–12 obvious publisher domains; not exhaustive by design. Users can pass `--accept-tdm-responsibility` to override per-invocation.

## Risks

- **Audit-log dependency.** If `~/.local/share/andamentum/` is unwritable (rare, but possible on locked-down systems), the audit log fails. Plan handles this: stderr warning, never blocks. Confirm tests cover this.
- **robots.txt slowdown.** Every fetch incurs an extra HTTP call. Mitigation: in-process cache (per host). Worth one round-trip on first fetch.
- **Confidentiality-marker false positives.** Some manuscripts legitimately contain words like "Editorial Office" in their references. False-positive rate needs measurement on a small corpus before final tuning. Plan: ship with the strict list, gather feedback, loosen if needed.
- **`--apply-patches` watermark default.** Banner-OFF default trusts the user to disclose. The XMP metadata layer is the recovery mechanism. If we discover users frequently fail to disclose, revisit the default.
