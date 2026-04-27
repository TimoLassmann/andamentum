# Chunker benchmark

Tracks boundary accuracy of `andamentum.chunker.extract_units` across diverse text types. Defaults to local model `ollama:gemma4:31b-nvfp4`; override via the `CHUNKER_BENCH_MODEL` env var or `--model` flag.

## Running

```bash
# Pytest mode — strict per-case F1 floors gate the run
uv run pytest benchmarks/chunker -m benchmark -v

# Standalone CLI — produces a markdown report
uv run python -m benchmarks.chunker.cli --output report.md

# One specific case only
uv run python -m benchmarks.chunker.cli --case academic_short

# Compare two models
uv run python -m benchmarks.chunker.cli --model openai:gpt-4o-mini --output gpt.md
uv run python -m benchmarks.chunker.cli --model ollama:gemma4:31b-nvfp4 --output gemma.md

# Visual case-authoring helper (FastAPI app at http://127.0.0.1:8765)
uv run python -m benchmarks.chunker.app
```

## Adding a case

For each new case, create a pair of files in `cases/`:

- `cases/<name>.input.<ext>` — the source document (`.md`, `.txt`, `.html`, `.py`, `.rst`, `.json`)
- `cases/<name>.truth.json` — the expected unit boundaries

### Faster path: use the helper app

Run `uv run python -m benchmarks.chunker.app`, paste your candidate text, click **Chunk it**, see the result with each unit highlighted in a different colour. Click **Download truth.json** for a draft annotation file with titles + anchors auto-filled — then edit it (especially `convention`, `expected_f1_floor`, and the unit `title`s) and save it into `cases/`.

### `truth.json` schema

```json
{
  "convention": "Brief description of what 'unit' means for THIS case",
  "expected_f1_floor": 0.7,
  "boundary_tolerance_chars": 50,
  "domain": "academic",
  "units": [
    {
      "title": "Introduction",
      "start_anchor": "first 5-10 verbatim words",
      "end_anchor": "last 5-10 verbatim words"
    }
  ]
}
```

### How to choose unit boundaries

Use the principle: **"smallest passage that could be reviewed or cited independently."** This varies by content:

- **Academic papers:** paragraphs, or subsections if they're self-contained
- **Book / narrative chapters:** scenes, topical shifts
- **Q&A transcripts:** each Q+A pair as one unit
- **Code:** functions, classes, distinct top-level blocks
- **Web pages:** article paragraphs only — nav / ads / footers should be SKIPPED, not units

### Anchors must be verbatim

Copy 5-10 words exactly from the source. The runner uses the same tiered matcher the chunker uses (`andamentum.chunker.validation.find_anchor`). Same anchor appearing twice resolves by document order — each unit's `start_anchor` is searched starting AFTER the previous unit's `end_anchor`.

### Setting `expected_f1_floor`

Conservative starting points:

- Easy (clear headings, distinct sections): **0.85+**
- Medium (some structure, some noise): **0.65-0.85**
- Hard (continuous prose, ambiguous boundaries): **0.45-0.65**

Run the benchmark once on your case to see what the model actually achieves, then set the floor a little below that as the regression bar.

### Setting `boundary_tolerance_chars`

How close a predicted boundary must be to a ground-truth boundary to count as "matched":

- Default: **50** (~10 words)
- Dense / short text: **30**
- Loose / long-paragraph text: **80**

### `domain`

Pick the matching domain hint for the chunker prompt: `academic`, `web`, `code`, `transcript`, or `general`. The hint is a 2-3 sentence string substituted into the prompt — it doesn't change algorithm, just the model's framing.

## Result interpretation

The runner reports per case:

| Metric | Meaning |
|---|---|
| `boundary_f1` | Headline accuracy — how well predicted boundaries match truth, within tolerance |
| `boundary_precision` / `recall` | Did we add spurious boundaries / miss real ones? |
| `coverage` | Fraction of source text claimed by some unit |
| `gap_fraction` | Fraction of source text in unclaimed gaps |
| `granularity_ratio` | `predicted_units / truth_units`. 1.0 = perfect; <1 = under-segmented; >1 = over-segmented |
| `fragmentation_rate` | Fraction of units marked `complete=False` (truncated at window boundary) |
| `anchor_method_*` | How many units matched via `exact` / `whitespace_normalised` / `fuzzy` |
| `wall_clock_seconds`, `model_calls` | Cost proxies |

A case **passes its floor** when `boundary_f1 >= expected_f1_floor`. Anything below fails the pytest test loudly.
