# andamentum.vision_critique — notes for Claude

Moved from the root `CLAUDE.md`, word for word. Claude Code loads this file when it reads files in this directory.

## Overview

- `andamentum.vision_critique` — bounded vision critique of rendered figures. Single async API: `await critique_figure(image, *, model) -> FigureCritique`. Image is bytes / Path / http(s) URL; model is any pydantic-ai multimodal id (Ollama, Anthropic, OpenAI). Schema is a tight pydantic model with close-set enums (`label_overlap`, `labels_legible`, `legend_blocks_data`, `aspect_ratio_issue`, `suggested_fixes` from a fixed `Literal` set, `confidence`, `one_line_summary`) so small local vision models can fill it reliably, plus `has_issues` — a computed field the model never fills, derived from the other seven and present in the serialised JSON but deliberately absent from the validation schema (a `@computed_field` enters the serialization schema only, so pydantic-ai keeps asking for seven fields; `tests/test_api.py` pins both halves). Calibration is a **list, not a single model**: `CALIBRATION_MODELS` in `tests/test_api.py` names every model certified against the committed `broken_bar.png` fixture, the `@pytest.mark.ollama` test is parametrised over it, and a model the local Ollama does not serve **skips rather than fails** — the list records what is certified, one machine holds a subset. Certified: `ollama:qwen3.8` (measured 2026-09-06 — `label_overlap=True`, `labels_legible=False`, confidence 0.98, 23.5 s cold; already pulled for `harvest` accurate mode, so the module adds no new download). `gemma4:e4b-it-q4_K_M` was the previous certified model and is retired — it is no longer in use and no longer pulled here. `gemma4:e2b` never qualified: it fails the fixture, confidently reporting the figure is fine. Add a model to the list before recommending it anywhere. Used standalone, by `figures` for figure refinement, or by other modules that want to lint rendered output.

## Layering

- `vision_critique` depends on `core` (for `resolve_model`) and `pydantic` / `pydantic-ai` / `httpx`. MUST NOT depend on `epistemic`, `deep_research`, `document_store`, `whetstone`, `compose`, `figures`, `typeset`, `chunker`, or `harvest`. Other modules MAY depend on `vision_critique` (e.g. `figures` for refinement loops, `whetstone` for figure review during manuscript checking).

## Interface notes

- The caller-supplied extra text is `context` in Python and `--context` on the command line (it was `extra_context` in Python).
- Only PNG, JPEG, WEBP and GIF are accepted. A supported file extension or content type is trusted; otherwise the type is read from the content. Anything else (PDF, SVG, TIFF) raises `UnsupportedImageError` (CLI exit 3) instead of being sent as PNG.
- Errors derive from `VisionCritiqueError`.
- CLI exit codes: 0 no issue found / 1 the critique found at least one issue (JSON still written) / 2 usage / 3 image unreadable or unsupported / 4 run failed. `-o/--out` writes the JSON to a file.
