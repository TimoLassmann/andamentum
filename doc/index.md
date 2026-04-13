# Andamentum documentation

Andamentum is a single Python package containing three sub-modules:

- [epistemic](./epistemic/overview.md) — claim analysis, multi-agent verification
- [deep_research](./deep_research/overview.md) — web research pipeline
- [document_store](./document_store/overview.md) — SQLite + FTS5 + vector search

## Design principles

Each public function takes its model as an explicit keyword-only argument.
There are no hidden defaults, no silent fallbacks, and no dependence on
shared configuration modules.

## Package layout

```
andamentum/
    epistemic/
    deep_research/
    document_store/
```

Sub-modules are imported by their full dotted name:

```python
from andamentum.epistemic import EpistemicRepository
from andamentum.deep_research import ResearchState
from andamentum.document_store import ingest, search
```

## CLIs

- `andamentum-epistemic` — epistemic analysis and agent operations
- `andamentum-research` — web research pipeline

Both take a `--model` argument or read `$ANDAMENTUM_MAIN_LLM_MODEL` from the environment.
