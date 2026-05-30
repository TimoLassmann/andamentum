"""andamentum.chunker — structural-first semantic chunking.

Pipeline (literature consensus, 2026):
  1. Structural split on markdown headings (deterministic, free).
  2. Semantic split for over-budget sections via cosine drops between
     paragraph embeddings.
  3. Optional LLM judge for grey-zone boundaries.

Output units' ``text`` is byte-identical to a source span. The LLM is
used only as a boundary judge, never as the primary segmenter.
"""

from .embeddings import EmbeddingFn, make_ollama_embedder
from .extractor import ExecutorFn, extract_units, make_runner_executor
from .judge import JudgeVerdict
from .types import (
    ChunkingResult,
    Gap,
    Unit,
)


__all__ = [
    # Functions / callables
    "extract_units",
    "make_ollama_embedder",
    "make_runner_executor",
    # Data types
    "ChunkingResult",
    "EmbeddingFn",
    "ExecutorFn",
    "Gap",
    "JudgeVerdict",
    "Unit",
]
