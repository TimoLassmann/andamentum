"""andamentum.chunker — verifiable semantic chunking of long text.

The LLM only points at boundaries (start/end anchor strings) — it
never rewrites text. Extraction is byte-identical to the source.
Validation drives ModelRetry; failures escalate through window
halving → executor escalation → loud failure. No heuristic fallbacks.
"""

# === Functions you can wrap as agent tools ===
# `extract_units` is the main entry point.
# `make_runner_executor` builds a production executor from an AgentRunner.
from .extractor import ExtractionAttempt, extract_units, make_runner_executor

# === Result/data types (returned by the above; not tools themselves) ===
from .refinement import EscalationOutcome
from .types import (
    ChunkingFailedError,
    ChunkingResult,
    Gap,
    NextUnitResult,
    Unit,
)

__version__ = "0.1.0"

__all__ = [
    # Functions / callables
    "extract_units",
    "make_runner_executor",
    # Data types
    "ChunkingFailedError",
    "ChunkingResult",
    "EscalationOutcome",
    "ExtractionAttempt",
    "Gap",
    "NextUnitResult",
    "Unit",
]
