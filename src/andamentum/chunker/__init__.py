"""andamentum.chunker — verifiable semantic chunking of long text.

The LLM only points at boundaries (start/end anchor strings) — it
never rewrites text. Extraction is byte-identical to the source.
Validation drives ModelRetry; failures escalate through window
halving → model escalation → loud failure. No heuristic fallbacks.
"""

__version__ = "0.1.0"
__all__: list[str] = []  # populated as the public surface lands
