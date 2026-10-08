"""andamentum.chunker — deterministic markdown chunking.

``chunk(text, max_chars=N)`` splits at headings, then paragraphs, then
sentences. Fenced code blocks, tables and YAML front matter are never cut.
Each chunk carries its source offsets and its heading path. No model is
involved and nothing outside the standard library is imported.
"""

from .splitter import Chunk, ChunkerError, chunk

__all__ = ["Chunk", "ChunkerError", "chunk"]
