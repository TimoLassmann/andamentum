"""Extraction backends.

Each backend is a thin async function with the same shape:

    async def extract(data: bytes, source_url: str) -> str

so the orchestrator can race them generically. A backend either returns
clean markdown or raises an exception — never returns a partial / fallback
result.
"""

from .docling_backend import extract as extract_with_docling
from .plain_backend import extract as extract_passthrough
from .trafilatura_backend import extract as extract_with_trafilatura

__all__ = [
    "extract_with_docling",
    "extract_with_trafilatura",
    "extract_passthrough",
]
