"""andamentum.harvest — universal source → markdown extraction.

Single public API: ``extract(source) -> str``. Accepts URLs and file paths.
Internally dispatches to the best backend for the detected format and,
for ambiguous HTML, runs multiple backends and picks the highest-quality
output by structural scoring.

Loud failure: every backend exhaustion or fetch problem raises a typed
``HarvestError`` — the function never silently returns empty markdown.
"""

from .api import extract, extract_from_bytes
from .errors import (
    ExtractionError,
    FetchError,
    HarvestError,
    UnsupportedFormatError,
)


__all__ = [
    "extract",
    "extract_from_bytes",
    "ExtractionError",
    "FetchError",
    "HarvestError",
    "UnsupportedFormatError",
]
