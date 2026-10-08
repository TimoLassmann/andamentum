"""andamentum.harvest — universal source → markdown extraction.

Single public API: ``extract(source) -> str``. Accepts URLs and file paths.
Internally dispatches to the best backend for the detected format and,
for ambiguous HTML, runs multiple backends and picks the highest-quality
output by structural scoring.

Two extraction modes, selected with ``mode=``. ``"fast"`` (the default)
runs docling's standard pipeline. ``"accurate"`` runs its VLM pipeline on
PDFs and writes a description for every figure. See `api.extract`.

Loud failure: every backend exhaustion or fetch problem raises a typed
``HarvestError`` — the function never silently returns empty markdown.
"""

from .api import ExtractionMode, extract, extract_from_bytes
from .errors import (
    ExtractionError,
    FetchError,
    FigureModelRequired,
    HarvestError,
    OllamaUnavailableError,
    UnsupportedFormatError,
)


__all__ = [
    "extract",
    "extract_from_bytes",
    "ExtractionMode",
    "ExtractionError",
    "FetchError",
    "FigureModelRequired",
    "HarvestError",
    "OllamaUnavailableError",
    "UnsupportedFormatError",
]
