"""Passthrough for inputs that are already markdown or plain text.

There's nothing to extract — the source IS the markdown. This backend
exists so the orchestrator's dispatch table is uniform across all formats.
"""

from __future__ import annotations

from ..errors import ExtractionError


async def extract(data: bytes, source_url: str) -> str:
    """Decode bytes as UTF-8 and return them unchanged.

    Tries a couple of fallback encodings before giving up loudly.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ExtractionError(
            f"could not decode {source_url} as UTF-8 or Latin-1",
            attempted=["plain"],
        )
    if not text.strip():
        raise ExtractionError(
            f"source {source_url} is empty",
            attempted=["plain"],
        )
    return text
