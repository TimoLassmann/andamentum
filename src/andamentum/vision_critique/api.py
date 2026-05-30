"""Public entry point: ``critique_figure``.

Async function that takes an image (bytes / Path / URL string) and a
pydantic-ai model id, and returns a structured critique. The model id is
required and explicit — no hidden default — matching the andamentum-wide
convention that every public LLM-calling function takes ``model=`` as a
keyword-only argument.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypeVar

import httpx
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent

from andamentum.core.models import resolve_model
from andamentum.core.url_safety import fetch_with_safe_redirects

from .prompts import build_prompt
from .schemas import FigureCritique


T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


async def critique_figure(
    image: bytes | str | Path,
    *,
    model: str,
    schema: type[T] = FigureCritique,  # type: ignore[assignment]
    extra_context: str | None = None,
) -> T:
    """Vision-critique a rendered figure against a bounded schema.

    Parameters
    ----------
    image:
        Raw PNG/JPEG bytes, a local file path (``str`` or ``Path``), or
        an ``http(s)://`` URL. URLs are fetched once via httpx; local
        paths are read; bytes pass straight through.
    model:
        pydantic-ai model id — e.g. ``ollama:gemma4:e4b-it-q4_K_M``,
        ``anthropic:claude-haiku-4-5``, ``openai:gpt-5.4-nano``. Must be
        a multimodal model. No default — caller picks.
    schema:
        Pydantic ``BaseModel`` subclass that the model output is parsed
        into. Defaults to :class:`FigureCritique`.
    extra_context:
        Optional caller-supplied context appended to the prompt — e.g.
        "this is a panel from a Cell-format manuscript figure" so the
        model knows what aspect-ratio / layout norms to apply.

    Returns
    -------
    An instance of ``schema``.

    Raises
    ------
    FileNotFoundError
        If ``image`` is a path that does not exist.
    httpx.HTTPError
        If ``image`` is a URL that cannot be fetched.
    andamentum.core.url_safety.SsrfBlocked
        If ``image`` is a URL (or redirect target) that resolves to a
        private / loopback / cloud-metadata address.
    andamentum.core.url_safety.ResponseTooLarge
        If the fetched image exceeds the response size cap.
    pydantic_ai.UnexpectedModelBehavior
        If the model fails to produce schema-conforming output after
        pydantic-ai's built-in retries.
    """
    image_bytes, media_type = await _normalise_image(image)
    prompt = build_prompt(extra_context=extra_context)

    resolved = resolve_model(model)
    agent: Agent[None, T] = Agent(model=resolved, output_type=schema)

    logger.debug("critique_figure: model=%s bytes=%d", model, len(image_bytes))
    result = await agent.run(
        [prompt, BinaryContent(data=image_bytes, media_type=media_type)]
    )
    return result.output


async def _normalise_image(
    image: bytes | str | Path,
) -> tuple[bytes, str]:
    """Resolve ``image`` to ``(bytes, media_type)``."""
    if isinstance(image, bytes):
        return image, _sniff_media_type(image)

    if isinstance(image, Path):
        return image.read_bytes(), _media_type_from_suffix(image.suffix)

    # str — could be a URL or a path
    if image.startswith(("http://", "https://")):
        # SSRF-protected fetch: every redirect hop is re-validated and the
        # body is size-capped (shared with harvest / deep_research).
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
            response = await fetch_with_safe_redirects(client, image)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(";")[0]
            return response.content, content_type or "image/png"

    path = Path(image)
    return path.read_bytes(), _media_type_from_suffix(path.suffix)


def _media_type_from_suffix(suffix: str) -> str:
    """Map a file extension to an image media type. Defaults to PNG."""
    suffix = suffix.lower().lstrip(".")
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
    }.get(suffix, "image/png")


def _sniff_media_type(data: bytes) -> str:
    """Guess image media type from magic bytes; default to PNG."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/png"
