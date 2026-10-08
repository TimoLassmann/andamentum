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

from .errors import UnsupportedImageError
from .prompts import build_prompt
from .schemas import FigureCritique

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


async def critique_figure(
    image: bytes | str | Path,
    *,
    model: str,
    schema: type[T] = FigureCritique,  # type: ignore[assignment]
    context: str | None = None,
) -> T:
    """Vision-critique a rendered figure against a bounded schema.

    The model must be multimodal. The only locally validated model is
    ``ollama:qwen3.8`` (it passes the committed ``broken_bar.png`` fixture,
    see ``CLAUDE.md``); that is a record of what was tested, not a default:
    ``model`` is always required.

    Parameters
    ----------
    image:
        Raw PNG, JPEG, WEBP or GIF bytes, a local file path (``str`` or ``Path``), or
        an ``http(s)://`` URL. URLs are fetched once via httpx; local
        paths are read; bytes pass straight through.
    model:
        pydantic-ai model id — e.g. ``ollama:qwen3.8``,
        ``anthropic:claude-haiku-4-5``, ``openai:gpt-5.4-nano``. Must be
        a multimodal model. No default — caller picks.
    schema:
        Pydantic ``BaseModel`` subclass that the model output is parsed
        into. Defaults to :class:`FigureCritique`.
    context:
        Optional caller-supplied context appended to the prompt — e.g.
        "this is a panel from a Cell-format manuscript figure" so the
        model knows what aspect-ratio / layout norms to apply.

    Returns
    -------
    An instance of ``schema``.

    Raises
    ------
    UnsupportedImageError
        If the image is not a PNG, JPEG, WEBP or GIF (e.g. a PDF or SVG);
        render it to PNG first. A ``VisionCritiqueError`` and a ``ValueError``.
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
    prompt = build_prompt(context=context)

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
    """Resolve ``image`` to ``(bytes, media_type)``.

    A supported file extension or content type is trusted; otherwise the type
    is read from the content (magic bytes). Anything that is not a PNG, JPEG,
    WEBP or GIF raises :class:`UnsupportedImageError`.
    """
    if isinstance(image, bytes):
        return image, _sniff_media_type(image)

    if isinstance(image, Path):
        data = image.read_bytes()
        return data, _media_type_from_suffix(image.suffix) or _sniff_media_type(
            data, name=str(image)
        )

    # str — could be a URL or a path
    if image.startswith(("http://", "https://")):
        # SSRF-protected fetch: every redirect hop is re-validated and the
        # body is size-capped (shared with harvest / deep_research).
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
            response = await fetch_with_safe_redirects(client, image)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(";")[0]
            return response.content, (
                content_type
                if content_type in _SUPPORTED
                else _sniff_media_type(response.content, name=image)
            )

    path = Path(image)
    data = path.read_bytes()
    return data, _media_type_from_suffix(path.suffix) or _sniff_media_type(
        data, name=image
    )


_SUPPORTED = ("image/png", "image/jpeg", "image/webp", "image/gif")


def _media_type_from_suffix(suffix: str) -> str | None:
    """Map a supported file extension to its media type, else None."""
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
    }.get(suffix.lower().lstrip("."))


def _sniff_media_type(data: bytes, *, name: str = "the image") -> str:
    """Media type from magic bytes: PNG, JPEG, WEBP or GIF, else an error."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    raise UnsupportedImageError(
        f"{name} is not a PNG, JPEG, WEBP or GIF image; "
        "render other formats (PDF, SVG, TIFF) to PNG first"
    )
