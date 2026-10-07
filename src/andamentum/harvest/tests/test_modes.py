"""Tests for the fast / accurate extraction modes.

Nothing here imports docling. The orchestration tests stub the backend and
assert what mode reached it; the backend's own pure helpers are tested
directly. The tests that actually run a model live in `test_vlm_live.py`
behind the `vlm` marker.
"""

from __future__ import annotations

import importlib.util

import pytest

from andamentum.harvest import api as api_mod
from andamentum.harvest import extract, extract_from_bytes
from andamentum.harvest.backends import docling_backend as backend
from andamentum.harvest.errors import ExtractionError
from andamentum.harvest.fetch import Fetched


# ---------- helpers --------------------------------------------------------


def _make_fetched(
    *, fmt: str, data: bytes = b"", url: str = "https://x.com"
) -> Fetched:
    return Fetched(data=data, format=fmt, source_url=url)  # type: ignore[arg-type]


@pytest.fixture
def spy_docling(monkeypatch):
    """Replace the docling backend and record the mode + format it was given."""
    calls: list[dict[str, str]] = []

    async def fake_docling(data, *, source_url, fmt="html", mode="fast", **_kwargs):
        calls.append({"fmt": fmt, "mode": mode})
        return "## Layout\n\nbody\n"

    async def fake_traf(data, source_url):
        return "## Article\n\nbody\n"

    monkeypatch.setattr(api_mod, "extract_with_docling", fake_docling)
    monkeypatch.setattr(api_mod, "extract_with_trafilatura", fake_traf)
    return calls


@pytest.fixture
def patched_resolve(monkeypatch):
    holder: dict[str, Fetched] = {}

    async def fake_resolve(source, **_kwargs):
        return holder["fetched"]

    monkeypatch.setattr(api_mod, "resolve", fake_resolve)

    def _set(fetched: Fetched) -> Fetched:
        holder["fetched"] = fetched
        return fetched

    return _set


# ---------- mode reaches the backend ---------------------------------------


async def test_default_mode_is_fast(patched_resolve, spy_docling):
    patched_resolve(_make_fetched(fmt="pdf", data=b"%PDF-1.4"))
    await extract("https://x.com/p.pdf")
    assert spy_docling == [{"fmt": "pdf", "mode": "fast"}]


async def test_accurate_mode_reaches_backend(patched_resolve, spy_docling):
    patched_resolve(_make_fetched(fmt="pdf", data=b"%PDF-1.4"))
    await extract("https://x.com/p.pdf", mode="accurate")
    assert spy_docling == [{"fmt": "pdf", "mode": "accurate"}]


@pytest.mark.parametrize("fmt", ["pdf", "docx", "pptx"])
async def test_mode_reaches_every_docling_format(patched_resolve, spy_docling, fmt):
    patched_resolve(_make_fetched(fmt=fmt))
    await extract(f"https://x.com/p.{fmt}", mode="accurate")
    assert spy_docling == [{"fmt": fmt, "mode": "accurate"}]


async def test_extract_from_bytes_carries_mode(spy_docling):
    await extract_from_bytes(b"%PDF-1.4", format="pdf", mode="accurate")
    assert spy_docling == [{"fmt": "pdf", "mode": "accurate"}]


async def test_html_not_article_carries_mode(patched_resolve, spy_docling):
    html = b'<html><head><script type="application/ld+json">{"@type":"WebPage"}</script></head></html>'
    patched_resolve(_make_fetched(fmt="html", data=html))
    await extract("https://x.com/index", mode="accurate")
    assert spy_docling == [{"fmt": "html", "mode": "accurate"}]


async def test_html_race_carries_mode(patched_resolve, spy_docling):
    patched_resolve(
        _make_fetched(fmt="html", data=b"<html><body><p>x</p></body></html>")
    )
    await extract("https://x.com/?", mode="accurate")
    assert spy_docling == [{"fmt": "html", "mode": "accurate"}]


async def test_markdown_ignores_mode(patched_resolve, spy_docling):
    """A markdown source never reaches docling, whatever the mode."""
    patched_resolve(_make_fetched(fmt="markdown", data=b"## hi\n\nbody\n"))
    md = await extract("https://x.com/p.md", mode="accurate")
    assert "## hi" in md
    assert spy_docling == []


# ---------- the empty-output guard -----------------------------------------


def test_has_text_rejects_placeholder_only_output():
    """A scan read without OCR comes back as image placeholders and nothing else."""
    assert backend._has_text("## Heading\n\nreal words\n") is True
    assert backend._has_text("<!-- image -->\n\n<!-- image -->\n") is False
    assert backend._has_text("   \n\n  ") is False
    assert backend._has_text("") is False


def test_empty_pdf_message_names_the_disabled_ocr_first():
    """With --no-ocr the likely cause is the flag, so name that before the mode."""
    message = backend._empty_message(fmt="pdf", mode="fast", ocr=False)
    assert "--no-ocr" in message
    assert "no text layer" in message
    assert "accurate" in message


def test_empty_pdf_message_with_ocr_on_points_at_accurate_mode():
    """OCR ran and still found nothing, so the flag is not the problem."""
    message = backend._empty_message(fmt="pdf", mode="fast", ocr=True)
    assert "even with OCR enabled" in message
    assert "--mode accurate" in message
    assert "--no-ocr" not in message


def test_empty_message_stays_generic_for_other_cases():
    assert backend._empty_message(fmt="docx", mode="fast", ocr=True) == (
        "docling returned empty markdown"
    )
    assert backend._empty_message(fmt="pdf", mode="accurate", ocr=True) == (
        "docling returned empty markdown"
    )


# ---------- the missing-extra guard ----------------------------------------


def test_require_vlm_extra_names_the_missing_package(monkeypatch):
    """Absent dependencies must fail before docling starts loading weights."""
    monkeypatch.setattr(backend, "_is_arm64", lambda: True)
    monkeypatch.setattr(backend.sys, "platform", "darwin")
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name in ("mlx_vlm", "transformers") else object(),
    )
    with pytest.raises(ImportError) as exc_info:
        backend._require_vlm_extra()
    message = str(exc_info.value)
    assert "mlx-vlm" in message
    assert "transformers" in message
    assert "uv sync --extra vlm" in message


def test_require_vlm_extra_ignores_mlx_off_apple_silicon(monkeypatch):
    """mlx-vlm does not exist for non-Apple hardware, so its absence is not a fault."""
    monkeypatch.setattr(backend, "_is_arm64", lambda: False)
    monkeypatch.setattr(backend.sys, "platform", "linux")
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name == "mlx_vlm" else object(),
    )
    backend._require_vlm_extra()  # must not raise


def test_require_vlm_extra_passes_when_everything_present(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    backend._require_vlm_extra()  # must not raise


# ---------- figure_model ---------------------------------------------------


@pytest.fixture
def spy_figure_model(monkeypatch):
    """Record the figure_model the backend was handed."""
    seen: list[str] = []

    async def fake_docling(data, *, source_url, fmt="html", mode="fast", **kwargs):
        seen.append(kwargs.get("figure_model", "<absent>"))
        return "## Layout\n\nbody\n"

    monkeypatch.setattr(api_mod, "extract_with_docling", fake_docling)
    return seen


async def test_figure_model_defaults_to_the_measured_best(
    patched_resolve, spy_figure_model
):
    patched_resolve(_make_fetched(fmt="pdf", data=b"%PDF-1.4"))
    await extract("https://x.com/p.pdf", mode="accurate")
    assert spy_figure_model == [backend.DEFAULT_FIGURE_MODEL]


def test_the_default_is_an_ollama_model():
    """Pinned deliberately: the default is NOT self-contained.

    It scored 8/8 against smolvlm's 4/8 on a figure with known values, which
    is why accurate mode uses it. The cost is that a machine with no Ollama
    cannot run accurate mode without passing `figure_model`, so this fact is
    asserted rather than left as a surprise.
    """
    assert backend.DEFAULT_FIGURE_MODEL.startswith(backend._OLLAMA_PREFIX)


async def test_figure_model_reaches_the_backend(patched_resolve, spy_figure_model):
    patched_resolve(_make_fetched(fmt="pdf", data=b"%PDF-1.4"))
    await extract(
        "https://x.com/p.pdf", mode="accurate", figure_model="ollama:minicpm-v4.6:1b"
    )
    assert spy_figure_model == ["ollama:minicpm-v4.6:1b"]


async def test_extract_from_bytes_carries_figure_model(spy_figure_model):
    await extract_from_bytes(
        b"%PDF-1.4", format="pdf", mode="accurate", figure_model="qwen"
    )
    assert spy_figure_model == ["qwen"]


def test_ollama_spec_with_no_model_name_is_rejected():
    """'ollama:' alone would otherwise reach Ollama as an empty model id.

    Validation happens before the docling import, so this stays cheap — the
    assertion is as much about that ordering as about the message.
    """
    with pytest.raises(ValueError, match="names no model"):
        backend._picture_description_options("ollama:")
    with pytest.raises(ValueError, match="names no model"):
        backend._picture_description_options("ollama:   ")


# ---------- the Ollama preflight -------------------------------------------


def _unreachable(*_args, **_kwargs):
    raise OSError("connection refused")


def test_unreachable_ollama_names_the_remedy(monkeypatch):
    """The default depends on an external service, so its absence must be clear."""
    monkeypatch.setattr(backend.urllib.request, "urlopen", _unreachable)
    with pytest.raises(backend.OllamaUnavailableError) as exc_info:
        backend._check_ollama_model("qwen3.8")
    message = str(exc_info.value)
    assert "ollama serve" in message
    assert "smolvlm" in message


def test_missing_model_is_an_error_not_a_substitution(monkeypatch):
    """Quietly using another model would change the descriptions unannounced."""
    monkeypatch.setattr(
        backend, "_installed_ollama_models", lambda: {"llama3:8b", "minicpm-v4.6:1b"}
    )
    with pytest.raises(backend.OllamaUnavailableError) as exc_info:
        backend._check_ollama_model("qwen3.8")
    message = str(exc_info.value)
    assert "ollama pull qwen3.8" in message
    assert "minicpm-v4.6:1b" in message  # reports what IS there


def test_installed_model_passes_the_preflight(monkeypatch):
    # Ollama lists a model pulled as "qwen3.8" under its implied tag.
    monkeypatch.setattr(backend, "_installed_ollama_models", lambda: {"qwen3.8:latest"})
    backend._check_ollama_model("qwen3.8")  # must not raise
    backend._check_ollama_model("qwen3.8:latest")  # must not raise


def test_another_tag_of_the_same_model_is_not_installed(monkeypatch):
    """A bare name means ':latest' and nothing else; never a different build."""
    monkeypatch.setattr(backend, "_installed_ollama_models", lambda: {"qwen3.8:27b-q4"})
    with pytest.raises(backend.OllamaUnavailableError):
        backend._check_ollama_model("qwen3.8")


def test_ollama_unavailable_is_an_extraction_error():
    """Callers already matching ExtractionError keep working."""
    assert issubclass(backend.OllamaUnavailableError, ExtractionError)


# ---------- error typing ---------------------------------------------------


async def test_extraction_error_survives_the_mode_parameter(
    patched_resolve, monkeypatch
):
    """The typed-failure contract is unchanged by the new parameter."""

    async def boom(data, *, source_url, fmt="html", mode="fast", **_kwargs):
        raise ExtractionError("docling gave up", attempted=[f"docling[{mode}]"])

    monkeypatch.setattr(api_mod, "extract_with_docling", boom)
    patched_resolve(_make_fetched(fmt="pdf", data=b"%PDF-1.4"))
    with pytest.raises(ExtractionError) as exc_info:
        await extract("https://x.com/p.pdf", mode="accurate")
    assert "docling[accurate]" in str(exc_info.value)
