"""Tests for the ``critique_figure`` public API.

Two layers:
  * unit — mocks ``Agent.run`` so we exercise input normalisation
    (bytes / Path / URL) and output unwrapping without an LLM call.
  * integration — marked ``@pytest.mark.ollama`` (deselected by
    default) hits a real local model on a committed broken-bar
    fixture and asserts the calibration property: any candidate
    vision model MUST flag ``label_overlap=True``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from andamentum.vision_critique import FigureCritique, critique_figure
from andamentum.vision_critique.api import _normalise_image, _sniff_media_type


FIXTURES = Path(__file__).parent / "fixtures"
BROKEN_BAR = FIXTURES / "broken_bar.png"


# ── unit: input normalisation ──────────────────────────────────────────


async def test_normalise_image_bytes_input() -> None:
    """Bytes pass through verbatim and the media type is sniffed from magic."""
    png_magic = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
    out, media = await _normalise_image(png_magic)
    assert out == png_magic
    assert media == "image/png"


async def test_normalise_image_path_input(tmp_path: Path) -> None:
    """A Path gets read and the media type comes from its suffix."""
    fpath = tmp_path / "x.jpg"
    fpath.write_bytes(b"\xff\xd8\xff\xe0fake")
    out, media = await _normalise_image(fpath)
    assert out == b"\xff\xd8\xff\xe0fake"
    assert media == "image/jpeg"


async def test_normalise_image_str_path(tmp_path: Path) -> None:
    """A str that's not a URL is treated as a local path."""
    fpath = tmp_path / "y.png"
    fpath.write_bytes(b"data")
    out, media = await _normalise_image(str(fpath))
    assert out == b"data"
    assert media == "image/png"


def test_sniff_media_type_jpeg() -> None:
    assert _sniff_media_type(b"\xff\xd8\xff\xe0\x00\x10") == "image/jpeg"


def test_sniff_media_type_unknown_defaults_png() -> None:
    assert _sniff_media_type(b"some-random-bytes-no-magic") == "image/png"


# ── unit: critique_figure end-to-end with mocked Agent.run ─────────────


async def test_critique_figure_returns_parsed_schema() -> None:
    """End-to-end: agent output is unwrapped and returned as the schema."""
    fake_critique = FigureCritique(
        label_overlap=True,
        labels_legible=False,
        legend_blocks_data=False,
        aspect_ratio_issue="ok",
        suggested_fixes=["rotate_x_labels"],
        confidence=0.95,
        one_line_summary="x labels are mush",
    )

    fake_result = type("FakeResult", (), {"output": fake_critique})()

    with patch(
        "andamentum.vision_critique.api.Agent",
        autospec=True,
    ) as mock_agent_cls:
        mock_agent = mock_agent_cls.return_value
        mock_agent.run = AsyncMock(return_value=fake_result)

        out = await critique_figure(
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 8,
            model="ollama:fake-model",
        )

    assert isinstance(out, FigureCritique)
    assert out.label_overlap is True
    assert out.suggested_fixes == ["rotate_x_labels"]


async def test_critique_figure_passes_extra_context() -> None:
    """``extra_context`` lands in the prompt sent to the agent."""
    fake_critique = FigureCritique(
        label_overlap=False,
        labels_legible=True,
        legend_blocks_data=False,
        aspect_ratio_issue="ok",
        suggested_fixes=["no_change_needed"],
        confidence=0.9,
        one_line_summary="figure is fine",
    )
    fake_result = type("FakeResult", (), {"output": fake_critique})()

    with patch(
        "andamentum.vision_critique.api.Agent",
        autospec=True,
    ) as mock_agent_cls:
        mock_agent = mock_agent_cls.return_value
        mock_agent.run = AsyncMock(return_value=fake_result)

        await critique_figure(
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 8,
            model="ollama:fake-model",
            extra_context="manuscript panel — Cell format",
        )

    sent = mock_agent.run.call_args.args[0]
    prompt_text = sent[0]
    assert "manuscript panel — Cell format" in prompt_text


# ── unit: schema convenience ──────────────────────────────────────────


def test_has_issues_true_on_label_overlap() -> None:
    c = FigureCritique(
        label_overlap=True,
        labels_legible=True,
        legend_blocks_data=False,
        aspect_ratio_issue="ok",
        suggested_fixes=["no_change_needed"],
        confidence=0.9,
        one_line_summary="overlap",
    )
    assert c.has_issues is True


def test_has_issues_false_on_clean_figure() -> None:
    c = FigureCritique(
        label_overlap=False,
        labels_legible=True,
        legend_blocks_data=False,
        aspect_ratio_issue="ok",
        suggested_fixes=["no_change_needed"],
        confidence=0.95,
        one_line_summary="figure is fine",
    )
    assert c.has_issues is False


# ── integration: real model on calibration fixture ────────────────────


@pytest.mark.ollama
async def test_critique_flags_broken_bar_chart() -> None:
    """The committed broken-bar fixture MUST be flagged by any candidate model.

    This is the calibration test from the architecture work — gemma4:e2b
    fails it (confidently says "fine"); gemma4:e4b-it-q4_K_M passes it
    (flags label_overlap=True). Any new default vision model must pass
    this before being recommended.
    """
    critique = await critique_figure(
        BROKEN_BAR,
        model="ollama:gemma4:e4b-it-q4_K_M",
    )
    assert critique.label_overlap is True, (
        f"vision model failed calibration: {critique.model_dump_json(indent=2)}"
    )
    assert critique.labels_legible is False
    assert critique.has_issues is True
