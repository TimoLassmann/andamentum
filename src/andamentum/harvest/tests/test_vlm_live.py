"""End-to-end tests that actually run a model. Deselected by default.

Run them with::

    uv run pytest -m vlm src/andamentum/harvest/tests/test_vlm_live.py -s

These are the only tests in the module that import docling, download model
weights, and hold a gigabyte or more of memory. Everything else stubs the
backend. Keep it that way: the default suite must stay cheap enough to run
while other models are loaded on the same machine.
"""

from __future__ import annotations

import importlib.util
from io import BytesIO

import pytest

from andamentum.harvest import extract_from_bytes

pytestmark = pytest.mark.vlm


# ---------- options construction (imports docling, runs no model) ----------


def test_ollama_spec_builds_api_options_pinned_to_one_request():
    """An Ollama figure model must never fan out — it serialises anyway."""
    from andamentum.harvest.backends import docling_backend as backend

    options = backend._picture_description_options("ollama:minicpm-v4.6:1b")
    assert options.kind == "api"
    assert options.params["model"] == "minicpm-v4.6:1b"
    assert options.concurrency == 1
    assert str(options.url) == backend._OLLAMA_URL
    assert options.prompt == backend._FIGURE_PROMPT


def test_preset_spec_is_asked_the_same_question_as_the_api_path():
    """Holding the prompt constant is what makes a model comparison honest."""
    from andamentum.harvest.backends import docling_backend as backend

    options = backend._picture_description_options("smolvlm")
    assert options.kind != "api"
    assert options.model_spec.prompt == backend._FIGURE_PROMPT


def test_remote_services_are_enabled_only_for_the_ollama_path(monkeypatch):
    """Turning it on unconditionally would widen what the pipeline may do."""
    from andamentum.harvest.backends import docling_backend as backend

    monkeypatch.setattr(backend, "has_text_layer", lambda _data: True)
    local = backend._accurate_options(b"%PDF-1.4", figure_model="smolvlm")
    served = backend._accurate_options(
        b"%PDF-1.4", figure_model="ollama:minicpm-v4.6:1b"
    )
    assert local.enable_remote_services is False
    assert served.enable_remote_services is True


# ---------- fast mode must not change what existing callers got ------------
#
# These import docling (hence the module's marker) but run no model.


def test_fast_mode_keeps_doclings_own_defaults():
    """Regression guard.

    Every caller that passes no `mode` lands here, so this path has to behave
    as calling docling with no options did. An earlier revision set
    do_ocr=False and TableFormerMode.FAST, which silently broke scanned PDFs
    and downgraded every table in document_store, deep_research and whetstone.
    """
    from docling.datamodel.pipeline_options import TableFormerMode

    from andamentum.harvest.backends import docling_backend as backend

    options = backend._fast_options(ocr=True)
    assert options.do_ocr is True
    assert options.do_table_structure is True
    # `.mode` is declared on the concrete TableStructureOptions, not on the
    # base type the field is annotated with, so read it dynamically.
    assert getattr(options.table_structure_options, "mode") is TableFormerMode.ACCURATE


def test_ocr_false_is_the_opt_in_speed_path():
    from andamentum.harvest.backends import docling_backend as backend

    options = backend._fast_options(ocr=False)
    assert options.do_ocr is False


# ---------- end-to-end -----------------------------------------------------


def _paper_like_pdf() -> bytes:
    """A page with a heading, body text and a chart — the shape we care about."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42

    buffer = BytesIO()
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.1, 0.93, "Growth of the observed population", fontsize=16)
    fig.text(
        0.1,
        0.82,
        "We measured the population at four time points and found a rise at "
        "every step. The chart below plots the counts against the sampling "
        "week. All values are means of three replicates.",
        fontsize=10,
        wrap=True,
    )
    axes = fig.add_axes((0.15, 0.35, 0.7, 0.35))
    axes.bar(["week 1", "week 2", "week 3", "week 4"], [4, 9, 17, 28])
    axes.set_ylabel("cells per field")
    axes.set_xlabel("sampling week")
    fig.savefig(buffer, format="pdf")
    plt.close(fig)
    return buffer.getvalue()


@pytest.mark.skipif(
    importlib.util.find_spec("mlx_vlm") is None
    and importlib.util.find_spec("transformers") is None,
    reason="needs the optional vlm extra: uv sync --extra vlm",
)
async def test_accurate_mode_reads_a_pdf_and_describes_its_figure():
    markdown = await extract_from_bytes(
        _paper_like_pdf(), format="pdf", source_url="paper.pdf", mode="accurate"
    )
    print("\n----- accurate mode output -----\n" + markdown)
    # The body text survives.
    assert "population" in markdown.lower()
    # The figure got a written description, marked as generated rather than
    # blended into the author's own prose.
    assert "<annotation" in markdown


async def test_fast_mode_reads_the_same_pdf_without_a_model():
    """The comparison baseline.

    It needs no optional extra, but it still loads docling and torch, so it
    carries the module's `vlm` marker like the rest of the file.
    """
    markdown = await extract_from_bytes(
        _paper_like_pdf(), format="pdf", source_url="paper.pdf"
    )
    assert "population" in markdown.lower()


async def test_fast_mode_with_ocr_off_fails_loudly_on_a_scan():
    """A PDF with no text layer must name the way out, not return a blank page.

    Only with ocr=False: the default path runs OCR, exactly as before this
    change, so a scan still extracts for every existing caller.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from andamentum.harvest.errors import ExtractionError

    buffer = BytesIO()
    fig = plt.figure(figsize=(8.5, 11))
    axes = fig.add_axes((0, 0, 1, 1))
    axes.imshow(np.tile(np.array([[0.2, 0.8], [0.8, 0.2]]), (32, 32)), cmap="gray")
    axes.axis("off")
    fig.savefig(buffer, format="pdf")
    plt.close(fig)

    with pytest.raises(ExtractionError) as exc_info:
        await extract_from_bytes(
            buffer.getvalue(), format="pdf", mode="fast", ocr=False
        )
    message = str(exc_info.value)
    assert "--no-ocr" in message
    assert "accurate" in message
