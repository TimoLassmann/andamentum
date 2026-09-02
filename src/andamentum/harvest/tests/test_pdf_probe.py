"""Tests for the PDF text-layer probe.

The probe decides one thing in accurate mode: whether the VLM should trust
the PDF's own text or read the page itself. Getting it wrong is expensive in
both directions — a wrong True on a scan returns almost nothing, a wrong
False on a paper hands body text to a 258M model that can invent words.

Fixtures are generated with matplotlib (already a core dependency) rather
than checked in, so the bytes are real PDFs and no binary lives in the repo.
"""

from __future__ import annotations

from io import BytesIO

import pytest

from andamentum.harvest.pdf_probe import has_text_layer

#: Comfortably above the 100-chars-per-page threshold.
_BODY_TEXT = (
    "The probe samples the leading pages of the document and counts the "
    "characters that the text layer yields. A typeset paper runs to several "
    "thousand characters on every page, so the threshold has a wide margin."
)


def _pdf_with_text(*, pages: int = 1, text: str = _BODY_TEXT) -> bytes:
    """A born-digital PDF: real glyphs, real text layer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # TrueType keeps the text extractable rather than rendering it as paths.
    matplotlib.rcParams["pdf.fonttype"] = 42

    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        for _ in range(pages):
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.1, 0.5, text, wrap=True, fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)
    return buffer.getvalue()


def _pdf_without_text() -> bytes:
    """Stands in for a scan: one image, no text layer at all."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    buffer = BytesIO()
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(np.tile(np.array([[0.2, 0.8], [0.8, 0.2]]), (32, 32)), cmap="gray")
    ax.axis("off")
    fig.savefig(buffer, format="pdf")
    plt.close(fig)
    return buffer.getvalue()


# ---------- the two real cases ---------------------------------------------


def test_born_digital_pdf_has_a_text_layer():
    assert has_text_layer(_pdf_with_text()) is True


def test_image_only_pdf_has_no_text_layer():
    assert has_text_layer(_pdf_without_text()) is False


def test_multi_page_document_is_judged_on_the_sample():
    assert has_text_layer(_pdf_with_text(pages=7)) is True


# ---------- thresholds -----------------------------------------------------

def test_sparse_text_falls_below_the_threshold():
    """A title page with a dozen characters is not a usable text layer."""
    assert has_text_layer(_pdf_with_text(text="Title")) is False


def test_threshold_is_adjustable():
    sparse = _pdf_with_text(text="Title")
    assert has_text_layer(sparse, min_chars_per_page=3) is True


def test_probe_pages_limits_the_sample():
    """A one-page probe still answers on a seven-page document."""
    assert has_text_layer(_pdf_with_text(pages=7), probe_pages=1) is True


# ---------- failure is not an exception ------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"not a pdf at all", id="garbage"),
        pytest.param(b"%PDF-1.4\nbut truncated", id="truncated"),
    ],
)
def test_unreadable_input_answers_false_without_raising(data):
    """The probe only picks between two working configurations.

    It must never be the thing that fails a run, and False is the branch
    that still produces output.
    """
    assert has_text_layer(data) is False


def test_probe_logs_when_it_fails(caplog, monkeypatch):
    """A silent fallback would hide a broken probe. It has to say so.

    `harvest.cli._configure_logging` turns propagation off on this logger, and
    a CLI test earlier in the session leaves it that way, so restore it here
    rather than depend on the order tests happen to run in.
    """
    import logging

    monkeypatch.setattr(logging.getLogger("andamentum.harvest"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="andamentum.harvest"):
        has_text_layer(b"not a pdf at all")
    assert any("text-layer probe" in record.message for record in caplog.records)
