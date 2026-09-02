"""Deterministic check for whether a PDF carries a usable text layer.

``mode="accurate"`` routes PDFs through docling's VLM pipeline. When the PDF
already has a text layer, the correct setting is ``force_backend_text=True``:
normal text regions come from deterministic extraction and only figures and
tables go through the model. A 258M-parameter model reading body text token
by token will occasionally invent a word, and in a paper nobody would notice.

When the PDF is a scan there is no text layer to trust, so the model has to
read the page itself. This module decides which of the two a document is.

The probe uses pypdfium2 — docling's own PDF backend — so it costs a few
milliseconds and does not import docling (or torch) to answer the question.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("andamentum.harvest")

#: How many leading pages to sample. A title page can be sparse, so one page
#: is not enough evidence either way.
PROBE_PAGES = 5

#: Mean characters per sampled page required to call the text layer usable.
#: A typeset paper runs 2000-4000; a scan with no OCR runs 0. The threshold
#: sits far below the first and far above the second on purpose.
MIN_CHARS_PER_PAGE = 100


def has_text_layer(
    data: bytes,
    *,
    probe_pages: int = PROBE_PAGES,
    min_chars_per_page: int = MIN_CHARS_PER_PAGE,
) -> bool:
    """Report whether a PDF's own text layer is worth trusting.

    Parameters
    ----------
    data:
        Raw PDF bytes.
    probe_pages:
        Number of leading pages to sample.
    min_chars_per_page:
        Mean characters per sampled page required to answer True.

    Returns
    -------
    bool
        True when the sampled pages carry enough extractable text. False for
        a scan, an empty document, or a PDF that pypdfium2 cannot open — in
        every False case the caller lets the model read the page, which is
        the answer that still produces output.
    """
    import pypdfium2 as pdfium

    document = None
    try:
        document = pdfium.PdfDocument(data)
        page_count = len(document)
        sampled = min(page_count, probe_pages)
        if sampled == 0:
            logger.warning("pdf text-layer probe: document has no pages")
            return False

        total_chars = 0
        for index in range(sampled):
            page = document[index]
            textpage = page.get_textpage()
            try:
                total_chars += len(textpage.get_text_bounded().strip())
            finally:
                textpage.close()
                page.close()

        verdict = total_chars >= min_chars_per_page * sampled
        logger.debug(
            "pdf text-layer probe: %d chars over %d/%d pages -> %s",
            total_chars,
            sampled,
            page_count,
            "text layer" if verdict else "no text layer",
        )
        return verdict
    except Exception:
        # Not an extraction failure — the probe only chooses between two
        # working configurations. Log it loudly and take the branch that
        # always produces output.
        logger.warning(
            "pdf text-layer probe failed; assuming no text layer", exc_info=True
        )
        return False
    finally:
        if document is not None:
            document.close()
