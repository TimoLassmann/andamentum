"""PDF / HTML / DOCX / PPTX → markdown via Docling, in one of two modes.

Docling preserves layout and emits a typed `DoclingDocument` tree which
exports cleanly to markdown with `##` headings. Two configurations:

``fast`` (the default)
    Docling's standard pipeline on its own defaults — OCR on, table
    structure ACCURATE — so this mode behaves exactly as calling docling
    with no options did. (That pipeline already runs its stages threaded;
    the only thing tuned here is the stage batch size, 4 -> 8.)

    Pass ``ocr=False`` for the born-digital speed path: for a PDF that
    already has a text layer, OCR is the expensive stage and buys nothing.
    A scan read with ``ocr=False`` yields no text, and the backend raises
    `ExtractionError` naming the remedy rather than returning a blank page.

``accurate``
    Docling's VLM pipeline. One model (granite-docling-258M) replaces the
    whole layout + OCR + TableFormer stack and emits DocTags, which
    round-trips into a `DoclingDocument` properly rather than being
    reconstructed from markdown. Figures additionally get a written
    description from whichever model `figure_model` names, marked in the
    output so a generated description never reads as the author's own words.
    There is no default model: the caller names it. ``ollama:<model>`` reaches
    a vision model served by a local Ollama; see the comment above
    `_OLLAMA_PREFIX` for the measured accuracy of the alternatives.

    ``force_backend_text`` is decided per document by `pdf_probe`: a PDF
    with a text layer keeps its deterministic text and sends only figures
    and tables to the model; a scan is read by the model end to end.

    PDF only. The VLM pipeline works from page images, and docling's DOCX
    / PPTX / HTML backends are not paginated, so those formats run the
    standard pipeline in either mode.

Cost: on Apple silicon the VLM path is roughly an order of magnitude
slower per page than ``fast``. It is not the thing to point at a directory
of 500 PDFs.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sys
import urllib.error
import urllib.request
from io import BytesIO
from typing import TYPE_CHECKING, Literal

from ..errors import ExtractionError, FigureModelRequired, OllamaUnavailableError
from ..pdf_probe import has_text_layer

if TYPE_CHECKING:
    from docling.datamodel.pipeline_options import (
        ThreadedPdfPipelineOptions,
        VlmPipelineOptions,
    )

DoclingFormat = Literal["pdf", "html", "docx", "pptx"]
ExtractionMode = Literal["fast", "accurate"]

logger = logging.getLogger("andamentum.harvest")

#: Preset ids from docling's own registries.
_VLM_CONVERT_PRESET = "granite_docling"

#: The model that writes the figure descriptions has no default: the caller
#: names it (``figure_model``; ``--figure-model`` on the command line). Either a docling
#: picture-description preset id (`smolvlm`, `granite_vision`, `qwen`,
#: `pixtral` — weights are downloaded and run in-process) or `ollama:<model>`
#: to reach a vision model already served by a local Ollama.
#:
#: Measured on a bar chart with known values, same prompt for each, scoring
#: eight facts a correct description must recover:
#:
#:     ollama:qwen3.8           8/8   27.3 s   1.8 GB in this process
#:     ollama:minicpm-v4.6:1b   6/8   27.7 s   1.8 GB
#:     smolvlm (256M, bundled)  4/8   12.0 s   2.7 GB
#:
#: The 27B model is no slower than the 1B one — the description is a small
#: part of the wall time, most of which is docling converting the page — and
#: it costs this process LESS memory than the bundled 256M preset, because
#: the weights live in Ollama rather than here.
#:
#: An ``ollama:`` model is not self-contained. It must already be pulled into
#: a running local Ollama, so a machine without it fails; `_check_ollama_model`
#: makes that failure say exactly what is wrong and what to do. A caller
#: wanting no external service passes a preset id instead — at the accuracy
#: shown above.

_OLLAMA_PREFIX = "ollama:"

#: Ollama's OpenAI-compatible chat endpoint. Not configurable by env var, by
#: house rule; pass a different `figure_model` or extend this function.
_OLLAMA_URL = "http://localhost:11434/v1/chat/completions"

#: Where the preflight asks which models are installed.
_OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

#: A 27B model on Ollama takes far longer per figure than docling's 20s default.
_OLLAMA_TIMEOUT_SECONDS = 300.0

#: One prompt for every figure model, so swapping the model compares the model
#: and not the prompt. docling's presets each ship their own wording
#: ("Describe this image in a few sentences" for SmolVLM, "What is shown in
#: this image?" for Granite), which would confound any comparison between them.
_FIGURE_PROMPT = (
    "Describe this figure. State what kind of chart it is, what each axis "
    "measures, and what the data shows."
)

#: Stage batch sizes. Docling's own default is 4; this is the only setting
#: `fast` changes from stock, and it affects throughput, never output.
_BATCH_SIZE = 8

#: What `mode="fast"` leaves behind on a page it could not read. Docling
#: emits a placeholder per picture, so a scanned page comes back as markdown
#: that is non-empty but carries no text.
_IMAGE_PLACEHOLDER = "<!-- image -->"


async def extract(
    data: bytes,
    *,
    source_url: str,
    fmt: DoclingFormat = "html",
    mode: ExtractionMode = "fast",
    ocr: bool = True,
    figure_model: str | None = None,
) -> str:
    """Run Docling on raw bytes; return markdown.

    `fmt` tells Docling which input backend to use. The orchestrator passes
    in the format detected upstream so we don't need to re-sniff here.
    `mode` selects the pipeline — see the module docstring. `ocr` applies to
    fast mode only. `figure_model` names the model that describes figures in
    accurate mode and is required there (`FigureModelRequired` otherwise).
    """
    if mode == "accurate" and figure_model is None:
        raise FigureModelRequired(
            'mode="accurate" needs figure_model: name the model that describes '
            "figures (CLI: --figure-model), e.g. 'smolvlm' or 'ollama:<model>'"
        )
    return await asyncio.to_thread(
        _extract_sync, data, source_url, fmt, mode, ocr, figure_model or ""
    )


def _extract_sync(
    data: bytes,
    source_url: str,
    fmt: DoclingFormat,
    mode: ExtractionMode,
    ocr: bool,
    figure_model: str,
) -> str:
    # Imports are deferred so this module is cheap to import at app startup.
    from docling.datamodel.base_models import DocumentStream, InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    fmt_map = {
        "pdf": InputFormat.PDF,
        "html": InputFormat.HTML,
        "docx": InputFormat.DOCX,
        "pptx": InputFormat.PPTX,
    }
    if fmt not in fmt_map:
        raise ExtractionError(
            f"docling backend does not support format {fmt!r}",
            attempted=["docling"],
        )

    input_format = fmt_map[fmt]
    use_vlm = mode == "accurate" and fmt == "pdf"
    if mode == "accurate" and fmt != "pdf":
        logger.info(
            "mode='accurate' has no effect on %s: docling's %s backend is not "
            "paginated, so the VLM pipeline has no page images to read. "
            "Running the standard pipeline.",
            fmt,
            fmt,
        )

    if use_vlm:
        # An Ollama-served figure model runs outside this process, so the
        # local inference stack is only needed for the page-reading model.
        _require_vlm_extra()
        converter = DocumentConverter(
            allowed_formats=[input_format],
            format_options={
                input_format: PdfFormatOption(
                    pipeline_cls=_describing_vlm_pipeline(),
                    pipeline_options=_accurate_options(data, figure_model=figure_model),
                )
            },
        )
    elif input_format == InputFormat.PDF:
        converter = DocumentConverter(
            allowed_formats=[input_format],
            format_options={
                input_format: PdfFormatOption(pipeline_options=_fast_options(ocr=ocr))
            },
        )
    else:
        # DOCX / PPTX / HTML take no pipeline options — there is no layout,
        # OCR or table-structure stage to configure.
        converter = DocumentConverter(allowed_formats=[input_format])

    # Give docling a stable filename hint so it picks the right backend.
    name = f"document.{fmt}"
    stream = DocumentStream(name=name, stream=BytesIO(data))
    try:
        result = converter.convert(stream)
    except Exception as exc:  # docling raises a bag of types
        raise ExtractionError(
            f"docling conversion failed: {exc}",
            attempted=[f"docling[{mode}]"],
            diagnostics={
                "docling": f"{type(exc).__name__}: {exc}",
                "source": source_url or "<bytes>",
            },
        ) from exc

    if use_vlm:
        marked = _mark_figure_descriptions(result.document)
        logger.info("marked %d model-written figure description(s)", marked)

    # `include_annotations` defaults to True, so figure descriptions written
    # by the picture-description stage land beside the image placeholder.
    # `mark_annotations` covers docling's deprecated `item.annotations` path;
    # the live path writes to `item.meta`, which `_mark_figure_descriptions`
    # has already marked above. The two are mutually exclusive — docling-core
    # skips the legacy branch whenever `item.meta` is set — so nothing is
    # marked twice.
    md = result.document.export_to_markdown(mark_annotations=use_vlm)
    if not _has_text(md):
        raise ExtractionError(
            _empty_message(fmt=fmt, mode=mode, ocr=ocr),
            attempted=[f"docling[{mode}]"],
            diagnostics={
                "docling": f"empty output ({len(md)} chars)",
                "source": source_url or "<bytes>",
            },
        )
    return md


#: The wrapper docling-core itself uses for a marked annotation. Reused verbatim
#: so a consumer needs one vocabulary, not two.
ANNOTATION_OPEN = '<!--<annotation kind="description">-->'
ANNOTATION_CLOSE = "<!--<annotation/>-->"


def _mark_figure_descriptions(document) -> int:
    """Wrap every model-written figure description in an annotation marker.

    A description of a figure is written by SmolVLM, not by the author. Left
    bare it reads as one more paragraph of the document, and anything
    downstream — a reviewer, a claim extractor, a search index — would treat
    a model's guess about a chart as the author's own statement.

    docling-core's own `mark_annotations` export flag only reaches the
    deprecated `item.annotations` list; the picture-description stage writes
    to `item.meta.description` instead, and that path ignores the flag
    (`docling_core/transforms/serializer/common.py:733` returns False as soon
    as `item.meta` is set). Marking the stored text is what actually works.

    Returns the number of descriptions marked.
    """
    # Selected by the attribute rather than by `isinstance(item, PictureItem)`:
    # docling-core re-exports `PictureItem` only from a private module path,
    # and carrying a `meta.description` is the exact property being marked —
    # a description on any item is model-written and wants the same wrapper.
    marked = 0
    for item, _depth in document.iterate_items():
        description = getattr(getattr(item, "meta", None), "description", None)
        if description is None:
            continue
        text = getattr(description, "text", None)
        if not text or text.startswith(ANNOTATION_OPEN):
            continue
        description.text = f"{ANNOTATION_OPEN}{text}{ANNOTATION_CLOSE}"
        marked += 1
    return marked


def _has_text(markdown: str) -> bool:
    """Report whether the markdown carries anything beyond image placeholders.

    A scanned page read without OCR comes back as one placeholder per
    picture: non-empty as a string, empty as a document.
    """
    return bool(markdown.replace(_IMAGE_PLACEHOLDER, "").strip())


def _empty_message(*, fmt: DoclingFormat, mode: ExtractionMode, ocr: bool) -> str:
    if fmt == "pdf" and mode == "fast" and not ocr:
        return (
            "docling found no text in this PDF. It has no text layer, and "
            "OCR was disabled with ocr=False (CLI: --no-ocr). Drop that flag "
            'to let OCR read it, or use mode="accurate" to read the page '
            "with a model."
        )
    if fmt == "pdf" and mode == "fast":
        return (
            "docling found no text in this PDF even with OCR enabled. Try "
            'mode="accurate" (CLI: --mode accurate), which reads the page '
            "with a vision model instead of the OCR + layout stack."
        )
    return "docling returned empty markdown"


def _fast_options(*, ocr: bool) -> ThreadedPdfPipelineOptions:
    """Docling's standard pipeline on its stock settings.

    Everything here except the batch size matches what docling does with no
    options at all, deliberately: this mode is what every existing caller
    gets by default, so it must not quietly change their output.
    `TableFormerMode.ACCURATE` is docling's own default and stays — the FAST
    model is a quality downgrade nobody asked for. `ocr` is the one knob,
    and it defaults to docling's `do_ocr=True`.
    """
    from docling.datamodel.pipeline_options import (
        TableFormerMode,
        TableStructureOptions,
        ThreadedPdfPipelineOptions,
    )

    return ThreadedPdfPipelineOptions(
        do_ocr=ocr,
        do_table_structure=True,
        table_structure_options=TableStructureOptions(mode=TableFormerMode.ACCURATE),
        ocr_batch_size=_BATCH_SIZE,
        layout_batch_size=_BATCH_SIZE,
        table_batch_size=_BATCH_SIZE,
    )


def _picture_description_options(figure_model: str):
    """Build the picture-description stage options for one model spec.

    Two shapes, one parameter:

    ``ollama:<model>``
        Reach a vision model already served by a local Ollama over its
        OpenAI-compatible endpoint. Nothing is downloaded and nothing is
        loaded in-process. `concurrency` is pinned to 1 — Ollama serialises
        same-model calls anyway, and firing several at once only competes
        for the machine's memory.

    anything else
        A docling picture-description preset id, whose weights are fetched
        from HuggingFace and run in-process through the auto-selected engine
        (MLX on Apple silicon, Transformers elsewhere).
    """
    # Parse and validate BEFORE importing docling: the import pulls torch and
    # costs about a gigabyte, and a malformed spec should not pay for that.
    served_model: str | None = None
    if figure_model.startswith(_OLLAMA_PREFIX):
        served_model = figure_model[len(_OLLAMA_PREFIX) :].strip()
        if not served_model:
            raise ValueError(
                f"figure_model={figure_model!r} names no model after "
                f"{_OLLAMA_PREFIX!r}. Use e.g. 'ollama:minicpm-v4.6:1b'."
            )

    from pydantic import AnyUrl

    from docling.datamodel.pipeline_options import (
        PictureDescriptionApiOptions,
        PictureDescriptionVlmEngineOptions,
    )
    from docling.datamodel.vlm_engine_options import AutoInlineVlmEngineOptions

    if served_model is not None:
        _check_ollama_model(served_model)
        return PictureDescriptionApiOptions(
            url=AnyUrl(_OLLAMA_URL),
            params={"model": served_model},
            prompt=_FIGURE_PROMPT,
            timeout=_OLLAMA_TIMEOUT_SECONDS,
            concurrency=1,
        )

    options = PictureDescriptionVlmEngineOptions.from_preset(
        figure_model, engine_options=AutoInlineVlmEngineOptions()
    )
    # The preset's prompt lives on the model spec, not on the options, so it
    # cannot be passed through `from_preset`. Setting it here is what keeps
    # every figure model answering the same question.
    options.model_spec.prompt = _FIGURE_PROMPT
    return options


def _installed_ollama_models() -> set[str]:
    """Names of the models a local Ollama currently has, via its tags endpoint.

    Raises OSError / ValueError when Ollama is unreachable or answers with
    something that is not the expected JSON; the caller turns that into a
    message naming the remedy.
    """
    with urllib.request.urlopen(_OLLAMA_TAGS_URL, timeout=5) as response:
        payload = json.load(response)
    return {entry.get("name", "") for entry in payload.get("models", [])}


def _ollama_name(model: str) -> str:
    """Ollama's own reading of a model name: no tag means ``:latest``.

    ``ollama list`` reports ``qwen3.8:latest`` for a model pulled as
    ``qwen3.8``, so an exact string comparison would call an installed model
    missing.
    """
    return model if ":" in model else f"{model}:latest"


def _check_ollama_model(model: str) -> None:
    """Fail early and specifically when Ollama cannot serve `model`.

    An ``ollama:`` figure model has to be pulled into a running
    local Ollama, so the common failure here is environmental, not a bug.
    Without this check the run gets as far as docling posting an image and
    dies on a connection refusal or a 404 from inside the enrichment stage,
    with nothing pointing at the real cause.

    A reachable Ollama that lacks the model is a hard error, not a fallback:
    quietly substituting a different model would change what every figure
    description says without the caller ever knowing.
    """
    try:
        installed = _installed_ollama_models()
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise OllamaUnavailableError(
            f"figure_model={_OLLAMA_PREFIX + model!r} needs a local Ollama at "
            f"{_OLLAMA_TAGS_URL}, which is not answering ({exc}).\n"
            "Start it with 'ollama serve', or pass a self-contained preset "
            "instead: figure_model='smolvlm' (CLI: --figure-model smolvlm).",
            attempted=["ollama"],
        ) from exc

    if _ollama_name(model) not in {_ollama_name(name) for name in installed}:
        raise OllamaUnavailableError(
            f"Ollama is running but does not have {model!r}.\n"
            f"Pull it with 'ollama pull {model}', pick one it does have "
            "('ollama list', and 'ollama show <model>' must report the vision "
            "capability), or pass a self-contained preset such as "
            "figure_model='smolvlm'.",
            attempted=["ollama"],
            diagnostics={"installed": ", ".join(sorted(installed)) or "<none>"},
        )


def _accurate_options(data: bytes, *, figure_model: str) -> VlmPipelineOptions:
    """VLM pipeline with figure descriptions, tuned to this document."""
    from docling.datamodel.pipeline_options import (
        VlmConvertOptions,
        VlmPipelineOptions,
    )
    from docling.datamodel.vlm_engine_options import AutoInlineVlmEngineOptions

    # AUTO_INLINE picks MLX on macOS and Transformers elsewhere. On Apple
    # silicon the difference is not marginal — docling's own benchmark puts
    # MLX at 6.2 s/page against 102 s/page for Transformers on an M3 Max —
    # so log which engine was selected rather than assume.
    engine = AutoInlineVlmEngineOptions()
    logger.info("vlm engine selected: %s", engine.engine_type)

    trust_text_layer = has_text_layer(data)
    logger.info(
        "pdf text layer: %s -> force_backend_text=%s",
        "present" if trust_text_layer else "absent",
        trust_text_layer,
    )

    return VlmPipelineOptions(
        vlm_options=VlmConvertOptions.from_preset(
            _VLM_CONVERT_PRESET,
            engine_options=engine,
            force_backend_text=trust_text_layer,
        ),
        do_picture_description=True,
        picture_description_options=_picture_description_options(figure_model),
        # docling refuses to call an API-backed description model without
        # this, and it treats a localhost Ollama as a remote service like any
        # other. Set only for that path: an in-process preset needs no
        # network, and turning it on unconditionally would quietly widen what
        # the whole pipeline is allowed to do.
        enable_remote_services=figure_model.startswith(_OLLAMA_PREFIX),
        # The description model reads a crop of each picture, and
        # `VlmPipeline` only cuts those crops when generate_picture_images is
        # set (pipeline/vlm_pipeline.py:325). Without it every figure reaches
        # the model with no image and comes back undescribed.
        generate_picture_images=True,
        generate_page_images=True,
        force_backend_text=trust_text_layer,
    )


def _describing_vlm_pipeline() -> type:
    """Docling's VLM pipeline, with the figure-description stage put back.

    `BasePipeline.__init__` builds `enrichment_pipe` from
    `do_picture_description` / `picture_description_options` — and
    `VlmPipeline.__init__` then overwrites the whole list with `[]` after
    calling it (docling 2.121.0, `pipeline/vlm_pipeline.py:95`). The model is
    therefore constructed, loaded, and thrown away: the options are accepted
    and inert, with no warning. `StandardPdfPipeline` keeps the list instead
    (`*self.enrichment_pipe`), which is why descriptions work there.

    Restoring the stage is three lines and it is the only reason accurate
    mode can offer figure descriptions at all. Both names it leans on are
    docling internals, so this is checked by a live test rather than trusted:
    if a docling upgrade moves them, `test_vlm_live.py` fails loudly instead
    of the output quietly losing its figure text.
    """
    from docling.pipeline.vlm_pipeline import VlmPipeline

    class DescribingVlmPipeline(VlmPipeline):
        def __init__(self, pipeline_options):
            super().__init__(pipeline_options)
            model = self._get_picture_description_model(
                artifacts_path=self.artifacts_path
            )
            if model is None:
                raise ExtractionError(
                    "docling could not build a picture-description model for "
                    f"{pipeline_options.picture_description_options.kind!r}",
                    attempted=["docling[accurate]"],
                )
            self.enrichment_pipe = [model]

    return DescribingVlmPipeline


def _require_vlm_extra() -> None:
    """Fail loudly and early when the optional VLM dependencies are absent.

    Without this the run reaches the model-loading step deep inside docling
    and fails there, or — worse on Apple silicon — silently falls back to a
    Transformers engine that is an order of magnitude slower.
    """
    missing: list[str] = []
    if importlib.util.find_spec("transformers") is None:
        missing.append("transformers")
    on_apple_silicon = sys.platform == "darwin" and _is_arm64()
    if on_apple_silicon and importlib.util.find_spec("mlx_vlm") is None:
        missing.append("mlx-vlm")
    if not missing:
        return
    raise ImportError(
        'mode="accurate" needs the optional VLM dependencies, and '
        f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not "
        "installed.\n"
        "Install them with:  uv sync --extra vlm\n"
        '(or use mode="fast", which needs nothing extra)'
    )


def _is_arm64() -> bool:
    import platform

    return platform.machine() in ("arm64", "aarch64")
