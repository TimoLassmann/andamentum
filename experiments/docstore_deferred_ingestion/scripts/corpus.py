"""The corpus, the retrieval probes and the deliberately broken sources.

Kept in one guard-free module so the offline harness tests can check the probe
set is well-formed (every probe names a paper that is actually in the corpus)
without touching a database, a model or the network.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Paper:
    """One version-pinned arXiv paper."""

    arxiv_id: str
    short: str
    title: str

    @property
    def pdf_url(self) -> str:
        """Direct PDF URL. The arXiv *API* is blocked in this environment; the
        direct URL is verified working, and pinning the version is what makes
        'the same experiment' mean the same documents (the unversioned URL
        silently serves the latest revision)."""
        return f"https://arxiv.org/pdf/{self.arxiv_id}"


PAPERS: tuple[Paper, ...] = (
    Paper("1706.03762v7", "attention", "Attention Is All You Need"),
    Paper("1810.04805v2", "bert", "BERT: Pre-training of Deep Bidirectional Transformers"),
    Paper("1512.03385v1", "resnet", "Deep Residual Learning for Image Recognition"),
    Paper("1412.6980v9", "adam", "Adam: A Method for Stochastic Optimization"),
)

#: The smallest paper — used wherever one cheap PDF suffices.
SMALL_PAPER = PAPERS[3]

BY_ID = {p.arxiv_id: p for p in PAPERS}
BY_SHORT = {p.short: p for p in PAPERS}


@dataclass(frozen=True)
class Probe:
    """A retrieval probe and the paper it should retrieve.

    Probes deliberately AVOID each paper's distinctive vocabulary. "the
    architecture that replaced recurrence with attention" must not contain the
    word "Transformer", or the probe measures string matching rather than
    semantics — and the pre-drain FTS arm would score above zero for the wrong
    reason.
    """

    query: str
    expect_short: str
    rationale: str


PROBE_QUERIES: tuple[Probe, ...] = (
    Probe(
        "the architecture that replaced recurrent processing of sequences with a "
        "purely alignment-based mechanism",
        "attention",
        "no 'transformer', no 'attention is all you need'",
    ),
    Probe(
        "parallelising sequence-to-sequence translation by removing the step-by-step "
        "dependency between positions",
        "attention",
        "describes the motivation without naming the model",
    ),
    Probe(
        "pre-training a language encoder by hiding random words and asking the model "
        "to guess them",
        "bert",
        "describes masked language modelling without the acronym",
    ),
    Probe(
        "using left and right context jointly in every layer instead of reading text "
        "in one direction",
        "bert",
        "describes bidirectionality without naming the model",
    ),
    Probe(
        "making very deep vision networks trainable by letting layers learn a "
        "correction to the identity mapping",
        "resnet",
        "describes skip connections without 'residual network'",
    ),
    Probe(
        "why accuracy got worse when researchers stacked more layers, and the shortcut "
        "that fixed it for image classification",
        "resnet",
        "describes the degradation problem obliquely",
    ),
    Probe(
        "a first-order optimiser that keeps running averages of both the gradient and "
        "its square to scale each parameter's step",
        "adam",
        "describes the update rule without the optimiser's name",
    ),
    Probe(
        "correcting the initialisation bias of exponential moving averages when "
        "adapting learning rates per weight",
        "adam",
        "describes bias correction without the optimiser's name",
    ),
)


@dataclass(frozen=True)
class BadSource:
    """A source that genuinely fails — no monkeypatching, no fault injection."""

    key: str
    description: str
    expected_stage: str
    why: str


BAD_SOURCES: tuple[BadSource, ...] = (
    BadSource(
        "missing_path",
        "a file path that does not exist",
        "convert",
        "the commonest real-world failure: the file moved after being queued",
    ),
    BadSource(
        "garbage_pdf",
        "400 bytes of random data saved with a .pdf extension",
        "convert",
        "exercises the extraction backend's own error, not a pre-flight check",
    ),
    BadSource(
        "missing_url",
        "https://arxiv.org/pdf/0000.00000 — a clearly non-existent arXiv id",
        "convert",
        (
            "a real 404 over the real network. NOT a localhost URL: core.url_safety's "
            "SSRF guard would reject that for a different reason than the one under "
            "test and muddy the result"
        ),
    ),
    BadSource(
        "whitespace_markdown",
        "a .md file containing only whitespace",
        "convert",
        (
            "drives the ValueError('Conversion produced no content') branch inside "
            "_convert_document, which raises BEFORE the write and therefore correctly "
            "routes back to pending_source on retry"
        ),
    ),
)


def probes_for(short: str) -> tuple[Probe, ...]:
    """All probes whose correct answer is the given paper."""
    return tuple(p for p in PROBE_QUERIES if p.expect_short == short)


def validate_corpus() -> None:
    """Structural self-check: every probe names a real paper, ids are unique."""
    shorts = {p.short for p in PAPERS}
    unknown = sorted({p.expect_short for p in PROBE_QUERIES} - shorts)
    if unknown:
        raise ValueError(f"PROBE_QUERIES reference unknown papers: {unknown}")
    if len(BY_ID) != len(PAPERS):
        raise ValueError("duplicate arxiv_id in PAPERS")
    if len(BY_SHORT) != len(PAPERS):
        raise ValueError("duplicate short name in PAPERS")
    uncovered = sorted(shorts - {p.expect_short for p in PROBE_QUERIES})
    if uncovered:
        raise ValueError(f"papers with no probe: {uncovered}")
