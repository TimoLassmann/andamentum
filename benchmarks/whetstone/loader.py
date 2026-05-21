"""Fetch v1 preprints (bioRxiv + arXiv) and harvest them to markdown ONCE.

Both review arms must consume identical text, so harvesting happens here, in
one place, cached to ``corpus/``. The PDF download and harvest are the only
network/IO in the harness besides the model calls.

NOTE: the bioRxiv/arXiv endpoints below are the public ones but have not been
exercised from this checkout — validate on the first live run and adjust if an
endpoint shape has changed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

from .types import PaperRef

logger = logging.getLogger("whetstone.bench")

_BIORXIV_PUBS = "https://api.biorxiv.org/pubs/biorxiv/{start}/{end}/{cursor}"
_BIORXIV_PDF = "https://www.biorxiv.org/content/{doi}v{version}.full.pdf"
_ARXIV_PDF = "https://arxiv.org/pdf/{id}v{version}"

_HEADERS = {"User-Agent": "andamentum-whetstone-eval/0.1 (research evaluation)"}


# ── Discovery (a helper to find candidates; never the reproducible source) ──


class Candidate(PaperRef):
    """A discovered candidate, carrying the journal it was later published in.

    The published journal is a SELECTION SIGNAL only — a preprint good enough
    to be published is a fair test. We still only ever download the open v1
    preprint (``fetch_pdf``); the published version is never fetched.
    """

    published_journal: str = ""
    published_doi: str = ""


def discover_biorxiv_published(*, n: int, days: int = 365) -> list[Candidate]:
    """bioRxiv preprints that were LATER PUBLISHED, with the journal name.

    Uses the public ``/pubs/`` endpoint (preprint→publication links). Field
    names are read defensively; validate on the first live run. Returns v1
    candidates for you to curate into a committed ``seeds.json``.
    """
    import datetime as _dt

    end = _dt.date.today()
    start = end - _dt.timedelta(days=days)
    found: list[Candidate] = []
    cursor = 0
    with httpx.Client(timeout=30.0, headers=_HEADERS) as client:
        while len(found) < n and cursor < 3000:
            resp = client.get(_BIORXIV_PUBS.format(start=start, end=end, cursor=cursor))
            resp.raise_for_status()
            batch = resp.json().get("collection", [])
            if not batch:
                break
            for e in batch:
                doi = e.get("biorxiv_doi") or e.get("preprint_doi") or e.get("doi")
                if not doi:
                    continue
                found.append(
                    Candidate(
                        source="biorxiv",
                        id=doi,
                        version=1,
                        title=e.get("preprint_title") or e.get("title", ""),
                        subfield=e.get("preprint_category") or e.get("category", ""),
                        published_journal=e.get("published_journal", ""),
                        published_doi=e.get("published_doi", ""),
                    )
                )
                if len(found) >= n:
                    break
            cursor += len(batch)
    logger.info("[loader] discovered %d published bioRxiv preprint(s)", len(found))
    return found


# ── Fetch ─────────────────────────────────────────────────────────────────


def fetch_pdf(ref: PaperRef, corpus_dir: Path) -> PaperRef:
    """Download *ref*'s v1 PDF into ``corpus/`` and set ``pdf_path``."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    out = corpus_dir / f"{ref.slug}.pdf"
    if out.exists():
        ref.pdf_path = str(out)
        return ref

    if ref.source == "biorxiv":
        url = _BIORXIV_PDF.format(doi=ref.id, version=ref.version)
    else:
        url = _ARXIV_PDF.format(id=ref.id, version=ref.version)

    with httpx.Client(timeout=60.0, headers=_HEADERS, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        out.write_bytes(resp.content)
    ref.pdf_path = str(out)
    logger.info("[loader] fetched %s (%d bytes)", ref.slug, len(resp.content))
    return ref


async def harvest_paper(ref: PaperRef, corpus_dir: Path) -> PaperRef:
    """Harvest the fetched PDF to markdown ONCE, cached, set ``markdown_path``."""
    from andamentum.harvest import extract

    md_path = corpus_dir / f"{ref.slug}.md"
    if md_path.exists():
        ref.markdown_path = str(md_path)
        return ref
    if not ref.pdf_path:
        raise ValueError(f"{ref.slug}: fetch the PDF before harvesting")
    markdown = await extract(Path(ref.pdf_path))
    md_path.write_text(markdown, encoding="utf-8")
    ref.markdown_path = str(md_path)
    logger.info("[loader] harvested %s → %d chars", ref.slug, len(markdown))
    return ref


# ── Seeds (curated id lists) ────────────────────────────────────────────────


def load_seeds(path: Path) -> list[PaperRef]:
    """Read a curated seed list. Format: JSON array of
    ``{"source": "biorxiv"|"arxiv", "id": "...", "subfield": "..."}``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [PaperRef.model_validate({"version": 1, **entry}) for entry in data]


def save_manifest(refs: list[PaperRef], path: Path) -> None:
    path.write_text(
        json.dumps([r.model_dump() for r in refs], indent=2), encoding="utf-8"
    )


def load_manifest(path: Path) -> list[PaperRef]:
    return [PaperRef.model_validate(e) for e in json.loads(path.read_text("utf-8"))]
