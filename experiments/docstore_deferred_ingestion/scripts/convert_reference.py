"""Standalone Docling conversion of each PDF — the checkpoint's price tag.

Two jobs:
  1. Produce the markdown corpus that LINEAGE MAIN ingests (so the main lineage
     measures the *enrichment* stage without a conversion confound).
  2. Measure per-PDF conversion wall-time. ``checkpoint_savings_seconds`` is that
     constant multiplied by the number of already-converted documents the resume
     drain did NOT re-convert — which is how "conversion is checkpointed" gets an
     honest price in seconds instead of a slogan.

No LLM is involved. Docling/RapidOCR write noisy progress to stderr; that is
NORMAL and must not be read as failure. A conversion that exceeds the configured
timeout RAISES.

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.convert_reference
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from . import _common as C

SCHEMA = "andamentum.experiment.docstore_deferred.conversion_baseline/1"


def _docling_version() -> str | None:
    from importlib import metadata

    try:
        return metadata.version("docling")
    except metadata.PackageNotFoundError:
        return None


async def convert_all() -> dict[str, Any]:
    from andamentum.harvest import extract

    registry = C.read_json(C.REGISTRY_PATH)
    C.MARKDOWN.mkdir(parents=True, exist_ok=True)
    timeout = float(C.CONFIG["timeouts"]["conversion_seconds"])

    conversions: list[dict[str, Any]] = []
    for paper in registry["papers"]:
        pdf_path = C.EXP_DIR / paper["path"]
        md_path = C.MARKDOWN / f"{paper['arxiv_id']}.md"

        print(f"convert {paper['arxiv_id']} ({paper['bytes']} bytes) ...", flush=True)
        started = time.monotonic()
        markdown = await asyncio.wait_for(extract(pdf_path), timeout=timeout)
        seconds = time.monotonic() - started

        if not markdown.strip():
            raise ValueError(
                f"harvest.extract produced no content for {pdf_path}. Failing loud: an "
                "empty markdown corpus would make every downstream measurement vacuous."
            )
        md_path.write_text(markdown)
        conversions.append(
            {
                "arxiv_id": paper["arxiv_id"],
                "short": paper["short"],
                "pdf_path": paper["path"],
                "pdf_bytes": paper["bytes"],
                "markdown_path": str(md_path.relative_to(C.EXP_DIR)),
                "markdown_chars": len(markdown),
                "markdown_sha256": C.sha256_text(markdown),
                "conversion_seconds": seconds,
            }
        )
        print(f"  -> {len(markdown)} chars in {seconds:.1f}s", flush=True)

    # THE COLD/WARM SPLIT. The four numbers above fall monotonically in CALL order
    # and are uncorrelated with document size (adam 38,978 chars converts in a
    # third of the time attention's 48,959 chars took, purely because attention
    # went first). One-time Docling/RapidOCR initialisation dominates the first
    # call. Publishing the first call as "the per-document conversion cost" — which
    # is what `checkpoint_savings_seconds` did — overstates the marginal cost by
    # roughly 2x. So convert the FIRST paper a second time, now warm, and publish
    # the difference as a named constant rather than leaving it folded in.
    first = registry["papers"][0]
    print(f"convert {first['arxiv_id']} AGAIN (warm) ...", flush=True)
    warm_started = time.monotonic()
    await asyncio.wait_for(extract(C.EXP_DIR / first["path"]), timeout=timeout)
    warm_seconds = time.monotonic() - warm_started
    cold_seconds = conversions[0]["conversion_seconds"]
    print(f"  -> warm {warm_seconds:.1f}s vs cold {cold_seconds:.1f}s", flush=True)

    seconds_list = sorted(c["conversion_seconds"] for c in conversions)
    return {
        "docling_version": _docling_version(),
        "n_converted": len(conversions),
        "docling_warm_conversion_seconds": warm_seconds,
        "docling_cold_conversion_seconds": cold_seconds,
        "docling_init_seconds": max(0.0, cold_seconds - warm_seconds),
        "warm_cold_arxiv_id": first["arxiv_id"],
        "warm_cold_note": (
            "the SAME PDF converted twice in the same process. The difference is "
            "one-time Docling/RapidOCR initialisation, which every first conversion in "
            "a process pays and no subsequent one does"
        ),
        "conversion_seconds": {
            "values": seconds_list,
            "min": seconds_list[0] if seconds_list else None,
            "median": seconds_list[len(seconds_list) // 2] if seconds_list else None,
            "max": seconds_list[-1] if seconds_list else None,
            "mean": (sum(seconds_list) / len(seconds_list)) if seconds_list else None,
        },
        "conversions": conversions,
    }


def main() -> int:
    payload = asyncio.run(convert_all())
    C.write_json(C.RESULTS / "conversion_baseline.json", payload, schema=SCHEMA)
    print(f"wrote {C.RESULTS / 'conversion_baseline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
