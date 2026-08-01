"""Download the four version-pinned arXiv PDFs and write the data registry.

ONE rule produces ALL four PDFs — deliberately not a wildcard rule — so the
downloads stay strictly sequential with a polite delay regardless of ``--cores``.

The arXiv *API* (export.arxiv.org/api/query) returns empty in this environment
and is never used. Direct versioned PDF URLs are used instead, and were verified
to resolve: https://arxiv.org/pdf/1706.03762v7 -> 200, bytes identical to the
unversioned form. Pinning the version is what makes "the same experiment" mean
the same documents; the unversioned URL silently serves the latest revision.

RE-FETCH SEMANTICS
------------------
matching sha256 -> skip. MISMATCH -> RegistryDriftError, never overwrite.
Re-registration requires an explicit ``--accept-new-hash`` which bumps
``registry_version`` and appends the old record to ``supersedes``. A drift is a
provenance event a human must see, not something a pipeline quietly absorbs.

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.fetch_pdfs
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import httpx

from . import _common as C
from .corpus import PAPERS

SCHEMA = "andamentum.experiment.docstore_deferred.registry/1"

PDF_MAGIC = b"%PDF"


class RegistryDriftError(RuntimeError):
    """A previously-registered PDF now hashes differently. Human decision needed."""


def _download(url: str, *, user_agent: str, timeout: float, retries: int) -> dict[str, Any]:
    """Fetch a PDF with retries. Returns the bytes plus the response provenance."""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            started = time.monotonic()
            with httpx.Client(
                follow_redirects=True,
                timeout=timeout,
                headers={"User-Agent": user_agent},
            ) as client:
                response = client.get(url)
            elapsed = time.monotonic() - started
            response.raise_for_status()
            data = response.content
            if not data.startswith(PDF_MAGIC):
                raise ValueError(
                    f"{url} did not return a PDF (first bytes: {data[:16]!r}). "
                    "Refusing to register a non-PDF as a paper."
                )
            return {
                "content": data,
                "requested_url": url,
                "final_url": str(response.url),
                "http_status": response.status_code,
                "content_type": response.headers.get("content-type"),
                "etag": response.headers.get("etag"),
                "last_modified": response.headers.get("last-modified"),
                "download_seconds": elapsed,
                "attempts": attempt + 1,
            }
        except Exception as exc:  # noqa: BLE001 — retried, then re-raised below
            last_error = exc
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url} after {retries + 1} attempts") from last_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accept-new-hash",
        action="store_true",
        help="Explicitly accept that a paper's bytes changed upstream (bumps registry_version)",
    )
    args = parser.parse_args()

    cfg = C.CONFIG["download"]
    C.PDFS.mkdir(parents=True, exist_ok=True)

    previous: dict[str, Any] = {}
    registry_version = 1
    supersedes: list[Any] = []
    if C.REGISTRY_PATH.exists():
        old = C.read_json(C.REGISTRY_PATH)
        previous = {p["arxiv_id"]: p for p in old.get("papers", [])}
        registry_version = int(old.get("registry_version", 1))
        supersedes = list(old.get("supersedes", []))

    records: list[dict[str, Any]] = []
    drift: list[str] = []
    downloaded = 0

    for index, paper in enumerate(PAPERS):
        target = C.PDFS / f"{paper.arxiv_id}.pdf"
        known = previous.get(paper.arxiv_id)

        if target.exists() and known is not None:
            current_sha = C.sha256_file(target)
            if current_sha == known.get("sha256"):
                print(f"skip     {paper.arxiv_id} (sha256 matches registry)")
                records.append({**known, "revalidated_at": C.utc_now()})
                continue
            drift.append(
                f"{paper.arxiv_id}: on-disk sha256 {current_sha} != registry "
                f"{known.get('sha256')}"
            )
            if not args.accept_new_hash:
                continue

        if index > 0 and downloaded > 0:
            time.sleep(float(cfg["polite_delay_seconds"]))

        print(f"download {paper.arxiv_id} <- {paper.pdf_url}")
        result = _download(
            paper.pdf_url,
            user_agent=cfg["user_agent"],
            timeout=float(cfg["timeout_seconds"]),
            retries=int(cfg["retries"]),
        )
        downloaded += 1
        content = result.pop("content")
        target.write_bytes(content)
        sha = C.sha256_file(target)

        if known is not None and known.get("sha256") not in (None, sha):
            if not args.accept_new_hash:
                drift.append(
                    f"{paper.arxiv_id}: upstream sha256 {sha} != registry "
                    f"{known.get('sha256')}"
                )
                continue
            supersedes.append({"superseded_at": C.utc_now(), "record": known})
            registry_version += 1

        records.append(
            {
                "arxiv_id": paper.arxiv_id,
                "short": paper.short,
                "title": paper.title,
                "path": str(target.relative_to(C.EXP_DIR)),
                "sha256": sha,
                "bytes": target.stat().st_size,
                "downloaded_at": C.utc_now(),
                **result,
            }
        )

    if drift:
        raise RegistryDriftError(
            "Registered PDF bytes changed:\n  "
            + "\n  ".join(drift)
            + "\n\nThe registry is NOT auto-refreshed: a drift is a provenance event a "
            "human should see. Re-run with --accept-new-hash to register the new bytes "
            "(this bumps registry_version and records the old entry under 'supersedes')."
        )

    C.write_json(
        C.REGISTRY_PATH,
        {
            "registry_version": registry_version,
            "n_papers": len(records),
            "api_note": (
                "export.arxiv.org/api/query returns empty in this environment and is "
                "never used; PDFs come from direct versioned URLs."
            ),
            "papers": records,
            "supersedes": supersedes,
        },
        schema=SCHEMA,
    )
    print(f"wrote {C.REGISTRY_PATH} ({len(records)} papers, {downloaded} downloaded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
