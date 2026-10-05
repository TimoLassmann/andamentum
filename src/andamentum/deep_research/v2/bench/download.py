"""Fetching benchmark data into a directory the user names.

Nothing is downloaded into the repository and nothing has a default location:
every downloader takes the directory. Each file is written atomically and
recorded in ``MANIFEST.json`` beside the data with its source URL, size,
sha256 and the licence card of its dataset, so a results file can be traced to
the exact bytes it was computed from.

The licence cards were checked on 2026-10-05 against the sources named below.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue

MANIFEST_NAME = "MANIFEST.json"
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_S = 2.0
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

Sleep = Callable[[float], Awaitable[None]]


class DownloadError(RuntimeError):
    """A download failed for good (after retries, or at once on a client error)."""

    def __init__(self, url: str, reason: str, *, status: int | None = None) -> None:
        self.url = url
        self.reason = reason
        self.status = status
        super().__init__(f"{url}: {reason}")


class DatasetMissing(FileNotFoundError):
    """A dataset is not in the directory; the message names the command that fetches it."""


class DatasetCard(BaseModel):
    """Where a dataset comes from and on what terms it may be used."""

    model_config = ConfigDict(frozen=True)

    key: str
    title: str
    source: str
    licence: str
    licence_evidence: str
    terms: str
    citation: str


LITSEARCH_CARD = DatasetCard(
    key="litsearch",
    title="LitSearch: a retrieval benchmark for scientific literature search",
    source="https://huggingface.co/datasets/princeton-nlp/LitSearch",
    licence="MIT",
    licence_evidence=(
        "the LICENSE of the code repository https://github.com/princeton-nlp/LitSearch "
        "is MIT (GitHub licence API, 2026-10-05); the Hugging Face dataset card "
        "declares no licence of its own"
    ),
    terms=(
        "The 597 queries come from the Hugging Face dataset-viewer API. Their gold "
        "papers are Semantic Scholar corpus ids, resolved to titles and identifiers "
        "through the Semantic Scholar API: that metadata stays in the data directory "
        "for local evaluation and is never committed or redistributed."
    ),
    citation=(
        "Ajith, Xia, Chevalier, Goyal, Chen, Gao. LitSearch: A Retrieval Benchmark "
        "for Scientific Literature Search. EMNLP 2024."
    ),
)

SYNERGY_CARD = DatasetCard(
    key="synergy",
    title="SYNERGY: open dataset on study selection in systematic reviews (v1.0)",
    source="https://github.com/asreview/synergy-dataset (tag v1.0), https://doi.org/10.34894/HE6NAQ",
    licence="CC0-1.0",
    licence_evidence=(
        "the DataverseNL record reports the licence CC0-1.0 (API, 2026-10-05) and the "
        "repository README states CC0 1.0; its records are OpenAlex metadata, also CC0"
    ),
    terms=(
        "Public domain dedication. The authors ask to be cited and the original review "
        "authors are credited in the dataset's own ATTRIBUTION.md."
    ),
    citation=(
        "De Bruin, Ma, Ferdinands, Teijema, Van de Schoot (2023). SYNERGY - Open "
        "machine learning dataset on study selection in systematic reviews. "
        "DataverseNL, V1. https://doi.org/10.34894/HE6NAQ"
    ),
)


class Downloader:
    """A polite HTTP client with retries on 429 and 5xx.

    *client* and *sleep* are injectable so tests run without a network or a
    wait. A 404 is a ``DownloadError`` with ``status=404`` (``get_optional``
    turns it into None); any other client error raises at once.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        retries: int = DEFAULT_RETRIES,
        backoff_s: float = DEFAULT_BACKOFF_S,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.client = client
        self.retries = retries
        self.backoff_s = backoff_s
        self._sleep = sleep

    async def get(self, url: str, *, params: dict[str, str] | None = None) -> bytes:
        return await self._request("GET", url, params=params)

    async def get_optional(self, url: str) -> bytes | None:
        try:
            return await self.get(url)
        except DownloadError as error:
            if error.status == 404:
                return None
            raise

    async def post_json(
        self,
        url: str,
        body: JsonValue,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> bytes:
        return await self._request(
            "POST", url, params=params, json=body, headers=headers
        )

    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, str] | None = None,
        json: JsonValue = None,
        headers: dict[str, str] | None = None,
    ) -> bytes:
        reason = "no attempt was made"
        for attempt in range(self.retries + 1):
            if attempt:
                await self._sleep(self.backoff_s * 2 ** (attempt - 1))
            try:
                response = await self.client.request(
                    method, url, params=params, json=json, headers=headers
                )
            except httpx.TransportError as error:
                reason = f"{type(error).__name__}: {error}"
                continue
            if response.status_code in _RETRY_STATUSES:
                reason = f"HTTP {response.status_code}"
                continue
            if response.status_code >= 400:
                raise DownloadError(
                    url,
                    f"HTTP {response.status_code}",
                    status=response.status_code,
                )
            return response.content
        raise DownloadError(url, f"{reason} after {self.retries + 1} attempts")


def write_atomic(path: Path, data: bytes) -> None:
    """Write *data* to *path* through a temporary name, so a crash leaves no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    temporary.write_bytes(data)
    temporary.replace(path)


class ManifestEntry(BaseModel):
    path: str
    url: str
    bytes: int
    sha256: str
    dataset: str
    downloaded_at: float


class Manifest(BaseModel):
    cards: list[DatasetCard] = Field(default_factory=list)
    files: list[ManifestEntry] = Field(default_factory=list)


def read_manifest(root: Path) -> Manifest:
    path = root / MANIFEST_NAME
    if not path.exists():
        return Manifest()
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))


def save_file(
    root: Path, relative: str, data: bytes, *, url: str, card: DatasetCard
) -> Path:
    """Write one downloaded file under *root* and record it in the manifest."""
    target = root / relative
    write_atomic(target, data)
    manifest = read_manifest(root)
    manifest.cards = [c for c in manifest.cards if c.key != card.key] + [card]
    manifest.files = [f for f in manifest.files if f.path != relative] + [
        ManifestEntry(
            path=relative,
            url=url,
            bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            dataset=card.key,
            downloaded_at=time.time(),
        )
    ]
    write_atomic(
        root / MANIFEST_NAME,
        json.dumps(manifest.model_dump(mode="json"), indent=2).encode("utf-8"),
    )
    return target
