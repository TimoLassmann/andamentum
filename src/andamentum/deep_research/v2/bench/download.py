"""Fetching benchmark data into a directory the user names.

Nothing is downloaded into the repository and nothing has a default location:
every downloader takes the directory. Each file is written atomically and
recorded in ``MANIFEST.json`` beside the data with its source URL, size,
sha256 and the licence card of its dataset, so a results file can be traced to
the exact bytes it was computed from.

The licence card of the dataset is passed in by the caller (``litsearch_local``
holds the one the bench downloads).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

MANIFEST_NAME = "MANIFEST.json"


class DownloadError(RuntimeError):
    """A download failed for good."""

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
