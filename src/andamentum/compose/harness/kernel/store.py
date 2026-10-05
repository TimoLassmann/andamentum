"""ProjectStore: the folder next to a document (``<stem>.harness/``).

Plain files only: ``project.yaml``, ``checks.yaml`` (optional user config),
``events.jsonl`` (append-only), ``cache/`` (check and HTTP caches) and
``view/`` (text snapshots under a git repository created on first snapshot).
Git is driven through ``subprocess`` with the user's global and system config
switched off and the author set per command, so nothing outside the project
folder is read or written.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from .view import DocumentView

SCHEMA_VERSION = 1
# How long a Crossref answer stays good: a work can be retracted after it was checked.
ANSWER_MAX_AGE = 7 * 24 * 3600.0
GIT_AUTHOR_NAME = "andamentum harness"
GIT_AUTHOR_EMAIL = "harness@localhost"


class ProjectMeta(BaseModel):
    id: str
    created: str
    schema_version: int = SCHEMA_VERSION


class CachedResponse(BaseModel):
    """One stored HTTP answer: enough to replay a lookup without the network."""

    model_config = ConfigDict(extra="forbid")

    url: str
    status: int
    body: str
    content_type: str | None = None
    fetched: float = (
        0.0  # epoch seconds; 0 (unknown) counts as stale for a ProjectCache
    )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


class HttpCache:
    """Response cache for network checks, stored in ``<project>/cache/http/``.

    A check builds a key with :meth:`key`, asks :meth:`get`, and on a miss makes
    the request and calls :meth:`put`. ``hits`` and ``misses`` count the
    lookups so the report can say how many network calls were made.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(method: str, url: str, params: dict[str, str] | None = None) -> str:
        query = json.dumps(sorted((params or {}).items()))
        return _digest(method.upper(), url, query)

    def get(self, key: str) -> CachedResponse | None:
        path = self.directory / f"{key}.json"
        if not path.is_file():
            self.misses += 1
            return None
        self.hits += 1
        return CachedResponse.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, key: str, response: CachedResponse) -> None:
        _atomic_write(self.directory / f"{key}.json", response.model_dump_json())


class ProjectCache:
    """The project's HTTP cache seen as the bytes cache ``ReferenceHttp`` takes.

    The project cache outlives sessions, so an answer older than *max_age*
    seconds is fetched again: a work checked once may be retracted later.
    """

    def __init__(
        self,
        cache: HttpCache,
        *,
        max_age: float,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._cache = cache
        self._max_age = max_age
        self._clock = clock

    def get(self, key: str) -> bytes | None:
        found = self._cache.get(HttpCache.key("GET", key))
        if found is None:
            return None
        if self._clock() - found.fetched > self._max_age:
            self._cache.hits -= 1
            self._cache.misses += 1
            return None
        return found.body.encode("utf-8")

    def put(self, key: str, value: bytes) -> None:
        self._cache.put(
            HttpCache.key("GET", key),
            CachedResponse(
                url=key,
                status=200,
                body=value.decode("utf-8"),
                fetched=self._clock(),
            ),
        )


class ProjectStore:
    """Files: project.yaml, checks.yaml (optional), events.jsonl (append-only),
    cache/ (check and http cache), view/ (text snapshots under a lazily created git repo)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.http_cache = HttpCache(path / "cache" / "http")

    @property
    def cache_dir(self) -> Path:
        return self.path / "cache"

    @property
    def view_dir(self) -> Path:
        return self.path / "view"

    # -- project metadata ------------------------------------------------

    def init(self, docx: Path, *, project_id: str | None = None) -> ProjectMeta:
        """Create the folder and ``project.yaml`` for *docx*; idempotent.

        An existing project keeps its id (a different *project_id* raises).
        A new project takes *project_id* (for a document that already carries
        one) or a fresh random id.
        """
        if not docx.is_file():
            raise FileNotFoundError(f"document not found: {docx}")
        if (self.path / "project.yaml").is_file():
            meta = self.meta()
            if project_id is not None and project_id != meta.id:
                raise ValueError(
                    f"{self.path} belongs to project {meta.id}, not {project_id}"
                )
            return meta
        meta = ProjectMeta(
            id=project_id or uuid.uuid4().hex,
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.path.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            self.path / "project.yaml",
            yaml.safe_dump(meta.model_dump(), sort_keys=False),
        )
        return meta

    def meta(self) -> ProjectMeta:
        path = self.path / "project.yaml"
        if not path.is_file():
            raise FileNotFoundError(
                f"{self.path} is not a harness project (no project.yaml)"
            )
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        meta = ProjectMeta.model_validate(data)
        if meta.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: schema_version {meta.schema_version} is not supported "
                f"(this code reads {SCHEMA_VERSION})"
            )
        return meta

    def config(self) -> dict[str, object]:
        """The user's ``checks.yaml``; empty when the file is absent."""
        path = self.path / "checks.yaml"
        if not path.is_file():
            return {}
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} must hold a mapping")
        return data

    # -- event log -------------------------------------------------------

    def append_event(self, event: dict[str, object]) -> None:
        """Append one event line; a ``ts`` field is added when missing."""
        record = dict(event)
        record.setdefault(
            "ts", datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        line = json.dumps(record, ensure_ascii=False) + "\n"
        self.path.mkdir(parents=True, exist_ok=True)
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line)

    def read_events(self) -> list[dict[str, object]]:
        path = self.path / "events.jsonl"
        if not path.is_file():
            return []
        events: list[dict[str, object]] = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: not valid JSON ({exc})") from exc
        return events

    # -- generic cache ---------------------------------------------------

    def _cache_path(self, namespace: str, key: str) -> Path:
        for part in (namespace, key):
            if not part or "/" in part or part.startswith("."):
                raise ValueError(f"invalid cache name {part!r}")
        return self.cache_dir / namespace / f"{key}.json"

    def cache_get(self, namespace: str, key: str) -> object | None:
        path = self._cache_path(namespace, key)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def cache_put(self, namespace: str, key: str, value: object) -> None:
        _atomic_write(self._cache_path(namespace, key), json.dumps(value))

    # -- view snapshots (shadow git) -------------------------------------

    def _git(self, *args: str) -> str:
        env = dict(os.environ)
        env.update(
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_CONFIG_SYSTEM=os.devnull,
            GIT_CONFIG_NOSYSTEM="1",
        )
        command = [
            "git",
            "-c",
            f"user.name={GIT_AUTHOR_NAME}",
            "-c",
            f"user.email={GIT_AUTHOR_EMAIL}",
            "-c",
            "commit.gpgsign=false",
            "-c",
            f"core.hooksPath={os.devnull}",
            *args,
        ]
        done = subprocess.run(
            command,
            cwd=self.view_dir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed in {self.view_dir}: {done.stderr.strip()}"
            )
        return done.stdout

    def snapshot_view(self, view: DocumentView, *, message: str) -> None:
        """Write the view as text and commit it: one commit per call.

        The commit is made even when nothing changed, so every run leaves a
        mark in the history. The git repository is created on first use.
        """
        self.view_dir.mkdir(parents=True, exist_ok=True)
        if not (self.view_dir / ".git").exists():
            self._git("init", "--quiet", "--initial-branch=main")
        _atomic_write(self.view_dir / "document.txt", view.text() + "\n")
        _atomic_write(
            self.view_dir / "paragraphs.json",
            json.dumps(
                [p.model_dump() for p in view.paragraphs], indent=1, ensure_ascii=False
            ),
        )
        self._git("add", "-A")
        self._git("commit", "--quiet", "--allow-empty", "-m", message)

    def snapshot_log(self) -> list[str]:
        """Commit messages of the view history, newest first."""
        if not (self.view_dir / ".git").exists():
            return []
        out = self._git("log", "--format=%s")
        return out.splitlines()
