"""The folder bundle: scribe's source of truth.

A bundle is a plain directory — ``bundle.yaml``, ``sections/sNN.md``
(YAML frontmatter + markdown body), ``figures/``, ``references.yaml`` and
``out/``. Every read goes to disk and every write is atomic (temp file in the
same directory, then ``os.replace``), so a crashed run leaves either the old
file or the new one, and writing the same input twice leaves identical bytes.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Callable, Literal

import yaml

from .models import (
    Reference,
    ScribeError,
    Section,
    StaleRevisionError,
    ValidationIssue,
)
from .scaffolds import SCAFFOLDS

FORMAT_VERSION = 1
ORDER_STEP = 10

_SECTION_FILE_RE = re.compile(r"^s(\d+)\.md$")
_FRONTMATTER_RE = re.compile(r"\A---\n(?P<meta>.*?)\n---\n?(?P<body>.*)\Z", re.DOTALL)
_LABEL_CLEAN_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _atomic_write(path: Path, text: str) -> None:
    """Write *text* to *path* via a temp file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _dump_yaml(data: Any) -> str:
    return yaml.safe_dump(
        data, sort_keys=False, allow_unicode=True, default_flow_style=False
    )


def _normalise_body(body: str) -> str:
    return body.strip("\n")


def _section_text(section: Section) -> str:
    meta = {
        "id": section.id,
        "title": section.title,
        "order": section.order,
        "revision": section.revision,
        "status": section.status,
    }
    return f"---\n{_dump_yaml(meta)}---\n{section.body}\n"


def _parse_section(path: Path) -> Section:
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise ScribeError(f"{path}: section file has no YAML frontmatter")
    meta = yaml.safe_load(match.group("meta"))
    if not isinstance(meta, dict):
        raise ScribeError(f"{path}: frontmatter is not a mapping")
    try:
        return Section(
            id=str(meta["id"]),
            title=str(meta["title"]),
            order=int(meta["order"]),
            revision=int(meta["revision"]),
            status=str(meta.get("status", "planned")),
            body=_normalise_body(match.group("body")),
        )
    except KeyError as exc:
        raise ScribeError(f"{path}: frontmatter is missing {exc.args[0]!r}") from exc


class Bundle:
    """A document held as a folder of plain files."""

    def __init__(self, path: Path, *, title: str, metadata: dict[str, Any]):
        self.path = path
        self.title = title
        self.metadata = metadata

    # -- lifecycle -------------------------------------------------------

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        title: str,
        scaffold: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Bundle:
        """Create a new bundle at *path*; refuses an existing non-empty directory."""
        root = Path(path)
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise ScribeError(f"{root} already exists and is not an empty directory")
        if scaffold is not None and scaffold not in SCAFFOLDS:
            raise ScribeError(
                f"unknown scaffold {scaffold!r}; valid scaffolds: {sorted(SCAFFOLDS)}"
            )
        for sub in ("sections", "figures", "out"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        bundle = cls(root, title=title, metadata=dict(metadata or {}))
        _atomic_write(
            root / "bundle.yaml",
            _dump_yaml(
                {
                    "title": title,
                    "created": date.today().isoformat(),
                    "format_version": FORMAT_VERSION,
                    "metadata": bundle.metadata,
                }
            ),
        )
        _atomic_write(root / "references.yaml", _dump_yaml([]))
        for section_title, _guide in SCAFFOLDS.get(scaffold or "", []):
            bundle.add_section(title=section_title)
        return bundle

    @classmethod
    def open(cls, path: str | Path) -> Bundle:
        """Open an existing bundle."""
        root = Path(path)
        manifest = root / "bundle.yaml"
        if not manifest.is_file():
            raise ScribeError(f"{root} is not a scribe bundle (no bundle.yaml)")
        data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "title" not in data:
            raise ScribeError(f"{manifest} is malformed (needs a title)")
        version = data.get("format_version")
        if version != FORMAT_VERSION:
            raise ScribeError(
                f"{manifest}: format_version {version!r} is not supported "
                f"(this scribe reads {FORMAT_VERSION})"
            )
        return cls(
            root, title=str(data["title"]), metadata=dict(data.get("metadata") or {})
        )

    def _write_manifest(self) -> None:
        manifest = self.path / "bundle.yaml"
        data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        data["title"] = self.title
        data["metadata"] = self.metadata
        _atomic_write(manifest, _dump_yaml(data))

    def set_title(self, title: str) -> None:
        """Replace the document title (the Word ``Title`` paragraph on render)."""
        self.title = title
        self._write_manifest()

    def set_metadata(self, **kv: Any) -> None:
        """Merge *kv* into the bundle's free metadata dict."""
        self.metadata.update(kv)
        self._write_manifest()

    # -- sections --------------------------------------------------------

    def _section_path(self, section_id: str) -> Path:
        return self.path / "sections" / f"{section_id}.md"

    def _save(self, section: Section) -> None:
        _atomic_write(self._section_path(section.id), _section_text(section))

    def sections(self) -> list[Section]:
        """All sections, sorted by ``order``."""
        directory = self.path / "sections"
        found = [
            _parse_section(p)
            for p in sorted(directory.iterdir())
            if _SECTION_FILE_RE.match(p.name)
        ]
        return sorted(found, key=lambda s: (s.order, s.id))

    def section(self, section_id: str) -> Section:
        path = self._section_path(section_id)
        if not _SECTION_FILE_RE.match(path.name) or not path.is_file():
            raise ScribeError(f"no section with id {section_id!r}")
        return _parse_section(path)

    def section_by_title(self, title: str) -> Section:
        matches = [s for s in self.sections() if s.title == title]
        if not matches:
            raise ScribeError(f"no section titled {title!r}")
        return matches[0]

    def _next_id(self) -> str:
        numbers = [
            int(m.group(1))
            for p in (self.path / "sections").iterdir()
            if (m := _SECTION_FILE_RE.match(p.name))
        ]
        return f"s{max(numbers, default=0) + 1:02d}"

    def add_section(
        self,
        *,
        title: str,
        body: str = "",
        after: str | None = None,
        status: str = "planned",
    ) -> Section:
        """Add a section at the end, or directly after the section id *after*."""
        existing = self.sections()
        if after is None:
            index = len(existing)
        else:
            ids = [s.id for s in existing]
            if after not in ids:
                raise ScribeError(f"no section with id {after!r}")
            index = ids.index(after) + 1
        new = Section(
            id=self._next_id(),
            title=title,
            order=0,
            revision=1,
            status=status,
            body=_normalise_body(body),
        )
        previous = existing[index - 1].order if index > 0 else 0
        following = existing[index].order if index < len(existing) else None
        if following is None:
            new.order = previous + ORDER_STEP
        elif following - previous >= 2:
            new.order = (previous + following) // 2
        else:
            # No gap: respace every section by ORDER_STEP, new one included.
            ordered = existing[:index] + [new] + existing[index:]
            for position, section in enumerate(ordered, start=1):
                section.order = position * ORDER_STEP
                if section is not new:
                    self._save(section)
        self._save(new)
        return new

    def _checked(self, section_id: str, expected_revision: int) -> Section:
        section = self.section(section_id)
        if section.revision != expected_revision:
            raise StaleRevisionError(section_id, expected_revision, section.revision)
        return section

    def write_section(
        self,
        section_id: str,
        *,
        body: str,
        expected_revision: int,
        status: str | None = None,
    ) -> Section:
        """Replace a section's body; ``revision`` goes up by one."""
        section = self._checked(section_id, expected_revision)
        section.body = _normalise_body(body)
        section.revision += 1
        if status is not None:
            section.status = status
        self._save(section)
        return section

    def rename_section(
        self, section_id: str, *, title: str, expected_revision: int
    ) -> Section:
        section = self._checked(section_id, expected_revision)
        section.title = title
        section.revision += 1
        self._save(section)
        return section

    def remove_section(self, section_id: str) -> None:
        self.section(section_id)
        self._section_path(section_id).unlink()

    def reorder(self, section_ids: list[str]) -> None:
        """Set the section order; *section_ids* must be a permutation of all ids."""
        current = {s.id: s for s in self.sections()}
        if len(section_ids) != len(set(section_ids)) or set(section_ids) != set(
            current
        ):
            raise ScribeError(
                "reorder needs every section id exactly once; "
                f"have {sorted(current)}, got {sorted(section_ids)}"
            )
        for position, section_id in enumerate(section_ids, start=1):
            section = current[section_id]
            if section.order != position * ORDER_STEP:
                section.order = position * ORDER_STEP
                self._save(section)

    # -- figures ---------------------------------------------------------

    def add_figure(
        self,
        section_id: str,
        *,
        image: str | Path,
        caption: str,
        label: str,
        expected_revision: int,
    ) -> Section:
        """Copy *image* into ``figures/`` and append it to the section body."""
        section = self._checked(section_id, expected_revision)
        source = Path(image)
        if not source.is_file():
            raise ScribeError(f"figure image not found: {source}")
        label = label.removeprefix("fig:")
        stem = _LABEL_CLEAN_RE.sub("-", label).strip("-")
        if not stem:
            raise ScribeError(f"figure label {label!r} has no usable characters")
        target = self.path / "figures" / f"{stem}{source.suffix.lower()}"
        if target.resolve() != source.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=target.parent, prefix=f".{target.name}."
            )
            os.close(fd)
            try:
                shutil.copyfile(source, tmp_name)
                os.replace(tmp_name, target)
            except BaseException:
                Path(tmp_name).unlink(missing_ok=True)
                raise
        clean_caption = " ".join(caption.split()).replace("[", "(").replace("]", ")")
        line = f"![{clean_caption}](figures/{target.name}){{#fig:{label}}}"
        section.body = f"{section.body}\n\n{line}" if section.body else line
        section.revision += 1
        self._save(section)
        return section

    # -- references ------------------------------------------------------

    def references(self) -> list[Reference]:
        path = self.path / "references.yaml"
        if not path.is_file():
            return []
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        if not isinstance(data, list):
            raise ScribeError(f"{path} must hold a list of references")
        return [Reference(**item) for item in data]

    def add_reference(
        self, *, key: str, text: str, url: str | None = None
    ) -> Reference:
        """Add a reference; a reference with the same key is replaced."""
        new = Reference(key=key, text=text, url=url)
        refs = [r for r in self.references() if r.key != key]
        position = next(
            (i for i, r in enumerate(self.references()) if r.key == key), len(refs)
        )
        refs.insert(position, new)
        _atomic_write(
            self.path / "references.yaml",
            _dump_yaml([r.model_dump(exclude_none=True) for r in refs]),
        )
        return new

    # -- validation and rendering ----------------------------------------

    def validate(self) -> list[ValidationIssue]:
        from .validate import validate_bundle

        return validate_bundle(self)

    def render(
        self,
        output_path: str | Path,
        *,
        format: Literal["docx", "md", "html", "txt"],
        text_filter: Callable[[str], str] | None = None,
        image_root: str | Path | None = None,
    ) -> Path:
        """Render to *output_path*. *image_root* is the folder bundle-relative
        image paths resolve against (default: this bundle) — set it when
        rendering a copy of the bundle."""
        from .render import render_bundle

        return render_bundle(
            self,
            Path(output_path),
            format=format,
            text_filter=text_filter,
            image_root=Path(image_root) if image_root is not None else None,
        )
