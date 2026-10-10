"""Output versions in a Word command's folder, and which outputs the user changed.

A Word command run with ``-o DIR`` names its own files (``paper.checked.docx``
and its report). Each file it writes is fingerprinted (SHA-256 of its bytes) in
``state/outputs.json``. On the next run into the folder:

- an earlier output whose bytes still match its fingerprint was not touched
  since the tool wrote it, so the new run writes over it;
- one that changed (edited in Word, or only saved again), or that has no
  fingerprint, is kept, and the run writes the next free version:
  ``paper.checked-2.docx`` with ``paper.checked-2.md`` and ``.json``, and so on.

The untouched files are moved into ``state/.replaced/`` before the run and
deleted only once the new copy exists; a run that writes no copy puts them back.

A caller that names an exact path (the MCP tools, the harness params) does not
come here: the verbs refuse a taken name (``project.refuse_taken``), because a
program that named a file must find its result there.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("andamentum.word")

FINGERPRINTS = "outputs.json"
ASIDE = ".replaced"

#: The files of one output version (1 is the plain name), the copy first.
Names = Callable[[int], tuple[Path, ...]]


def copy_names(folder: Path, stem: str, *, report: bool = True) -> Names:
    """``<stem>.docx`` with ``<stem>.md`` and ``.json`` (*report*); version n
    adds ``-n`` to the stem."""

    def names(version: int) -> tuple[Path, ...]:
        base = stem if version == 1 else f"{stem}-{version}"
        files = [folder / f"{base}.docx"]
        if report:
            files += [folder / f"{base}.md", folder / f"{base}.json"]
        return tuple(files)

    return names


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(state: Path) -> dict[str, str]:
    path = state / FINGERPRINTS
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _untouched(path: Path, records: dict[str, str]) -> bool:
    return records.get(path.name) == fingerprint(path)


@dataclass
class OutputPlan:
    """The files this run writes, and what happened to the earlier ones."""

    files: tuple[Path, ...]  # the copy first
    kept: list[Path] = field(default_factory=list)  # changed since written, left alone
    replaced: list[Path] = field(default_factory=list)  # untouched, written over
    aside: list[tuple[Path, Path]] = field(default_factory=list)  # (file, moved to)
    state: Path = Path()


def plan_outputs(names: Names, *, state: Path) -> OutputPlan:
    """Choose the first version whose existing files are all untouched outputs
    and move those files aside; earlier versions that changed are kept."""
    records = _records(state)
    kept: list[Path] = []
    version = 1
    while True:
        files = names(version)
        existing = [p for p in files if p.exists()]
        changed = [p for p in existing if not _untouched(p, records)]
        if not changed:
            break
        kept += changed
        version += 1
    plan = OutputPlan(files=files, kept=kept, replaced=existing, state=state)
    if existing:
        aside = state / ASIDE
        aside.mkdir(parents=True, exist_ok=True)
        for path in existing:
            moved = aside / path.name
            moved.unlink(missing_ok=True)
            path.rename(moved)
            plan.aside.append((path, moved))
    if kept:
        logger.warning(
            "kept %s: changed after andamentum wrote it; writing %s",
            ", ".join(str(p) for p in kept),
            files[0],
        )
    return plan


def finish_outputs(plan: OutputPlan) -> None:
    """Delete the replaced files and fingerprint the new ones when the copy was
    written; otherwise put the replaced files back."""
    wrote = plan.files[0].exists()
    for original, moved in plan.aside:
        if wrote:
            moved.unlink(missing_ok=True)
        elif not original.exists():
            moved.rename(original)
    if not wrote:
        return
    records = _records(plan.state)
    for path in plan.files:
        if path.exists():
            records[path.name] = fingerprint(path)
    plan.state.mkdir(parents=True, exist_ok=True)
    (plan.state / FINGERPRINTS).write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@contextmanager
def versioned(names: Names, *, state: Path) -> Iterator[OutputPlan]:
    """:func:`plan_outputs` before the run, :func:`finish_outputs` after it,
    also when the run raises."""
    plan = plan_outputs(names, state=state)
    try:
        yield plan
    finally:
        finish_outputs(plan)


def changed_copies(file: Path, *, folder: Path, state: Path) -> list[Path]:
    """Copies of *file* in *folder* that the tool wrote and that changed since.

    A copy of ``paper.docx`` is a ``paper.<something>.docx`` with a fingerprint;
    *file* itself is never listed. A run on *file* while one of these exists may
    miss the edits made in it.
    """
    records = _records(state)
    found = []
    for name in sorted(records):
        path = folder / name
        if not (name.startswith(f"{file.stem}.") and name.endswith(".docx")):
            continue
        if not path.exists() or path.resolve() == file.resolve():
            continue
        if not _untouched(path, records):
            found.append(path)
    return found


def warn_changed_copies(file: Path, *, folder: Path, state: Path) -> list[Path]:
    """:func:`changed_copies`, each logged as a warning naming the file to use."""
    found = changed_copies(file, folder=folder, state=state)
    for path in found:
        logger.warning(
            "%s changed after andamentum wrote it; if it holds your newer edits, "
            "run on it instead of %s",
            path,
            file,
        )
    return found
