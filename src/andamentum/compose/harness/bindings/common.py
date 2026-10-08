"""What the two verbs share: opening the document and project."""

from __future__ import annotations

from pathlib import Path

from ..docx.reader import DocxReadError
from ..docx.writer import DocxAdapter
from ..kernel import DocumentView, ProjectStore
from ..errors import WordUsageError
from ..project import project_dir

EXIT_OK, EXIT_ERRORS, EXIT_USAGE = 0, 1, 2


class UsageError(WordUsageError):
    """A bad invocation (missing file, unmet need): exit code 2."""


def open_document(
    file: Path, *, out: Path, project: Path | None, results: Path
) -> tuple[DocxAdapter, DocumentView, ProjectStore, str]:
    """The adapter, the view, the project store and the project id."""
    if not file.is_file() or file.suffix.lower() != ".docx":
        raise UsageError(f"{file} is not a .docx file")
    if not results.is_dir():
        raise UsageError(f"--results {results} is not a folder")
    adapter = DocxAdapter()
    try:
        view = adapter.open(file)
    except DocxReadError as exc:
        raise UsageError(str(exc)) from exc
    store = ProjectStore(project_dir(out, project))
    try:
        meta = store.init(file, project_id=view.project_id)
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    return adapter, view, store, meta.id


def output_path(file: Path, out: Path) -> Path:
    if out.resolve() == file.resolve():
        raise UsageError("the output must not be the input file; it is never edited")
    return out
