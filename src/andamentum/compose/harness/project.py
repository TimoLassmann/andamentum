"""Where a document's project lives and how a copy of the document is written.

Every verb that opens a ``.docx`` finds its project folder here, and every verb
that writes a copy writes it here, so a copy always carries the project id and
a later run into the same folder finds the same ledger, event log and snapshots.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .docx.writer import DocxAdapter
from .kernel import ApplyResult, DocumentView, Transaction


STATE_DIR = "state"


def written_files(out: Path, report: Path | None) -> tuple[Path, Path, Path]:
    """The files a verb writes beside ``state/``: the copy *out* and the report
    pair (*report*, default ``out`` with ``.md``, and the same with ``.json``)."""
    report_md = report or out.with_suffix(".md")
    return out, report_md, report_md.with_suffix(".json")


def refuse_taken(paths: Iterable[Path], *, error: type[Exception]) -> None:
    """Raise *error* naming each of *paths* that already exists.

    A verb never writes over a file: a copy from an earlier run may hold edits
    made in Word since. Called before anything is written, ``state/`` included.
    """
    taken = [str(p) for p in paths if p.exists()]
    if taken:
        raise error(
            f"{', '.join(taken)} already exists and may hold your edits; the tool "
            "never overwrites a file. Move or rename it, or give --out another folder"
        )


def project_dir(out: Path, project: Path | None) -> Path:
    """The project folder: *project* when given, else ``state/`` in the folder
    that holds the written copy (*out*). Nothing is ever kept beside the input.

    A later run that is given the same output folder finds the ledger, event
    log and snapshots of the earlier one there.
    """
    return project if project is not None else out.parent / STATE_DIR


def write_copy(
    adapter: DocxAdapter,
    view: DocumentView,
    transaction: Transaction,
    *,
    out: Path,
    project_id: str | None,
) -> ApplyResult:
    """Apply *transaction* to a copy at *out*; the copy carries *project_id*
    (when given and the document carries none yet)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if view.project_id is not None or project_id is None:
        return adapter.apply(view, transaction, output=out)
    staged = out.with_name(f".{out.name}.staged.docx")
    result = adapter.apply(view, transaction, output=staged)
    try:
        adapter.write_project_id(staged, project_id=project_id, output=out)
    finally:
        staged.unlink(missing_ok=True)
    return result.model_copy(update={"output": out})
