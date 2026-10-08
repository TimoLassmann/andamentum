"""Where a document's project lives and how a copy of the document is written.

Every verb that opens a ``.docx`` finds its project folder here, and every verb
that writes a copy writes it here, so a copy always carries the project id and
a later run into the same folder finds the same ledger, event log and snapshots.
"""

from __future__ import annotations

from pathlib import Path

from .docx.writer import DocxAdapter
from .kernel import ApplyResult, DocumentView, Transaction


STATE_DIR = "state"


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
