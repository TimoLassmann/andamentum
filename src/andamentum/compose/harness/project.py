"""Where a document's project lives and how a copy of the document is written.

Every verb that opens a ``.docx`` finds its project folder here, and every verb
that writes a copy writes it here, so a copy always carries the project id and
a later run on that copy finds the same ledger, event log and snapshots.
"""

from __future__ import annotations

from pathlib import Path

from .docx.writer import DocxAdapter
from .kernel import ApplyResult, DocumentView, ProjectStore, Transaction


def project_dir(file: Path, view: DocumentView, project: Path | None) -> Path:
    """``--project``, else ``<stem>.harness`` beside the file, else the sibling
    project folder whose id the document carries.

    ``bind-numbers`` writes ``paper.bound.docx`` and the ledger lives with the
    project of ``paper.docx``; looking the id up lets any verb run on
    ``paper.bound.docx`` find it without a flag.
    """
    if project is not None:
        return project
    default = file.with_name(f"{file.stem}.harness")
    if default.exists() or view.project_id is None:
        return default
    for sibling in sorted(file.parent.glob("*.harness")):
        try:
            if ProjectStore(sibling).meta().id == view.project_id:
                return sibling
        except (FileNotFoundError, ValueError):
            continue
    return default


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
