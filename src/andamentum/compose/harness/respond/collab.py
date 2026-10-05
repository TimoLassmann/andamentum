"""Collaborators' own edits: which paragraphs changed since the harness last looked.

The project's shadow git holds the text view of the last run (``check`` and
``respond`` both snapshot the document they read). Paragraph ids do not
survive edits, so paragraphs are compared by text: a paragraph of the current
view that is not part of a matching run of the snapshot is changed.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher

from ..kernel import DocumentView, ProjectStore


def previous_texts(store: ProjectStore) -> list[str] | None:
    """Paragraph texts of the last snapshot, or None when there is none."""
    path = store.view_dir / "paragraphs.json"
    if not path.is_file():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [str(row["text"]) for row in rows]


def changed_paragraph_ids(view: DocumentView, previous: list[str]) -> list[str]:
    """Ids of the paragraphs in *view* that differ from the snapshot."""
    current = [p.text for p in view.paragraphs]
    matcher = SequenceMatcher(None, previous, current, autojunk=False)
    changed: list[str] = []
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            changed += [view.paragraphs[j].id for j in range(j1, j2)]
    return changed
