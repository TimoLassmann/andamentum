"""Hand-written synthetic labelled sets for calibrating decision models.

Two tasks, 30 pairs each, 15 positive and 15 negative, labels fixed by
construction (invented content only):

- ``needs_citation.jsonl`` — ``{id, text, label}``; label is True when the
  sentence states an empirical factual claim that needs a citation, False for
  roadmap, definition, plan and acknowledgement sentences.
- ``supported_by_passage.jsonl`` — ``{id, passage, sentence, label}``; label
  is True when the passage supports the sentence, False when it contradicts
  it or does not mention it.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Literal

Task = Literal["needs_citation", "supported_by_passage"]


def load_labelled_set(task: Task) -> list[dict[str, object]]:
    """Read one calibration set from package data."""
    path = resources.files(__package__) / f"{task}.jsonl"
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
