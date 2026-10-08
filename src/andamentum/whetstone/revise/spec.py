"""The frozen revision specification — built once, before the loop starts.

Nothing inside the loop can hand back an edited copy: the condition a round is
judged against is exactly the condition the caller asked for, never rewritten
by a round that would rather change what "done" means than change the draft.
That is the concrete defence against a documented failure mode in iterative
LLM refinement: a loop checked against its own judge can learn to satisfy
the judge without improving the underlying work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, model_validator

RevisionCheck = Literal["judged", "structure", "citations", "figures"]

# The condition a report names for the checks that do not read a text of the
# caller's: ``structure``, ``citations`` and ``figures`` are decided by a fixed
# check, so their wording is fixed too. ``judged`` has none — its text is what
# the judge scores, and only the caller can state it.
FIXED_CONDITION_TEXT: dict[str, str] = {
    "structure": "no major or moderate problems remain",
    "citations": "every citation matches the reference list",
    "figures": "every named figure matches its caption and the text",
}


class FigureRef(BaseModel):
    """One named figure and the image file that shows it.

    ``label`` is matched against ``"Figure N"`` / ``"Fig. N"`` mentions in
    the manuscript text (see ``revise.figures.locate_figure_mentions``) — it
    is not required to be the literal caption text, only to name the figure
    the same way the prose does.
    """

    model_config = {"frozen": True}

    label: str
    path: Path


class RevisionSpec(BaseModel):
    """The condition that ends a revision, fixed for the life of one run.

    ``check="judged"`` needs ``text``: a condition in the caller's words, which
    a separate model call scores every round. The other checks are decided by
    a fixed check, so ``text`` may be left out; it is then filled with the
    check's fixed wording (:data:`FIXED_CONDITION_TEXT`), which is what a
    report says the loop was trying to achieve.
    """

    model_config = {"frozen": True}

    text: str = ""
    check: RevisionCheck = "judged"
    max_rounds: int = 4
    figures: tuple[FigureRef, ...] = ()
    figure_model: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _fill_fixed_text(cls, data: Any) -> Any:
        if isinstance(data, dict):
            check = data.get("check", "judged")
            if not data.get("text") and check in FIXED_CONDITION_TEXT:
                data = {**data, "text": FIXED_CONDITION_TEXT[check]}
        return data

    @model_validator(mode="after")
    def _validate(self) -> "RevisionSpec":
        if self.max_rounds < 1:
            raise ValueError(f"max_rounds must be at least 1, got {self.max_rounds}.")
        if self.check == "judged" and not self.text.strip():
            raise ValueError("check='judged' needs `text`: the condition to judge.")
        if self.check == "figures":
            if not self.figures:
                raise ValueError(
                    "check='figures' needs at least one entry in `figures`."
                )
            if not self.figure_model:
                raise ValueError("check='figures' needs `figure_model` set.")
        elif self.figures or self.figure_model:
            raise ValueError(
                f"`figures`/`figure_model` are only used with check='figures', "
                f"not check={self.check!r}."
            )
        return self
