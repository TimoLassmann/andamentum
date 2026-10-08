"""The frozen goal specification — built once, before the loop starts.

Nothing inside the loop can hand back an edited copy: the goal a round is
judged against is exactly the goal the caller asked for, never rewritten by
a round that would rather change what "done" means than change the draft.
That is the concrete defence against a documented failure mode in iterative
LLM refinement: a loop checked against its own judge can learn to satisfy
the judge without improving the underlying work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

GoalCheck = Literal["judged", "structure", "citations", "figures"]


class FigureRef(BaseModel):
    """One named figure and the image file that shows it.

    ``label`` is matched against ``"Figure N"`` / ``"Fig. N"`` mentions in
    the manuscript text (see ``goal.figures.locate_figure_mentions``) — it is
    not required to be the literal caption text, only to name the figure the
    same way the prose does.
    """

    model_config = {"frozen": True}

    label: str
    path: Path


class GoalSpec(BaseModel):
    """A win condition, fixed for the life of one goal-loop run.

    ``text`` is always kept, even for checks that do not read it as a judge
    criterion (``structure``, ``citations``, ``figures``) — it is what a
    report says the loop was trying to achieve.
    """

    model_config = {"frozen": True}

    text: str
    check: GoalCheck = "judged"
    max_rounds: int = 4
    figures: tuple[FigureRef, ...] = ()
    figure_model: str | None = None

    @model_validator(mode="after")
    def _figures_only_with_figure_check(self) -> "GoalSpec":
        if self.max_rounds < 1:
            raise ValueError(f"max_rounds must be at least 1, got {self.max_rounds}.")
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
