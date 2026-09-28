"""Goal-driven loop over whetstone's own review/edit/re-review machinery.

Set a win condition and let review -> edit -> re-review run until it holds
for two rounds running, no round changes anything, or a round cap is hit.
Every check kind is built from data whetstone already produces, or a small,
narrowly-scoped new deterministic function — see each submodule's docstring.
"""

from .figures import (
    FigureCheckResult,
    FigureTextGoalReport,
    FigureTextMatch,
    check_figure_text_match,
    locate_figure_mentions,
)
from .references import ReferenceGoalReport, check_reference_goal
from .loop import GoalRunResult, RoundRecord, StoppedReason, run_goal_loop
from .spec import FigureRef, GoalCheck, GoalSpec
from .structure import StructureGoalReport, structure_goal_met
from .verdict import GoalVerdict

__all__ = [
    "FigureCheckResult",
    "FigureRef",
    "FigureTextGoalReport",
    "FigureTextMatch",
    "GoalCheck",
    "GoalRunResult",
    "GoalSpec",
    "GoalVerdict",
    "ReferenceGoalReport",
    "RoundRecord",
    "StoppedReason",
    "StructureGoalReport",
    "check_figure_text_match",
    "check_reference_goal",
    "locate_figure_mentions",
    "run_goal_loop",
    "structure_goal_met",
]
