"""The goal-driven loop: review, edit, re-review until a win condition holds.

A plain bounded loop over functions whetstone already has — not a second
pydantic-graph graph beside ``v3/graph.py``. The house style
(``agentic_dialect``'s laws) allows this explicitly: a bounded loop that
generates, checks, and retries within one run is ordinary control flow, not
an external driver, as long as the stop decision is a deterministic function
of typed state and the bound traces to a named value (here,
``GoalSpec.max_rounds``, checked through :func:`_loop_allowed`, structurally
the same guard as ``andamentum.forge.runtime.loop_allowed`` — mirrored, not
imported, since whetstone must not depend on ``forge`` for a two-line helper).

Each round:

1. Run the ordinary ``review_document_v3(..., editor=True)`` over the
   document as it currently stands.
2. Compute the goal verdict for THAT document (from the round's findings,
   for ``structure``; from the reference checker, for ``citations``, whose
   problems are also handed to the round's editor beside the review's own
   findings; from the document text directly, for ``figures``; from a separate ``llm_judge.judge_score`` call, for
   ``judged`` — never the reviewer/editor pass that produced the edit).
3. Stop once the verdict is satisfied twice in a row — confirming the next
   round's ordinary edits, which may touch criteria unrelated to the goal,
   did not undo it. When the goal already holds and the editor finds
   nothing to change, the SAME text is reviewed again for that second
   confirmation, rather than treating "nothing left to change" as
   stalling — it is only stalling when the goal is NOT yet met. Otherwise,
   apply this round's edits and continue.

The loop never trusts a single pass, never lets the goal itself change once
the run has started (:class:`~.spec.GoalSpec` is frozen), and always stops
with a typed, visible reason: ``goal_met``, ``no_progress`` (no edits while
the goal is unmet, edits with no net effect, or the draft alternating
between two prior states),
``cap_reached``, or ``unresolvable`` (checked once, before spending a single
round, for a check that cannot be evaluated at all).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..reviewed_document import ReviewedDocument, carry_forward_edits
from ..schemas import AuthorQuestion, ReviewResult
from ..v3.criteria import Criterion
from ..v3.graph import review_document_v3
from ..v3.reviewers import DEFAULT_PROFILE, ReviewerProfile, author_for
from .figures import check_figure_text_match, unmentioned_figures
from .references import ReferenceGoalReport, check_reference_goal
from .spec import GoalSpec
from .structure import structure_goal_met
from .verdict import GoalVerdict

StoppedReason = Literal["goal_met", "no_progress", "cap_reached", "unresolvable"]

# How many times running does a check have to report "satisfied" before the
# loop trusts it — see the module docstring. Not user-configurable: it is
# the loop's own defence against a single lucky pass, not a cost/quality
# knob like `max_rounds`.
_CONFIRMATIONS_NEEDED = 2


@dataclass(frozen=True)
class RoundRecord:
    """What happened in one round, kept in full — nothing summarised away."""

    index: int
    verdict: GoalVerdict
    review: ReviewResult
    document_text_before: str


@dataclass(frozen=True)
class GoalRunResult:
    goal: GoalSpec
    rounds: list[RoundRecord]
    goal_met: bool
    stopped_reason: StoppedReason
    reason: str
    final_document: ReviewedDocument
    final_review: ReviewResult


def _loop_allowed(count: int, cap: int) -> bool:
    """True while a bounded loop may run again (dialect Law 5).

    Structurally identical to ``andamentum.forge.runtime.loop_allowed`` —
    mirrored rather than imported, since whetstone must not depend on
    ``forge`` for a two-line helper. Termination is a comparison against a
    value on ``GoalSpec``, never the model's own judgment.
    """
    return count < cap


def _unresolvable_reason(
    spec: GoalSpec, document: ReviewedDocument, *, references: ReferenceGoalReport | None
) -> str | None:
    """Checked once, before spending a round. ``None`` means resolvable."""
    if references is not None and references.unresolvable is not None:
        return references.unresolvable
    if spec.check == "figures":
        missing = unmentioned_figures(document.text, spec.figures)
        if missing:
            return (
                f"{', '.join(missing)} never appear in the text (no "
                '"Figure N" / "Fig. N" match), so there is nothing to '
                "check them against."
            )
    return None


async def _compute_verdict(
    spec: GoalSpec,
    *,
    document: ReviewedDocument,
    result: ReviewResult,
    judge_model: str,
    references: ReferenceGoalReport | None,
) -> GoalVerdict:
    """Dispatch on ``spec.check`` — a closed, caller-supplied value, never a
    classifier's guess (dialect Law 4). *references* is the round's reference
    check, already run on *document* before the review so its problems could
    reach the editor; set exactly when ``spec.check == "citations"``."""
    if spec.check == "structure":
        report = structure_goal_met(result)
        if report.satisfied:
            reason = "No major or moderate findings this round."
        else:
            reason = (
                f"{report.blocking_count} major/moderate finding(s) remain: "
                + "; ".join(report.blocking_titles)
            )
        return GoalVerdict(satisfied=report.satisfied, reason=reason)

    if spec.check == "citations":
        assert references is not None
        return GoalVerdict(satisfied=references.satisfied, reason=references.reason)

    if spec.check == "judged":
        # Imported here, not at module level: this is the one branch that
        # needs llm_judge, and it is a separate call from the reviewer/editor
        # pass above — never the same pass certifying its own edit.
        from andamentum.llm_judge import Criterion as JudgeCriterion
        from andamentum.llm_judge import judge_score

        score = await judge_score(
            document.text,
            criteria=[JudgeCriterion(name="goal", description=spec.text)],
            model=judge_model,
        )
        satisfied = score.overall == "meets" and not score.needs_review
        reason = score.per_criterion[0].reasoning if score.per_criterion else score.overall
        return GoalVerdict(satisfied=satisfied, reason=reason, progress_score=score.expected_score)

    if spec.check == "figures":
        # GoalSpec's own validator guarantees figure_model is set whenever
        # check == "figures" — asserted here only so the type checker knows
        # it too.
        assert spec.figure_model is not None
        report = await check_figure_text_match(
            document.text, spec.figures, model=spec.figure_model
        )
        if report.satisfied:
            reason = f"All {len(report.checks)} figure(s) match their caption and text."
        else:
            problems = []
            for check in report.checks:
                if check.matched is None:
                    problems.append(f"{check.label}: {check.error}")
                elif not check.matched.matches_description:
                    detail = "; ".join(check.matched.discrepancies) or "does not match"
                    problems.append(f"{check.label}: {detail}")
            reason = "; ".join(problems)
        return GoalVerdict(satisfied=report.satisfied, reason=reason)

    raise NotImplementedError(
        f"goal check {spec.check!r} is not wired into the loop yet."
    )


def _with_author_question(review: ReviewResult, *, goal_text: str, reason: str) -> ReviewResult:
    """A non-``goal_met`` stop gets one ``AuthorQuestion`` naming what the
    loop could not settle — the same schema whetstone already carries for
    "only the author can answer this", given its first real producer here."""
    question = AuthorQuestion(
        question=f'The goal loop stopped without confirming "{goal_text}": {reason}',
        sections_involved=[],
        why=reason,
    )
    return review.model_copy(update={"author_questions": [*review.author_questions, question]})


async def run_goal_loop(
    document: ReviewedDocument,
    goal: GoalSpec,
    *,
    model: str,
    reviewer: ReviewerProfile | None = None,
    criteria: list[Criterion] | None = None,
    guidelines_text: str | None = None,
    document_type: str = "auto",
    confirm_own_draft: bool = False,
    editor_criteria: list[str] | None = None,
    check_novelty: bool = False,
    novelty_search_depth: int = 2,
    judge_model: str | None = None,
    patch_author: str | None = None,
    workdir: Path | None = None,
) -> GoalRunResult:
    """Review, edit, and re-review *document* until *goal* holds.

    *document* is a document the caller already opened (see
    :func:`~..reviewed_document.open_source`) and still owns — this function
    never closes it. Every later round's document is this function's own
    working state, written under *workdir* (a caller-owned directory, or a
    temporary one this function creates and removes).

    *judge_model* is the model used for ``goal.check == "judged"``; ``None``
    reuses *model*. It is deliberately a separate parameter, not implicitly
    the same call, so a caller can hand the arbiter a different model from
    the one drafting the edits.

    See :mod:`.loop`'s module docstring for the stop conditions.
    """
    if workdir is None:
        with tempfile.TemporaryDirectory(prefix="whetstone-goal-") as tmp:
            return await _run_goal_loop(
                document,
                goal,
                model=model,
                reviewer=reviewer,
                criteria=criteria,
                guidelines_text=guidelines_text,
                document_type=document_type,
                confirm_own_draft=confirm_own_draft,
                editor_criteria=editor_criteria,
                check_novelty=check_novelty,
                novelty_search_depth=novelty_search_depth,
                judge_model=judge_model,
                patch_author=patch_author,
                workdir=Path(tmp),
            )
    return await _run_goal_loop(
        document,
        goal,
        model=model,
        reviewer=reviewer,
        criteria=criteria,
        guidelines_text=guidelines_text,
        document_type=document_type,
        confirm_own_draft=confirm_own_draft,
        editor_criteria=editor_criteria,
        check_novelty=check_novelty,
        novelty_search_depth=novelty_search_depth,
        judge_model=judge_model,
        patch_author=patch_author,
        workdir=workdir,
    )


async def _run_goal_loop(
    document: ReviewedDocument,
    goal: GoalSpec,
    *,
    model: str,
    reviewer: ReviewerProfile | None,
    criteria: list[Criterion] | None,
    guidelines_text: str | None,
    document_type: str,
    confirm_own_draft: bool,
    editor_criteria: list[str] | None,
    check_novelty: bool,
    novelty_search_depth: int,
    judge_model: str | None,
    patch_author: str | None,
    workdir: Path,
) -> GoalRunResult:
    resolved_judge_model = judge_model or model
    profile = reviewer if reviewer is not None else DEFAULT_PROFILE
    base_author = author_for(profile=profile, patch_author=patch_author)

    # Before anything reads the draft: the citations check below may call the
    # model before round 1's review would have run its own tripwire.
    if not confirm_own_draft:
        from .._confidentiality import check_confidentiality

        check_confidentiality(document.confidentiality_text)

    # The citations check runs BEFORE each round's review, not after: its
    # problems are findings the round's editor answers. The first round's
    # reading doubles as the resolvability check, so it is not made twice.
    references = (
        await check_reference_goal(document, model=model) if goal.check == "citations" else None
    )
    unresolvable = _unresolvable_reason(goal, document, references=references)
    if unresolvable is not None:
        return GoalRunResult(
            goal=goal,
            rounds=[],
            goal_met=False,
            stopped_reason="unresolvable",
            reason=unresolvable,
            final_document=document,
            final_review=_with_author_question(
                ReviewResult(), goal_text=goal.text, reason=unresolvable
            ),
        )

    rounds: list[RoundRecord] = []
    recent_texts: list[str] = [document.text]
    consecutive_met = 0
    round_idx = 0

    while _loop_allowed(round_idx, goal.max_rounds):
        round_idx += 1
        if goal.check == "citations" and round_idx > 1:
            references = await check_reference_goal(document, model=model)
        result = await review_document_v3(
            document,
            model=model,
            reviewer=reviewer,
            criteria=criteria,
            guidelines_text=guidelines_text,
            document_type=document_type,
            confirm_own_draft=confirm_own_draft,
            editor=True,
            editor_criteria=editor_criteria,
            check_novelty=check_novelty and round_idx == 1,
            flag_novelty=round_idx == 1,
            novelty_search_depth=novelty_search_depth,
            patch_author=patch_author,
            extra_findings=references.findings if references is not None else None,
        )
        verdict = await _compute_verdict(
            goal,
            document=document,
            result=result,
            judge_model=resolved_judge_model,
            references=references,
        )
        rounds.append(
            RoundRecord(
                index=round_idx,
                verdict=verdict,
                review=result,
                document_text_before=document.text,
            )
        )

        if verdict.satisfied:
            consecutive_met += 1
            if consecutive_met >= _CONFIRMATIONS_NEEDED:
                return GoalRunResult(
                    goal=goal,
                    rounds=rounds,
                    goal_met=True,
                    stopped_reason="goal_met",
                    reason=verdict.reason,
                    final_document=document,
                    final_review=result,
                )
            if not result.edits:
                # The goal already holds and the editor found nothing else
                # to fix — re-review the SAME text next round for the second
                # confirmation, rather than treating "nothing left to
                # change" as stalling, which it is only when the goal is NOT
                # yet met (the branch below).
                continue
        else:
            consecutive_met = 0
            if not result.edits:
                reason = "The editor found nothing left to change, and the goal was not met."
                return GoalRunResult(
                    goal=goal,
                    rounds=rounds,
                    goal_met=False,
                    stopped_reason="no_progress",
                    reason=reason,
                    final_document=document,
                    final_review=_with_author_question(
                        result, goal_text=goal.text, reason=reason
                    ),
                )

        interim = workdir / f"round{round_idx}.docx"
        new_document = await carry_forward_edits(
            result,
            document=document,
            output_path=interim,
            author=f"{base_author} (round {round_idx})",
            model=model,
            visible_watermark=False,
        )

        if new_document.text == document.text:
            reason = "This round's edits had no net effect on the text."
            return GoalRunResult(
                goal=goal,
                rounds=rounds,
                goal_met=False,
                stopped_reason="no_progress",
                reason=reason,
                final_document=new_document,
                final_review=_with_author_question(result, goal_text=goal.text, reason=reason),
            )
        if len(recent_texts) >= 2 and new_document.text == recent_texts[-2]:
            reason = "The draft is alternating between two states across rounds."
            return GoalRunResult(
                goal=goal,
                rounds=rounds,
                goal_met=False,
                stopped_reason="no_progress",
                reason=reason,
                final_document=new_document,
                final_review=_with_author_question(result, goal_text=goal.text, reason=reason),
            )

        recent_texts.append(new_document.text)
        document = new_document

    reason = f"Reached the {goal.max_rounds}-round cap with the goal not yet met."
    last_review = rounds[-1].review if rounds else ReviewResult()
    return GoalRunResult(
        goal=goal,
        rounds=rounds,
        goal_met=False,
        stopped_reason="cap_reached",
        reason=reason,
        final_document=document,
        final_review=_with_author_question(last_review, goal_text=goal.text, reason=reason),
    )
