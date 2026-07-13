"""Prompt text and builders, isolated here for inspection and snapshot
testing. Both prompts follow the derive-then-judge (G-Eval form-filling)
pattern validated in ``experiments/dirichlet_confidence`` and
``experiments/pairwise_judge``: the model writes its reasoning BEFORE the
numbers, and the numbers are always a verbalized distribution over discrete
outcomes rather than a bare confidence.
"""

from __future__ import annotations

from .schemas import Criterion

# ── judge_score: one criterion per call ─────────────────────────────────

SCORE_INSTRUCTIONS = (
    "You are a careful, impartial judge scoring one candidate output against "
    "a single evaluation criterion. You are given the criterion, the output, "
    "and (if provided) the task/context the output was answering.\n\n"
    "First, reason briefly: (a) restate what a good output must do to satisfy "
    "THIS criterion; (b) check the given output against that standard, "
    "treating a confident factual or logical error as a serious fault even "
    "if the output otherwise reads well; (c) reach a judgment.\n\n"
    "Then express your judgment as three integers from 0 to 100 that sum to "
    "100: how much the output MEETS this criterion, PARTIALLY meets it, and "
    "FAILS it. These are a probability distribution over the three outcomes, "
    "not a single label: reserve extremes like 0 or 100 for clear-cut cases, "
    "and use intermediate values to express genuine uncertainty. Report the "
    "reasoning first, then the three numbers."
)

_SCORE_PROMPT_WITH_CONTEXT = (
    "CRITERION: {criterion_name} — {criterion_description}\n\n"
    "CONTEXT / TASK:\n{context}\n\n"
    "OUTPUT TO JUDGE:\n{output}\n\n"
    "Judge the output against the criterion above: reason first, then "
    "distribute 100 points across {{meets, partial, fails}}."
)

_SCORE_PROMPT_NO_CONTEXT = (
    "CRITERION: {criterion_name} — {criterion_description}\n\n"
    "OUTPUT TO JUDGE:\n{output}\n\n"
    "Judge the output against the criterion above: reason first, then "
    "distribute 100 points across {{meets, partial, fails}}."
)


def build_score_prompt(output: str, criterion: Criterion, context: str | None) -> str:
    """Build the per-criterion judge_score prompt.

    Includes a CONTEXT/TASK block only when ``context`` is not ``None`` —
    the public API's design says context is "strongly used by the prompt
    when present", so its presence changes the prompt shape rather than
    being folded in as an always-there-but-sometimes-empty field.
    """
    if context is not None:
        return _SCORE_PROMPT_WITH_CONTEXT.format(
            criterion_name=criterion.name,
            criterion_description=criterion.description,
            context=context,
            output=output,
        )
    return _SCORE_PROMPT_NO_CONTEXT.format(
        criterion_name=criterion.name,
        criterion_description=criterion.description,
        output=output,
    )


# ── judge_compare: both orders, position-neutral labels ────────────────
#
# Adapted near-verbatim from the validated experiments/pairwise_judge/elicit.py
# prompt. Candidates are ALWAYS labelled 'Response 1' / 'Response 2' — never
# A/B — so the schema itself carries no position information; the caller
# (panel.py) controls which of output_a/output_b is shown as Response 1 via
# the `order` argument, and always runs both orders.

COMPARE_INSTRUCTIONS_TEMPLATE = (
    "You are a careful, impartial judge comparing two candidate responses to "
    "the same task. You are given the original INPUT (if provided) and two "
    "responses, labelled Response 1 and Response 2.\n\n"
    "First, reason briefly: (a) restate what the input is actually asking "
    "for and what a good answer must do; (b) name the criteria that matter "
    "for THIS input — {criteria_sentence} — weighting correctness most, and "
    "treating a confident factual or logical error as a serious fault; "
    "(c) judge each response against those criteria and say why one is "
    "better.\n\n"
    "Then express your judgment as three integers from 0 to 100 that sum to "
    "100: how much you believe Response 1 is better, that the two are "
    "equally good (tie), and that Response 2 is better. These are a "
    "probability distribution over the three outcomes, not a single choice: "
    "reserve extremes like 0 or 100 for clear-cut cases, and use "
    "intermediate values to express genuine uncertainty. Do not favour a "
    "response for being longer, more confident, or listed first — judge "
    "only on the merits. Report the reasoning first, then the three "
    "numbers."
)

_COMPARE_PROMPT_WITH_CONTEXT = (
    "INPUT:\n{context}\n\n"
    "--- RESPONSE 1 ---\n{r1}\n\n"
    "--- RESPONSE 2 ---\n{r2}\n\n"
    "Judge which response is the better answer to the INPUT, following the "
    "instructions: reason first, then distribute 100 points across "
    "{{Response 1 better, tie, Response 2 better}}."
)

_COMPARE_PROMPT_NO_CONTEXT = (
    "--- RESPONSE 1 ---\n{r1}\n\n"
    "--- RESPONSE 2 ---\n{r2}\n\n"
    "Judge which response is the better one, following the instructions: "
    "reason first, then distribute 100 points across "
    "{{Response 1 better, tie, Response 2 better}}."
)


def build_compare_instructions(criteria: list[Criterion]) -> str:
    """Fold criterion names into the 'criteria that matter' sentence."""
    names = ", ".join(c.name.replace("_", " ") for c in criteria)
    criteria_sentence = (
        f"typically {names}" if names else "correctness, completeness, and clarity"
    )
    return COMPARE_INSTRUCTIONS_TEMPLATE.format(criteria_sentence=criteria_sentence)


def build_compare_prompt(output_1: str, output_2: str, context: str | None) -> str:
    """Build the judge_compare prompt for one presentation order.

    ``output_1``/``output_2`` are already order-arranged by the caller
    (panel.py) — Response 1 is whichever output is being shown first for
    this call, never a fixed output_a/output_b mapping.
    """
    if context is not None:
        return _COMPARE_PROMPT_WITH_CONTEXT.format(
            context=context, r1=output_1, r2=output_2
        )
    return _COMPARE_PROMPT_NO_CONTEXT.format(r1=output_1, r2=output_2)


__all__ = [
    "SCORE_INSTRUCTIONS",
    "build_score_prompt",
    "build_compare_instructions",
    "build_compare_prompt",
]
