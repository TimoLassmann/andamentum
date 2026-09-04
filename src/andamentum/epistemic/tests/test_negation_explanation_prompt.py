"""The enumeration prompt asks for the negation-explanation, and says so once.

REFUTATION_FIRST_PLAN.md §3 (D5, D6) with §7 A7-6 applied. The prompt is the whole
of change 2: the two-genuine-rival gate is untouched, and only the supply of real
candidates changes. These tests pin the wording against silent drift, which the
version constant alone cannot do.
"""

from __future__ import annotations

from ..agents.integration import (
    PROPOSE_ONE_CANDIDATE_PROMPT,
    PROPOSE_ONE_CANDIDATE_PROMPT_VERSION,
)


class TestNegationExplanationPromptStaysCoherent:
    """A7-6: the prompt's unconditional quota ("include at least one...")
    and its closing hedge ("only where it is a genuine reading") used to
    contradict each other. The quota is gone; the request and the hedge
    stay, and the version constant is pinned against the wording so a
    future edit that drops the request is caught here rather than
    silently at runtime."""

    def test_version_is_bumped_and_travels_in_the_prompt(self) -> None:
        assert PROPOSE_ONE_CANDIDATE_PROMPT_VERSION >= 2
        assert f"[v{PROPOSE_ONE_CANDIDATE_PROMPT_VERSION}]" in PROPOSE_ONE_CANDIDATE_PROMPT

    def test_prompt_still_asks_for_the_negation_explanation(self) -> None:
        lowered = PROPOSE_ONE_CANDIDATE_PROMPT.lower()
        assert "negation" in lowered
        assert "take the claim's negation seriously" in lowered
        # The hedge survives...
        assert "genuine reading of this evidence" in lowered
        # ...but the unconditional quota it used to contradict is gone.
        assert "include at least one" not in lowered

    def test_the_shapes_are_offered_as_prompts_not_as_a_menu(self) -> None:
        """A model told a rival exists will produce one. The wording must ask
        the question and leave the answer open, which is what separates
        finding a rival from being handed one (plan §4, second risk)."""
        lowered = PROPOSE_ONE_CANDIDATE_PROMPT.lower()
        assert "prompts for that question, not a menu" in lowered
        assert "judge which,\nif any, fits this evidence" in lowered or "if any, fits this evidence" in lowered


