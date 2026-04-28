"""Lexicon of overclaim / "reviewer 2 bait" patterns.

Used by the ``overclaim`` lens prompt as a recall substrate. The patterns
are deliberately broad — the lens prompt narrows them down by asking
the LLM to verify each candidate against the section's evidence.
Pure-regex flagging would generate too many false positives in a
literature-review context where these words have legitimate uses.

Categorised so the lens can name the failure mode in its rationale.
"""

from __future__ import annotations

import re

# Words that overclaim novelty without citation context. Most frequently
# load-bearing in abstracts and introductions where authors stake their
# contribution.
NOVELTY_WORDS = [
    "novel",
    "first",
    "unprecedented",
    "landmark",
    "groundbreaking",
    "paradigm-shifting",
    "paradigm shifting",
    "revolutionary",
    "breakthrough",
    "pioneering",
    "seminal",
    "transformative",
    "game-changing",
    "game changing",
]

# Strong-effect language whose use should match the data. Often paired
# with reported effect sizes; frequently used WITHOUT them.
STRENGTH_WORDS = [
    "dramatic",
    "dramatically",
    "robust",
    "robustly",
    "remarkable",
    "remarkably",
    "striking",
    "strikingly",
    "compelling",
    "profound",
    "profoundly",
    "substantial",
    "substantially",
    "marked",
    "markedly",
    "considerable",
    "considerably",
]

# Causal-mechanism language. Legitimate when the design supports causal
# inference (RCT, instrumental variable, etc.); a lurking risk in
# observational/correlational work.
CAUSAL_WORDS = [
    "causes",
    "caused",
    "leading to",
    "leads to",
    "resulting in",
    "results in",
    "due to",
    "because of",
    "drives",
    "driven by",
    "underlying",
    "mechanism of",
    "mechanism for",
]

# Generalisation language — "humans" / "the population" / "in general"
# claims that sometimes outrun a small or non-representative sample.
GENERALISATION_WORDS = [
    "in humans",
    "in mammals",
    "in clinical practice",
    "in general",
    "broadly",
    "across the population",
    "for all",
    "always",
    "universally",
    "fundamental",
    "fundamentally",
]


# Compiled, case-insensitive, word-boundary-respecting matchers.
def _compile(words: list[str]) -> re.Pattern[str]:
    """Build a single regex matching any phrase in ``words``."""
    # Sort longest-first so multi-word phrases match before their words
    sorted_words = sorted(set(words), key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(re.escape(w) for w in sorted_words) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


NOVELTY_RE = _compile(NOVELTY_WORDS)
STRENGTH_RE = _compile(STRENGTH_WORDS)
CAUSAL_RE = _compile(CAUSAL_WORDS)
GENERALISATION_RE = _compile(GENERALISATION_WORDS)


def find_overclaim_candidates(text: str) -> list[tuple[str, str]]:
    """Return (category, matched_phrase) pairs found in the text.

    The lens uses these as candidates to verify against the actual
    evidence. A candidate isn't a finding — most of these words have
    legitimate uses. The lens decides per-instance.
    """
    out: list[tuple[str, str]] = []
    for category, pattern in (
        ("novelty", NOVELTY_RE),
        ("strength", STRENGTH_RE),
        ("causal", CAUSAL_RE),
        ("generalisation", GENERALISATION_RE),
    ):
        for m in pattern.finditer(text):
            out.append((category, m.group(0)))
    return out
