"""K prompt phrasings for the SciFact claim-verification task.

Lever 1 of the method: ask the same question several ways. Each template means
the same thing but reads differently, so the spread between them measures how
much the answer depended on wording (phrasing brittleness). All templates ask
for the same 0-100 integer scale over the three classes so the rows are
directly comparable; the variation is in framing and reasoning style, including
one chain-of-thought variant.

Each template takes ``{claim}`` and ``{evidence}`` and is rendered into the
user prompt; the standardised output contract (three integers) lives in the
agent instructions, not the phrasing, so extraction is uniform.
"""

from __future__ import annotations

CLASSES = ["SUPPORT", "CONTRADICT", "NOINFO"]

# The shared system framing. The output contract is constant; only the phrasing
# of the *question* (the user prompt) varies across K.
SYSTEM_INSTRUCTIONS = (
    "You are a careful scientific claim-verification assistant. You are given a "
    "CLAIM and an EVIDENCE passage drawn from a scientific abstract. Decide how "
    "the evidence bears on the claim, across exactly three categories:\n"
    "  SUPPORT    — the evidence supports the claim\n"
    "  CONTRADICT — the evidence contradicts the claim\n"
    "  NOINFO     — the evidence neither supports nor contradicts the claim\n\n"
    "Express your judgment as three integers from 0 to 100 (one per category) "
    "that together sum to 100, reflecting how the total weight of belief is "
    "distributed across the categories. Report them in the structured fields "
    "support, contradict, noinfo."
)

# K = 5 phrasings (4 direct framings + 1 chain-of-thought).
PHRASINGS: list[str] = [
    # 1. Points-distribution framing.
    "CLAIM:\n{claim}\n\nEVIDENCE:\n{evidence}\n\n"
    "Distribute 100 points across the categories SUPPORT, CONTRADICT, and "
    "NOINFO according to how the evidence bears on the claim.",
    # 2. Probability framing.
    "CLAIM:\n{claim}\n\nEVIDENCE:\n{evidence}\n\n"
    "Give the probability (as a percentage from 0 to 100) that the evidence "
    "SUPPORTS, CONTRADICTS, or gives NO INFORMATION about the claim. The three "
    "percentages should sum to 100.",
    # 3. Confidence framing.
    "CLAIM:\n{claim}\n\nEVIDENCE:\n{evidence}\n\n"
    "How confident are you, as percentages summing to 100, that the relationship "
    "between the evidence and the claim is SUPPORT, CONTRADICT, or NOINFO?",
    # 4. Likelihood-rating framing.
    "CLAIM:\n{claim}\n\nEVIDENCE:\n{evidence}\n\n"
    "Rate the likelihood of each category — SUPPORT, CONTRADICT, NOINFO — on a "
    "0 to 100 scale, normalised so the three ratings sum to 100.",
    # 5. Chain-of-thought framing.
    "CLAIM:\n{claim}\n\nEVIDENCE:\n{evidence}\n\n"
    "First reason step by step about what the evidence actually establishes "
    "relative to the claim. Then, having reasoned, assign 0-100 scores summing "
    "to 100 across SUPPORT, CONTRADICT, and NOINFO.",
]


def render(phrasing: str, claim: str, evidence: str) -> str:
    """Render one phrasing template with the claim and evidence."""
    return phrasing.format(claim=claim, evidence=evidence)
