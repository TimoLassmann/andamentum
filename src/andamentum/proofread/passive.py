"""Passive-voice detector.

Heuristic from Matt Might's ``passive-voice.el``: a "be" verb (am, is,
are, was, were, be, being, been) followed by a past participle. Past
participles are detected as either (a) a regular ``-ed`` ending or (b)
membership in a curated list of common irregular past participles.

This is a *heuristic*. False positives include adjectival uses ("the
result is interesting"), false negatives include split passives ("was
quickly eaten"). It's good enough for advisory linting; treat findings as
candidates to review, not as errors.
"""

from __future__ import annotations

import re

from .models import PassiveFinding, Span
from .sentences import Sentence, index_for_offset

_BE_VERBS = (
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "'s",
    "'re",
    "'m",
)

# Curated list of frequent irregular past participles. Not exhaustive —
# the regular -ed branch catches the majority of cases. Listed forms here
# are those that are unambiguously participles (not also bare nouns/verbs)
# so false-positive risk is low.
_IRREGULAR_PARTICIPLES = (
    "arisen",
    "awoken",
    "been",
    "borne",
    "born",
    "beaten",
    "become",
    "begun",
    "bent",
    "bet",
    "bidden",
    "bitten",
    "bled",
    "blown",
    "broken",
    "brought",
    "built",
    "burnt",
    "burst",
    "bought",
    "caught",
    "chosen",
    "clung",
    "come",
    "cost",
    "crept",
    "cut",
    "dealt",
    "dug",
    "done",
    "drawn",
    "dreamt",
    "driven",
    "drunk",
    "eaten",
    "fallen",
    "fed",
    "felt",
    "fought",
    "found",
    "fled",
    "flown",
    "forbidden",
    "forgotten",
    "forgiven",
    "frozen",
    "given",
    "gone",
    "grown",
    "hung",
    "had",
    "heard",
    "hidden",
    "hit",
    "held",
    "hurt",
    "kept",
    "knelt",
    "known",
    "laid",
    "led",
    "leapt",
    "learnt",
    "left",
    "lent",
    "let",
    "lain",
    "lost",
    "made",
    "meant",
    "met",
    "paid",
    "proven",
    "put",
    "quit",
    "read",
    "ridden",
    "risen",
    "run",
    "said",
    "seen",
    "sold",
    "sent",
    "set",
    "shaken",
    "shed",
    "shone",
    "shot",
    "shown",
    "shut",
    "sung",
    "sunk",
    "sat",
    "slept",
    "slid",
    "spoken",
    "spent",
    "spread",
    "stood",
    "stolen",
    "struck",
    "sworn",
    "swept",
    "swum",
    "swung",
    "taken",
    "taught",
    "torn",
    "told",
    "thought",
    "thrown",
    "trodden",
    "understood",
    "woken",
    "worn",
    "won",
    "written",
)

_BE_GROUP = "|".join(re.escape(v) for v in _BE_VERBS)
_IRR_GROUP = "|".join(re.escape(p) for p in _IRREGULAR_PARTICIPLES)

# be-verb, optional adverb (single -ly token), then a participle (regular -ed
# or irregular). We also allow "by <noun>" later but don't require it.
_PATTERN = re.compile(
    rf"\b({_BE_GROUP})\b(\s+\w+ly)?\s+(\w+ed|{_IRR_GROUP})\b",
    re.IGNORECASE,
)


def scan(text: str, sentences: list[Sentence]) -> list[PassiveFinding]:
    """Return passive-voice candidates in ``text``."""
    findings: list[PassiveFinding] = []
    for m in _PATTERN.finditer(text):
        findings.append(
            PassiveFinding(
                matched_text=m.group(0),
                span=Span(start=m.start(), end=m.end()),
                sentence_index=index_for_offset(sentences, m.start()),
            )
        )
    return findings
