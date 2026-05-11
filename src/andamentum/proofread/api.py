"""Single public entry point: ``analyze(text) -> ProofreadResult``."""

from __future__ import annotations

from . import extras, passive, readability, weasel
from .models import ProofreadResult
from .sentences import segment


def analyze(text: str) -> ProofreadResult:
    """Run every deterministic proofreading check on ``text``.

    Pure function, no I/O, no LLM. Same input → same output.

    Parameters
    ----------
    text:
        The raw input prose. Markdown is fine; markup tokens won't fool
        the readability scores but may produce extra weasel/passive matches
        if they happen to contain trigger words.

    Returns
    -------
    ProofreadResult
        A pydantic object with five readability scores, lists of findings
        (weasel words, passive voice, duplicate words, weak openers),
        adverb stats, and a one-line ``summary``.
    """
    sentences = segment(text)
    scores = readability.compute(text)
    weasels = weasel.scan(text, sentences)
    passives = passive.scan(text, sentences)
    duplicates = extras.scan_duplicates(text, sentences)
    openers = extras.scan_weak_openers(sentences)
    adv = extras.adverb_stats(text)

    summary = _summarise(
        scores.smog_index,
        scores.flesch_reading_ease,
        scores.word_count,
        len(weasels),
        len(passives),
        len(duplicates),
        len(openers),
        adv.adverb_density,
    )

    return ProofreadResult(
        readability=scores,
        weasel_words=weasels,
        passive_voice=passives,
        duplicate_words=duplicates,
        weak_openers=openers,
        adverbs=adv,
        summary=summary,
    )


def _summarise(
    smog: float,
    flesch_ease: float,
    words: int,
    weasels: int,
    passives: int,
    duplicates: int,
    openers: int,
    adverb_density: float,
) -> str:
    """Build the one-line human-readable summary string."""
    if words == 0:
        return "Empty input — nothing to score."

    parts = [
        f"SMOG {smog:.1f} (grade ≈{smog:.0f}); Flesch ease {flesch_ease:.0f}.",
        f"{words} words.",
    ]
    flags: list[str] = []
    if weasels:
        flags.append(f"{weasels} weasel")
    if passives:
        flags.append(f"{passives} passive")
    if duplicates:
        flags.append(f"{duplicates} duplicate")
    if openers:
        flags.append(f"{openers} weak opener")
    if adverb_density > 0.05:
        flags.append(f"adverb density {adverb_density:.0%}")
    if flags:
        parts.append("Flags: " + ", ".join(flags) + ".")
    else:
        parts.append("No style flags.")
    return " ".join(parts)
