"""End-to-end tests for ``analyze()``."""

from andamentum.proofread import (
    ProofreadResult,
    ReadabilityScores,
    analyze,
)


def test_analyze_returns_proofread_result():
    r = analyze("The cat sat. The dog ran.")
    assert isinstance(r, ProofreadResult)
    assert isinstance(r.readability, ReadabilityScores)


def test_analyze_empty_text():
    r = analyze("")
    assert r.readability.word_count == 0
    assert r.weasel_words == []
    assert r.passive_voice == []
    assert r.duplicate_words == []
    assert r.weak_openers == []
    assert r.adverbs.adverb_count == 0
    assert "Empty" in r.summary or "nothing" in r.summary.lower()


def test_analyze_kitchen_sink():
    text = (
        "There are many studies. The data was eaten by the dog. "
        "It is clear that the the duplicate exists. "
        "Cats quickly run very fast."
    )
    r = analyze(text)

    weasel_words = {f.word.lower() for f in r.weasel_words}
    assert "many" in weasel_words
    assert "very" in weasel_words
    assert "clearly" not in weasel_words  # "clear" not "clearly"

    assert any("eaten" in f.matched_text.lower() for f in r.passive_voice)

    duplicate_words = {f.word.lower() for f in r.duplicate_words}
    assert "the" in duplicate_words

    weak = {f.matched_text.lower() for f in r.weak_openers}
    assert any("there are" in w for w in weak)
    assert any("it is" in w for w in weak)

    assert r.adverbs.adverb_count >= 1  # "quickly"

    assert isinstance(r.summary, str)
    assert r.summary  # non-empty


def test_analyze_is_deterministic():
    text = "Many studies were performed quickly. There are several issues."
    r1 = analyze(text)
    r2 = analyze(text)
    assert r1 == r2


def test_analyze_summary_mentions_smog():
    r = analyze("A simple sentence. Another simple sentence. One more.")
    assert "SMOG" in r.summary


def test_analyze_clean_prose_reports_no_flags():
    # Carefully written: no weasels, no passives, no duplicates, no weak
    # openers, low adverb density.
    text = "Scientists ran the experiment three times. The results matched."
    r = analyze(text)
    assert r.weasel_words == []
    assert r.passive_voice == []
    assert r.duplicate_words == []
    assert r.weak_openers == []
    assert "No style flags" in r.summary
