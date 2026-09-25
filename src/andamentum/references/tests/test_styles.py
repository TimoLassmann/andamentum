"""Reading how a document cites, and which candidates are citations.

No network and no real model: the model calls are replaced where a test
needs one. The style test set in ``benchmarks/references`` measures the same
code on real styles; these pin the rules one by one.
"""

from __future__ import annotations

from andamentum.references import Hint, check_references
from andamentum.references import llm as llm_mod
from andamentum.references.scan import scan

AUTHOR_YEAR = """Introduction
Rare diseases are common (Nguengang Wakap et al., 2020). Costs are high (Yang and Cintina, 2022; Smith, 2019).
Sequencing helps, as Turro et al. (2019) showed. See Table 2 (Table, 2020).

References
Nguengang Wakap, S., Lambert, D. M. (2020). Estimating cumulative point prevalence of rare diseases. Eur J Hum Genet, 28, 165-173.
Turro, E., Astle, W. J. (2020). Whole-genome sequencing of patients with rare diseases. Nature, 583, 96-102.
Yang, G., Cintina, I. (2022). The national economic burden of rare disease. Orphanet J Rare Dis, 17, 163.
Zurynski, Y. (2017). Australian children living with rare diseases. Orphanet J Rare Dis, 12, 68.
"""


async def _read(text: str, **kw):
    return await check_references(text, lookup=False, **kw)


# ── author-year ─────────────────────────────────────────────────────────────


async def test_an_author_year_document_is_read_from_the_evidence() -> None:
    report = await _read(AUTHOR_YEAR)
    assert report.style is not None
    assert (report.style.style, report.style.decided_by) == ("author_year", "evidence")


async def test_author_year_problems_are_found() -> None:
    report = await _read(AUTHOR_YEAR)
    by_kind = {p.kind: p for p in report.problems}
    # Smith 2019 has no entry; Turro is 2020 in the list; Zurynski is never cited.
    assert AUTHOR_YEAR[by_kind["cited_not_listed"].start : by_kind["cited_not_listed"].end].startswith("(Yang")
    assert "2019" in by_kind["year_mismatch"].issue and "2020" in by_kind["year_mismatch"].issue
    assert AUTHOR_YEAR[by_kind["listed_not_cited"].start :].startswith("Zurynski")
    # The Turro entry is cited, with the wrong year: reported once, not also as never cited.
    assert sum(1 for p in report.problems if p.kind == "listed_not_cited") == 1


def test_names_that_are_not_authors_are_not_candidates() -> None:
    text = "as shown (Table, 2020) and in (Figure 3, 2019) during (January 2020)"
    assert scan(text, start=0, end=len(text)) == []


def test_three_named_authors_and_a_name_list_read_the_first_author() -> None:
    text = "(LeCun, Bengio and Hinton, 2015) and (Li & Durbin, 2009)"
    works = [w for c in scan(text, start=0, end=len(text)) for w in c.works]
    assert [(w.surname, w.year) for w in works] == [("LeCun", 2015), ("Li", 2009)]


async def test_two_entries_by_one_author_in_one_year_need_a_letter() -> None:
    text = (
        "A (Smith, 2020). B (Smith, 2020). C (Lee, 2019). D (Lee, 2019).\n\nReferences\n"
        "Smith, J. (2020). First paper. J 1, 1.\n"
        "Smith, J. (2020). Second paper. J 2, 2.\n"
        "Lee, K. (2019). Only paper. J 3, 3.\n"
    )
    report = await _read(text)
    assert [p.kind for p in report.problems] == ["ambiguous_citation"]


async def test_letters_choose_between_same_year_entries() -> None:
    text = (
        "A (Smith, 2020a). B (Smith, 2020b). C (Lee, 2019). D (Lee, 2019).\n\nReferences\n"
        "Smith, J. (2020a). First paper. J 1, 1.\n"
        "Smith, J. (2020b). Second paper. J 2, 2.\n"
        "Lee, K. (2019). Only paper. J 3, 3.\n"
    )
    report = await _read(text)
    assert report.problems == []


# ── superscript ─────────────────────────────────────────────────────────────

SUPERSCRIPT = (
    "A study\nAda Lovelace1, Alan Turing2\nIntroduction\n"
    "Diagnosis is slow1. Costs are high2. Sequencing helps3. We grew 105 cells on 3 m2 plates.\n"
    "References\n"
    "1. Wakap S. Prevalence of rare diseases. Eur J Hum Genet 2020;28:165.\n"
    "2. Yang G. Economic burden. Orphanet J Rare Dis 2022;17:163.\n"
    "3. Turro E. Genome sequencing. Nature 2020;583:96.\n"
)


def _hint_after(text: str, before: str, digit: str) -> Hint:
    start = text.index(before) + len(before)
    assert text[start : start + len(digit)] == digit
    return Hint(kind="superscript", start=start, end=start + len(digit))


async def test_raised_numbers_are_read_and_distractors_set_aside() -> None:
    hints = [
        _hint_after(SUPERSCRIPT, "Ada Lovelace", "1"),
        _hint_after(SUPERSCRIPT, "Alan Turing", "2"),
        _hint_after(SUPERSCRIPT, "Diagnosis is slow", "1"),
        _hint_after(SUPERSCRIPT, "Costs are high", "2"),
        _hint_after(SUPERSCRIPT, "Sequencing helps", "3"),
        _hint_after(SUPERSCRIPT, "We grew 10", "5"),
        _hint_after(SUPERSCRIPT, "3 m", "2"),
    ]
    report = await _read(SUPERSCRIPT, hints=hints)
    assert report.style is not None and report.style.style == "numeric_superscript"
    accepted = [SUPERSCRIPT[c.start - 4 : c.end] for c in report.citations]
    assert accepted == ["slow1", "high2", "elps3"]
    set_aside = {
        SUPERSCRIPT[j.candidate.start - 4 : j.candidate.end]: j.reason
        for j in report.judged
        if not j.is_citation
    }
    assert "affiliation" in set_aside["lace1"] and "affiliation" in set_aside["ring2"]
    exponents = [r for k, r in set_aside.items() if k.endswith(("105", "m2"))]
    assert len(exponents) == 2 and all("exponent" in r for r in exponents)
    assert report.problems == []


async def test_without_formatting_the_citation_order_reads_glued_digits() -> None:
    """A converted PDF keeps "slow1." with nothing to say the 1 was raised.
    A list numbered in citation order introduces 1, 2, 3 in turn."""
    report = await _read(SUPERSCRIPT)
    assert report.style is not None and report.style.style == "numeric_superscript"
    assert [c.numbers for c in report.citations] == [[1], [2], [3]]
    assert all(
        j.reason in ("rule: the next number in citation order", "rule: a raised number after a word or punctuation")
        or not j.is_citation
        for j in report.judged
    )


# ── the model: only for what the evidence leaves open ───────────────────────


async def test_what_the_rules_leave_open_is_unjudged_without_a_model() -> None:
    text = SUPERSCRIPT.replace("We grew 105 cells on 3 m2 plates.", "Reads were counted at day3.")
    report = await _read(text)
    unjudged = [j for j in report.judged if j.reason.startswith("not judged")]
    assert [text[j.candidate.start - 3 : j.candidate.end] for j in unjudged] == ["day3"]
    # With a candidate unjudged, "never cited" is not reported at all.
    assert not [p for p in report.problems if p.kind == "listed_not_cited"]


async def test_the_model_sees_only_the_open_cases_in_short_excerpts(monkeypatch) -> None:
    seen: list[str] = []

    async def fake(items: str, *, style: str, model: str, list_line: str = ""):
        seen.append(items)
        return llm_mod.CitationIds(citation_ids=[])

    monkeypatch.setattr(llm_mod, "ask_citations", fake)
    text = SUPERSCRIPT.replace("We grew 105 cells on 3 m2 plates.", "Reads were counted at day3.")
    report = await _read(text, model="stub:model")
    assert len(seen) == 1 and seen[0].count("⟦") == 1
    assert "day⟦3⟧" in seen[0]
    assert len(seen[0]) < 2 * llm_mod.CONTEXT_CHARS + 40
    assert [j.reason for j in report.judged if "day" in text[j.candidate.start - 3 : j.candidate.start]] == ["model"]


async def test_mixed_evidence_goes_to_the_model_and_its_answer_is_checked(monkeypatch) -> None:
    async def says_author_year(prompt: str, *, model: str):
        return llm_mod.StyleAnswer(style="author_year", reason="stub")

    monkeypatch.setattr(llm_mod, "ask_style", says_author_year)
    # Two bracket citations and two superscripts against a numbered list:
    # neither leads by three to one, so the evidence does not settle it.
    text = "Intro\nOne [1]. Two [2]. Three3. Four4.\nReferences\n1. A B (2020) T. J\n2. C D (2021) T. J\n3. E F (2022) T. J\n4. G H (2023) T. J\n"
    report = await _read(text, model="stub:model")
    # The model named a style with no candidates in the text: refused.
    assert report.style is not None and report.style.style == "undecided"
    assert report.style.decided_by == "model"


async def test_without_a_model_mixed_evidence_is_undecided_and_checks_nothing() -> None:
    text = "Intro\nOne [1]. Two [2]. Three3. Four4.\nReferences\n1. A B (2020) T. J\n2. C D (2021) T. J\n3. E F (2022) T. J\n4. G H (2023) T. J\n"
    report = await _read(text)
    assert report.style is not None and report.style.style == "undecided"
    assert report.problems == []


# ── reference-manager fields ────────────────────────────────────────────────


async def test_a_reference_manager_field_settles_the_style() -> None:
    text = "Intro\nClaim (Smith, 2020).\nReferences\nSmith, J. (2020). Paper. J 1, 1.\n"
    start = text.index("(Smith")
    hint = Hint(kind="field_citation", start=start, end=start + len("(Smith, 2020)"), payload="ADDIN ZOTERO_ITEM CSL_CITATION {}")
    report = await _read(text, hints=[hint])
    assert report.style is not None
    assert (report.style.style, report.style.decided_by) == ("author_year", "evidence")
    assert [j.reason for j in report.judged] == ["rule: a reference-manager field"]
