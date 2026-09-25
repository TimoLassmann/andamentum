"""Notes styles: citations in footnotes or endnotes (Chicago notes and kin)."""

from __future__ import annotations

from andamentum.references import Hint, check_references
from andamentum.references import llm as llm_mod
from andamentum.references.notes import read_note

# ── reading one note ────────────────────────────────────────────────────────


def test_the_forms_of_a_note() -> None:
    cases = {
        "Sophie Gudmundsson et al., “Variant Interpretation,” Human Mutation 43, no. 8 (2021): 1012–30.": ("full", [("Gudmundsson", 2021)]),
        "Gudmundsson et al., “Variant Interpretation,” 1015.": ("short", [("Gudmundsson", None)]),
        "Michael Pollan, The Omnivore’s Dilemma (New York: Penguin, 2006), 99–100.": ("full", [("Pollan", 2006)]),
        "Pollan, Omnivore’s Dilemma, 3.": ("short", [("Pollan", None)]),
        "Ibid., 22.": ("ibid", []),
        "This point is discussed further below.": ("comment", []),
        "Interview with the author, March 2019.": ("unclear", []),
    }
    for text, (form, works) in cases.items():
        got_form, got_works = read_note(text)
        assert (got_form, [(w.surname, w.year) for w in got_works]) == (form, works), text


def test_a_page_after_a_short_title_is_not_a_year() -> None:
    """"…,” 1755." is page 1755 of a short form, not the year 1755."""
    form, works = read_note("Li and Durbin, “Fast and Accurate Short Read Alignment,” 1755.")
    assert form == "short" and works[0].year is None


def test_signal_words_and_several_works_in_one_note() -> None:
    form, works = read_note(
        "See also Love, Huber, and Anders, “Moderated Estimation,” 550; Robinson et al., “Integrative Genomics Viewer,” 24."
    )
    assert form == "short"
    assert [w.surname for w in works] == ["Love", "Robinson"]


# ── a document ──────────────────────────────────────────────────────────────

BIBLIOGRAPHY = (
    "\nBibliography\n"
    "Gudmundsson, Sophie. 2021. “Variant Interpretation Using Population Databases.” Human Mutation 43: 1012–30.\n"
    "Pollan, Michael. 2006. The Omnivore’s Dilemma. New York: Penguin.\n"
    "Zurynski, Yvonne. 2017. “Australian Children Living with Rare Diseases.” Orphanet 12: 68.\n"
)


def _document(notes: list[str], *, bibliography: bool = True) -> tuple[str, list[Hint]]:
    body = "Introduction\n"
    hints: list[Hint] = []
    for n, note in enumerate(notes, start=1):
        body += f"Sentence number {n}."
        hints.append(Hint(kind="note", start=len(body), end=len(body), payload=note))
        body += "\n"
    text = body.rstrip("\n") + (BIBLIOGRAPHY if bibliography else "")
    return text, hints


FULL_G = "Sophie Gudmundsson, “Variant Interpretation Using Population Databases,” Human Mutation 43 (2021): 1012."
FULL_P = "Michael Pollan, The Omnivore’s Dilemma (New York: Penguin, 2006), 99."
FULL_Z = "Yvonne Zurynski, “Australian Children Living with Rare Diseases,” Orphanet 12 (2017): 68."


async def _read(text: str, hints: list[Hint], **kw):
    return await check_references(text, hints=hints, lookup=False, **kw)


async def test_a_clean_notes_document_has_no_problems() -> None:
    text, hints = _document([FULL_G, "Ibid., 1015.", FULL_P, "Gudmundsson, “Variant Interpretation,” 1020.", FULL_Z])
    report = await _read(text, hints)
    assert report.style is not None and report.style.style == "notes"
    assert report.problems == []


async def test_each_notes_problem_is_found_on_its_note() -> None:
    text, hints = _document(
        [
            FULL_G,
            FULL_P.rstrip(".") + "; " + FULL_G,  # two works
            "Ibid., 5.",  # unclear: two sources before it
            "Ada Lovelace, “Notes on the Analytical Engine,” Taylor’s Scientific Memoirs 3 (1843): 666.",  # not listed
            "This point is discussed further below.",
        ]
    )
    report = await _read(text, hints)
    by_kind = {p.kind: p for p in report.problems}
    assert set(by_kind) == {"ibid_unclear", "cited_not_listed", "listed_not_cited"}
    assert text[by_kind["ibid_unclear"].start : by_kind["ibid_unclear"].end] == "3."
    assert text[by_kind["cited_not_listed"].start : by_kind["cited_not_listed"].end] == "4."
    assert text[by_kind["listed_not_cited"].start :].startswith("Zurynski")


async def test_ibid_after_a_comment_is_unclear() -> None:
    text, hints = _document([FULL_G, FULL_P, "This point is discussed further below.", "Ibid.", FULL_Z])
    report = await _read(text, hints)
    assert [p.kind for p in report.problems] == ["ibid_unclear"]


async def test_without_a_bibliography_a_short_form_needs_an_earlier_full_one() -> None:
    text, hints = _document(
        ["Pollan, Omnivore’s Dilemma, 3.", FULL_P, FULL_G, "Gudmundsson, “Variant Interpretation,” 1020."],
        bibliography=False,
    )
    report = await _read(text, hints)
    assert report.found_list is False
    assert [(p.kind, text[p.start : p.end]) for p in report.problems] == [("short_before_full", "1.")]


async def test_without_a_bibliography_the_full_notes_are_what_is_looked_up() -> None:
    import httpx

    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url))
        return httpx.Response(200, json={"message": {"items": []}})

    text, hints = _document([FULL_G, "Ibid., 3.", FULL_P], bibliography=False)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await check_references(text, hints=hints, client=client)
    assert report.looked_up
    assert len(report.lookups) == 2  # the two full notes, not the Ibid.
    assert all("Sentence" not in url for url in sent)  # never the draft's text


async def test_a_note_the_rules_cannot_read_goes_to_the_model_as_note_text(monkeypatch) -> None:
    seen: list[str] = []

    async def fake(items: str, *, style: str, model: str, list_line: str = ""):
        seen.append(items)
        return llm_mod.CitationIds(citation_ids=[])

    monkeypatch.setattr(llm_mod, "ask_citations", fake)
    text, hints = _document([FULL_G, FULL_P, FULL_Z, "Interview with the author, March 2019."])
    report = await _read(text, hints, model="stub:model")
    assert len(seen) == 1 and "a note reading: ⟦Interview with the author" in seen[0]
    assert report.problems == []


async def test_markdown_footnotes_are_notes() -> None:
    text = (
        "Introduction\nRare diseases are common.[^1] Costs are high.[^2] More.[^3]\n\n"
        "[^1]: " + FULL_G + "\n[^2]: Ibid., 3.\n[^3]: " + FULL_P + "\n"
    )
    report = await check_references(text, lookup=False)
    assert report.style is not None and report.style.style == "notes"
    assert len(report.citations) == 3
    assert report.problems == []
