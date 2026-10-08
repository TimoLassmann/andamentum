"""The reference checker, with Crossref replaced by a stub transport."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import httpx

from andamentum.references import check_references
from andamentum.references.parse import find_reference_list, split_entries
from andamentum.references.scan import scan

DRAFT = """Introduction
Rare diseases are common [1]. Diagnosis is slow [2, 3]. Costs are high [5].
Sequencing helps [3-4].

References
1. Nguengang Wakap S, Lambert DM (2020) Estimating cumulative point prevalence of rare diseases. Eur J Hum Genet 28:165-173
2. Zurynski Y, Deverell M (2017) Australian children living with rare diseases: experiences of diagnosis. Orphanet J Rare Dis 12:68
3. Yang G, Cintina I (2022) The national economic burden of rare disease in the United States in 2019. Orphanet J Rare Dis 17:163
4. Nisar H, Wajid B (2021) Whole-genome sequencing as a first-tier diagnostic framework. Front Genet 12:1
4. Smith J (2019) An unrelated duplicate number. J Test 1:1
6. Nguengang Wakap S, Lambert DM (2020) Estimating cumulative point prevalence of rare diseases. Eur J Hum Genet 28:165-173
"""


async def _offline(text: str = DRAFT):
    return await check_references(text, lookup=False)


# ── parsing ─────────────────────────────────────────────────────────────────


def test_the_list_starts_after_the_last_references_heading() -> None:
    text = "See the References below.\nBody.\nReferences\n1. A (2020) T. J\n"
    span = find_reference_list(text)
    assert span is not None and text[span[0] :].lstrip().startswith("1. A")


def test_entries_keep_their_numbers_and_offsets() -> None:
    start, end = find_reference_list(DRAFT) or (0, 0)
    entries = split_entries(DRAFT, start=start, end=end)
    assert [e.number for e in entries] == [1, 2, 3, 4, 4, 6]
    assert DRAFT[entries[0].start : entries[0].end] == entries[0].text


def test_a_converter_list_item_and_a_wrapped_entry_are_one_entry() -> None:
    text = "References\n- [1] Author A (2020) A long title\n  that wraps. J 1:2\n- [2] B (2021) T. J\n"
    start, end = find_reference_list(text) or (0, 0)
    entries = split_entries(text, start=start, end=end)
    assert [e.number for e in entries] == [1, 2]
    assert entries[0].text.endswith("that wraps. J 1:2")


def test_markers_expand_ranges_and_skip_implausible_ones() -> None:
    text = "a [2–4, 8] b [1-400] c [9]"
    assert [
        m.numbers for m in scan(text, start=0, end=len(text)) if m.source == "brackets"
    ] == [
        [2, 3, 4, 8],
        [9],
    ]


# ── the offline checks ──────────────────────────────────────────────────────


async def test_every_number_problem_is_found_and_located() -> None:
    report = await _offline()
    kinds = sorted(p.kind for p in report.problems)
    assert kinds == [
        "cited_not_listed",  # [5]
        "duplicate_entry",  # 6 is the same work as 1
        "duplicate_number",  # two 4s
        "listed_not_cited",  # 6
    ]
    missing = next(p for p in report.problems if p.kind == "cited_not_listed")
    assert DRAFT[missing.start : missing.end] == "[5]"
    assert missing.fix


async def test_a_citation_out_of_first_citation_order_is_reported_once() -> None:
    text = "a [1] b [2] c [4] d [3] e [5]\n\nReferences\n" + "".join(
        f"{n}. Author{n} ({2000 + n}) Title number {n}. J\n" for n in range(1, 6)
    )
    report = await check_references(text, lookup=False)
    order = [p for p in report.problems if p.kind == "out_of_order"]
    assert len(order) == 1
    assert text[order[0].start : order[0].end] == "[4]"


async def test_an_alphabetical_list_is_not_checked_for_order() -> None:
    text = "a [4] b [2] c [5] d [1] e [3]\n\nReferences\n" + "".join(
        f"{n}. Author{n} ({2000 + n}) Title number {n}. J\n" for n in range(1, 6)
    )
    report = await check_references(text, lookup=False)
    assert not [p for p in report.problems if p.kind == "out_of_order"]


async def test_no_list_means_nothing_is_checked_and_it_says_so() -> None:
    report = await check_references("A draft with no reference list [1].", lookup=False)
    assert report.found_list is False
    assert report.problems == []


# ── the lookup ──────────────────────────────────────────────────────────────


def _item(
    title: str,
    year: int,
    family: str,
    kind: str = "journal-article",
    doi: str = "10.1/x",
):
    return {
        "title": [title],
        "issued": {"date-parts": [[year]]},
        "author": [{"family": family}],
        "type": kind,
        "DOI": doi,
    }


def _crossref(route):
    """A transport answering like Crossref; *route* maps a request to a
    (status, json) pair. Records every request for inspection."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        status, body = route(request)
        return httpx.Response(status, json=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen


def _query(request: httpx.Request) -> str:
    return parse_qs(urlparse(str(request.url)).query).get("query.bibliographic", [""])[
        0
    ]


async def _lookup(text: str, route):
    client, seen = _crossref(route)
    async with client:
        report = await check_references(text, client=client)
    return report, seen


ONE = "Body [1].\n\nReferences\n1. Yang G, Cintina I (2022) The national economic burden of rare disease in the United States in 2019. Orphanet J Rare Dis 17:163\n"


async def test_a_matching_work_is_found_and_only_the_entry_is_sent() -> None:
    title = "The national economic burden of rare disease in the United States in 2019"
    report, seen = await _lookup(
        ONE, lambda r: (200, {"message": {"items": [_item(title, 2022, "Yang")]}})
    )
    assert [r.status for r in report.lookups] == ["found"]
    assert report.problems == []
    # What left the machine is the entry, not the draft.
    assert "Body" not in _query(seen[0])
    assert "national economic burden" in _query(seen[0])


async def test_an_unrelated_nearest_work_is_not_a_match() -> None:
    """Crossref always returns something; the nearest work to an invented
    reference is an unrelated one."""
    report, _ = await _lookup(
        ONE,
        lambda r: (
            200,
            {
                "message": {
                    "items": [
                        _item("This jellyfish makes glowing proteins", 2020, "Doe")
                    ]
                }
            },
        ),
    )
    assert [p.kind for p in report.problems] == ["not_found"]
    assert report.problems[0].severity == "moderate"


async def test_a_different_year_on_the_published_version_is_reported() -> None:
    title = "The national economic burden of rare disease in the United States in 2019"
    report, _ = await _lookup(
        ONE, lambda r: (200, {"message": {"items": [_item(title, 2021, "Yang")]}})
    )
    assert [p.kind for p in report.problems] == ["details_differ"]
    assert "2021" in report.problems[0].issue


async def test_a_later_repost_does_not_contradict_the_year() -> None:
    title = "The national economic burden of rare disease in the United States in 2019"
    report, _ = await _lookup(
        ONE,
        lambda r: (
            200,
            {"message": {"items": [_item(title, 2025, "Yang", kind="posted-content")]}},
        ),
    )
    assert [r.status for r in report.lookups] == ["found"]
    assert report.problems == []


async def test_a_different_first_author_is_reported_when_the_numbers_tie_it() -> None:
    title = "The national economic burden of rare disease in the United States in 2019"
    item = _item(title, 2022, "Brown")
    item.update(volume="17", page="163")
    report, _ = await _lookup(ONE, lambda r: (200, {"message": {"items": [item]}}))
    assert "Brown" in report.problems[0].issue


async def test_a_same_year_namesake_with_other_numbers_is_not_the_entry() -> None:
    """Measured: two 2016 papers titled "Squamous cell carcinoma of the
    conjunctiva", by different authors in different journals."""
    title = "The national economic burden of rare disease in the United States in 2019"
    item = _item(title, 2022, "Liu")
    item.update(volume="88", page="901")
    report, _ = await _lookup(ONE, lambda r: (200, {"message": {"items": [item]}}))
    assert [p.kind for p in report.problems] == ["not_found"]


DOI_ENTRY = "Body [1].\n\nReferences\n1. Lee K (2021) A study of variants. Hum Mutat 42:1 doi:10.1002/humu.24380\n"


async def test_a_doi_that_does_not_resolve_is_major() -> None:
    report, seen = await _lookup(DOI_ENTRY, lambda r: (404, {}))
    assert "/works/10.1002/humu.24380" in str(seen[0].url)
    assert [(p.kind, p.severity) for p in report.problems] == [
        ("doi_not_found", "major")
    ]


async def test_a_doi_for_a_different_work_is_major() -> None:
    report, _ = await _lookup(
        DOI_ENTRY,
        lambda r: (
            200,
            {"message": _item("Something else entirely here", 2019, "Nobody")},
        ),
    )
    assert [(p.kind, p.severity) for p in report.problems] == [
        ("details_differ", "major")
    ]


async def test_a_service_failure_is_unchecked_not_wrong(monkeypatch) -> None:
    import andamentum.references.lookup as lookup_mod

    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr(lookup_mod.asyncio, "sleep", no_wait)
    report, seen = await _lookup(ONE, lambda r: (503, {}))
    assert [r.status for r in report.lookups] == ["service_error"]
    assert report.problems == []
    assert len(seen) == 2  # one retry


def test_the_module_is_a_leaf() -> None:
    """It must not import another andamentum sub-module but core (for the
    model calls). The one exception is ``cli.py``, which reads a Word or PDF
    file through ``harvest`` (a lazy import)."""
    import pathlib
    import re

    root = pathlib.Path(__file__).parents[1]
    for path in root.glob("*.py"):
        imports = re.findall(
            r"^\s*(?:from|import)\s+andamentum\.(\w+)", path.read_text(), re.M
        )
        allowed = {"references", "core"} | ({"harvest"} if path.name == "cli.py" else set())
        assert not [m for m in imports if m not in allowed], path.name


async def test_the_print_year_counts_as_well_as_the_online_year() -> None:
    """Online in 2019, in the 2020 issue: a list that says 2020 is right."""
    title = "The national economic burden of rare disease in the United States in 2019"
    item = _item(title, 2021, "Yang")
    item["published-print"] = {"date-parts": [[2022, 3]]}
    report, _ = await _lookup(ONE, lambda r: (200, {"message": {"items": [item]}}))
    assert [r.status for r in report.lookups] == ["found"]


async def test_a_title_search_finds_what_the_whole_entry_search_missed() -> None:
    """Crossref ranked two Nature/Science papers out of the first page for
    the whole-entry search; the field search by title found them first."""
    title = "The national economic burden of rare disease in the United States in 2019"

    def route(request: httpx.Request):
        params = parse_qs(urlparse(str(request.url)).query)
        if "query.title" in params:
            assert params["query.title"] == [title]
            return 200, {"message": {"items": [_item(title, 2022, "Yang")]}}
        return 200, {"message": {"items": [_item("Something unrelated", 2022, "X")]}}

    report, seen = await _lookup(ONE, route)
    assert [r.status for r in report.lookups] == ["found"]
    assert len(seen) == 2


async def test_a_consortium_author_is_not_compared() -> None:
    title = "The national economic burden of rare disease in the United States in 2019"
    item = _item(title, 2022, "")
    item["author"] = [{"name": "The Rare Disease Consortium"}]
    report, _ = await _lookup(ONE, lambda r: (200, {"message": {"items": [item]}}))
    assert report.problems == []


async def test_a_same_titled_work_by_other_authors_in_another_year_is_not_the_entry() -> (
    None
):
    """Generic titles repeat. A record with the entry's title but another
    year AND another first author is a namesake, and must not be reported as
    the entry's details being wrong."""
    title = "The national economic burden of rare disease in the United States in 2019"
    report, _ = await _lookup(
        ONE, lambda r: (200, {"message": {"items": [_item(title, 2016, "Liu")]}})
    )
    assert [p.kind for p in report.problems] == ["not_found"]


async def test_an_exact_match_in_a_later_search_beats_a_partial_one_earlier() -> None:
    title = "The national economic burden of rare disease in the United States in 2019"

    def route(request: httpx.Request):
        params = parse_qs(urlparse(str(request.url)).query)
        if "query.title" in params:
            return 200, {"message": {"items": [_item(title, 2022, "Yang")]}}
        # First search: right title and year, wrong author.
        return 200, {"message": {"items": [_item(title, 2022, "Liu")]}}

    report, _ = await _lookup(ONE, route)
    assert [r.status for r in report.lookups] == ["found"]


# ── lists a PDF converter merged into paragraphs ────────────────────────────


def _entries(text: str):
    start, end = find_reference_list(text) or (0, 0)
    return split_entries(text, start=start, end=end)


def test_a_merged_numbered_list_is_split_at_the_next_number_even_across_a_gap() -> None:
    text = (
        "References\n"
        "1.Rentzsch, P. CADD: predicting deleteriousness. Nucleic Acids Research vol. 47 D886 (2018). "
        "2.Gudmundsson, S. et al. Variant interpretation. Human Mutation vol. 43, no. 8. Wiley (2021). "
        "4.LeCun, Y. Deep learning. Nature vol. 521 436 (2015).\n"
    )
    assert [(e.number, e.year) for e in _entries(text)] == [
        (1, 2018),
        (2, 2021),
        (4, 2015),
    ]


def test_numbers_inside_an_entry_do_not_split_it() -> None:
    """ "no. 9." or "no. 4." sit before an entry's year: nothing before them is
    a whole entry yet."""
    text = (
        "References\n"
        "[1]B. Langmead, 'Fast gapped-read alignment', Nature Methods, vol. 9, no. 4. Springer, 2012. "
        "[2]M. I. Love, 'Moderated estimation', Genome Biology, vol. 15, no. 12. Springer, 2014.\n"
    )
    assert [e.number for e in _entries(text)] == [1, 2]


def test_a_merged_author_year_list_is_split_before_each_new_author() -> None:
    text = (
        "References\n"
        "Rentzsch, P., Witten, D., & Cooper, G. M. (2018). CADD: predicting deleteriousness. "
        "Nucleic Acids Research, 47, D886. https://doi.org/10.1093/nar/gky1016 "
        "Gudmundsson, S., Singer-Berk, M. (2021). Variant interpretation. Human Mutation, 43, 1012.\n"
        "https://doi.org/10.1002/humu.24309\n"
    )
    entries = _entries(text)
    assert [e.lead[:11] for e in entries] == ["Rentzsch, P", "Gudmundsson"]
    # A line that does not open like an entry continues the one above.
    assert entries[1].text.endswith("humu.24309")


def test_an_author_opened_line_without_a_year_is_still_an_entry() -> None:
    """A converter can move an entry's year into the paragraph before it."""
    text = (
        "Bibliography\n"
        "Li, H., and R. Durbin. 'Fast and Accurate Short Read Alignment'. Bioinformatics, 2009.\n"
        "Turro, E., W. J. Astle, K. Megy, et al. 'Whole-genome Sequencing of Patients'. Nature.\n"
    )
    assert [e.lead[:5] for e in _entries(text)] == ["Li, H", "Turro"]


def test_an_author_list_broken_across_lines_stays_one_entry() -> None:
    """Elsevier lists every author; a converter breaks the list over lines,
    and the second line opens with a name. An entry with no year yet is not
    finished, so the line continues it."""
    text = (
        "References\n"
        "Turro, E., Astle, W.J., Megy, K., Gräf, S., Greene, D., Shamardina, O.\n"
        "Syrris, P., Tait, R.C., Talks, K., 2020. Whole-genome sequencing of patients. Nature 583, 96.\n"
        "LeCun, Y., Bengio, Y., Hinton, G., 2015. Deep learning. Nature 521, 436.\n"
    )
    assert [e.lead[:5] for e in _entries(text)] == ["Turro", "LeCun"]


async def test_a_list_that_starts_late_is_one_problem_not_one_per_citation() -> None:
    text = (
        "Body [1], [2], [3], [4].\n\nReferences\n"
        "3. Turro E. Whole-genome sequencing. Nature 2020;583:96.\n"
        "4. LeCun Y. Deep learning. Nature 2015;521:436.\n"
    )
    report = await check_references(text, lookup=False)
    assert [p.headline for p in report.problems] == ["The list starts at entry 3"]
    assert "entries 1–2 are missing" in report.problems[0].issue


def test_a_fragment_out_of_alphabetical_order_is_not_an_entry() -> None:
    """A converter moved the tail of one entry before its head; the fragment
    opens with a capital and a year, but breaks the list's alphabetical
    order and has no author."""
    text = (
        "Bibliography\n"
        "Langmead, B., and S. L. Salzberg. 'Fast Gapped-read Alignment'. Nature Methods, 2012.\n"
        "Love, M. I., and W. Huber. 'Moderated Estimation'. Genome Biology, 2014.\n"
        "Robinson, J. T., H. Thorvaldsdóttir. 'Integrative Genomics Viewer'. Nature, 2011.\n"
        "Media LLC, 2020. Originally published as []. https://doi.org/10.1038/s41586-020-\n"
        "Turro, E., W. J. Astle. 'Whole-genome Sequencing of Patients'. Nature.\n"
    )
    assert [e.lead[:5] for e in _entries(text)] == ["Langm", "Love,", "Robin", "Turro"]


def test_an_unsorted_list_keeps_every_entry() -> None:
    """Only a sorted list says anything about a line out of order."""
    text = (
        "References\n"
        "Turro, E. (2020). Whole-genome sequencing. Nature, 583, 96.\n"
        "Adams, K. (2019). A study. Journal, 1, 1.\n"
        "Love, M. I. (2014). Moderated estimation. Genome Biology, 15, 550.\n"
        "Baker, J. (2018). Another study. Journal, 2, 2.\n"
    )
    assert len(_entries(text)) == 4


# ── Pandoc drafts ───────────────────────────────────────────────────────────

PANDOC = """Introduction
Rare diseases are common [@wakap2020]. Diagnosis is slow [see @zurynski2017, p. 4;
@yang2022]. Costs are high [@missing2021]. Mail me at a@b.org.

References
[@wakap2020] Nguengang Wakap S, Lambert DM (2020) Estimating cumulative point prevalence of rare diseases. Eur J Hum Genet 28:165-173
[@zurynski2017] Zurynski Y, Deverell M (2017) Australian children living with rare diseases. Orphanet J Rare Dis 12:68
[@yang2022] Yang G, Cintina I (2022) The national economic burden of rare disease. Orphanet J Rare Dis 17:163
[@nisar2021] Nisar H, Wajid B (2021) Whole-genome sequencing as a first-tier diagnostic framework. Front Genet 12:1
[@yang2022] Yang G (2022) A second entry under the same key. J Test 1:1
"""


def test_a_keyed_list_keeps_each_key_and_drops_it_from_the_text() -> None:
    entries = _entries(PANDOC)
    assert [e.key for e in entries] == [
        "wakap2020",
        "zurynski2017",
        "yang2022",
        "nisar2021",
        "yang2022",
    ]
    assert entries[0].text.startswith("Nguengang Wakap S")
    assert entries[0].number is None


def test_a_pandoc_group_is_one_candidate_with_every_key() -> None:
    candidates = [
        c
        for c in scan(PANDOC, start=0, end=PANDOC.index("References"))
        if c.source == "key"
    ]
    assert [c.keys for c in candidates] == [
        ["wakap2020"],
        ["zurynski2017", "yang2022"],
        ["missing2021"],
    ]
    # An e-mail address is not a citation.
    assert not any("b.org" in k for c in candidates for k in c.keys)


async def test_every_key_problem_is_found_by_evidence() -> None:
    report = await _offline(PANDOC)
    assert report.style is not None
    assert (report.style.style, report.style.decided_by) == ("pandoc_keys", "evidence")
    by_kind = {(p.kind, p.headline) for p in report.problems}
    assert ("cited_not_listed", "[@missing2021] is not in the list") in by_kind
    assert ("duplicate_key", "Two entries have the key @yang2022") in by_kind
    assert ("listed_not_cited", "Reference @nisar2021 is never cited") in by_kind
    assert len(report.problems) == 3
    missing = next(p for p in report.problems if p.kind == "cited_not_listed")
    assert PANDOC[missing.start : missing.end] == "[@missing2021]"


async def test_a_consistent_pandoc_draft_has_no_problems() -> None:
    text = (
        "Body [@a2020] and [@b2021; @a2020].\n\nReferences\n"
        "[@a2020] Adams J (2020) First paper. J 1:1\n"
        "[@b2021]: Brown K (2021) Second paper. J 2:2\n"
    )
    report = await _offline(text)
    assert report.style is not None and report.style.style == "pandoc_keys"
    assert report.problems == []
