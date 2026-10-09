"""``draft_document`` end to end: stubbed models, offline unless stated, synthetic SaltLog project."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import docx
import pytest
import yaml
from lxml import etree  # type: ignore

from andamentum.compose.brief import BriefDraft
from andamentum.compose.harness.bench.fixture_http import (
    fixture_transport,
    scheme_only_url_check,
)
from andamentum.compose.harness.bench.planted import load_fixtures
from andamentum.compose.harness.docx.reader import open_docx
from andamentum.compose.harness.draft.cite import CiteVerdict
from andamentum.compose.harness.draft import (
    DraftResult,
    DraftUsageError,
    draft_document,
)
from andamentum.compose.harness.kernel import ProjectStore
from andamentum.compose.schemas import ReaderQuestion, Roles
from andamentum.compose.write import SectionBody

from ._draft_helpers import (
    BODIES,
    ROLES,
    SECTIONS,
    default_script,
    install_script,
    make_project,
    saltlog_cite_tools,
)

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
TASK = "Write the SaltLog pilot article"


async def write(tmp_path: Path, **overrides: object) -> DraftResult:
    notes, results = make_project(tmp_path)
    arguments: dict[str, object] = dict(
        out=tmp_path / "paper.docx",
        roles=ROLES,
        sources=[str(notes)],
        results=results,
        offline=True,
    )
    arguments.update(overrides)
    if "cite" in arguments:
        return await draft_document(TASK, **arguments)  # type: ignore[arg-type]
    async with saltlog_cite_tools(tmp_path) as cite:
        return await draft_document(TASK, cite=cite, **arguments)  # type: ignore[arg-type]


@pytest.fixture
async def written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[DraftResult, Path]:
    install_script(monkeypatch)
    result = await write(tmp_path)
    return result, tmp_path / "paper.docx"


def listed(event: dict[str, object], key: str) -> list[str]:
    value = event[key]
    assert isinstance(value, list)
    return [str(item) for item in value]


def document_xml(path: Path) -> etree._Element:
    with zipfile.ZipFile(path) as archive:
        return etree.fromstring(archive.read("word/document.xml"))


# ---------------------------------------------------------------- a new document


async def test_the_sections_and_the_back_matter(
    written: tuple[DraftResult, Path],
) -> None:
    result, out = written
    view = open_docx(out)
    assert [p.text for p in view.paragraphs if p.heading_level] == [
        *SECTIONS,
        "References",
        "Statement on the use of AI",
    ]
    assert result.sections == list(SECTIONS)
    assert view.paragraphs[0].text == "SaltLog pilot"
    assert view.paragraphs[0].style.lower() == "title"


async def test_everything_is_a_tracked_insertion(
    written: tuple[DraftResult, Path],
) -> None:
    _, out = written
    root = document_xml(out)
    inserted = root.findall(f".//{{{W_NS}}}p/{{{W_NS}}}pPr/{{{W_NS}}}rPr/{{{W_NS}}}ins")
    paragraphs = root.findall(f".//{{{W_NS}}}p")
    assert len(inserted) == len(paragraphs) - 1  # all but the template's title
    authors = {i.get(f"{{{W_NS}}}author") for i in root.iter(f"{{{W_NS}}}ins")}
    assert authors == {"andamentum compose"}


async def test_the_citation_is_inserted_with_a_reference_entry(
    written: tuple[DraftResult, Path],
) -> None:
    result, out = written
    view = open_docx(out)
    text = view.text()
    assert "improves delivery fidelity (Okafor 2020)." in text
    assert "Okafor, A. (2020)." in text
    assert result.citations == 1
    assert result.works == ["doi:10.5555/saltlog.2020.1"]


async def test_numbers_from_results_are_bound(
    written: tuple[DraftResult, Path],
) -> None:
    result, out = written
    view = open_docx(out)
    assert result.numbers_bound == 2
    assert sorted(
        view.paragraphs[int(a.paragraph_id[1:].split("-")[0])].text[a.start : a.end]
        for a in view.anchors
    ) == ["41.2", "53.8"]
    assert {a.kind for a in view.anchors} == {"number"}


async def test_questions_for_the_author_are_anchored_comments(
    written: tuple[DraftResult, Path],
) -> None:
    result, out = written
    view = open_docx(out)
    assert {c.author for c in view.comments} == {"andamentum compose"}
    by_text = {c.text: c for c in view.comments}
    fill = by_text["Needs the author: control group sample size."]
    paragraph = next(p for p in view.paragraphs if p.id == fill.paragraph_id)
    assert (
        paragraph.text[fill.start : fill.end] == "[TO FILL: control group sample size]"
    )
    interpretations = [c for c in view.comments if c.text.startswith("Interpretation")]
    assert len(interpretations) == 2
    assert any("control group sample size" in q for q in result.author_questions)


async def test_the_ledger_has_claims_and_bound_numbers(
    written: tuple[DraftResult, Path],
) -> None:
    result, _ = written
    claims = yaml.safe_load(result.ledger.read_text())["claims"]
    drafted = [c for c in claims if "section" in c]
    bound = [c for c in claims if "binding" in c]
    assert len(bound) == 2
    # The row label travels with the binding so refresh can see a reordered csv.
    assert all(c["binding"]["context"] for c in bound)
    assert {c["kind"] for c in drafted} == {
        "method",
        "result",
        "background",
        "interpretation",
    }
    assert all(c["approved_by"] is None for c in claims)
    interpretation = next(c for c in drafted if c["kind"] == "interpretation")
    assert interpretation["status"] == "needs-author"
    sentence = next(c for c in drafted if c["text"].startswith("The treatment group"))
    assert sorted(sentence["anchors"]) == sorted(c["id"] for c in bound)
    background = next(c for c in drafted if c["kind"] == "background")
    assert background["evidence"][0]["work"] == "doi:10.5555/saltlog.2020.1"


async def test_the_statement_comes_from_the_event_log(
    written: tuple[DraftResult, Path],
) -> None:
    result, out = written
    view = open_docx(out)
    assert view.paragraphs[-1].text == result.statement
    assert (
        'Sections "Abstract", "Introduction", "Methods", "Results" and "Discussion"'
        in result.statement
    )
    assert "stub:drafter" in result.statement
    assert "numbers" in result.statement and "crossrefs" in result.statement
    assert "revised" not in result.statement  # nothing needed revising
    events = ProjectStore(result.project).read_events()
    assert [e["step"] for e in events if e.get("type") == "draft_step"].count(
        "draft"
    ) == 5
    (run,) = [e for e in events if e.get("type") == "draft_run"]
    assert run["citations"] == 1 and run["numbers_bound"] == 2


async def test_provenance_and_project_id(written: tuple[DraftResult, Path]) -> None:
    result, out = written
    view = open_docx(out)
    assert view.project_id is not None
    assert view.project_id == ProjectStore(result.project).meta().id
    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()
        core = archive.read("docProps/core.xml").decode()
    assert any(n.startswith("customXml/") for n in names)
    assert "andamentum:ai-generated" in core


async def test_the_report_and_exit_code(written: tuple[DraftResult, Path]) -> None:
    result, out = written
    # the scripted draft leaves a question for the authors, so it exits 4
    assert result.author_questions and result.exit_code == 1
    report = result.report.read_text()
    assert result.report == out.with_name("paper.report.md")
    for expected in (
        "# Write report: paper.docx",
        "citations inserted: 1",
        "numbers bound: 2",
        "## Questions for the authors",
        "control group sample size",
        "## Statement on the use of AI",
    ):
        assert expected in report
    assert result.report.with_suffix(".json").is_file()
    assert result.usage["drafter"] > 0 and result.usage["judge"] > 0


async def test_the_checks_ran_on_the_draft(written: tuple[DraftResult, Path]) -> None:
    result, _ = written
    runs = [
        e
        for e in ProjectStore(result.project).read_events()
        if e.get("type") == "check_run"
    ]
    assert len(runs) == 1
    assert set(listed(runs[0], "checks")) >= {
        "crossrefs",
        "statistics",
        "abstract_body",
        "numbers",
    }
    assert "citations" in listed(runs[0], "skipped")  # offline


# ------------------------------------------------------------- without extras


async def test_without_the_cite_loop_nothing_is_cited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_script(monkeypatch)
    result = await write(tmp_path, cite=None)
    view = open_docx(tmp_path / "paper.docx")
    assert result.citations == 0
    assert "References" not in [p.text for p in view.paragraphs]
    assert "Citations were inserted" not in result.statement


async def test_without_results_no_number_is_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = install_script(monkeypatch)
    result = await write(tmp_path, results=None)
    assert result.numbers_bound == 0
    from andamentum.compose.harness.checks.numbers import NumericClaims

    assert script.count(NumericClaims) == 0


async def test_a_genre_that_does_not_cite_asks_no_literature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handlers = default_script()
    handlers[BriefDraft] = lambda _: BriefDraft(
        genre="memo",
        title="SaltLog memo",
        audience="the team",
        purpose="decide",
        key_messages=["Keep logging"],
        must_include=[],
        constraints=[],
        reader_questions=[
            ReaderQuestion(question="What now?", expected_answer="Keep logging"),
            ReaderQuestion(question="Who decides?", expected_answer="The team"),
        ],
    )
    install_script(monkeypatch, handlers)
    asked: list[str] = []
    async with saltlog_cite_tools(tmp_path, asked=asked) as cite:
        await write(tmp_path, cite=cite, genre="memo")
    assert asked == []


# --------------------------------------------------------------------- online


async def test_online_the_citation_check_runs_against_the_reference_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = load_fixtures()
    doi, found = next(
        (d, f)
        for d, f in fixtures.crossref.works.items()
        if f.status == 200 and not f.body["message"].get("updated-by")  # type: ignore[index, union-attr]
    )
    message = found.body["message"]  # type: ignore[index]
    csl = {
        "id": doi,
        "type": "article-journal",
        "title": message["title"][0],
        "author": [
            {"family": a["family"], "given": a["given"]} for a in message["author"]
        ],
        "issued": message["issued"],
        "container-title": message["container-title"][0],
        "volume": message["volume"],
        "issue": message["issue"],
        "page": message["page"],
        "DOI": doi,
        "abstract": "Structured logging of reading sessions improves fidelity of delivery in school interventions.",
    }
    handlers = default_script()
    handlers[CiteVerdict] = lambda m: (
        CiteVerdict(supports="yes", quote=str(csl["abstract"]))
        if "improves delivery fidelity" in m
        else CiteVerdict(supports="no")
    )
    install_script(monkeypatch, handlers)
    paper = f"# {csl['title']}\n\n{csl['abstract']}\n"
    async with saltlog_cite_tools(tmp_path, csl=csl, paper=paper) as cite:
        result = await write(
            tmp_path,
            cite=cite,
            offline=False,
            transport=fixture_transport(fixtures),
            url_check=scheme_only_url_check,
        )
    runs = [
        e
        for e in ProjectStore(result.project).read_events()
        if e.get("type") == "check_run"
    ]
    assert "citations" in listed(runs[-1], "checks")
    assert int(str(runs[-1]["network_calls"])) > 0
    assert result.open_errors == []
    assert result.citations == 1


# ------------------------------------------------------------ repair loop


FLIPPED = "The groups differed, t(28) = 0.50, p = .01 [[f2]]."
FIXED = "The groups did not differ, t(28) = 0.50, p = .62 [[f2]]."


def results_with(body: str, *, fixed: str | None) -> dict:
    handlers = default_script()
    original = handlers[SectionBody]

    def section_body(message: str) -> SectionBody:
        if "ISSUES TO RESOLVE" in message:
            assert "statistics" in message and "p" in message
            return SectionBody(body=fixed if fixed is not None else body)
        if "SECTION: Results" in message:
            return SectionBody(body=body)
        return original(message)  # type: ignore[return-value]

    handlers[SectionBody] = section_body
    return handlers


async def test_an_error_in_a_drafted_paragraph_sends_the_section_to_the_reviser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = install_script(monkeypatch, results_with(FLIPPED, fixed=FIXED))
    result = await write(tmp_path)
    assert [(r.round, r.errors, r.revised) for r in result.rounds] == [
        (0, 1, []),
        (1, 0, ["Results"]),
    ]
    assert result.open_errors == [] and result.exit_code == 1  # author questions remain
    text = open_docx(tmp_path / "paper.docx").text()
    assert "p = .62" in text and "p = .01" not in text
    assert (
        "revised by stub:drafter after automated checks reported errors"
        in result.statement
    )
    assert (
        sum(1 for t, m in script.calls if t is SectionBody and "ISSUES TO RESOLVE" in m)
        == 1
    )


async def test_an_error_that_stays_is_a_comment_and_exit_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_script(monkeypatch, results_with(FLIPPED, fixed=None))
    result = await write(tmp_path)
    assert result.exit_code == 1
    assert len(result.open_errors) == 1 and result.open_errors[0].startswith(
        "statistics:"
    )
    # the reviser made no change, so the loop stopped after one revision attempt
    assert [(r.round, r.errors) for r in result.rounds] == [(0, 1)]
    view = open_docx(tmp_path / "paper.docx")
    flagged = [c for c in view.comments if c.text.startswith("[error] statistics")]
    assert len(flagged) == 1 and flagged[0].author == "andamentum compose"
    assert any(q.startswith("statistics:") for q in result.author_questions)
    assert "## Open check errors" in result.report.read_text()


async def test_max_rounds_bounds_the_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}
    handlers = default_script()
    original = handlers[SectionBody]

    def section_body(message: str) -> SectionBody:
        if "ISSUES TO RESOLVE" in message:
            calls["n"] += 1
            return SectionBody(body=FLIPPED.replace("0.50", f"0.5{calls['n']}"))
        if "SECTION: Results" in message:
            return SectionBody(body=FLIPPED)
        return original(message)  # type: ignore[return-value]

    handlers[SectionBody] = section_body
    install_script(monkeypatch, handlers)
    result = await write(tmp_path, max_rounds=2)
    assert calls["n"] == 2
    assert [r.round for r in result.rounds] == [0, 1, 2]
    assert result.exit_code == 1


async def test_max_rounds_zero_only_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = install_script(monkeypatch, results_with(FLIPPED, fixed=FIXED))
    result = await write(tmp_path, max_rounds=0)
    assert result.exit_code == 1
    assert not any(
        "ISSUES TO RESOLVE" in m for t, m in script.calls if t is SectionBody
    )


TYPED = (
    "Earlier work was explored by Marlowe and Idris (2021) in school settings [[f1]]."
)


def typed_citation_script(*, fixed: bool):
    handlers = default_script()
    original = handlers[SectionBody]

    def section_body(message: str) -> SectionBody:
        if "SECTION: Introduction" in message:
            if "ISSUES TO RESOLVE" in message and fixed:
                return SectionBody(
                    body="Earlier work looked at school settings [[f1]]."
                )
            return SectionBody(body=TYPED)
        return original(message)  # type: ignore[return-value]

    handlers[SectionBody] = section_body
    return handlers


async def test_a_typed_citation_goes_back_to_the_reviser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = install_script(monkeypatch, typed_citation_script(fixed=True))
    result = await write(tmp_path)
    assert [(r.round, r.errors, r.revised) for r in result.rounds] == [
        (0, 1, []),
        (1, 0, ["Introduction"]),
    ]
    assert result.open_errors == []
    assert "Marlowe" not in open_docx(tmp_path / "paper.docx").text()
    assert any(
        "types the citation 'Marlowe and Idris (2021)'" in m
        for t, m in script.calls
        if t is SectionBody
    )


async def test_a_typed_citation_that_stays_is_an_error_comment_and_never_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_script(monkeypatch, typed_citation_script(fixed=False))
    result = await write(tmp_path)
    assert result.exit_code == 1
    assert any("Marlowe and Idris (2021)" in e for e in result.open_errors)
    assert any("typed citation" in q for q in result.author_questions)
    view = open_docx(tmp_path / "paper.docx")
    flagged = [c for c in view.comments if c.text.startswith("[error] Typed citation")]
    assert len(flagged) == 1 and flagged[0].author == "andamentum compose"
    claims = yaml.safe_load((result.project / "claims.yaml").read_text())["claims"]
    typed = [c for c in claims if "Marlowe" in c["text"]]
    assert [c["status"] for c in typed] == ["unsupported"]
    assert typed[0]["strength"] == "typed citation, not verified"


# ----------------------------------------------------------------------- into


def collaborator(path: Path) -> Path:
    document = docx.Document()
    document.add_heading("Collaborator draft", 0)
    document.add_heading("Introduction", 1)
    document.add_paragraph(
        "Reading support matters for early learners in every school."
    )
    document.add_heading("Methods", 1)
    document.add_paragraph(
        "We enrolled children from three schools and logged every session."
    )
    document.save(str(path))
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def test_into_adds_the_named_section_and_leaves_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = install_script(monkeypatch)
    source = collaborator(tmp_path / "collab.docx")
    before = digest(source)
    notes, results = make_project(tmp_path)
    async with saltlog_cite_tools(tmp_path) as cite:
        result = await draft_document(
            "Write the discussion section into the draft",
            cite=cite,
            out=tmp_path / "out" / "paper.docx",
            into=source,
            roles=ROLES,
            sources=[str(notes)],
            results=results,
            offline=True,
        )
    assert digest(source) == before  # the input is never touched
    assert result.sections == ["Discussion"]
    view = open_docx(tmp_path / "out" / "paper.docx")
    texts = [p.text for p in view.paragraphs]
    assert texts[:5] == [
        "Collaborator draft",
        "Introduction",
        "Reading support matters for early learners in every school.",
        "Methods",
        "We enrolled children from three schools and logged every session.",
    ]
    assert texts[5] == "Discussion"
    assert texts[6] == BODIES["Discussion"]
    # one draft call only: the other planned sections were not written
    assert sum(1 for t, _ in script.calls if t is SectionBody) == 1
    # the existing text went to the workers as a source
    assert any("Reading support matters" in m for _, m in script.calls)
    # new paragraphs are tracked insertions, existing ones are not
    root = document_xml(tmp_path / "out" / "paper.docx")
    inserted = [
        "".join(p.itertext())
        for p in root.iter(f"{{{W_NS}}}p")
        if p.find(f"{{{W_NS}}}pPr/{{{W_NS}}}rPr/{{{W_NS}}}ins") is not None
    ]
    assert "Discussion" in inserted and "Introduction" not in inserted


async def test_into_missing_sections_are_written_when_none_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_script(monkeypatch)
    source = collaborator(tmp_path / "collab.docx")
    notes, results = make_project(tmp_path)
    result = await draft_document(
        "Finish the paper",
        out=tmp_path / "paper.docx",
        into=source,
        roles=ROLES,
        sources=[str(notes)],
        results=results,
        offline=True,
    )
    assert result.sections == ["Abstract", "Results", "Discussion"]
    headings = [
        p.text for p in open_docx(tmp_path / "paper.docx").paragraphs if p.heading_level
    ]
    assert headings == [
        "Abstract",
        "Introduction",
        "Methods",
        "Results",
        "Discussion",
        "Statement on the use of AI",
    ]


async def test_into_only_errors_in_new_text_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_script(monkeypatch)
    document = docx.Document()
    document.add_heading("Draft", 0)
    document.add_heading("Introduction", 1)
    document.add_paragraph(
        "An old claim: t(28) = 0.50, p = .01 stays as the author wrote it."
    )
    document.save(str(tmp_path / "collab.docx"))
    notes, results = make_project(tmp_path)
    result = await draft_document(
        "Write the discussion",
        out=tmp_path / "paper.docx",
        into=tmp_path / "collab.docx",
        roles=ROLES,
        sources=[str(notes)],
        offline=True,
    )
    assert result.open_errors == [] and result.exit_code == 1  # author questions remain
    assert (
        result.counts["error"] == 1
    )  # the check saw the author's error; it is not ours


# --------------------------------------------------------------------- usage


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"out": "x.txt"}, "--out must be a .docx"),
        ({"max_rounds": -1}, "must not be negative"),
        ({"results": "nowhere"}, "--results"),
        ({"into": "missing.docx"}, "--into"),
        ({"sources": ["missing.md"]}, "source not found"),
        ({"template": "missing.dotx"}, "--template"),
    ],
)
async def test_usage_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict, message: str
) -> None:
    install_script(monkeypatch)
    paths = {k: tmp_path / v if isinstance(v, str) else v for k, v in overrides.items()}
    with pytest.raises(DraftUsageError, match=message):
        await draft_document(
            TASK,
            **{"out": tmp_path / "p.docx", "roles": ROLES, **paths},  # type: ignore[arg-type]
        )


async def test_an_empty_task_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DraftUsageError, match="empty"):
        await draft_document("  ", out=tmp_path / "p.docx", roles=ROLES)


async def test_into_and_template_exclude_each_other(tmp_path: Path) -> None:
    source = collaborator(tmp_path / "c.docx")
    with pytest.raises(DraftUsageError, match="cannot be combined"):
        await draft_document(
            TASK, out=tmp_path / "p.docx", roles=ROLES, into=source, template=source
        )


async def test_out_must_not_be_the_input(tmp_path: Path) -> None:
    source = collaborator(tmp_path / "c.docx")
    with pytest.raises(DraftUsageError, match="never edited"):
        await draft_document(TASK, out=source, roles=ROLES, into=source)


async def test_roles_are_required_by_type() -> None:
    assert set(Roles.model_fields) == {"drafter", "critic", "judge", "reader"}


async def test_into_a_zotero_document_the_citation_is_a_zotero_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from andamentum.compose.harness.spike.make import make_spike_docx

    handlers = default_script()
    original = handlers[SectionBody]
    handlers[SectionBody] = lambda m: (
        SectionBody(body="Logging improves delivery fidelity.")  # f1 is the notes
        if "SECTION: Discussion" in m
        else original(m)  # type: ignore[return-value]
    )
    install_script(monkeypatch, handlers)
    source = make_spike_docx(tmp_path / "spike.docx")
    notes, results = make_project(tmp_path)
    async with saltlog_cite_tools(tmp_path) as cite:
        result = await draft_document(
            "Write the discussion",
            cite=cite,
            out=tmp_path / "paper.docx",
            into=source,
            roles=ROLES,
            sources=[str(notes)],
            offline=True,
        )
    view = open_docx(tmp_path / "paper.docx")
    zotero = [f for f in view.fields if f.kind == "citation" and f.manager == "zotero"]
    assert [f.display for f in zotero] == ["(Doe, 2020)", "(Okafor, 2020)"]
    assert any(f.kind == "bibliography" and "Okafor" in f.display for f in view.fields)
    assert result.citations == 1 and result.exit_code == 0


async def test_write_never_overwrites_a_file(tmp_path: Path) -> None:
    existing = tmp_path / "paper.docx"
    existing.write_bytes(b"someone's edits")
    with pytest.raises(DraftUsageError, match="never overwrites"):
        await draft_document(TASK, out=existing, roles=ROLES)
    assert existing.read_bytes() == b"someone's edits"


async def test_a_document_with_no_paragraphs_cannot_be_written_into(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty.docx"
    docx.Document().save(str(empty))
    with pytest.raises(DraftUsageError, match="no paragraphs"):
        await draft_document(TASK, out=tmp_path / "paper.docx", roles=ROLES, into=empty)


async def test_a_second_pass_appends_to_the_section_and_the_statement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_script(monkeypatch)
    first = await write(tmp_path)
    notes, _ = tmp_path / "notes.md", None
    second = await draft_document(
        "Rewrite the discussion",
        out=tmp_path / "second" / "second.docx",
        into=first.output,
        roles=ROLES,
        sources=[str(notes)],
        offline=True,
    )
    view = open_docx(second.output)
    before = open_docx(first.output)
    headings = [p.text for p in view.paragraphs if p.heading_level]
    assert headings.count("Statement on the use of AI") == 1
    assert headings.count("Discussion") == 1
    texts = [p.text for p in view.paragraphs]
    start = texts.index("Discussion")
    assert texts[start + 1 : start + 3] == [BODIES["Discussion"]] * 2
    statements = texts[texts.index("Statement on the use of AI") + 1 :]
    assert len(statements) == 2 and statements[0] == before.paragraphs[-1].text
    # the numbers bound in the first pass are still bound
    assert {a.id for a in before.anchors} <= {a.id for a in view.anchors}
    assert view.project_id == before.project_id
    assert first.project != second.project


async def test_the_exit_code_is_1_while_anything_is_left_for_the_authors(
    written: tuple[DraftResult, Path],
) -> None:
    result, _ = written
    clean = result.model_copy(
        update={"open_errors": [], "author_questions": [], "refused": []}
    )
    assert clean.exit_code == 0
    for update in (
        {"author_questions": ["Results: needs the author: sample size"]},
        {"author_questions": ["not cited, could not be verified: X"]},
        {"open_errors": ["numbers: p changed"]},
        {"refused": ["insert p0@999: range ends past the paragraph"]},
    ):
        assert clean.model_copy(update=update).exit_code == 1
