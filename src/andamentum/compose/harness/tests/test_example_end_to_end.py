"""write, refresh, respond and the MCP server end to end on the SaltLog example project.

Models are stubbed at the call seam and Crossref is served from the example's
fixtures, so nothing here touches a network. The project is synthetic.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from mcp import types
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import BaseModel
from pydantic_ai.usage import RunUsage

from andamentum.word import cli
from andamentum.compose.brief import BriefDraft
from andamentum.compose.evidence import RawFact, RawFacts
from andamentum.compose.harness import mcp_server
from andamentum.compose.harness.bench.fixture_http import scheme_only_url_check
from andamentum.compose.harness.bindings import bind as bind_module
from andamentum.compose.harness.bindings.bind import BindParams, run_bind
from andamentum.compose.harness.bindings.refresh import RefreshParams, run_refresh
from andamentum.compose.harness.checks.numbers import NumericClaims
from andamentum.compose.harness.docx.reader import open_docx
from andamentum.compose.harness.draft.claims import ClaimLabel, ClaimLabels
from andamentum.compose.harness.draft.verb import DraftParams, run_draft
from andamentum.compose.harness.examples import (
    ExampleProject,
    fixture_transport_for_example,
    make_example,
)
from andamentum.compose.harness.kernel.verbs import load_builtin_verbs
from andamentum.compose.harness.respond import RespondParams, run_respond
from andamentum.compose.harness.respond.items import ReviewPoint, ReviewPoints
from andamentum.compose.harness.respond.triage import (
    AnswerProposal,
    EditProposal,
    Triage,
)
from andamentum.compose.schemas import Outline, ReaderQuestion, SectionPlan
from andamentum.compose.write import SectionBody

from ._helpers import copy_of
from .test_bindings_bind import StubModel

USAGE = RunUsage(input_tokens=11, output_tokens=7)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def project(tmp_path: Path) -> ExampleProject:
    return make_example(tmp_path / "saltlog")


# ------------------------------------------------------------------------ write

OKAFOR = "Higher dietary sodium intake was associated with higher blood pressure"
NOTES_QUOTE = (
    "48 participants completed the four weeks; 55 enrolled (completion 87.3%)."
)
RESULT_FACT = "mean_reduction_mg,412"


class WriteScript:
    """Answers the compose workers by output type, from what each prompt holds."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, model: str, **kw: object) -> tuple[BaseModel, RunUsage]:
        kind = kw["output_type"]
        message = str(kw["user_message"])
        assert isinstance(kind, type)
        self.calls.append(kind.__name__)
        return self.answer(kind, message), USAGE

    def answer(self, kind: type, message: str) -> BaseModel:
        if kind is BriefDraft:
            return BriefDraft(
                genre="article",
                title="SaltLog pilot",
                audience="nutrition researchers",
                purpose="report the SaltLog pilot",
                key_messages=["Dietary sodium and blood pressure in community cohorts"],
                must_include=[],
                constraints=[],
                reader_questions=[
                    ReaderQuestion(question="What fell?", expected_answer="sodium"),
                    ReaderQuestion(question="How many finished?", expected_answer="48"),
                ],
            )
        if kind is RawFacts:
            return self.facts(message)
        if kind is Outline:
            ids = dict(self.statements(message))
            intro = [i for i, s in ids.items() if "blood pressure" in s]
            results = [i for i, s in ids.items() if "mean reduction" in s or "48" in s]
            return Outline(
                sections=[
                    SectionPlan(
                        title=title,
                        purpose=f"{title} of the pilot",
                        key_points=[],
                        fact_ids=fact_ids,
                        target_words=300,
                    )
                    for title, fact_ids in (
                        ("Introduction", intro),
                        ("Results", results),
                    )
                ]
            )
        if kind is SectionBody:
            return SectionBody(body=self.body(message))
        if kind is ClaimLabels:
            numbers = [int(n) for n in re.findall(r"^(\d+): ", message, re.MULTILINE)]
            return ClaimLabels(
                labels=[ClaimLabel(number=n, kind="method") for n in numbers]
            )
        if kind is NumericClaims:
            return NumericClaims(claims=[])
        raise AssertionError(f"unscripted call for {kind.__name__}")

    @staticmethod
    def statements(message: str) -> list[tuple[str, str]]:
        return re.findall(r"^(f\d+): (.*)$", message, re.MULTILINE)

    @staticmethod
    def facts(message: str) -> RawFacts:
        if "completed the four weeks" in message:
            return RawFacts(
                facts=[
                    RawFact(
                        statement="48 participants completed the four weeks of 55 enrolled.",
                        quote=NOTES_QUOTE,
                    )
                ]
            )
        if RESULT_FACT in message:
            return RawFacts(
                facts=[
                    RawFact(
                        statement="The mean reduction in daily sodium was 412 mg.",
                        quote=RESULT_FACT,
                    )
                ]
            )
        if OKAFOR in message:
            return RawFacts(facts=[RawFact(statement=OKAFOR + ".", quote=OKAFOR)])
        return RawFacts(facts=[])

    def body(self, message: str) -> str:
        section = re.search(r"^SECTION: (.+)$", message, re.MULTILINE)
        assert section
        marked = {s: i for i, s in self.statements(message)}
        if section.group(1) == "Introduction":
            (marker,) = [i for s, i in marked.items() if "blood pressure" in s]
            return f"Higher sodium intake goes with higher blood pressure [[{marker}]]."
        mean = next(i for s, i in marked.items() if "mean reduction" in s)
        done = next(i for s, i in marked.items() if "48 participants" in s)
        return (
            f"Mean daily sodium fell by 412 mg [[{mean}]]. "
            f"Of those enrolled, 48 participants completed the four weeks [[{done}]]."
        )


async def test_write_into_a_new_word_file_from_the_example(
    project: ExampleProject, monkeypatch: pytest.MonkeyPatch
):
    script = WriteScript()
    monkeypatch.setattr("andamentum.compose._call.run_agent_with_usage", script)
    monkeypatch.setattr(
        "andamentum.compose.harness.checks.numbers.run_agent_with_usage", script
    )
    notes_before, refs_before = sha(project.notes), sha(project.library)
    out = project.root / "paper.docx"
    result = await run_draft(
        DraftParams(
            task="Write the SaltLog pilot article",
            out=out,
            source=[str(project.notes)],
            results=project.results,
            library=project.library,
            model="stub:m",
        ),
        transport=fixture_transport_for_example(),
        url_check=scheme_only_url_check,
    )
    assert (sha(project.notes), sha(project.library)) == (notes_before, refs_before)
    assert result.exit_code == 0, (result.open_errors, result.refused)

    view = open_docx(out)
    text = view.text()
    assert [p.text for p in view.paragraphs if p.heading_level] == [
        "Introduction",
        "Results",
        "References",
        "Statement on the use of AI",
    ]
    # the library work was verified against the fixtures, then cited and listed
    assert result.works == ["doi:10.5555/saltlog.001"]
    assert result.citations == 1 and "Okafor" in text
    assert "Dietary sodium and blood pressure in community cohorts" in text
    # the number that came from summary.csv is bound; the one from the notes is not
    bound = [
        view.paragraphs[int(a.paragraph_id[1:].split("-")[0])].text[a.start : a.end]
        for a in view.anchors
        if a.kind == "number"
    ]
    assert bound == ["412"]
    assert result.statement in text
    ledger = (result.ledger).read_text()
    assert "mean_reduction_mg" in ledger or "summary.csv" in ledger
    assert result.report.is_file()
    assert {c.author for c in view.comments} <= {"andamentum compose"}


def test_draft_cli_refuses_research_flags_without_research(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    # a usage error proves the Word path ran and refused (no model is called)
    code = cli.main(
        ["draft", "t", "--out", str(tmp_path / "d"), "--model", "m", "--browser"]
    )
    assert code == 2
    assert "need --research" in capsys.readouterr().err


# ---------------------------------------------------------------------- refresh


BOUND_SENTENCE = (
    "Mean daily sodium fell by 412 mg, from 3,310 mg in week 0 to 2,898 mg in "
    "week 4 (t(47) = 3.12, p = .003)."
)


def set_summary(project: ExampleProject, **changes: str) -> None:
    path = project.results / "summary.csv"
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, _ = line.partition(",")
        lines.append(f"{key},{changes[key]}" if key in changes else line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def test_a_changed_csv_value_is_a_tracked_change_and_a_comment(
    project: ExampleProject, monkeypatch: pytest.MonkeyPatch
):
    picks = {
        BOUND_SENTENCE: [
            ("412", "mean_reduction_mg"),
            ("3,310", "sodium_week0_mg"),
            ("2,898", "sodium_week4_mg"),
            ("3.12", "paired_t"),
            (".003", "p_value"),
        ]
    }
    monkeypatch.setattr(bind_module, "run_agent_with_usage", StubModel(picks))
    before = sha(project.collaborator)
    bound = await run_bind(
        BindParams(out=copy_of(project.collaborator, "bound"), file=project.collaborator, results=project.results, model="stub:m")
    )
    assert sha(project.collaborator) == before
    assert len(bound.report.bound) == 5, bound.report.unmatched
    # the abstract's numbers were not proposed: listed by code, not guessed
    assert any(u.written == "412" and u.paragraph != 8 for u in bound.report.unmatched)

    set_summary(project, mean_reduction_mg="398", p_value="0.081")
    bound_before = sha(bound.output)
    refreshed = run_refresh(
        RefreshParams(
            out=copy_of(bound.output, "refreshed"), file=bound.output,
            results=project.results,
            project=Path(bound.report.project_dir),
        )
    )
    assert sha(bound.output) == bound_before and sha(project.collaborator) == before
    changes = {(c.old, c.new) for c in refreshed.report.changes}
    assert changes == {("412", "398"), (".003", ".081")}
    view = open_docx(refreshed.output)
    results_paragraph = next(p for p in view.paragraphs if "sodium fell by" in p.text)
    assert "398 mg" in results_paragraph.text and "p = .081" in results_paragraph.text
    ours = [c for c in view.comments if c.author == "andamentum refresh"]
    # one comment per sentence: both changes, flagged as an error for the p flip
    (comment,) = ours
    assert comment.text.startswith("[error]")
    assert "412 → 398" in comment.text and "0.05" in comment.text
    # the unbound abstract number is left alone
    assert any("412 mg" in p.text for p in view.paragraphs if "Abstract" not in p.text)
    assert refreshed.report.exit_code == 1  # a significance flip is an error


# ---------------------------------------------------------------------- respond

CAUSAL_EDIT = (
    "In this single-arm pilot, 52 adults logged every meal for four weeks. "
    "Reported sodium intake fell by 412 mg over the four weeks of daily logging."
)


class RespondScript:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, model: str, **kw: object) -> tuple[BaseModel, RunUsage]:
        kind = kw["output_type"]
        message = str(kw["user_message"])
        assert isinstance(kind, type)
        self.calls.append(kind.__name__)
        if kind is ReviewPoints:
            return (
                ReviewPoints(
                    points=[
                        ReviewPoint(
                            label="Reviewer 2, point 2",
                            quote="Please say whether the study was registered before "
                            "recruitment started.",
                        ),
                        ReviewPoint(
                            label="Reviewer 1, point 3",
                            quote="Please add them.",
                        ),
                    ]
                ),
                USAGE,
            )
        if kind is Triage:
            item = message.split("ITEM:")[-1]
            if "Is this causal" in item:
                return Triage(category="edit", reason="the wording over-claims"), USAGE
            if "food-composition table" in item and "Reviewer" not in item[:40]:
                return Triage(category="answer", reason="stated in the text"), USAGE
            return Triage(category="author", reason="needs the authors"), USAGE
        if kind is EditProposal:
            return (
                EditProposal(
                    new_text=CAUSAL_EDIT,
                    explanation="the sentence now reports the change without a causal claim.",
                ),
                USAGE,
            )
        if kind is AnswerProposal:
            return (
                AnswerProposal(
                    answerable=True,
                    answer="Sodium was estimated from the logged meals; the table is not named.",
                    evidence_quotes=["Sodium was estimated from the logged meals."],
                ),
                USAGE,
            )
        raise AssertionError(f"unscripted call for {kind.__name__}")


async def test_respond_answers_comments_and_writes_a_response_letter(
    project: ExampleProject, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "andamentum.compose._call.run_agent_with_usage", RespondScript()
    )
    before = sha(project.collaborator)
    original = {c.id: c for c in open_docx(project.collaborator).comments}
    result = await run_respond(
        RespondParams(
            out=copy_of(project.collaborator, "responded"), file=project.collaborator,
            reviews=project.reviews,
            model="stub:m",
            offline=True,
        )
    )
    assert sha(project.collaborator) == before
    out = open_docx(result.output)
    replies = {c.parent_id: c for c in out.comments if c.author == "andamentum respond"}
    # thread 0 (causal) was edited, thread 2 (food-composition table) answered
    assert replies["0"].text.startswith("Changed in section")
    assert replies["2"].text.startswith("Sodium was estimated")
    # the resolved thread is not touched and no human comment was resolved
    assert "3" not in replies
    for comment in out.comments:
        if comment.id in original:
            assert comment.resolved == original[comment.id].resolved
    assert any(
        "Reported sodium intake fell by 412 mg" in p.text for p in out.paragraphs
    )
    # reviewer points become a letter beside the output copy
    letter = result.letter_md
    assert letter is not None and letter.name == "response-letter.md"
    text = letter.read_text()
    assert "## Reviewer 2, point 2" in text
    assert "> Please say whether the study was registered" in text
    assert result.letter_docx is not None and result.letter_docx.is_file()
    assert result.report.exit_code == 1  # items for the authors remain


# -------------------------------------------------------------------------- MCP


async def test_mcp_lists_the_tools_and_runs_check_on_the_example(
    project: ExampleProject,
):
    load_builtin_verbs()
    before = sha(project.collaborator)
    async with create_connected_server_and_client_session(
        mcp_server.build_server()
    ) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert set(tools) == {
            "open_document",
            "show_section",
            "apply_edits",
            "check",
            "draft",
            "refresh",
            "respond",
            "find_citation",
            "insert_citation",
        }
        assert not any(
            "not registered" in (t.description or "").lower() for t in tools.values()
        )
        called = await client.call_tool(
            "check",
            {
                "file": str(project.collaborator),
                "offline": True,
                "out": str(project.root / "checked.docx"),
            },
        )
        assert called.structuredContent is not None
        run = called.structuredContent
        assert run["exit_code"] == 1
        report = await client.read_resource(mcp_server.REPORT_URI)  # type: ignore[arg-type]
        text = report.contents[0].text  # type: ignore[union-attr]
        assert "abstract_body" in text and "52" in text
        summary = await client.call_tool(
            "open_document", {"path": str(project.collaborator)}
        )
        block = summary.content[0]
        assert isinstance(block, types.TextContent)
        assert "Abstract" in block.text and "Results" in block.text
    assert sha(project.collaborator) == before


async def test_mcp_find_and_insert_citation_verify_before_citing(
    project: ExampleProject,
):
    from andamentum.compose.harness.literature.verbs import (
        FindParams,
        InsertParams,
        run_find,
        run_insert,
    )

    found = await run_find(
        FindParams(query="smartphone food diaries reporting", library=project.library),
        transport=fixture_transport_for_example(),
    )
    verified = found["verified"]
    assert isinstance(verified, list) and verified
    assert json.dumps(found)  # JSON-ready
    view = open_docx(project.collaborator)
    paragraph = next(p for p in view.paragraphs if p.text.startswith("Mean daily"))
    before = sha(project.collaborator)
    code, message = await run_insert(
        InsertParams(
            file=project.collaborator,
            paragraph_id=paragraph.id,
            at=len(paragraph.text) - 1,
            key=["lindqvist2022"],
            library=project.library,
            out=project.root / "cited.docx",
        ),
        transport=fixture_transport_for_example(),
    )
    assert code == 0, message
    assert sha(project.collaborator) == before
    assert (project.root / "cited.docx").is_file()
    # a DOI that Crossref does not know is refused and nothing is written
    code, message = await run_insert(
        InsertParams(
            file=project.collaborator,
            paragraph_id=paragraph.id,
            at=0,
            doi=["10.0000/fake"],
            out=project.root / "fake.docx",
        ),
        transport=fixture_transport_for_example(),
    )
    assert code == 1 and "nothing written" in message and "10.0000/fake" in message
    assert not (project.root / "fake.docx").exists()
    # an offset outside the paragraph is a usage error and nothing is written
    from andamentum.compose.harness.bindings.common import UsageError

    with pytest.raises(UsageError, match="outside the paragraph"):
        await run_insert(
            InsertParams(
                file=project.collaborator,
                paragraph_id=paragraph.id,
                at=len(paragraph.text) + 1,
                key=["lindqvist2022"],
                library=project.library,
                out=project.root / "far.docx",
            ),
            transport=fixture_transport_for_example(),
        )
    assert not (project.root / "far.docx").exists()
