"""respond end to end on a synthetic collaborator document (models stubbed at the call seam)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import BaseModel

from andamentum.compose import cli
from andamentum.compose.harness.docx.reader import open_docx
from andamentum.compose.harness.kernel import ProjectStore
from andamentum.compose.harness.kernel.verbs import _clear_verbs, load_builtin_verbs
from andamentum.compose.harness.respond import (
    RespondParams,
    RespondUsageError,
    run_respond,
)
from andamentum.compose.harness.respond.items import ReviewPoint, ReviewPoints
from andamentum.compose.harness.respond.items import Item
from andamentum.compose.harness.respond.triage import (
    AUTHOR_REASON,
    CATEGORIES,
    AnswerProposal,
    EditProposal,
    Triage,
    templated_triage,
)
from andamentum.compose.harness.spike.make import (
    build_docx,
    comment_part,
    comments_extended_part,
    paragraph,
    run,
)
from andamentum.core.decisions.testing import StubDecisions, choice_answer

from ...tests._stub import USAGE
from ._helpers import make_view

APP = "Participants logged every meal for four weeks using the SaltLog app."
PAPER = "Participants logged every meal for four weeks using a paper diary."
FELL = "Intake fell by 12% over the study."
CAUSAL = "We think the effect is causal."
CLINICS = "The pilot enrolled 48 participants across three clinics."
STAT_BAD = "Mean sodium fell (t(20) = 1.5, p = .001)."
STAT_OK = "Mean sodium fell (t(20) = 2.5, p = .021)."


def commented(ids: list[int], text: str, *, para_id: str) -> str:
    start = "".join(f'<w:commentRangeStart w:id="{i}"/>' for i in ids)
    end = "".join(f'<w:commentRangeEnd w:id="{i}"/>' for i in ids)
    refs = "".join(f'<w:r><w:commentReference w:id="{i}"/></w:r>' for i in ids)
    return paragraph(f"{start}{run(text)}{end}{refs}", para_id=para_id)


def make_doc(path: Path, *, stat: str = STAT_OK) -> Path:
    body = "".join(
        [
            paragraph(run("SaltLog pilot"), style="Title", para_id="10000001"),
            paragraph(run("Methods"), style="Heading1", para_id="10000002"),
            commented([0, 5], PAPER, para_id="10000003"),
            commented([1, 2, 4], FELL, para_id="10000004"),
            commented([3], CAUSAL, para_id="10000005"),
            paragraph(run("Results"), style="Heading1", para_id="10000006"),
            commented([6], CLINICS, para_id="10000007"),
            paragraph(run(stat), para_id="10000008"),
        ]
    )
    comments = [
        (
            0,
            "Jane Doe",
            "2026-10-01T09:00:00Z",
            "Please say it was an app, not paper.",
            "20000001",
        ),
        (
            1,
            "Roe Reviewer",
            "2026-10-01T09:05:00Z",
            "How large was the fall?",
            "20000002",
        ),
        (
            2,
            "Jane Doe",
            "2026-10-01T09:06:00Z",
            "Already fine, never mind the old one.",
            "20000003",
        ),
        (
            3,
            "Roe Reviewer",
            "2026-10-01T09:10:00Z",
            "Can you justify causality?",
            "20000004",
        ),
        (4, "Jane Doe", "2026-10-01T09:11:00Z", "Resolved earlier.", "20000005"),
        (
            5,
            "andamentum check",
            "2026-10-01T09:12:00Z",
            "[warning] citations: nothing.",
            "20000006",
        ),
        (
            6,
            "Roe Reviewer",
            "2026-10-01T09:20:00Z",
            "Please add the number of clinics.",
            "20000007",
        ),
    ]
    extended = [
        ("20000001", None, False),
        ("20000002", None, False),
        ("20000003", "20000002", False),
        ("20000004", None, False),
        ("20000005", None, True),  # a resolved thread
        ("20000006", None, False),
        ("20000007", None, False),
    ]
    return build_docx(
        path,
        body,
        comments_xml=comment_part(comments),
        comments_extended_xml=comments_extended_part(extended),
    )


class Scripted:
    """Stands in for the model call: answers by output type and by what the prompt holds."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.edit_text: dict[str, str] = {PAPER: APP}
        self.triage: dict[str, Triage] = {
            "Please say it was an app": Triage(
                category="edit", reason="the method used an app"
            ),
            "How large was the fall": Triage(
                category="answer", reason="stated in the text"
            ),
            "justify causality": Triage(
                category="author",
                reason="the authors must give the argument or soften the claim",
            ),
            "number of clinics": Triage(
                category="done",
                reason="the text already names the clinics",
                evidence_quote="across three clinics",
            ),
        }
        self.answer = AnswerProposal(
            answerable=True, answer="A 12% fall.", evidence_quotes=[FELL]
        )

    async def __call__(self, model: str, **kw) -> tuple[BaseModel, object]:
        kind = kw["output_type"]
        message = kw["user_message"]
        self.calls.append((kind.__name__, message))
        if kind is Triage:
            for key, out in self.triage.items():
                if key in message.split("ITEM:")[-1]:
                    return out, USAGE
            raise AssertionError(f"no triage scripted for {message[-200:]}")
        if kind is EditProposal:
            current = message.split("PARAGRAPH TO CHANGE:\n")[1].split("\n\n")[0]
            return (
                EditProposal(
                    new_text=self.edit_text[current],
                    explanation="We changed the text: the diary was an app, so it now says so.",
                ),
                USAGE,
            )
        if kind is AnswerProposal:
            return self.answer, USAGE
        raise AssertionError(f"unexpected call {kind}")

    def kinds(self) -> list[str]:
        return [k for k, _ in self.calls]


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> Scripted:
    stub = Scripted()
    monkeypatch.setattr("andamentum.compose._call.run_agent_with_usage", stub)
    return stub


@pytest.fixture
def doc(tmp_path: Path) -> Path:
    return make_doc(tmp_path / "paper.docx")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def respond(doc: Path, *, decision_handle=None, **kw):
    kw.setdefault("model", "test:model")
    kw.setdefault("offline", True)
    return await run_respond(
        RespondParams(file=doc, **kw), decision_handle=decision_handle
    )


async def test_every_category_is_handled_and_nothing_human_is_resolved(
    doc: Path, script: Scripted
):
    before = sha(doc)
    result = await respond(doc)
    assert sha(doc) == before  # the input is never modified
    report = result.report
    assert report.counts == {"edit": 1, "answer": 1, "author": 1, "done": 1}
    assert report.exit_code == 4  # an item needs the authors
    assert result.output == doc.with_name("paper.responded.docx")

    out = open_docx(result.output)
    # the edit is a tracked change: the paragraph reads as the new text
    assert [p.text for p in out.paragraphs if "Participants" in p.text] == [APP]
    # replies, one per thread, signed by the harness, in the right threads
    replies = {c.parent_id: c for c in out.comments if c.author == "andamentum respond"}
    assert set(replies) == {"0", "1", "3", "6"}
    assert replies["0"].text.startswith('Changed in section "Methods", paragraph 2:')
    assert "tracked" in replies["0"].text
    assert replies["1"].text.startswith(
        'A 12% fall. See "Intake fell by 12% over the study."'
    )
    assert replies["3"].text.startswith("Needs the authors: the authors must give")
    assert replies["6"].text.startswith("Already addressed:")
    assert "across three clinics" in replies["6"].text
    # no human comment was resolved, and the resolved thread was left alone
    human = {c.id: c for c in out.comments if not c.author.startswith("andamentum")}
    original = {c.id: c for c in open_docx(doc).comments}
    for cid, comment in human.items():
        assert comment.resolved == original[cid].resolved
    assert not any(c.parent_id == "4" for c in out.comments)
    assert any("resolved" in s for s in report.skipped_threads)
    # the thread already carrying a human reply is answered once, at its root
    assert sum(1 for c in out.comments if c.author == "andamentum respond") == 4


async def test_models_are_called_in_order_and_only_as_needed(
    doc: Path, script: Scripted
):
    await respond(doc)
    assert script.kinds() == [
        "Triage",
        "EditProposal",
        "Triage",
        "AnswerProposal",
        "Triage",
        "Triage",
    ]


async def test_report_and_events_and_snapshot(doc: Path, script: Scripted):
    result = await respond(doc)
    text = result.report_md.read_text()
    assert "## Needs the authors" in text and "Comment by Roe Reviewer" in text
    assert result.report.model_calls == 6
    assert result.report.input_tokens == 6 * USAGE.input_tokens
    store = ProjectStore(doc.with_name("paper.harness"))
    kinds = [e["type"] for e in store.read_events()]
    assert kinds == ["transaction", "respond_run"]
    assert store.snapshot_log()[0].startswith(
        "respond: 1 edit(s), 1 answer(s), 1 for the authors"
    )


async def test_a_second_run_does_not_answer_again(
    doc: Path, script: Scripted, tmp_path: Path
):
    first = await respond(doc)
    again = await respond(first.output, project=tmp_path / "other.harness")
    assert again.report.outcomes == []
    assert len(again.report.skipped_threads) >= 4
    assert again.report.exit_code == 0


async def test_an_edit_that_introduces_a_check_error_is_not_written(
    doc: Path, script: Scripted
):
    script.edit_text[PAPER] = PAPER + " The effect was large (t(20) = 1.5, p = .001)."
    result = await respond(doc)
    report = result.report
    assert report.counts["edit"] == 0 and report.counts["author"] == 2
    assert (
        len(report.gate_rejections) == 1 and "statistics" in report.gate_rejections[0]
    )
    out = open_docx(result.output)
    assert [p.text for p in out.paragraphs if "Participants" in p.text] == [PAPER]
    reply = next(c for c in out.comments if c.parent_id == "0")
    assert reply.text.startswith("Needs the authors: the edit was not written")
    demoted = next(o for o in report.outcomes if o.item.comment_id == "0")
    assert demoted.proposed == "edit" and demoted.category == "author"


async def test_an_edit_that_changes_nothing_or_a_missing_paragraph_goes_to_the_authors(
    doc: Path, script: Scripted
):
    script.edit_text[PAPER] = PAPER
    result = await respond(doc)
    demoted = next(o for o in result.report.outcomes if o.item.comment_id == "0")
    assert demoted.category == "author" and "changes nothing" in demoted.demoted


async def test_an_answer_without_a_quotable_passage_goes_to_the_authors(
    doc: Path, script: Scripted
):
    script.answer = AnswerProposal(
        answerable=True,
        answer="Large.",
        evidence_quotes=["a sentence that is not there"],
    )
    result = await respond(doc)
    outcome = next(o for o in result.report.outcomes if o.item.comment_id == "1")
    assert outcome.category == "author" and "quoted" in outcome.demoted


async def test_a_done_without_verified_evidence_goes_to_the_authors(
    doc: Path, script: Scripted
):
    script.triage["number of clinics"] = Triage(
        category="done", reason="it is there", evidence_quote="made-up passage"
    )
    result = await respond(doc)
    outcome = next(o for o in result.report.outcomes if o.item.comment_id == "6")
    assert outcome.category == "author"


async def test_two_edits_in_one_paragraph_compose(doc: Path, script: Scripted):
    # the second thread's paragraph is the same paragraph as the first one
    script.triage["How large was the fall"] = Triage(
        category="edit",
        reason="make it a number",
        target_quote="Participants logged every meal",
    )
    # a comment anchored in the intake paragraph may not retarget another paragraph
    script.edit_text[FELL] = "Intake fell by 12% over the four weeks of the study."
    result = await respond(doc)
    assert result.report.counts["edit"] == 2
    out = open_docx(result.output)
    texts = [p.text for p in out.paragraphs]
    assert (
        APP in texts and "Intake fell by 12% over the four weeks of the study." in texts
    )


async def test_no_items_needs_no_model(tmp_path: Path):
    plain = build_docx(
        tmp_path / "plain.docx", paragraph(run("Nothing to answer here."))
    )
    result = await run_respond(RespondParams(file=plain, offline=True))
    assert result.report.outcomes == [] and result.report.exit_code == 0
    assert result.output.is_file()


async def test_items_without_a_model_is_a_usage_error(doc: Path, script: Scripted):
    with pytest.raises(RespondUsageError, match="--model"):
        await run_respond(RespondParams(file=doc, offline=True))
    with pytest.raises(RespondUsageError, match="must not be the input"):
        await run_respond(RespondParams(file=doc, out=doc, model="m", offline=True))
    assert script.calls == []


# ------------------------------------------------------------------ reviews


REVIEWS = """Reviewer 1

1. Please say it was an app, not paper. The methods read as if diaries were kept on paper.

2. The authors claim a causal effect. Can you justify causality?

Reviewer 2

1. The paper is wonderful.
"""


class ScriptedReviews(Scripted):
    async def __call__(self, model: str, **kw):
        if kw["output_type"] is ReviewPoints:
            self.calls.append(("ReviewPoints", kw["user_message"]))
            return (
                ReviewPoints(
                    points=[
                        ReviewPoint(
                            label="Reviewer 1, point 1",
                            quote="Please say it was an app, not paper. The methods read as if diaries were kept on paper.",
                        ),
                        ReviewPoint(
                            label="Reviewer 1, point 2",
                            quote="The authors claim a causal effect. Can you justify causality?",
                        ),
                        ReviewPoint(
                            label="Reviewer 2, point 1",
                            quote="The paper is not wonderful at all.",  # not in the text
                        ),
                    ]
                ),
                USAGE,
            )
        return await super().__call__(model, **kw)


async def test_reviews_file_gives_a_response_letter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stub = ScriptedReviews()
    stub.triage = {
        "diaries were kept on paper": Triage(
            category="edit", reason="app", target_quote="using a paper diary"
        ),
        "justify causality": Triage(
            category="author",
            reason="the authors must give the argument",
            target_quote="the effect is causal",
        ),
    }
    # the document's own comment threads are items as well; park them
    for key in (
        "Please say it was an app",
        "How large was the fall",
        "number of clinics",
    ):
        stub.triage[key] = Triage(category="author", reason="parked")
    monkeypatch.setattr("andamentum.compose._call.run_agent_with_usage", stub)
    doc = make_doc(tmp_path / "paper.docx")
    reviews = tmp_path / "reviews.txt"
    reviews.write_text(REVIEWS)

    result = await run_respond(
        RespondParams(
            file=doc,
            reviews=reviews,
            model="m",
            offline=True,
            out=tmp_path / "out" / "paper.docx",
        )
    )
    report = result.report
    # the comment threads are items too: only the review points are in the letter
    reviews_items = [o for o in report.outcomes if o.item.source == "review"]
    assert [o.item.label for o in reviews_items] == [
        "Reviewer 1, point 1",
        "Reviewer 1, point 2",
    ]
    assert report.unverified_points == ["Reviewer 2, point 1"]
    assert report.exit_code == 4

    assert result.letter_docx == tmp_path / "out" / "response-letter.docx"
    assert result.letter_md == tmp_path / "out" / "response-letter.md"
    letter = (tmp_path / "out" / "response-letter.md").read_text()
    assert "## Reviewer 1, point 1" in letter
    assert "> Please say it was an app, not paper." in letter
    assert "**Response.** We changed the text:" in letter
    assert (
        'section "Methods", paragraph 2; tracked change by andamentum respond' in letter
    )
    assert (
        "[Authors to complete] Needs the authors: the authors must give the argument"
        in letter
    )
    assert "Reviewer 2, point 1" in letter.split("could not be read reliably")[1]
    import docx

    paragraphs = [p.text for p in docx.Document(str(result.letter_docx)).paragraphs]
    assert (
        "Reviewer 1, point 2" in paragraphs
        and "Response to the reviewers" in paragraphs
    )

    # the edit from the review point is a tracked change; no thread, no reply for it
    out = open_docx(result.output)
    assert any(APP == p.text for p in out.paragraphs)
    # an author point with a located passage gets an anchored comment
    ours = [
        c
        for c in out.comments
        if c.author == "andamentum respond" and c.parent_id is None
    ]
    assert len(ours) == 1 and ours[0].text.startswith(
        "Reviewer 1, point 2: Needs the authors"
    )
    anchor = next(p for p in out.paragraphs if p.id == ours[0].paragraph_id)
    assert anchor.text[ours[0].start : ours[0].end] == "the effect is causal"


async def test_an_existing_response_letter_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "andamentum.compose._call.run_agent_with_usage", ScriptedReviews()
    )
    doc = make_doc(tmp_path / "paper.docx")
    reviews = tmp_path / "reviews.txt"
    reviews.write_text(REVIEWS)
    out = tmp_path / "out" / "paper.docx"
    out.parent.mkdir()
    letter = out.parent / "response-letter.md"
    letter.write_text("my hand edits\n")
    before = sha(letter)
    with pytest.raises(RespondUsageError, match="response-letter.md"):
        await run_respond(
            RespondParams(file=doc, reviews=reviews, model="m", offline=True, out=out)
        )
    assert sha(letter) == before
    assert not out.exists() and not (out.parent / "response-letter.docx").exists()


async def test_a_copy_finds_the_project_of_the_original(
    tmp_path: Path, script: Scripted
):
    doc = make_doc(tmp_path / "paper.docx")
    first = await respond(doc)
    copy = tmp_path / "paper.bound.docx"
    copy.write_bytes(first.output.read_bytes())
    again = await respond(copy, out=tmp_path / "again.docx")
    assert again.report.project_id == first.report.project_id
    assert not (tmp_path / "paper.bound.harness").exists()


async def test_review_point_without_a_locatable_passage_is_for_the_authors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stub = ScriptedReviews()
    stub.triage = {"diaries were kept on paper": Triage(category="edit", reason="app")}
    stub.triage["justify causality"] = Triage(category="author", reason="x")
    for key in (
        "Please say it was an app",
        "How large was the fall",
        "number of clinics",
    ):
        stub.triage.setdefault(key, Triage(category="author", reason="x"))
    monkeypatch.setattr("andamentum.compose._call.run_agent_with_usage", stub)
    doc = make_doc(tmp_path / "paper.docx")
    reviews = tmp_path / "reviews.txt"
    reviews.write_text(REVIEWS)
    result = await run_respond(
        RespondParams(file=doc, reviews=reviews, model="m", offline=True)
    )
    first = next(
        o for o in result.report.outcomes if o.item.label == "Reviewer 1, point 1"
    )
    assert first.category == "author" and "could not be located" in first.demoted


async def test_docx_and_pdf_reviews_go_through_harvest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from andamentum.compose.harness.respond.items import read_reviews

    seen: list[Path] = []

    async def fake_extract(source):
        seen.append(Path(source))
        return "extracted text"

    monkeypatch.setattr("andamentum.harvest.extract", fake_extract)
    for name in ("r.docx", "r.pdf"):
        assert await read_reviews(tmp_path / name) == "extracted text"
    assert [p.name for p in seen] == ["r.docx", "r.pdf"]


async def test_reviews_must_be_a_known_type(tmp_path: Path, script: Scripted):
    doc = make_doc(tmp_path / "paper.docx")
    odd = tmp_path / "reviews.rtf"
    odd.write_text("x")
    with pytest.raises(ValueError, match="must be"):
        await run_respond(RespondParams(file=doc, reviews=odd, model="m", offline=True))


# ----------------------------------------------- collaborators' own edits


async def test_collaborators_edits_since_the_last_look_are_checked(tmp_path: Path):
    # first look: a document with no threads, so no model is needed
    first = build_docx(
        tmp_path / "paper2.docx",
        paragraph(run("Methods"), style="Heading1") + paragraph(run(STAT_OK)),
    )
    project = tmp_path / "shared.harness"
    one = await run_respond(RespondParams(file=first, project=project, offline=True))
    assert "no earlier snapshot" in one.report.collaborator_baseline
    # a collaborator hand-edits the statistic, the document is saved again
    second = build_docx(
        tmp_path / "paper3.docx",
        paragraph(run("Methods"), style="Heading1") + paragraph(run(STAT_BAD)),
    )
    two = await run_respond(RespondParams(file=second, project=project, offline=True))
    assert two.report.collaborator_changes == 1
    assert len(two.report.collaborator_errors) == 1
    assert two.report.collaborator_errors[0].check == "statistics"
    assert two.report.exit_code == 4
    out = open_docx(two.output)
    comment = next(c for c in out.comments if c.author == "andamentum respond")
    assert comment.text.startswith("Changed since the last look. [error] statistics:")
    # the same text again: nothing changed since that look, so nothing is flagged
    three = await run_respond(
        RespondParams(
            file=second, project=project, offline=True, out=tmp_path / "again.docx"
        )
    )
    assert (
        three.report.collaborator_changes == 0
        and three.report.collaborator_errors == []
    )


# ------------------------------------------------------------ decision model


BY_COMMENT = {
    "Please say it was an app": "edit",
    "How large was the fall": "answer",
    "justify causality": "author",
    "number of clinics": "done",
}


def decision_server(
    p: float, *, force: str | None = None, name: str = "nimble:9b"
) -> StubDecisions:
    """A decision model that picks the category by what the comment says."""

    def respond(state, questions):
        text = str(state)
        picked = force or next(c for k, c in BY_COMMENT.items() if k in text)
        return {k: choice_answer(list(CATEGORIES), picked, p) for k in questions}

    return StubDecisions(respond, name=name)


async def test_the_decision_model_saves_the_language_model_call_for_done_and_author(
    doc: Path, script: Scripted
):
    stub = decision_server(0.99)
    result = await respond(doc, decision_handle=stub.model())
    assert all(o.decision is not None for o in result.report.outcomes)
    # edit and answer still need the language model for the quote and the text
    assert script.kinds().count("Triage") == 2
    assert "EditProposal" in script.kinds() and "AnswerProposal" in script.kinds()
    saved = [
        o for o in result.report.outcomes if o.decision and o.decision.saved_llm_call
    ]
    assert {o.proposed for o in saved} == {"author", "done"}
    assert result.report.counts["edit"] == 1 and result.report.counts["answer"] == 1
    author = next(o for o in saved if o.proposed == "author")
    assert AUTHOR_REASON.rstrip(".") in (author.reply or "")
    assert author.decision is not None
    assert author.decision.choice == "author"
    assert "no language-model triage call" in result.report_md.read_text()


async def test_every_item_decided_as_author_makes_no_language_model_call(
    doc: Path, script: Scripted
):
    stub = decision_server(0.99, force="author")
    result = await respond(doc, decision_handle=stub.model())
    assert result.report.counts == {"edit": 0, "answer": 0, "author": 4, "done": 0}
    assert script.calls == []  # nothing was sent to a language model
    assert result.report.model_calls == 0


async def test_a_done_with_no_matching_passage_becomes_author(
    doc: Path, script: Scripted
):
    stub = decision_server(0.99, force="done")
    result = await respond(doc, decision_handle=stub.model())
    assert script.calls == []
    by_label = {o.item.text: o for o in result.report.outcomes}
    # "Please add the number of clinics" shares one word with the document,
    # not enough to point to a passage: demoted, as for any unquotable done
    clinics = next(o for t, o in by_label.items() if "number of clinics" in t)
    assert clinics.category == "author" and "no passage" in clinics.demoted


def test_templated_triage_points_at_the_best_matching_sentence():
    view = make_view(
        ("Methods", 1),
        ("We enrolled adults. Sodium intake was measured with a food diary.", None),
    )
    item = Item(
        id="i",
        source="comment",
        label="Comment",
        text="How was sodium intake measured?",
    )
    done = templated_triage("done", item, view)
    assert done.category == "done"
    assert done.evidence_quote == "Sodium intake was measured with a food diary."
    unrelated = Item(id="j", source="comment", label="Comment", text="Add a figure.")
    assert templated_triage("done", unrelated, view).evidence_quote == ""
    assert templated_triage("author", item, view).reason == AUTHOR_REASON
    with pytest.raises(ValueError, match="cannot be templated"):
        templated_triage("edit", item, view)


# ----------------------------------------------------------------- the verb


def test_verb_is_registered_and_the_cli_parses_it():
    _clear_verbs()
    load_builtin_verbs()
    parser = cli.build_parser()
    args = parser.parse_args(
        ["respond", "paper.docx", "--reviews", "r.txt", "--model", "m", "--offline"]
    )
    assert args.verb == "respond" and args.offline and str(args.reviews) == "r.txt"
    text = (
        parser.format_help()
        + next(
            a
            for a in parser._subparsers._group_actions  # type: ignore[union-attr]
        )
        .choices["respond"]
        .format_help()
    )
    assert "respond" in text and "--decision-model" in text


def test_cli_usage_error_exits_2(doc: Path, capsys: pytest.CaptureFixture[str]):
    _clear_verbs()
    code = cli.main(["respond", str(doc), "--offline"])
    assert code == 2
    assert "--model" in capsys.readouterr().err
