"""Real-model tests for compose: the one integration the offline suite cannot cover.

Deselected by default (marker ``compose_eval``); run with
``uv run pytest -m compose_eval``. They call ``ollama:gemma4:31b-cloud``, named
explicitly, and use only synthetic inputs. The Word tests run the example
project (``harness/examples``) with Crossref answered from the recorded
fixtures, and assert structure only: what a model wrote is not asserted.
"""

from __future__ import annotations

import hashlib

import pytest
from dotenv import load_dotenv

from andamentum.compose import Roles, write_document
from andamentum.compose.checks.critic import run_critic
from andamentum.compose.genres import genre_for
from andamentum.compose.log import ActionLog
from andamentum.compose.bundle import Bundle
from andamentum.compose.harness.bench.fixture_http import scheme_only_url_check
from andamentum.compose.harness.bindings.bind import BindParams, run_bind
from andamentum.compose.harness.bindings.refresh import RefreshParams, run_refresh
from andamentum.compose.harness.docx.reader import open_docx
from andamentum.compose.harness.draft.verb import WriteParams, run_write
from andamentum.compose.harness.examples import (
    ExampleProject,
    fixture_transport_for_example,
    make_example,
)
from andamentum.compose.harness.respond import RespondParams, run_respond

pytestmark = pytest.mark.compose_eval

MODEL = "ollama:gemma4:31b-cloud"
NOTES = (
    "# Demo notes\n\n"
    "The Apollo Labs demo moves to 19 March at 10:00 in Meeting Room 4.\n"
    "Harbor Foods must confirm the attendees by 10 March.\n"
)


@pytest.fixture(autouse=True)
def _dotenv() -> None:
    load_dotenv()


async def test_write_document_end_to_end_email(tmp_path):
    source = tmp_path / "notes.md"
    source.write_text(NOTES, encoding="utf-8")
    result = await write_document(
        "Email Jane Doe at Harbor Foods that the Apollo Labs demo moved to 19 March "
        "and ask her to confirm the attendees by 10 March",
        out=tmp_path / "email.compose",
        roles=Roles.all(MODEL),
        genre="email",
        sources=[str(source)],
        max_rounds=3,
        formats=["txt"],
    )
    email = (tmp_path / "email.compose" / "out" / "email.txt").read_text("utf-8")
    assert email.startswith("Subject:")
    assert "[[" not in email
    assert result.stop_reason in ("goal_met", "cap_reached", "no_progress")


async def test_real_critic_on_a_small_rendered_bundle(tmp_path):
    bundle = Bundle.create(tmp_path / "b.compose", title="Apollo pilot memo")
    bundle.add_section(
        title="Summary",
        body="The Apollo Labs pilot starts on 1 March. We recommend going ahead.",
        status="drafted",
    )
    bundle.add_section(
        title="Recommendation",
        body="Approve the pilot and name an owner by Friday.",
        status="drafted",
    )
    result = await run_critic(
        bundle,
        genre=genre_for("memo"),
        model=MODEL,
        round_no=1,
        log=ActionLog(bundle.path),
    )
    assert result.review_path.is_file()
    assert result.major >= 0 and result.check.name == "critic"


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def example(tmp_path) -> ExampleProject:
    return make_example(tmp_path / "example")


async def test_write_into_word_cites_and_binds(example: ExampleProject):
    out = example.root / "paper.docx"
    before = _sha(example.library), _sha(example.notes)
    result = await run_write(
        WriteParams(
            task="Introduction and results for the SaltLog pilot, citing the published "
            "works in the library for the background",
            out=out,
            genre="article",
            source=[str(example.notes)],
            results=example.results,
            library=example.library,
            model=MODEL,
        ),
        transport=fixture_transport_for_example(),
        url_check=scheme_only_url_check,
    )
    assert (_sha(example.library), _sha(example.notes)) == before
    assert result.exit_code in (0, 4)
    view = open_docx(out)
    assert any(p.heading_level for p in view.paragraphs)
    assert view.project_id is not None
    assert result.numbers_bound >= 1
    assert any(a.kind == "number" for a in view.anchors)
    # whether the model cites a library work varies between runs; when it does,
    # the work is listed, and nothing it cites may be unverified
    if result.citations:
        assert "References" in [p.text for p in view.paragraphs]
    assert not result.refused
    assert result.statement in view.text()


async def test_bind_numbers_then_refresh_after_a_result_changes(
    example: ExampleProject,
):
    before = _sha(example.collaborator)
    bound = await run_bind(
        BindParams(file=example.collaborator, results=example.results, model=MODEL)
    )
    assert _sha(example.collaborator) == before
    assert bound.report.bound, "the model matched no number to a result file"
    summary = example.results / "summary.csv"
    summary.write_text(summary.read_text().replace("412", "398"), encoding="utf-8")
    refreshed = run_refresh(RefreshParams(file=bound.output, results=example.results))
    report = refreshed.report
    assert report.exit_code in (0, 1)
    assert len(report.changes) + report.unchanged == len(bound.report.bound)
    assert open_docx(refreshed.output).project_id == bound.report.project_id


async def test_respond_replies_to_every_open_thread_and_writes_the_letter(
    example: ExampleProject,
):
    before = _sha(example.collaborator)
    out = example.root / "out" / "collaborator.responded.docx"
    run = await run_respond(
        RespondParams(
            file=example.collaborator,
            reviews=example.reviews,
            results=None,
            model=MODEL,
            offline=True,
            out=out,
        )
    )
    assert _sha(example.collaborator) == before
    assert run.report.exit_code in (0, 4)
    source = open_docx(example.collaborator)
    after = open_docx(run.output)
    open_threads = [
        c
        for c in source.comments
        if c.parent_id is None and not c.resolved and c.author != "andamentum compose"
    ]
    replied = {c.parent_id for c in after.comments if c.parent_id is not None}
    for thread in open_threads:
        assert thread.id in replied, f"no reply to {thread.id}"
    points = [o for o in run.report.outcomes if o.item.source == "review"]
    assert run.letter_md is not None and run.letter_docx is not None
    assert points and run.letter_md.read_text().count("**Response.**") == len(points)
