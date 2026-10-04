"""Real-model tests for compose: the one integration the offline suite cannot cover.

Deselected by default (marker ``compose_eval``); run with
``uv run pytest -m compose_eval``. They call ``ollama:gemma4:31b-cloud``, named
explicitly, and use only synthetic inputs.
"""

from __future__ import annotations

import pytest
from dotenv import load_dotenv

from andamentum.compose import Roles, write_document
from andamentum.compose.checks.critic import run_critic
from andamentum.compose.genres import genre_for
from andamentum.compose.log import ActionLog
from andamentum.compose.bundle import Bundle

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
