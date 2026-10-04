from __future__ import annotations

from andamentum.compose.genres import GENRES
from andamentum.compose.log import ActionLog
from andamentum.compose.revise import revise_section
from andamentum.compose.schemas import Fact, Outline, SectionPlan, WorkItem
from andamentum.compose.write import SectionBody, draft_section

from ._stub import install, make_brief

MEMO = GENRES["memo"]
FACTS = [
    Fact(
        id="f1",
        source_id="src1",
        statement="Pilot starts 1 March",
        quote="starts 1 March",
        start=0,
        end=5,
    ),
    Fact(id="f2", source_id="src1", statement="Unrelated", quote="zzz", start=0, end=1),
]
PLAN = SectionPlan(
    title="Summary",
    purpose="state the date",
    key_points=["start date"],
    fact_ids=["f1"],
    target_words=80,
)
OUTLINE = Outline(sections=[PLAN])


async def test_draft_section_prompt_and_result(monkeypatch, tmp_path):
    stub = install(
        monkeypatch, [SectionBody(body="  The pilot starts 1 March.[[f1]]  ")]
    )
    log = ActionLog(tmp_path)
    body = await draft_section(
        PLAN,
        brief=make_brief(),
        genre=MEMO,
        outline=OUTLINE,
        facts=FACTS,
        previous_summaries=["Intro: Apollo."],
        model="m",
        log=log,
    )
    assert body == "The pilot starts 1 March.[[f1]]"
    msg = stub.calls[0]["user_message"]
    assert "f1: Pilot starts 1 March" in msg and "f2" not in msg
    assert "Intro: Apollo." in msg
    assert "[[f12]]" in stub.calls[0]["instructions"]
    assert log.records()[0]["node"] == "draft"


async def test_draft_first_section_and_no_facts(monkeypatch):
    stub = install(monkeypatch, [SectionBody(body="[TO FILL: date]")])
    plan = PLAN.model_copy(update={"fact_ids": []})
    await draft_section(
        plan,
        brief=make_brief(),
        genre=MEMO,
        outline=OUTLINE,
        facts=FACTS,
        previous_summaries=[],
        model="m",
    )
    msg = stub.calls[0]["user_message"]
    assert "first section" in msg and "FACTS FOR THIS SECTION: none" in msg


async def test_revise_section_includes_items_and_body(monkeypatch):
    stub = install(monkeypatch, [SectionBody(body="Better text.[[f1]]")])
    items = [
        WorkItem(
            level="section",
            section_id="s01",
            reason="claim unsupported",
            source="grounding",
        ),
        WorkItem(
            level="section", section_id="s01", reason="too vague", source="critic"
        ),
    ]
    body = await revise_section(
        "Old text.",
        items,
        plan=PLAN,
        brief=make_brief(),
        genre=MEMO,
        facts=FACTS,
        model="m",
    )
    assert body == "Better text.[[f1]]"
    msg = stub.calls[0]["user_message"]
    assert "Old text." in msg and "(grounding) claim unsupported" in msg
    assert "(critic) too vague" in msg
