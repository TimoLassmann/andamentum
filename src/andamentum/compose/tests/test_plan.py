from __future__ import annotations

from andamentum.compose.genres import GENRES
from andamentum.compose.log import ActionLog
from andamentum.compose.plan import (
    MAX_OUTLINE_AMENDMENTS,
    amend_outline,
    check_outline,
    make_outline,
)
from andamentum.compose.schemas import (
    Fact,
    FigurePlan,
    Outline,
    SectionPlan,
    Source,
    WorkItem,
)

from ._stub import install, make_brief

MEMO = GENRES["memo"]
REPORT = GENRES["report"]
FACTS = [Fact(id="f1", source_id="src1", statement="s", quote="q", start=0, end=1)]


def _outline(words=(100, 100, 300, 100, 100), titles=None, **kw) -> Outline:
    titles = titles or [s.title for s in MEMO.sections]
    return Outline(
        sections=[
            SectionPlan(
                title=t,
                purpose="p",
                key_points=["start 1 March"] if i == 0 else [],
                fact_ids=["f1"],
                target_words=w,
                **kw,
            )
            for i, (t, w) in enumerate(zip(titles, words))
        ]
    )


def _by_name(results):
    return {r.name: r for r in results}


def _check(outline, brief=None, genre=MEMO, tabular=()):
    return _by_name(
        check_outline(
            outline,
            brief or make_brief(),
            genre,
            facts=FACTS,
            tabular_sources=list(tabular),
        )
    )


def test_good_outline_passes_everything():
    assert all(r.passed for r in _check(_outline()).values())


def test_missing_and_extra_sections_in_fixed_genre():
    r = _check(
        _outline(
            titles=[
                "Summary",
                "Background",
                "Options or Analysis",
                "Recommendation",
                "Extras",
            ]
        )
    )["outline_sections"]
    assert not r.passed
    assert any("Next steps".casefold() in d for d in r.details)
    assert any("extras" in d for d in r.details)


def test_flexible_genre_with_guidelines_skips_skeleton():
    brief = make_brief(
        genre="report", guidelines="Follow the call", word_range=(1500, 6000)
    )
    out = Outline(
        sections=[
            SectionPlan(
                title="Custom",
                purpose="p",
                key_points=["1 March"],
                fact_ids=[],
                target_words=2000,
            )
        ]
    )
    res = _by_name(check_outline(out, brief, REPORT, facts=FACTS, tabular_sources=[]))
    assert res["outline_sections"].passed
    no_guide = make_brief(genre="report", word_range=(1500, 6000))
    res = _by_name(
        check_outline(out, no_guide, REPORT, facts=FACTS, tabular_sources=[])
    )
    assert not res["outline_sections"].passed


def test_duplicate_titles_must_include_words_and_fact_ids():
    out = _outline(
        words=(1, 1, 1, 1, 1),
        titles=[
            "Summary",
            "summary",
            "Options or Analysis",
            "Recommendation",
            "Next steps",
        ],
    )
    out.sections[1].fact_ids = ["f99"]
    res = _check(out, make_brief(must_include=["1 March", "June 2028"]))
    assert not res["outline_unique_titles"].passed
    assert not res["outline_words"].passed
    assert res["outline_fact_ids"].details == ["unknown fact id: f99"]
    assert res["outline_must_include"].details == [
        "must-include item not in any section's key_points: June 2028"
    ]


def test_figure_rules():
    fig = FigurePlan(source_id="src9", kind="bar", x="a", y="b", caption="c", label="l")
    res = _check(_outline(figure=fig))
    assert not res["outline_figures"].passed  # memo takes no figures
    tab = Source(
        id="src2", path="d.csv", title="d", sha256="0" * 64, chars=1, kind="file"
    )
    brief = make_brief(genre="report", word_range=(1500, 6000))
    out = Outline(
        sections=[
            SectionPlan(
                title="Findings",
                purpose="p",
                key_points=["1 March"],
                fact_ids=[],
                target_words=2000,
                figure=fig,
            )
        ]
    )
    res = _by_name(
        check_outline(out, brief, REPORT, facts=FACTS, tabular_sources=[tab])
    )
    assert "not a tabular source" in res["outline_figures"].details[0]


async def test_make_outline_retries_once_with_problems(monkeypatch, tmp_path):
    bad = _outline(words=(1, 1, 1, 1, 1))
    good = _outline()
    stub = install(monkeypatch, [bad, good])
    log = ActionLog(tmp_path)
    out = await make_outline(
        make_brief(), MEMO, FACTS, tabular_sources=[], model="m", log=log
    )
    assert out == good
    assert len(stub.calls) == 2
    assert "target_words sum to 5" in stub.calls[1]["user_message"]
    assert any(r.get("event") == "outline_retry" for r in log.records())


async def test_make_outline_proceeds_with_warnings_after_retry(monkeypatch, tmp_path):
    bad = _outline(words=(1, 1, 1, 1, 1))
    stub = install(monkeypatch, [bad, bad])
    log = ActionLog(tmp_path)
    out = await make_outline(
        make_brief(), MEMO, FACTS, tabular_sources=[], model="m", log=log
    )
    assert out == bad and len(stub.calls) == 2
    assert any(r.get("event") == "outline_warnings" for r in log.records())


async def test_no_retry_when_clean(monkeypatch):
    stub = install(monkeypatch, [_outline()])
    await make_outline(make_brief(), MEMO, FACTS, tabular_sources=[], model="m")
    assert len(stub.calls) == 1


async def test_amend_outline_sends_items(monkeypatch):
    stub = install(monkeypatch, [_outline()])
    items = [
        WorkItem(level="outline", section_id=None, reason="add risks", source="critic")
    ]
    out = await amend_outline(
        _outline(),
        items,
        brief=make_brief(),
        genre=MEMO,
        facts=FACTS,
        tabular_sources=[],
        model="m",
    )
    assert out == _outline()
    assert "add risks" in stub.calls[0]["user_message"]
    assert MAX_OUTLINE_AMENDMENTS == 1
