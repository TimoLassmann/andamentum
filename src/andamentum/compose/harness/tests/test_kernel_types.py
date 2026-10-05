from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from andamentum.compose.harness.kernel import (
    Bind,
    CheckContext,
    CheckSpec,
    CommentOp,
    Delete,
    Diagnostic,
    DocumentView,
    Insert,
    Operation,
    Paragraph,
    Replace,
    Reply,
    Transaction,
    Verb,
    all_checks,
    all_verbs,
    register,
    register_verb,
    select,
)
from andamentum.compose.harness.kernel import checks as checks_mod
from andamentum.compose.harness.kernel import verbs as verbs_mod


def _view(*paragraphs: tuple[str, str, int | None]) -> DocumentView:
    return DocumentView(
        path=Path("x.docx"),
        project_id=None,
        paragraphs=[
            Paragraph(id=pid, index=i, style="Normal", heading_level=level, text=text)
            for i, (pid, text, level) in enumerate(paragraphs)
        ],
        anchors=[],
        fields=[],
        comments=[],
    )


def test_text_joins_with_newline():
    view = _view(("p0", "Title", 1), ("p1", "abc", None))
    assert view.text() == "Title\nabc"


def test_locate_maps_offsets():
    view = _view(
        ("p0", "Title", 1), ("p1", "abc", None), ("p2", "", None), ("p3", "xy", None)
    )
    assert view.locate(0) == ("p0", 0)
    assert view.locate(5) == ("p0", 5)  # the separator belongs to the paragraph before
    assert view.locate(6) == ("p1", 0)
    assert view.locate(8) == ("p1", 2)
    assert view.locate(9) == ("p1", 3)
    assert view.locate(10) == ("p2", 0)
    assert view.locate(11) == ("p3", 0)
    assert view.locate(len(view.text())) == ("p3", 2)


def test_locate_rejects_out_of_range():
    view = _view(("p0", "ab", None))
    with pytest.raises(ValueError):
        view.locate(-1)
    with pytest.raises(ValueError):
        view.locate(3)


def test_locate_round_trips_every_offset():
    view = _view(("p0", "Head", 1), ("p1", "one two", None), ("p2", "three", None))
    starts = {}
    pos = 0
    for p in view.paragraphs:
        starts[p.id] = pos
        pos += len(p.text) + 1
    for offset in range(len(view.text()) + 1):
        pid, local = view.locate(offset)
        assert starts[pid] + local == offset


def test_sections_by_heading():
    view = _view(
        ("p0", "Preamble", None),
        ("p1", "Intro", 1),
        ("p2", "a", None),
        ("p3", "b", None),
        ("p4", "Methods", 2),
        ("p5", "Results", 1),
        ("p6", "c", None),
    )
    assert view.sections() == [
        ("", ["p0"]),
        ("Intro", ["p2", "p3"]),
        ("Methods", []),
        ("Results", ["p6"]),
    ]


def test_sections_without_preamble():
    view = _view(("p0", "Intro", 1), ("p1", "a", None))
    assert view.sections() == [("Intro", ["p1"])]


def test_replace_validates_range_and_expect():
    Replace(paragraph_id="p", start=1, end=4, text="new", expect="old")
    with pytest.raises(ValidationError):
        Replace(paragraph_id="p", start=4, end=1, text="new", expect="")
    with pytest.raises(ValidationError):
        Replace(paragraph_id="p", start=0, end=3, text="new", expect="toolong")
    with pytest.raises(ValidationError):
        Replace(paragraph_id="p", start=-1, end=2, text="new", expect="ab")


def test_delete_requires_nonempty_range():
    Delete(paragraph_id="p", start=0, end=2, expect="ab")
    with pytest.raises(ValidationError):
        Delete(paragraph_id="p", start=2, end=2, expect="")


def test_insert_validation():
    Insert(paragraph_id="p", at=0, text="x")
    with pytest.raises(ValidationError):
        Insert(paragraph_id="p", at=-1, text="x")
    with pytest.raises(ValidationError):
        Insert(paragraph_id="p", at=0, text="")


def test_comment_reply_bind_validation():
    CommentOp(paragraph_id="p", start=0, end=0, text="note")
    with pytest.raises(ValidationError):
        CommentOp(paragraph_id="p", start=0, end=1, text=" ")
    with pytest.raises(ValidationError):
        Reply(parent_comment_id="c1", text="")
    Bind(paragraph_id="p", start=0, end=3, anchor_id="a1", kind="number")
    with pytest.raises(ValidationError):
        Bind(paragraph_id="p", start=0, end=3, anchor_id="a1", kind="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Bind(paragraph_id="p", start=2, end=2, anchor_id="a1", kind="span")


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        Insert(paragraph_id="p", at=0, text="x", surprise=1)  # type: ignore[call-arg]


def test_transaction_round_trips_through_json():
    txn = Transaction(
        ops=[
            Replace(paragraph_id="p1", start=0, end=3, text="new", expect="old"),
            Delete(paragraph_id="p1", start=5, end=6, expect="x"),
            Reply(parent_comment_id="c1", text="done"),
        ],
        author="andamentum check",
        reason="fix",
        source="citations",
    )
    again = Transaction.model_validate_json(txn.model_dump_json())
    assert again == txn
    assert [type(o) for o in again.ops] == [Replace, Delete, Reply]
    adapter = TypeAdapter(Operation)
    assert isinstance(
        adapter.validate_python(
            {"op": "delete", "paragraph_id": "p", "start": 0, "end": 1, "expect": "a"}
        ),
        Delete,
    )


def test_diagnostic_defaults():
    d = Diagnostic(
        check="c",
        tier=0,
        severity="error",
        paragraph_id=None,
        start=None,
        end=None,
        message="m",
        evidence=[],
    )
    assert d.data == {}
    with pytest.raises(ValidationError):
        Diagnostic(
            check="c",
            tier=3,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
            severity="error",
            paragraph_id=None,
            start=None,
            end=None,
            message="m",
            evidence=[],
        )


class _Fake:
    version = "1"

    def __init__(self, id: str, tier: int, needs: list[str]) -> None:
        self.spec = CheckSpec(id=id, tier=tier, scope="document", needs=needs)  # type: ignore[arg-type]

    def run(self, view: DocumentView, ctx: CheckContext) -> list[Diagnostic]:
        return []


@pytest.fixture
def clean_registry():
    checks_mod._clear_registry()
    yield
    checks_mod._clear_registry()


def test_registry_register_all_select(clean_registry):
    a, b, c = _Fake("a", 0, []), _Fake("b", 0, ["network"]), _Fake("c", 1, ["model"])
    for check in (a, b, c):
        register(check)
    assert [x.spec.id for x in all_checks()] == ["a", "b", "c"]
    assert [x.spec.id for x in select(ids=["b", "c"])] == ["b", "c"]
    assert [x.spec.id for x in select(tiers=[0])] == ["a", "b"]
    assert [x.spec.id for x in select(needs=[])] == ["a"]
    assert [x.spec.id for x in select(tiers=[0], needs=["network"])] == ["a", "b"]
    with pytest.raises(ValueError):
        register(_Fake("a", 0, []))
    with pytest.raises(KeyError):
        select(ids=["nope"])


def test_check_context_defaults():
    ctx = CheckContext(store=None, offline=True)
    assert ctx.models == {} and ctx.results_dir is None


def test_promote_at_constant():
    assert checks_mod.PROMOTE_AT == 0.9


def test_verb_registry():
    saved = dict(verbs_mod._VERBS)
    verbs_mod._clear_verbs()
    try:
        from pydantic import BaseModel

        class P(BaseModel):
            file: str

        verb = register_verb(
            Verb(name="check", summary="s", params=P, handler=lambda p: 0)
        )
        assert all_verbs() == [verb]
        assert verbs_mod.get_verb("check") is verb
        with pytest.raises(ValueError):
            register_verb(verb)
        with pytest.raises(KeyError):
            verbs_mod.get_verb("nope")
    finally:
        verbs_mod._clear_verbs()
        verbs_mod._VERBS.update(saved)
