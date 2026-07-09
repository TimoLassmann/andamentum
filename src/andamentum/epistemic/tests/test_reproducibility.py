"""Tests for the reproducibility tripwire (harness-backstops item 2a).

Covers the pure predicates (``warrants_reproducibility_check``, ``verdicts_agree``)
and the bounded, sequential ``check_claims_reproducibility`` driver with a fake
runner + repo (no live model).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..reproducibility import (
    check_claims_reproducibility,
    verdicts_agree,
    warrants_reproducibility_check,
)


@dataclass
class _Ev:
    entity_id: str
    support_judgment: str | None = None
    judgment_one_hot: bool | None = None
    extracted_content: str = "some evidence"
    source_ref: str = "src"
    invalidated: bool = False
    cluster_status: str = "representative"


@dataclass
class _Claim:
    entity_id: str
    statement: str = "X causes Y"
    scope: str = ""
    evidence_ids: list[str] = field(default_factory=list)


class _Repo:
    def __init__(self, evidence: dict[str, _Ev]):
        self._evidence = evidence

    async def get(self, kind: str, eid: str):
        return self._evidence.get(eid)


class _FakeRunner:
    """Returns a canned paraphrase, and a scripted verdict for re-judgment.

    ``rejudge_verdict`` is the verdict the re-judged evidence gets, letting a
    test force agreement or a flip.
    """

    def __init__(self, rejudge_verdict: str):
        self._rejudge_verdict = rejudge_verdict
        self.calls: list[str] = []

    async def run(self, agent_name: str, **kwargs):
        self.calls.append(agent_name)
        if agent_name == "epistemic_paraphrase_claim":
            return type("P", (), {"paraphrase": "Y is caused by X"})()
        raise AssertionError(f"unexpected agent {agent_name}")


class TestPurePredicates:
    def test_warrants_only_one_hot_directional(self) -> None:
        assert warrants_reproducibility_check(
            _Ev("e", support_judgment="supports", judgment_one_hot=True)
        )
        # Not one-hot → entropy is already informative, skip.
        assert not warrants_reproducibility_check(
            _Ev("e", support_judgment="supports", judgment_one_hot=False)
        )
        # One-hot but no_bearing → not directional / load-bearing.
        assert not warrants_reproducibility_check(
            _Ev("e", support_judgment="no_bearing", judgment_one_hot=True)
        )
        # No distribution measured → nothing to call brittle.
        assert not warrants_reproducibility_check(
            _Ev("e", support_judgment="supports", judgment_one_hot=None)
        )

    def test_verdicts_agree(self) -> None:
        assert verdicts_agree("supports", "supports")
        assert not verdicts_agree("supports", "contradicts")
        assert not verdicts_agree("contradicts", "supports")
        # Drift to/from no_bearing is not a directional flip.
        assert verdicts_agree("supports", "no_bearing")
        assert verdicts_agree("no_bearing", "contradicts")


class TestDriver:
    async def test_no_runner_is_noop(self) -> None:
        out = await check_claims_reproducibility([], _Repo({}), None)
        assert out.flip_found is False and out.checks_run == 0

    async def test_flip_detected(self, monkeypatch) -> None:
        ev = _Ev("e1", support_judgment="supports", judgment_one_hot=True)
        claim = _Claim("c1", evidence_ids=["e1"])
        runner = _FakeRunner(rejudge_verdict="contradicts")

        # Patch judge_evidence to return the scripted flipped verdict.
        async def fake_judge(**kwargs):
            return type("J", (), {"verdict": "contradicts"})()

        monkeypatch.setattr(
            "andamentum.epistemic.reproducibility.judge_evidence", fake_judge
        )
        out = await check_claims_reproducibility([claim], _Repo({"e1": ev}), runner)
        assert out.flip_found is True
        assert out.flipped_evidence_id == "e1"
        assert out.checks_run == 1

    async def test_no_flip_when_verdict_holds(self, monkeypatch) -> None:
        ev = _Ev("e1", support_judgment="supports", judgment_one_hot=True)
        claim = _Claim("c1", evidence_ids=["e1"])
        runner = _FakeRunner(rejudge_verdict="supports")

        async def fake_judge(**kwargs):
            return type("J", (), {"verdict": "supports"})()

        monkeypatch.setattr(
            "andamentum.epistemic.reproducibility.judge_evidence", fake_judge
        )
        out = await check_claims_reproducibility([claim], _Repo({"e1": ev}), runner)
        assert out.flip_found is False
        assert out.checks_run == 1

    async def test_skips_non_one_hot_evidence(self, monkeypatch) -> None:
        # A confident-but-graded (not one-hot) judgment is not checked at all.
        ev = _Ev("e1", support_judgment="supports", judgment_one_hot=False)
        claim = _Claim("c1", evidence_ids=["e1"])
        runner = _FakeRunner(rejudge_verdict="contradicts")

        async def fake_judge(**kwargs):  # pragma: no cover - must not be called
            raise AssertionError("judge should not run for non-one-hot evidence")

        monkeypatch.setattr(
            "andamentum.epistemic.reproducibility.judge_evidence", fake_judge
        )
        out = await check_claims_reproducibility([claim], _Repo({"e1": ev}), runner)
        assert out.checks_run == 0
        assert out.flip_found is False

    async def test_respects_max_checks_budget(self, monkeypatch) -> None:
        evs = {
            f"e{i}": _Ev(f"e{i}", support_judgment="supports", judgment_one_hot=True)
            for i in range(6)
        }
        claim = _Claim("c1", evidence_ids=list(evs))
        runner = _FakeRunner(rejudge_verdict="supports")

        async def fake_judge(**kwargs):
            return type("J", (), {"verdict": "supports"})()

        monkeypatch.setattr(
            "andamentum.epistemic.reproducibility.judge_evidence", fake_judge
        )
        out = await check_claims_reproducibility(
            [claim], _Repo(evs), runner, max_checks=2
        )
        assert out.checks_run == 2
