"""Model-facing tests for the probe runner — everything stubbed, no real model.

Two of these tests exist to pin design invariants rather than behaviour anyone
asked for, and both would look like missing features to a later reader:

* probes are INDEPENDENT — nothing an earlier probe found may reach a later
  probe's prompt. Review mode's criterion cascade does the opposite on purpose;
  wiring one in here would silently destroy the run-to-run comparability that
  is checklist mode's whole product;
* probe calls are SEQUENTIAL — never two model calls in flight at once. Local
  Ollama serialises them anyway, and concurrent local inference is forbidden
  project-wide.
"""

from __future__ import annotations

import asyncio
import logging
import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded

from andamentum.whetstone.v3.checklist.probes import Probe
from andamentum.whetstone.v3.checklist.run_probes import (
    _AbsenceFinding,
    _AbsenceFindings,
    _PresenceFinding,
    _PresenceFindings,
    run_probe,
    run_probes,
)
from andamentum.whetstone.v3.model import DocumentModel, Section

_AGENT_FACTORY = "andamentum.whetstone.v3.checklist.run_probes._build_agent"


@contextmanager
def _capture_v3_logs() -> Iterator[list[logging.LogRecord]]:
    """Attach a handler directly to the v3 logger. ``caplog`` is unreliable
    here because ``whetstone/cli.py`` flips ``andamentum.whetstone.propagate``
    to False, breaking propagation to the root caplog handler when CLI tests
    run earlier in the same session."""
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(level=logging.WARNING)
    logger = logging.getLogger("andamentum.whetstone.v3")
    prior_level = logger.level
    logger.setLevel(logging.WARNING)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)


# ── fakes ───────────────────────────────────────────────────────────────────


class _RecordingAgents:
    """A stand-in for ``run_probes._build_agent``.

    Records every prompt it is given (in call order) and how many calls were
    in flight at once, then hands back whatever *responder* returns for that
    prompt. A responder may raise to simulate a probe blowing up.
    """

    def __init__(self, responder: Callable[[str], Any]) -> None:
        self._responder = responder
        self.prompts: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0

    def factory(self, *, output_type: type, instructions: str, agent_model: str) -> Any:
        outer = self

        class _Agent:
            async def run(self, prompt: str, **_kwargs: Any) -> Any:
                outer.prompts.append(prompt)
                outer.in_flight += 1
                outer.max_in_flight = max(outer.max_in_flight, outer.in_flight)
                try:
                    # Yield to the event loop a few times so that any
                    # concurrent caller would be observed overlapping.
                    for _ in range(3):
                        await asyncio.sleep(0)
                    return types.SimpleNamespace(output=outer._responder(prompt))
                finally:
                    outer.in_flight -= 1

        return _Agent()


def _probe(
    probe_id: str,
    *,
    shape: str = "presence",
    quota: int | None = 3,
    tags: tuple[str, ...] = (),
) -> Probe:
    return Probe(
        id=probe_id,
        title=f"Probe {probe_id}",
        instruction=f"Find instances of {probe_id}.",
        shape=shape,  # type: ignore[arg-type]
        quota=quota,
        tags=tags,
    )


_SRC = "The study used n=50 participants. Recruitment ran for six weeks."


def _model(src: str = _SRC) -> DocumentModel:
    return DocumentModel(
        source=src,
        sections=[Section(id="s1", title="S", text=src, start=0, end=len(src))],
    )


def _by_title(mapping: dict[str, Any]) -> Callable[[str], Any]:
    """Responder that keys off the ``CHECK: <title>`` line of the prompt."""

    def responder(prompt: str) -> Any:
        title = prompt.splitlines()[0].removeprefix("CHECK: ").strip()
        value = mapping[title]
        if isinstance(value, Exception):
            raise value
        return value

    return responder


# ── shape conversion ────────────────────────────────────────────────────────


async def test_presence_probe_output_becomes_findings_with_quotes() -> None:
    probe = _probe("unmotivated_choice")
    agents = _RecordingAgents(
        lambda _p: _PresenceFindings(
            findings=[
                _PresenceFinding(
                    issue="no reason is given for the sample size",
                    quote="The study used n=50 participants.",
                    severity="major",
                )
            ]
        )
    )

    with patch(_AGENT_FACTORY, new=agents.factory):
        findings = await run_probe(probe, _model(), agent_model="stub")

    assert len(findings) == 1
    finding = findings[0]
    assert finding.probe_id == "unmotivated_choice"
    assert finding.quote == "The study used n=50 participants."
    assert finding.issue == "no reason is given for the sample size"
    assert finding.severity == "major"
    assert finding.shape == "presence"
    # The absence half of the schema stays empty…
    assert finding.missing_item == ""
    assert finding.search_terms == ()
    # …and anchoring is verify.py's job, not this module's.
    assert finding.span is None


async def test_absence_probe_output_populates_missing_item_and_search_terms() -> None:
    probe = _probe("missing_to_act", shape="absence")
    agents = _RecordingAgents(
        lambda _p: _AbsenceFindings(
            findings=[
                _AbsenceFinding(
                    issue="a reader cannot repeat the recruitment step",
                    missing_item="the inclusion criteria used to recruit",
                    # The blank term is dropped rather than carried: an empty
                    # needle matches everywhere and would refute the finding.
                    search_terms=["inclusion criteria", "  ", "eligibility"],
                )
            ]
        )
    )

    with patch(_AGENT_FACTORY, new=agents.factory):
        findings = await run_probe(probe, _model(), agent_model="stub")

    assert len(findings) == 1
    finding = findings[0]
    assert finding.shape == "absence"
    assert finding.missing_item == "the inclusion criteria used to recruit"
    assert finding.search_terms == ("inclusion criteria", "eligibility")
    assert finding.quote == ""
    assert finding.span is None


# ── tag normalisation ───────────────────────────────────────────────────────


async def test_out_of_set_tag_is_dropped_to_empty_and_logged() -> None:
    """A tag outside the probe's closed set becomes "" — and is reported.

    The drop rate is the cheapest available read on whether a model can run
    the bank at all, so it must not be absorbed silently.
    """
    probe = _probe("reach_beyond_scope", tags=("population", "setting"))
    agents = _RecordingAgents(
        lambda _p: _PresenceFindings(
            findings=[
                _PresenceFinding(
                    issue="in-set tag, sloppy casing",
                    quote="The study used n=50 participants.",
                    tag="  Population ",
                ),
                _PresenceFinding(
                    issue="out-of-set tag",
                    quote="Recruitment ran for six weeks.",
                    tag="galaxy_brain",
                ),
            ]
        )
    )

    with patch(_AGENT_FACTORY, new=agents.factory):
        with _capture_v3_logs() as records:
            findings = await run_probe(probe, _model(), agent_model="stub")

    # Both findings survive — a bad tag is a label problem, not a finding one.
    assert len(findings) == 2
    assert findings[0].tag == "population"  # canonicalised to the set's spelling
    assert findings[1].tag == ""  # dropped, not repaired

    log_text = "\n".join(r.getMessage() for r in records)
    assert "reach_beyond_scope" in log_text
    assert "outside the allowed set" in log_text
    assert "1/2" in log_text


async def test_untagged_probe_drops_any_volunteered_tag_without_warning() -> None:
    """With no closed set declared there is nothing to match against, so a tag
    the model volunteered anyway is dropped rather than carried through.

    It must NOT be reported as an out-of-set drop. The drop rate is meant to
    measure whether a model can follow a probe's closed set; a probe that never
    declared one cannot fail that, and counting it corrupts the measurement.
    Observed live before this was fixed: three of the nine default probes are
    untagged, and every run logged them as total tag failures.
    """
    probe = _probe("term_drift")
    agents = _RecordingAgents(
        lambda _p: _PresenceFindings(
            findings=[
                _PresenceFinding(
                    issue="drifted term",
                    quote="Recruitment ran for six weeks.",
                    tag="whatever",
                )
            ]
        )
    )

    with patch(_AGENT_FACTORY, new=agents.factory):
        with _capture_v3_logs() as records:
            findings = await run_probe(probe, _model(), agent_model="stub")

    assert [f.tag for f in findings] == [""]
    assert [f.issue for f in findings] == ["drifted term"]  # the finding survives
    assert "outside the allowed set" not in "\n".join(r.getMessage() for r in records)


# ── malformed-finding isolation ─────────────────────────────────────────────


async def test_malformed_presence_finding_is_dropped_without_killing_the_probe() -> (
    None
):
    """A finding that fails ``ProbeFinding``'s shape invariant is dropped on
    its own. Its siblings from the same probe still come through — a model
    that botches one of three findings has still produced two usable ones."""
    probe = _probe("unmotivated_choice")
    agents = _RecordingAgents(
        lambda _p: _PresenceFindings(
            findings=[
                # Malformed FIRST, so a per-probe (rather than per-finding)
                # gate would swallow everything after it.
                _PresenceFinding(issue="no quote at all", quote=""),
                _PresenceFinding(
                    issue="a real one",
                    quote="Recruitment ran for six weeks.",
                ),
            ]
        )
    )

    with patch(_AGENT_FACTORY, new=agents.factory):
        with _capture_v3_logs() as records:
            findings = await run_probe(probe, _model(), agent_model="stub")

    assert [f.issue for f in findings] == ["a real one"]
    log_text = "\n".join(r.getMessage() for r in records)
    assert "malformed finding" in log_text
    assert "no quote at all" in log_text


async def test_malformed_absence_finding_is_dropped_without_killing_the_probe() -> None:
    """An absence finding with no search terms is unverifiable, so it fails
    construction — and only it."""
    probe = _probe("missing_to_act", shape="absence")
    agents = _RecordingAgents(
        lambda _p: _AbsenceFindings(
            findings=[
                _AbsenceFinding(
                    issue="unverifiable",
                    missing_item="something",
                    search_terms=[],
                ),
                _AbsenceFinding(
                    issue="a real one",
                    missing_item="the eligibility rule",
                    search_terms=["eligibility"],
                ),
            ]
        )
    )

    with patch(_AGENT_FACTORY, new=agents.factory):
        findings = await run_probe(probe, _model(), agent_model="stub")

    assert [f.issue for f in findings] == ["a real one"]


# ── crash isolation across the bank ─────────────────────────────────────────


async def test_a_crashing_probe_is_recorded_and_the_others_still_run() -> None:
    probes = [_probe("p1"), _probe("p2"), _probe("p3")]

    def _found(probe_id: str) -> _PresenceFindings:
        return _PresenceFindings(
            findings=[
                _PresenceFinding(
                    issue=f"issue from {probe_id}",
                    quote="Recruitment ran for six weeks.",
                )
            ]
        )

    agents = _RecordingAgents(
        _by_title(
            {
                "Probe p1": _found("p1"),
                "Probe p2": RuntimeError("simulated probe crash"),
                "Probe p3": _found("p3"),
            }
        )
    )

    with patch(_AGENT_FACTORY, new=agents.factory):
        with _capture_v3_logs() as records:
            findings, failed = await run_probes(probes, _model(), agent_model="stub")

    assert failed == ["p2"]
    # The crash quarantines one probe and nothing else: p3 ran AFTER the crash.
    assert [f.probe_id for f in findings] == ["p1", "p3"]
    assert len(agents.prompts) == 3

    log_text = "\n".join(r.getMessage() for r in records)
    assert "p2 crashed" in log_text
    assert "simulated probe crash" in log_text
    assert "1/3 probe(s) produced no findings" in log_text


async def test_unexpected_model_behaviour_body_reaches_the_log() -> None:
    probes = [_probe("p1")]
    agents = _RecordingAgents(
        _by_title(
            {
                "Probe p1": UnexpectedModelBehavior(
                    "boom", body="upstream provider error payload"
                )
            }
        )
    )

    with patch(_AGENT_FACTORY, new=agents.factory):
        with _capture_v3_logs() as records:
            findings, failed = await run_probes(probes, _model(), agent_model="stub")

    assert findings == []
    assert failed == ["p1"]
    log_text = "\n".join(r.getMessage() for r in records)
    assert "model behaviour error" in log_text
    assert "upstream provider error payload" in log_text


async def test_usage_limit_exceeded_is_recorded_as_a_failed_probe() -> None:
    probes = [_probe("p1"), _probe("p2")]
    agents = _RecordingAgents(
        _by_title(
            {
                "Probe p1": UsageLimitExceeded("request_limit (60) exceeded"),
                "Probe p2": _PresenceFindings(
                    findings=[
                        _PresenceFinding(
                            issue="survivor",
                            quote="Recruitment ran for six weeks.",
                        )
                    ]
                ),
            }
        )
    )

    with patch(_AGENT_FACTORY, new=agents.factory):
        with _capture_v3_logs() as records:
            findings, failed = await run_probes(probes, _model(), agent_model="stub")

    assert failed == ["p1"]
    assert [f.probe_id for f in findings] == ["p2"]
    assert "usage limit hit" in "\n".join(r.getMessage() for r in records)


# ── DESIGN INVARIANT: probes cannot see each other ──────────────────────────


async def test_probe_independence_no_prompt_contains_another_probes_findings() -> None:
    """No probe's prompt may carry anything an earlier probe found.

    This is NOT the review-mode criterion cascade, which deliberately threads
    accumulated findings forward. Here a probe's output must be a function of
    the document and the probe alone, or two runs of the same bank stop
    producing the same table — and that comparability is the only reason
    checklist mode exists beside review mode. If this test is ever "fixed" by
    passing prior findings along, the mode has lost its product.
    """
    probes = [_probe(f"p{n}") for n in (1, 2, 3, 4)]

    def _found(prompt: str) -> _PresenceFindings:
        probe_id = prompt.splitlines()[0].removeprefix("CHECK: Probe ").strip()
        return _PresenceFindings(
            findings=[
                _PresenceFinding(
                    issue=f"ISSUE_MARKER_{probe_id}",
                    quote=f"QUOTE_MARKER_{probe_id}",
                    tag=f"TAG_MARKER_{probe_id}",
                )
            ]
        )

    agents = _RecordingAgents(_found)
    with patch(_AGENT_FACTORY, new=agents.factory):
        findings, failed = await run_probes(probes, _model(), agent_model="stub")

    assert failed == []
    assert len(findings) == len(probes)  # every probe reported something
    assert len(agents.prompts) == len(probes)

    for position, prompt in enumerate(agents.prompts):
        for other in probes:
            if other.id == probes[position].id:
                continue
            for marker in (
                f"ISSUE_MARKER_{other.id}",
                f"QUOTE_MARKER_{other.id}",
                f"TAG_MARKER_{other.id}",
            ):
                assert marker not in prompt, (
                    f"probe {probes[position].id}'s prompt carries {marker!r} "
                    f"from probe {other.id} — probes must be independent"
                )


async def test_probe_independence_prompts_are_identical_whatever_earlier_probes_found() -> (  # noqa: E501
    None
):
    """The stronger form of the same invariant: run the same bank twice over
    the same document, once with every probe silent and once with every probe
    loudly reporting, and the prompt sequence must be byte-identical."""
    probes = [_probe(f"p{n}") for n in (1, 2, 3)]
    model = _model()

    silent = _RecordingAgents(lambda _p: _PresenceFindings(findings=[]))
    with patch(_AGENT_FACTORY, new=silent.factory):
        await run_probes(probes, model, agent_model="stub")

    def _loud(prompt: str) -> _PresenceFindings:
        probe_id = prompt.splitlines()[0].removeprefix("CHECK: Probe ").strip()
        return _PresenceFindings(
            findings=[
                _PresenceFinding(
                    issue=f"very important issue from {probe_id}",
                    quote="Recruitment ran for six weeks.",
                    severity="major",
                )
                for _ in range(3)
            ]
        )

    noisy = _RecordingAgents(_loud)
    with patch(_AGENT_FACTORY, new=noisy.factory):
        await run_probes(probes, model, agent_model="stub")

    assert silent.prompts == noisy.prompts


# ── HARD PROJECT RULE: one model call at a time ─────────────────────────────


async def test_probe_calls_are_strictly_sequential_never_concurrent() -> None:
    """At most ONE probe call may be in flight at any moment.

    Concurrent local-model inference is forbidden project-wide (Ollama
    serialises same-model calls anyway, so an ``asyncio.gather`` here would
    buy nothing and break the rule). The fake agent yields to the event loop
    while "running", so a gathered implementation would be caught overlapping.
    """
    probes = [_probe(f"p{n}") for n in range(1, 7)]
    agents = _RecordingAgents(lambda _p: _PresenceFindings(findings=[]))

    with patch(_AGENT_FACTORY, new=agents.factory):
        await run_probes(probes, _model(), agent_model="stub")

    assert agents.max_in_flight == 1, (
        f"{agents.max_in_flight} probe calls were in flight at once — probe "
        f"calls must be strictly sequential"
    )
    # …and every probe ran, in bank order.
    ran = [p.splitlines()[0].removeprefix("CHECK: ").strip() for p in agents.prompts]
    assert ran == [p.title for p in probes]


async def test_a_crash_does_not_leak_an_in_flight_call() -> None:
    """Sanity check on the sequencing assertion above: the counter is not
    fooled by a probe that raises mid-call."""
    probes = [_probe("p1"), _probe("p2"), _probe("p3")]
    agents = _RecordingAgents(
        _by_title(
            {
                "Probe p1": _PresenceFindings(findings=[]),
                "Probe p2": RuntimeError("boom"),
                "Probe p3": _PresenceFindings(findings=[]),
            }
        )
    )

    with patch(_AGENT_FACTORY, new=agents.factory):
        await run_probes(probes, _model(), agent_model="stub")

    assert agents.max_in_flight == 1
    assert agents.in_flight == 0
