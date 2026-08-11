"""End-to-end tests for the checklist graph — every model call stubbed.

Markdown goes in, a ``ReviewResult`` the existing renderers consume comes out.
Beyond the happy path these pin three things the design pays for and would
otherwise be easy to lose:

* the summary is a DETERMINISTIC coverage table, and a crashed probe reads
  ``not run`` there — never ``0``;
* the quote offsets on a finding are SECTION-relative, not document-relative
  (invisible in markdown, corrupts the docx anchor);
* ``consolidate`` and ``gap_loop`` are NOT part of this pipeline. That is a
  deliberate omission, not an oversight — see the graph module docstring.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from andamentum.whetstone.renderers import render_html, render_markdown
from andamentum.whetstone.schemas import ReviewResult
from andamentum.whetstone.v3.checklist.graph import run_checklist_v3
from andamentum.whetstone.v3.checklist.probes import Probe
from andamentum.whetstone.v3.checklist.run_probes import (
    _AbsenceFinding,
    _AbsenceFindings,
    _PresenceFinding,
    _PresenceFindings,
)
from andamentum.whetstone.v3.extract import _ClaimSpans, _Requote
from andamentum.whetstone.v3.sectionize import sectionize

_AGENT_FACTORY = "andamentum.whetstone.v3.checklist.run_probes._build_agent"

SRC = (
    "# Introduction\n\nWe propose a fast method.\n\n"
    "# Results\n\nWe achieve 95% accuracy on the benchmark.\n"
)

_RESULTS_CLAIM = "We achieve 95% accuracy on the benchmark."


# ── the bank used by these tests ────────────────────────────────────────────

PRESENCE_PROBE = Probe(
    id="claim_outruns_evidence",
    title="Claims the evidence supports least well",
    instruction="Find claims the reported evidence supports least well.",
    shape="presence",
    quota=3,
    tags=("no_result", "weaker_than_claimed"),
)

ABSENCE_PROBE = Probe(
    id="missing_to_act",
    title="Information needed to act, and absent",
    instruction="Find things a reader needs in order to repeat this work.",
    shape="absence",
    quota=3,
)


# ── fakes ───────────────────────────────────────────────────────────────────


def _extract_router(defn: Any, _model: Any) -> Any:
    """Stub for the claim-extraction agents (the only other model users)."""
    name = defn.name

    class _Agent:
        def output_validator(self, fn: Any) -> Any:
            return fn

        async def run(self, _prompt: str, **_kwargs: Any) -> Any:
            if name == "v3_extract_claims":
                out: Any = _ClaimSpans(
                    claims=["We propose a fast method.", _RESULTS_CLAIM]
                )
            elif name == "v3_requote":
                out = _Requote(quote="")  # give up → unmatched claims drop
            else:  # pragma: no cover - the checklist graph uses no other agent
                raise AssertionError(f"unexpected agent in checklist run: {name}")
            return types.SimpleNamespace(output=out)

    return _Agent()


def _probe_agents(responder: Callable[[str], Any]) -> Callable[..., Any]:
    """Stub for ``run_probes._build_agent``; *responder* is keyed on the prompt."""

    def factory(*, output_type: type, instructions: str, agent_model: str) -> Any:
        class _Agent:
            async def run(self, prompt: str, **_kwargs: Any) -> Any:
                value = responder(prompt)
                if isinstance(value, Exception):
                    raise value
                return types.SimpleNamespace(output=value)

        return _Agent()

    return factory


def _by_title(mapping: dict[str, Any]) -> Callable[[str], Any]:
    def responder(prompt: str) -> Any:
        title = prompt.splitlines()[0].removeprefix("CHECK: ").strip()
        return mapping[title]

    return responder


async def _run(
    probes: list[Probe], responder: Callable[[str], Any], *, source: str = SRC
) -> ReviewResult:
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "andamentum.whetstone.v3.extract.build_pydantic_ai_agent",
                new=_extract_router,
            )
        )
        stack.enter_context(
            patch("andamentum.whetstone.v3.extract.resolve_model", new=lambda x: None)
        )
        stack.enter_context(patch(_AGENT_FACTORY, new=_probe_agents(responder)))
        return await run_checklist_v3(source, model="stub", probes=probes)


def _presence(issue: str, quote: str, *, severity: str = "major") -> _PresenceFindings:
    return _PresenceFindings(
        findings=[
            _PresenceFinding(
                issue=issue,
                quote=quote,
                severity=severity,  # type: ignore[arg-type]
            )
        ]
    )


_STANDARD_ABSENCE = _AbsenceFindings(
    findings=[
        _AbsenceFinding(
            issue="The reader cannot tell how many runs the number comes from.",
            missing_item="a statement of the sample size",
            search_terms=["sample size", "participants", "how many runs"],
            severity="moderate",
        )
    ]
)


def _standard_responder() -> Callable[[str], Any]:
    return _by_title(
        {
            PRESENCE_PROBE.title: _presence(
                "No baseline is reported next to the accuracy figure.",
                _RESULTS_CLAIM,
            ),
            ABSENCE_PROBE.title: _STANDARD_ABSENCE,
        }
    )


# ── end to end ──────────────────────────────────────────────────────────────


async def test_markdown_in_review_result_out() -> None:
    result = await _run([PRESENCE_PROBE, ABSENCE_PROBE], _standard_responder())

    assert isinstance(result, ReviewResult)
    assert len(result.findings) == 2
    assert result.edits == []
    assert result.failed_criteria == []
    assert not result.degraded
    assert result.degraded_reason == ""
    # The document map reflects the real sections of the source.
    assert [c.section_id for c in result.document_map] == ["s1", "s2"]
    # Cost is counted honestly: claim extraction + one call per probe.
    assert result.metrics.llm_calls > 0
    assert result.metrics.investigated_findings_count == 2
    # No gap loop by design.
    assert result.metrics.reflection_rounds_used == 0


async def test_summary_is_the_coverage_table() -> None:
    result = await _run([PRESENCE_PROBE, ABSENCE_PROBE], _standard_responder())

    summary = result.summary
    assert summary.startswith("## Checklist coverage")
    assert "| Check | Findings | Severity | Ran |" in summary
    # One row per probe, in bank order, with its findings count (against the
    # probe's quota, so a saturated row is visible as saturated) and severity.
    assert f"| {PRESENCE_PROBE.title} | 1 of 3 | 1 major | yes |" in summary
    assert f"| {ABSENCE_PROBE.title} | 1 of 3 | 1 moderate | yes |" in summary
    assert "not run" not in summary  # nothing crashed in this run
    assert "unverifiable" not in summary  # nothing was discarded either


async def test_summary_is_byte_identical_across_two_identical_runs() -> None:
    """Comparability is the product: the same bank over the same document
    must produce a summary you can ``diff`` to nothing."""
    first = await _run([PRESENCE_PROBE, ABSENCE_PROBE], _standard_responder())
    second = await _run([PRESENCE_PROBE, ABSENCE_PROBE], _standard_responder())
    assert first.summary == second.summary


async def test_findings_carry_source_probe_category_and_section_relative_quotes() -> (
    None
):
    """The offsets on a rendered quote are SECTION-relative.

    ``Span`` is document-relative and ``Quote`` is section-relative; mixing
    them up is invisible in markdown (which prints ``Quote.text``) and
    corrupts the docx anchor, which seeks by offset. The claim quoted here
    sits in the SECOND section, so a document-relative offset would resolve
    to the wrong text rather than merely to a different number.
    """
    result = await _run([PRESENCE_PROBE], _standard_responder())

    (finding,) = result.findings
    assert finding.source == "checklist"
    assert finding.category == PRESENCE_PROBE.id  # the probe id, verbatim
    assert finding.title == PRESENCE_PROBE.title
    assert finding.severity == "major"

    (quote,) = finding.quotes
    assert quote.text == _RESULTS_CLAIM
    assert quote.section_id == "s2"
    assert finding.sections_involved == ["s2"]

    section = next(s for s in sectionize(SRC) if s.id == quote.section_id)
    assert section.start > 0, "test would not discriminate on the first section"
    # The offsets resolve, against the SECTION text, back to the quote.
    assert section.text[quote.char_start : quote.char_end] == _RESULTS_CLAIM
    # …and would not resolve against the whole document.
    assert SRC[quote.char_start : quote.char_end] != _RESULTS_CLAIM


async def test_absence_finding_has_no_quote_and_a_self_sufficient_rationale() -> None:
    """An absence finding has nothing to point at, so its rationale must name
    the missing item itself — the reader gets no quote underneath it."""
    result = await _run([ABSENCE_PROBE], _standard_responder())

    (finding,) = result.findings
    assert finding.quotes == []
    assert finding.sections_involved == []
    assert finding.category == ABSENCE_PROBE.id
    rationale = finding.rationale
    assert "The reader cannot tell how many runs the number comes from." in rationale
    assert "a statement of the sample size" in rationale
    # It reads as one statement, not a title echoed twice.
    assert rationale != finding.title


async def test_absence_finding_refuted_by_the_source_is_dropped() -> None:
    """The absence gate is checkable: a search term that IS in the document
    refutes the finding, and it never reaches the reader."""
    refuted = _AbsenceFindings(
        findings=[
            _AbsenceFinding(
                issue="no accuracy is reported",
                missing_item="a reported accuracy figure",
                search_terms=["95% accuracy"],
            )
        ]
    )
    result = await _run([ABSENCE_PROBE], _by_title({ABSENCE_PROBE.title: refuted}))

    assert result.findings == []
    # The probe RAN, so this is not "not run" — but it kept nothing of what it
    # reported, so it is not the informative 0 either, and the row says so.
    assert f"| {ABSENCE_PROBE.title} | 0 of 3 (1 unverifiable) | — | yes |" in (
        result.summary
    )
    assert result.failed_criteria == []
    assert not result.degraded


async def test_hallucinated_quote_is_dropped_before_it_reaches_the_reader() -> None:
    """…and the row it leaves behind does not read as a clean check.

    A paraphrased quote is the commonest small-model failure, so the gate fires
    often. If the probe's row then said a bare ``0`` and the prose said the
    defect "was looked for and is not there", the mode would be committing the
    exact absence-of-checking/absence-of-defects conflation it exists to
    prevent, one gate later.
    """
    result = await _run(
        [PRESENCE_PROBE],
        _by_title(
            {
                PRESENCE_PROBE.title: _presence(
                    "invented", "We cured cancer in a single afternoon."
                )
            }
        ),
    )

    assert result.findings == []
    assert f"| {PRESENCE_PROBE.title} | 0 of 3 (1 unverifiable) | — | yes |" in (
        result.summary
    )
    assert "usually a result" in result.summary
    assert (
        "which is a result: that defect was looked for and is not there"
        not in result.summary
    )
    assert PRESENCE_PROBE.title in result.summary.split("were discarded")[0]


# ── degraded coverage ───────────────────────────────────────────────────────


async def test_a_crashed_probe_degrades_the_result_and_reads_not_run() -> None:
    """A probe that never checked must not be reported as a clean check.

    ``0`` and ``not run`` look almost identical in a table and mean opposite
    things: ``0`` says the defect was looked for and is not there, ``not run``
    says nobody looked.
    """
    result = await _run(
        [PRESENCE_PROBE, ABSENCE_PROBE],
        _by_title(
            {
                PRESENCE_PROBE.title: _presence(
                    "No baseline is reported.", _RESULTS_CLAIM
                ),
                ABSENCE_PROBE.title: RuntimeError("simulated probe crash"),
            }
        ),
    )

    # Degraded, with the crashed check named by its human-readable title.
    assert result.failed_criteria == [ABSENCE_PROBE.title]
    assert result.degraded
    assert "1/2" in result.degraded_reason
    assert ABSENCE_PROBE.title in result.degraded_reason

    # The coverage table says "not run" for it, and never "0".
    assert f"| {ABSENCE_PROBE.title} | not run | not run | no |" in result.summary
    assert "**Coverage is incomplete.**" in result.summary
    assert "ran 1 of 2 checks" in result.summary

    # Signal already acquired is still returned whole.
    assert [f.category for f in result.findings] == [PRESENCE_PROBE.id]


# ── the renderers are unchanged ─────────────────────────────────────────────


async def test_all_three_renderers_accept_a_checklist_result(tmp_path: Path) -> None:
    """ "Renderers unchanged" is a claim the plan makes; this is the check.

    A checklist ``ReviewResult`` differs from a review one in exactly two ways
    (a computed summary, ``source="checklist"`` on every finding), and all
    three renderers must swallow it without special-casing.
    """
    result = await _run([PRESENCE_PROBE, ABSENCE_PROBE], _standard_responder())

    markdown = render_markdown(result, tmp_path / "review.md")
    assert "Checklist coverage" in markdown
    assert PRESENCE_PROBE.title in markdown
    assert (tmp_path / "review.md").exists()

    html = render_html(result, tmp_path / "review.html")
    assert "<html" in html.lower()
    assert (tmp_path / "review.html").exists()

    from andamentum.whetstone.renderers import docx as docx_mod

    captured: dict[str, Any] = {}

    def fake_finalize(**kwargs: Any) -> tuple[str, object]:
        captured.update(kwargs)
        return (str(tmp_path / "out.docx"), object())

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "andamentum.whetstone.docx.finalization.finalize_reviewed_document",
                fake_finalize,
            )
        )
        stack.enter_context(
            patch.object(docx_mod, "stamp_docx_core_properties", lambda *a, **k: None)
        )
        docx_mod.render_docx(
            result,
            source_docx_path=tmp_path / "source.docx",
            output_path=tmp_path / "out.docx",
        )

    # The presence finding became an anchored comment patch; the absence
    # finding has no anchor and so contributes none.
    patches = captured["patches"]
    assert len(patches) == 1
    assert patches[0].text_pattern == _RESULTS_CLAIM
    assert "Checklist coverage" in captured["review_summary"]
    assert captured["issues_count"] == 2


# ── the deliberate omissions ────────────────────────────────────────────────


async def test_consolidate_and_gap_loop_are_never_invoked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checklist mode deliberately does NOT consolidate and does NOT gap-loop.

    Both are stochastic LLM stages. Merging near-duplicates makes per-probe
    counts wobble between runs of an unchanged document (and throws away
    co-detection, which is the mode's confidence signal); the gap loop adds an
    unpredictable number of extra rounds. Predictable, diffable output is the
    product, so this pins the omission rather than trusting a docstring.
    """

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(
            "checklist mode must not call the review pipeline's stochastic stages"
        )

    monkeypatch.setattr("andamentum.whetstone.v3.consolidate.consolidate", _boom)
    monkeypatch.setattr("andamentum.whetstone.v3.gaps.gap_loop", _boom)
    monkeypatch.setattr("andamentum.whetstone.v3.gaps.analyze_gaps", _boom)

    result = await _run([PRESENCE_PROBE, ABSENCE_PROBE], _standard_responder())

    assert len(result.findings) == 2
    assert result.metrics.reflection_rounds_used == 0

    # Belt and braces: a late import would rebind the name before the patch
    # could bite, so also assert no checklist module imports either one.
    import re

    from andamentum.whetstone.v3 import checklist as checklist_pkg

    offender = re.compile(r"^\s*(?:from|import)\s+.*\b(consolidate|gaps)\b", re.M)
    package_dir = Path(checklist_pkg.__file__).parent
    for module_path in sorted(package_dir.glob("*.py")):
        found = offender.search(module_path.read_text(encoding="utf-8"))
        assert found is None, (
            f"{module_path.name} imports a stage checklist mode deliberately "
            f"omits: {found.group(0).strip() if found else ''}"
        )
