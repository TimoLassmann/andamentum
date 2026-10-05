"""The checklist graph — quota-driven defect extraction, end to end.

Seven nodes, no loops, no branches:

    Sectionize(D) → ExtractClaims(A) → BuildModel(D) → RunProbes(A)
      → VerifyProbes(D) → Corroborate(D) → BuildResult(D) → End[ReviewResult]

D = deterministic, A = agent. The part of the cost this mode owns has no
variance in it: ``RunProbes`` makes exactly ``len(probes)`` calls, one per
probe, strictly sequential, with no retries, no loops and no branch that can
fire a second round — so it is known from the bank alone, before the run
starts. On top of that sits the shared substrate, whose cost is a function of
the document rather than of the bank (``extract.build_claims`` spends one call
per section plus a re-quote call for every claim it cannot anchor), and one
classifier call when the caller asks for ``document_type="auto"`` — which is
what the CLI passes unless told otherwise. Review mode cannot promise the
fixed part at all, and that is the reason this mode exists beside it rather
than inside it.

**Three things the review graph does are deliberately absent here. Do not add
them back.**

``consolidate``
    Review mode merges near-duplicate findings with a stochastic LLM call,
    which is right when the deliverable is a readable list of comments. Here
    the fixed slot structure IS the deliverable: a nondeterministic merger
    makes per-probe counts wobble between runs of an unchanged document and
    destroys the comparability the mode is built for. Worse, a passage that
    more than one check landed on is worth reading first — ``Corroborate``
    records that as ``corroborated_by`` and keeps both findings, where merging
    would have collapsed the fact that the passage was reached twice.

``gap_loop``
    Bounded (``V3Deps.cap`` rounds, 2 by default) but demand-driven: whether a
    round fires at all, and how many demands it raises, depends on what the
    first pass happened to return, so two runs of an unchanged document cost
    differently. Predictability is the product.

LLM synthesis
    The summary is :func:`~.summary.coverage_table`, computed from the bank and
    the verified findings rather than written. It is a pure function of those
    two, so the same findings produce byte-identical summaries — which is what
    makes ``diff`` a usable review tool here. What it will not do is round a bad
    run up to a clean one: a probe that crashed reads ``not run``, and a probe
    whose findings were all discarded by verification reads
    ``0 (n unverifiable)``, never a bare ``0``.

What IS reused is the substrate every mode needs and nothing above it:
``sectionize``, ``extract.build_claims``, ``digest.build_document_model``, the
``read_section`` / ``search_paper`` tools (inside :mod:`.run_probes`), and
``schemas.ReviewResult`` so all three renderers consume the output unchanged.

Entry points: :func:`run_checklist_v3` (already-harvested markdown) and
:func:`checklist_document_v3` (a path or URL, harvested first).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic_graph import BaseNode, End, Graph, GraphBuilder, GraphRunContext

from ...schemas import ReviewerHeader, ReviewResult
from ..digest import build_document_model
from ..extract import build_claims
from ..model import Claim, DocumentModel, Section
from ..sectionize import sectionize
from .bank import probe_bank_for
from .build_result import build_checklist_result
from .corroborate import corroborate
from .probes import Probe, ProbeFinding
from .run_probes import run_probes
from .verify import verify_probe_findings

if TYPE_CHECKING:
    from andamentum.core.decisions import DecisionModel

    from ...reviewed_document import ReviewedDocument
    from ..reviewers import ReviewerProfile

logger = logging.getLogger("andamentum.whetstone.v3")


@dataclass(frozen=True)
class ChecklistDeps:
    agent_model: str
    # The bank is data, not machinery (see .bank). Holding it on Deps rather
    # than resolving it inside a node keeps the graph indifferent to which
    # checklist is being run — a caller with a house bank passes it here and
    # every stage downstream behaves identically.
    probes: list[Probe] = field(default_factory=lambda: probe_bank_for("general"))
    # The active reviewer profile's stance, rendered at the top of every probe
    # prompt. Empty for the default reviewer, and then rendered as nothing at
    # all — which is what keeps two default runs comparable row for row with
    # every run this mode has produced so far.
    brief: str = ""
    # What the renderers print about the reviewer, or None when the default
    # reviewer ran and there is nothing to print.
    reviewer_header: ReviewerHeader | None = None


@dataclass
class ChecklistState:
    # ── input
    source: str
    # ── substrate
    sections: list[Section] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    document_model: DocumentModel | None = None
    # ── results
    findings: list[ProbeFinding] = field(default_factory=list)
    # Probes that crashed and checked nothing. Carried separately from
    # `findings` all the way to BuildResult because "found nothing" and
    # "never looked" must not collapse into the same empty row.
    failed_probe_ids: list[str] = field(default_factory=list)
    # probe_id -> how many of its findings verification discarded. Carried for
    # the same reason and to the same place: a probe that reported three
    # paraphrased quotes and lost all three has an empty row, and an empty row
    # that does not say why is indistinguishable from a clean one.
    unverifiable_counts: dict[str, int] = field(default_factory=dict)


def _stage(name: str, detail: str, *args: object) -> None:
    """Mark a completed pipeline stage in the log, with what it produced.

    A checklist run over a real manuscript on a local model spends minutes per
    probe, and without these the log jumps from the document-type line straight
    to tool calls with nothing saying which stage is underway. The count is the
    payload: "0 verified claim(s)" explains a thin run that would otherwise
    look like a bad bank.

    ``VerifyProbes`` deliberately has no call here — :mod:`.verify` emits its
    own kept/dropped summary, and two lines for one stage is worse than one.
    """
    logger.info("[v3.checklist] stage: %s — " + detail, name, *args)


# ── Node 1-3: the shared deterministic/extraction substrate ────────────────


@dataclass
class Sectionize(BaseNode[ChecklistState, ChecklistDeps, ReviewResult]):
    async def run(
        self, ctx: GraphRunContext[ChecklistState, ChecklistDeps]
    ) -> "ExtractClaims":
        # Already split when the source was a Word document: its headings live
        # in paragraph styles, which the ``#``-line sectioniser cannot see.
        if not ctx.state.sections:
            ctx.state.sections = sectionize(ctx.state.source)
        _stage("sectionize", "%d section(s)", len(ctx.state.sections))
        return ExtractClaims()


@dataclass
class ExtractClaims(BaseNode[ChecklistState, ChecklistDeps, ReviewResult]):
    async def run(
        self, ctx: GraphRunContext[ChecklistState, ChecklistDeps]
    ) -> "BuildModel":
        ctx.state.claims = await build_claims(
            ctx.state.sections, ctx.state.source, model=ctx.deps.agent_model
        )
        _stage("extract claims", "%d verified claim(s)", len(ctx.state.claims))
        return BuildModel()


@dataclass
class BuildModel(BaseNode[ChecklistState, ChecklistDeps, ReviewResult]):
    async def run(
        self, ctx: GraphRunContext[ChecklistState, ChecklistDeps]
    ) -> "RunProbes":
        ctx.state.document_model = build_document_model(
            ctx.state.source, ctx.state.sections, ctx.state.claims
        )
        _stage("build document model", "%d probe(s) queued", len(ctx.deps.probes))
        return RunProbes()


# ── Node 4: the only stage that calls a model per probe ────────────────────


@dataclass
class RunProbes(BaseNode[ChecklistState, ChecklistDeps, ReviewResult]):
    """One agent call per probe, strictly sequential (see :func:`.run_probes`).

    An empty bank is a configuration error the caller can only have made
    deliberately, but it produces an empty run that otherwise looks green, so
    it is logged loudly. The graph then continues as usual — on to
    ``VerifyProbes``, which has no findings to verify, and ``Corroborate``,
    which has none to cross-link — and :func:`~.summary.coverage_table`
    refuses to emit a header with no rows under it, saying no checks were
    configured instead of showing a document that passed everything.
    """

    async def run(
        self, ctx: GraphRunContext[ChecklistState, ChecklistDeps]
    ) -> "VerifyProbes":
        model = ctx.state.document_model
        assert model is not None
        if not ctx.deps.probes:
            logger.warning(
                "[v3.checklist] probe bank is empty — nothing will be checked "
                "and the coverage table will have no rows"
            )
            return VerifyProbes()

        ctx.state.findings, ctx.state.failed_probe_ids = await run_probes(
            ctx.deps.probes,
            model,
            agent_model=ctx.deps.agent_model,
            brief=ctx.deps.brief,
        )
        return VerifyProbes()


# ── Node 5-6: deterministic gates ──────────────────────────────────────────


@dataclass
class VerifyProbes(BaseNode[ChecklistState, ChecklistDeps, ReviewResult]):
    """Anchor presence findings, refute absence findings, drop the failures.

    Runs BEFORE ``Corroborate`` deliberately: cross-linking is a claim that two
    checks landed on the same real passage, so it must be computed over
    surviving findings only. Cross-linking first would let two hallucinated
    quotes link to each other and then have one of them dropped, leaving the
    survivor carrying a citation to a finding that no longer exists.

    What was dropped is kept alongside what survived (``unverifiable_counts``),
    because the coverage table has to distinguish a probe that found nothing
    from one whose findings were all discarded here.
    """

    async def run(
        self, ctx: GraphRunContext[ChecklistState, ChecklistDeps]
    ) -> "Corroborate":
        model = ctx.state.document_model
        assert model is not None
        ctx.state.findings, ctx.state.unverifiable_counts = verify_probe_findings(
            ctx.state.findings, model
        )
        return Corroborate()


@dataclass
class Corroborate(BaseNode[ChecklistState, ChecklistDeps, ReviewResult]):
    """Cross-link co-detections. Deterministic, no LLM, nothing merged."""

    async def run(
        self, ctx: GraphRunContext[ChecklistState, ChecklistDeps]
    ) -> "BuildResult":
        ctx.state.findings = corroborate(ctx.state.findings)
        _stage("corroborate", "%d finding(s)", len(ctx.state.findings))
        return BuildResult()


# ── Node 7: assemble the renderer-facing result ────────────────────────────


@dataclass
class BuildResult(BaseNode[ChecklistState, ChecklistDeps, ReviewResult]):
    async def run(
        self, ctx: GraphRunContext[ChecklistState, ChecklistDeps]
    ) -> End[ReviewResult]:
        from .._metrics import current as current_counters

        model = ctx.state.document_model
        assert model is not None
        counters = current_counters()
        result = build_checklist_result(
            model,
            ctx.state.findings,
            ctx.deps.probes,
            failed_probe_ids=ctx.state.failed_probe_ids,
            unverifiable_counts=ctx.state.unverifiable_counts,
            llm_calls=counters.llm_calls if counters else 0,
        )
        # Assigned rather than threaded through build_checklist_result: which
        # reviewer ran is configuration, not a property of the coverage table
        # that function computes.
        result.reviewer_header = ctx.deps.reviewer_header
        return End(result)


# ── Graph + public entry ───────────────────────────────────────────────────


def _build_checklist_graph() -> Graph[
    ChecklistState, ChecklistDeps, Sectionize, ReviewResult
]:
    """Assemble the checklist graph from the `BaseNode` classes above.

    pydantic-graph v2 dropped `Graph(nodes=[...])`; the builder is now the
    only construction path. Edges are still derived from each node's `run`
    return annotation, so the topology is declared by the return types.
    """
    g = GraphBuilder[ChecklistState, ChecklistDeps, Sectionize, ReviewResult](
        name="checklist_graph_v3",
        state_type=ChecklistState,
        deps_type=ChecklistDeps,
        input_type=Sectionize,
        output_type=ReviewResult,
    )
    g.add(
        g.edge_from(g.start_node).to(Sectionize),
        g.node(Sectionize),
        g.node(ExtractClaims),
        g.node(BuildModel),
        g.node(RunProbes),
        g.node(VerifyProbes),
        g.node(Corroborate),
        g.node(BuildResult),
    )
    return g.build()


# linear pipeline — topology test exempt
checklist_graph_v3 = _build_checklist_graph()


async def run_checklist_v3(
    markdown: str,
    *,
    model: str,
    sections: list[Section] | None = None,
    confidentiality_text: str | None = None,
    document_type: str = "general",
    probes: list[Probe] | None = None,
    confirm_own_draft: bool = False,
    reviewer: "ReviewerProfile | None" = None,
    patch_author: str | None = None,
    decision_model: "DecisionModel | None" = None,
) -> ReviewResult:
    """Run the checklist bank over already-harvested markdown.

    Returns a `ReviewResult` the existing renderers consume unchanged. Its
    ``summary`` is the deterministic coverage table rather than synthesised
    prose, and every finding carries ``source="checklist"`` plus the probe id
    in ``category``.

    Parameters
    ----------
    markdown:
        Pre-harvested document body (use `harvest.extract` to get this from a
        PDF / DOCX / URL, or call :func:`checklist_document_v3`).
    model:
        pydantic-ai model id (e.g. "openai:gpt-5.4-nano",
        "ollama:gemma4:31b-nvfp4").
    sections:
        Pre-resolved sections, when the caller knows the document's structure.
        A Word document declares its headings in their paragraph styles, which
        the ``#``-line sectioniser cannot see; see
        :func:`~andamentum.whetstone.reviewed_document.open_source`.
    confidentiality_text:
        What the marker scan reads, when that is more than the reviewed text —
        a Word document's running headers and footnotes are outside the body
        projection, and a "CONFIDENTIAL" stamp usually lives in a header.
    document_type:
        Selects the probe bank via :func:`~.bank.probe_bank_for`. Defaults to
        ``"general"`` — NOT to ``"auto"``, unlike review mode: the default path
        of a mode whose selling point is predictable cost must not spend an
        LLM call on a classifier the caller did not ask for. Pass ``"auto"``
        explicitly to get the same one-shot classifier review mode uses (one
        extra call, counted in the metrics). Ignored when *probes* is given.
    probes:
        An explicit bank, bypassing document-type routing entirely. This is the
        supported way to run a house checklist: the machinery is fixed and the
        probe list is data (see :mod:`.bank`).
    reviewer:
        The reviewer profile whose probe selection and stance this run uses
        (see :mod:`..reviewers`). ``None`` means the default reviewer, which
        selects no probes of its own and therefore leaves the document-type
        bank in place — the run is then byte-identical to the run the same
        arguments produced before profiles existed. A profile holds probe
        **ids**, and the same id names a differently-worded probe on different
        document types, so the profile is resolved here rather than by the
        caller: the type has to be known first.
    patch_author:
        The tracked-change author the caller will render with, recorded in the
        reviewer header so the report names the same author the ``.docx`` will
        carry. It configures nothing else.
    decision_model:
        An opened ``core.decisions.DecisionModel`` that may answer the
        document-type question under ``document_type="auto"`` before the LLM
        classifier is asked, instead of the LLM classifier. ``None`` leaves the run
        as it was.
    confirm_own_draft:
        Bypasses the confidentiality-marker tripwire. Checklist mode gets the
        same gate review mode has and for the same reason — the tool is for
        drafts you wrote — but NOT panel mode's mandatory authorship
        affirmation, which exists because panel output is shaped like a journal
        peer-review report and checklist output is not. The check runs BEFORE
        any LLM call.
    """
    if not confirm_own_draft:
        from ..._confidentiality import check_confidentiality

        check_confidentiality(
            markdown if confidentiality_text is None else confidentiality_text
        )

    # Per-run metric counters live in a contextvar; run_probes bumps them
    # after each agent.run(). Read at BuildResult time.
    from .._metrics import increment_llm_calls, start_run

    start_run()

    from ..reviewers import DEFAULT_PROFILE, may_misfit, resolve

    profile = reviewer if reviewer is not None else DEFAULT_PROFILE

    # Only classify when the answer is consumed. Two things consume it: the
    # bank — whether it comes from the document type or from the profile's
    # probe ids, both of which need the type — and the fit check, but only for
    # a reviewer some document type could fail.
    if document_type == "auto" and (probes is None or may_misfit(profile=profile)):
        from ..._document_type import classify

        # sectionize is deterministic; calling it here AND inside the
        # graph's first node is cheap (no LLM).
        titles = [s.title for s in (sections or sectionize(markdown))]
        document_type = await classify(
            model=model,
            section_titles=titles,
            markdown=markdown,
            decision_model=decision_model,
        )
        # classify() is one LLM call and does not touch the metrics module
        # itself (that would couple _document_type.py to v3 internals), so
        # it is counted here. A decision model makes no LLM call.
        if decision_model is None:
            increment_llm_calls(1)

    resolved = resolve(
        profile=profile,
        document_type=document_type,
        probes=probes,
        patch_author=patch_author,
    )
    if resolved.fit_warning:
        # The run proceeds; the same line goes into the report header. See
        # ``reviewers.fits`` for why a miss is not a refusal.
        logger.warning("[v3.checklist] %s", resolved.fit_warning)

    deps = ChecklistDeps(
        agent_model=model,
        probes=resolved.probes,
        brief=resolved.brief,
        reviewer_header=resolved.header,
    )
    state = ChecklistState(source=markdown, sections=list(sections or []))
    return await checklist_graph_v3.run(state=state, deps=deps, inputs=Sectionize())


async def checklist_document_v3(
    source: "str | ReviewedDocument",
    *,
    model: str,
    document_type: str = "general",
    probes: list[Probe] | None = None,
    confirm_own_draft: bool = False,
    reviewer: "ReviewerProfile | None" = None,
    patch_author: str | None = None,
    decision_model: "DecisionModel | None" = None,
) -> ReviewResult:
    """Checklist entry from a path / URL / markdown blob, or an open document.

    Thin wrapper over :func:`run_checklist_v3`; see it for argument semantics.
    Source resolution is
    :func:`~andamentum.whetstone.reviewed_document.open_source`, shared with
    review mode, so a ``.docx`` is sectionised by its own heading styles here
    too and a missing path loud-fails identically.
    """
    from ...reviewed_document import ReviewedDocument as _Opened
    from ...reviewed_document import open_source

    owned = not isinstance(source, _Opened)
    document = await open_source(source) if owned else source
    assert isinstance(document, _Opened)
    try:
        return await run_checklist_v3(
            document.text,
            model=model,
            sections=document.sections,
            confidentiality_text=document.confidentiality_text,
            document_type=document_type,
            probes=probes,
            confirm_own_draft=confirm_own_draft,
            reviewer=reviewer,
            patch_author=patch_author,
            decision_model=decision_model,
        )
    finally:
        if owned:
            document.cleanup()
