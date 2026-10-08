"""The review graph — clean pydantic-graph, one job per node.

Linear flow with the gap loop encapsulated in its own node:

    Sectionize(D) → ExtractClaims(A) → BuildModel(D) → ReviewCriteria(A)
      → VerifyFindings(D) → GapLoop(A/D) → Consolidate(A) → Gate(D)
      → Synthesise(A) → CritiqueRevise(A) → Finalize(D) → End[ReviewResult]

Each node wraps a tested phase function; the deterministic/agent split is kept
at the node level. Entry point: `run_review(markdown, *, model=…)`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic_graph import BaseNode, End, Graph, GraphBuilder, GraphRunContext

from ..schemas import (
    DeclinedFinding,
    Edit,
    Finding as PublicFinding,
    NoveltyClaim,
    ReviewerHeader,
    ReviewResult,
)
from .consolidate import consolidate
from .criteria import Criterion, criterion_set_for
from .editor import DEFAULT_EDITOR_CRITERIA, Declined, run_editor, to_editor_finding
from .extract import build_claims
from .digest import build_document_model, read_citations
from .gaps import gap_loop
from .gate import gate_and_aggregate
from .model import Claim, DocumentModel, Section
from .novelty import (
    NoveltyEvidence,
    NoveltyTarget,
    flag_novelty_targets,
    judge_novelty,
    run_novelty_searches,
    verdicts_to_findings,
)
from .review import Finding, run_criteria, verify_findings
from .sectionize import sectionize
from .synth import StructuredReview, critique_and_revise, synthesise, to_review_result

if TYPE_CHECKING:
    from andamentum.core.decisions import DecisionModel
    from andamentum.deep_research import NoveltyResearch
    from andamentum.references import Hint

    from ..reviewed_document import ReviewedDocument
    from .reviewers import ReviewerProfile

logger = logging.getLogger("andamentum.whetstone.pipeline")


@dataclass(frozen=True)
class ReviewDeps:
    agent_model: str
    cap: int = 3  # gap-loop round cap
    # Aggregate-failure gate: when this fraction (or more) of the active
    # criteria crash and contribute nothing, the result is flagged
    # degraded — a run that skipped most of its cascade is not green.
    soft_failure_threshold: float = 0.5
    criteria: list[Criterion] = field(
        default_factory=lambda: criterion_set_for("academic")
    )
    # The full markdown is no longer always-passed alongside the digest;
    # instead the criterion-review agents get layer-1 tools
    # (read_section / search_paper) and ask for source content on demand.
    editor_enabled: bool = False
    editor_criteria: list[str] = field(
        default_factory=lambda: list(DEFAULT_EDITOR_CRITERIA)
    )
    check_novelty: bool = False
    # Whether to LIST what the draft claims as new (the always-on half of the
    # novelty split; see FlagNoveltyTargets). On for every ordinary review.
    # The CLI turns it off for the second and later rounds of a reviewer
    # list: the draft's claims do not change because another reviewer read
    # it, so re-flagging spends a full-draft model call to produce a list
    # that is discarded.
    flag_novelty: bool = True
    novelty_target_cap: int = 8
    novelty_search_depth: int = 2
    # Where the novelty check searches (deep_research, scholarly sources)
    # and where its research store lives; None takes the defaults. The review
    # model judges what the search finds; an undetermined target gives no
    # finding. Only read when ``check_novelty`` is on.
    novelty_research: "NoveltyResearch | None" = None
    # The active reviewer profile's stance, rendered at the top of every
    # criterion and editor prompt. Empty for the default reviewer, and then
    # rendered as nothing at all.
    brief: str = ""
    # What the renderers print about the reviewer, or None when the default
    # reviewer ran and there is nothing to print. Carried on Deps rather than
    # rebuilt at the end: it is an input to the run, not a result of it.
    reviewer_header: ReviewerHeader | None = None
    # Findings the caller already has — the goal loop's reference problems —
    # that the editor answers beside the review's own, and that the result
    # lists after them. They are not reviewed, verified or consolidated: they
    # come from a deterministic check, not from a criterion.
    extra_findings: list[PublicFinding] = field(default_factory=list)
    # What the Word file knows about its citations and its text does not
    # (superscript runs, reference-manager fields, note marks), for the
    # citation summary. Empty for a caller holding only text.
    citation_hints: "list[Hint]" = field(default_factory=list)


@dataclass
class ReviewState:
    source: str
    sections: list[Section] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    document_model: DocumentModel | None = None
    findings: list[Finding] = field(default_factory=list)
    failed_criteria: list[str] = field(default_factory=list)
    review: StructuredReview | None = None
    edits: list[Edit] = field(default_factory=list)
    # Findings the editor could not answer with a rewrite, and why. A finding
    # with no edit is work left to the author; naming which ones is the
    # difference between a review that answers itself and one that looks
    # like it did.
    declined_findings: list[Declined] = field(default_factory=list)
    novelty_targets: list[NoveltyTarget] = field(default_factory=list)
    novelty_evidence: list[NoveltyEvidence] = field(default_factory=list)


@dataclass
class Sectionize(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Split the source into reviewable units — unless it arrived split.

    A Word document says which paragraphs are headings in their style, and
    :func:`~andamentum.whetstone.reviewed_document.open_source` has already
    used that. Re-deriving the sections here would fall back to looking for
    ``#`` lines in text that has none, and collapse the whole manuscript into
    one unit.
    """

    async def run(self, ctx: GraphRunContext[ReviewState, ReviewDeps]) -> "ExtractClaims":
        if not ctx.state.sections:
            ctx.state.sections = sectionize(ctx.state.source)
        return ExtractClaims()


@dataclass
class ExtractClaims(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[ReviewState, ReviewDeps]) -> "BuildModel":
        ctx.state.claims = await build_claims(
            ctx.state.sections, ctx.state.source, model=ctx.deps.agent_model
        )
        return BuildModel()


@dataclass
class BuildModel(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[ReviewState, ReviewDeps]) -> "ReviewCriteria":
        citations = await read_citations(
            ctx.state.source, ctx.state.sections, hints=ctx.deps.citation_hints
        )
        ctx.state.document_model = build_document_model(
            ctx.state.source, ctx.state.sections, ctx.state.claims, citations=citations
        )
        return ReviewCriteria()


@dataclass
class ReviewCriteria(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[ReviewState, ReviewDeps]) -> "VerifyFindings":
        model = ctx.state.document_model
        assert model is not None
        ctx.state.findings, ctx.state.failed_criteria = await run_criteria(
            ctx.deps.criteria,
            model,
            agent_model=ctx.deps.agent_model,
            brief=ctx.deps.brief,
        )
        return VerifyFindings()


@dataclass
class VerifyFindings(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[ReviewState, ReviewDeps]) -> "GapLoop":
        model = ctx.state.document_model
        assert model is not None
        ctx.state.findings = verify_findings(ctx.state.findings, model)
        return GapLoop()


@dataclass
class GapLoop(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[ReviewState, ReviewDeps]) -> "Consolidate":
        model = ctx.state.document_model
        assert model is not None
        if not ctx.deps.criteria:
            # Nothing was reviewed, so there is no coverage to re-examine. The
            # loop would still run its full round cap against "(no findings
            # yet)" and a summary listing every section as untouched — the
            # most expensive way available to find nothing. Skipping is a flow
            # decision, so it lives in the graph rather than inside gap_loop.
            return Consolidate()
        ctx.state.findings = await gap_loop(
            model,
            ctx.state.findings,
            agent_model=ctx.deps.agent_model,
            cap=ctx.deps.cap,
            criterion_names=[c.name for c in ctx.deps.criteria],
        )
        return Consolidate()


@dataclass
class Consolidate(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[ReviewState, ReviewDeps]) -> "FlagNoveltyTargets":
        ctx.state.findings = await consolidate(
            ctx.state.findings, agent_model=ctx.deps.agent_model
        )
        return FlagNoveltyTargets()


@dataclass
class FlagNoveltyTargets(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """List what the draft claims as new. Runs on EVERY review.

    Unconditional, unlike the two nodes after it, and the asymmetry is the
    whole point of the split. This step costs one call to the model that has
    already read every word of the draft for the review itself, so it
    discloses nothing that was not already disclosed. The next step sends
    those sentences to a search engine — a party that was NOT already reading
    the document — which is a decision to take deliberately on an unpublished
    manuscript, not a default to inherit.

    So the claims are always FOUND and never VERIFIED unless asked. The report
    lists them saying exactly that, which also puts ``--check-novelty`` in
    front of the reader at the moment it means something, holding the specific
    sentences that would be sent.

    A crash returns an empty list (see ``flag_novelty_targets``), so novelty
    being unavailable costs a review nothing.

    The one exception to "every review" is ``deps.flag_novelty=False`` — the
    second and later rounds of a reviewer list, where the list already exists
    from round one and flagging again would be a discarded call.
    """

    async def run(self, ctx: GraphRunContext[ReviewState, ReviewDeps]) -> "RunNoveltySearches":
        if not ctx.deps.flag_novelty:
            return RunNoveltySearches()
        ctx.state.novelty_targets = await flag_novelty_targets(
            ctx.state.sections,
            ctx.state.source,
            agent_model=ctx.deps.agent_model,
            cap=ctx.deps.novelty_target_cap,
        )
        return RunNoveltySearches()


@dataclass
class RunNoveltySearches(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """deep_research per target. Per-target failures are
    captured in NoveltyEvidence.error rather than aborting the loop."""

    async def run(self, ctx: GraphRunContext[ReviewState, ReviewDeps]) -> "JudgeNovelty":
        if not ctx.deps.check_novelty or not ctx.state.novelty_targets:
            return JudgeNovelty()
        ctx.state.novelty_evidence = await run_novelty_searches(
            ctx.state.novelty_targets,
            agent_model=ctx.deps.agent_model,
            search_depth=ctx.deps.novelty_search_depth,
            research=ctx.deps.novelty_research,
        )
        return JudgeNovelty()


@dataclass
class JudgeNovelty(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Pure adaptation: evidence → verdict → Finding. Verdicts where
    is_novel=True are silenced (no Finding emitted)."""

    async def run(self, ctx: GraphRunContext[ReviewState, ReviewDeps]) -> "Gate":
        if not ctx.deps.check_novelty or not ctx.state.novelty_evidence:
            return Gate()
        # A target the check could not assess is not a novel one: name it in
        # the result so the reader can tell "novel" (silence) from "not
        # assessed" (listed here).
        for ev in ctx.state.novelty_evidence:
            if ev.error is not None:
                ctx.state.failed_criteria.append(
                    f"novelty: '{ev.target.short_summary[:60]}' not assessed "
                    f"({ev.error[:200]})"
                )
        verdicts = judge_novelty(ctx.state.novelty_evidence)
        ctx.state.findings.extend(verdicts_to_findings(verdicts))
        return Gate()


@dataclass
class Gate(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[ReviewState, ReviewDeps]) -> "Synthesise":
        ctx.state.findings = gate_and_aggregate(ctx.state.findings)
        return Synthesise()


@dataclass
class Synthesise(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[ReviewState, ReviewDeps]) -> "CritiqueRevise":
        model = ctx.state.document_model
        assert model is not None
        ctx.state.review = await synthesise(
            model, ctx.state.findings, agent_model=ctx.deps.agent_model
        )
        return CritiqueRevise()


@dataclass
class CritiqueRevise(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[ReviewState, ReviewDeps]) -> "EditSections":
        model = ctx.state.document_model
        assert model is not None and ctx.state.review is not None
        ctx.state.review = await critique_and_revise(
            model,
            ctx.state.review,
            ctx.state.findings,
            agent_model=ctx.deps.agent_model,
        )
        return EditSections()


@dataclass
class EditSections(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Optional per-section editor pass. On by default.

    Gated on ``ctx.deps.editor_enabled``. When enabled, runs the
    editor agent once per section (≤5 concurrent) and populates
    ``ctx.state.edits`` with anchored ``Edit`` objects that the docx
    renderer turns into track-changes.
    """

    async def run(self, ctx: GraphRunContext[ReviewState, ReviewDeps]) -> "Finalize":
        if not ctx.deps.editor_enabled:
            return Finalize()
        ctx.state.edits, ctx.state.declined_findings = await run_editor(
            ctx.state.sections,
            criteria=ctx.deps.editor_criteria,
            agent_model=ctx.deps.agent_model,
            # The findings are final by now — Gate has run — so the editor
            # answers the review rather than guessing at prose in parallel
            # with it. This is the whole point of running the editor last.
            findings=[
                *ctx.state.findings,
                *(
                    f
                    for f in map(to_editor_finding, ctx.deps.extra_findings)
                    if f is not None
                ),
            ],
            brief=ctx.deps.brief,
        )
        return Finalize()


@dataclass
class Finalize(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[ReviewState, ReviewDeps]) -> End[ReviewResult]:
        from ._metrics import current as current_counters

        model = ctx.state.document_model
        assert model is not None and ctx.state.review is not None
        counters = current_counters()
        result = to_review_result(
            model,
            ctx.state.findings,
            ctx.state.review,
            ctx.state.edits,
            declined=[
                DeclinedFinding(
                    criterion=d.criterion,
                    issue=d.issue,
                    why=d.why,
                    section_id=d.section_id,
                    finding_id=d.finding_id,
                )
                for d in ctx.state.declined_findings
            ],
            novelty_claims=[
                NoveltyClaim(
                    claim_text=t.claim_text,
                    short_summary=t.short_summary,
                    why_load_bearing=t.why_load_bearing,
                    section_id=t.origin_section_id or "",
                )
                for t in ctx.state.novelty_targets
            ],
            novelty_checked=ctx.deps.check_novelty,
            llm_calls=counters.llm_calls if counters else 0,
            gap_rounds_used=counters.gap_rounds if counters else 0,
            failed_criteria=ctx.state.failed_criteria,
            total_criteria=len(ctx.deps.criteria),
            soft_failure_threshold=ctx.deps.soft_failure_threshold,
        )
        # Assigned rather than passed through `to_review_result`: which
        # reviewer ran is a property of the configuration, not of the findings
        # that function assembles, and every other caller of it would have to
        # learn a parameter it has nothing to say about.
        result.reviewer_header = ctx.deps.reviewer_header
        result.findings = [*result.findings, *ctx.deps.extra_findings]
        return End(result)


def _build_review_graph() -> Graph[ReviewState, ReviewDeps, Sectionize, ReviewResult]:
    """Assemble the graph from the `BaseNode` classes above.

    pydantic-graph v2 dropped `Graph(nodes=[...])`; the builder is now the
    only construction path. The nodes stay `BaseNode` subclasses — `g.node`
    reads each `run` return annotation and derives the edges, so the
    topology is still declared by the return types, not here.
    """
    g = GraphBuilder[ReviewState, ReviewDeps, Sectionize, ReviewResult](
        name="review_graph",
        state_type=ReviewState,
        deps_type=ReviewDeps,
        input_type=Sectionize,
        output_type=ReviewResult,
    )
    g.add(
        g.edge_from(g.start_node).to(Sectionize),
        g.node(Sectionize),
        g.node(ExtractClaims),
        g.node(BuildModel),
        g.node(ReviewCriteria),
        g.node(VerifyFindings),
        g.node(GapLoop),
        g.node(Consolidate),
        g.node(FlagNoveltyTargets),
        g.node(RunNoveltySearches),
        g.node(JudgeNovelty),
        g.node(Gate),
        g.node(Synthesise),
        g.node(CritiqueRevise),
        g.node(EditSections),
        g.node(Finalize),
    )
    return g.build()


# linear pipeline — topology test exempt
review_graph = _build_review_graph()


async def run_review(
    markdown: str,
    *,
    model: str,
    cap: int = 3,
    sections: list[Section] | None = None,
    confidentiality_text: str | None = None,
    document_type: str = "auto",
    confirm_own_draft: bool = False,
    criteria: list[Criterion] | None = None,
    guidelines_text: str | None = None,
    editor: bool = True,
    editor_criteria: list[str] | None = None,
    check_novelty: bool = False,
    flag_novelty: bool = True,
    novelty_target_cap: int = 8,
    novelty_search_depth: int = 2,
    novelty_research: "NoveltyResearch | None" = None,
    reviewer: "ReviewerProfile | None" = None,
    patch_author: str | None = None,
    extra_findings: list[PublicFinding] | None = None,
    citation_hints: "list[Hint] | None" = None,
    decision_model: DecisionModel | None = None,
) -> ReviewResult:
    """Run the review over markdown text you already hold. Returns a
    `ReviewResult` the renderers consume unchanged. :func:`review_document`
    is the same review starting from a file, URL or open document.

    The active criterion set is resolved with the following precedence:

    1. ``criteria=[...]`` (explicit caller-supplied list) — used as-is,
       skips both extractor and classifier
    2. ``guidelines_text="..."`` (free-text reviewer brief / journal
       guidelines) — decomposed into a ``list[Criterion]`` by one LLM
       call (see :func:`extract_criteria_from_guidelines`); skips
       classifier
    3. Default — ``criterion_set_for(document_type)``. When
       ``document_type="auto"`` a one-shot LLM classifier picks one of
       the six routed types (academic, external_communication, essay,
       tutorial, creative, general). Classifier failures default to
       ``"general"``.

    ``criteria`` and ``guidelines_text`` are mutually exclusive —
    passing both raises ``ValueError``. They represent two different
    ways to supply the active set and choosing between them must be
    explicit.

    ``sections`` lets a caller that already knows the document's structure
    supply it. A Word document declares its headings in their paragraph
    styles, which the ``#``-line sectioniser cannot see, so
    :func:`~andamentum.whetstone.reviewed_document.open_source` resolves them
    and passes them in. Omitted, the sections are derived from *markdown*.

    ``extra_findings`` are findings the caller already holds, in the same
    coordinates — the goal loop's reference problems. The editor answers them
    beside the review's own findings, and the result lists them after those.
    They are not reviewed, verified or consolidated; nothing reads them when
    the editor is off except the result.

    ``citation_hints`` are the Word file's superscript runs, reference-manager
    fields and note marks, for the citation line the ``citations``-facet
    criteria read. :func:`review_document` reads them from the file it
    opened; a caller with text only leaves them out.

    ``decision_model`` is an opened ``core.decisions.DecisionModel`` that may
    answer the document-type question under ``document_type="auto"`` instead of
    the LLM classifier (its top choice is the type). ``None`` (the default)
    leaves the run exactly as it was: the LLM classifier decides.

    ``novelty_research`` (a ``deep_research.NoveltyResearch``: the scholarly
    sources and an optional store; ``None`` takes the defaults) configures the
    novelty check, which runs on deep_research and judges with the review
    model; it needs ``check_novelty=True``. A claim whose check cannot conclude (a failing
    source, a thin search, a spent budget) produces no finding: the verdict is
    never guessed.

    ``reviewer`` is the reviewer profile whose stance, criteria and editor
    criteria this run uses (see :mod:`.reviewers`). ``None`` means the default
    reviewer, which selects nothing of its own and therefore reproduces the
    behaviour above exactly. A profile is resolved **inside** this function
    rather than by the caller because resolution needs the document type, and
    under ``document_type="auto"`` that type is only known once the classifier
    below has run — resolving earlier would mean classifying twice. Anything a
    profile supplies is still overridden field by field by the arguments
    above, which is what keeps a named reviewer and a typed-out one the same
    mechanism.

    ``patch_author`` is the tracked-change author the caller will render with,
    recorded in the reviewer header so the report names the same author the
    ``.docx`` will carry. It configures nothing else: this function writes no
    files.

    ``confidentiality_text`` is what the marker scan reads, when that is more
    than the reviewed text. A Word document's running headers and footnotes
    are outside the body projection, and a "CONFIDENTIAL" stamp usually lives
    in a header.

    Refuses to run on text containing confidentiality markers
    (``"Manuscript ID:"``, ``"Reviewer Instructions"``, etc.) — the
    safety gate that protects against accidentally using whetstone on
    peer-review material. Override with ``confirm_own_draft=True`` only
    when the matched marker is a legitimate false positive in the user's
    own draft. The check runs BEFORE any LLM call.
    """
    if criteria is not None and guidelines_text is not None:
        raise ValueError(
            "run_review: pass either `criteria=` (pre-decomposed) or "
            "`guidelines_text=` (free-text to extract from), not both. "
            "These are two different ways to supply the active criterion "
            "set and choosing between them must be explicit."
        )

    if novelty_research is not None and not check_novelty:
        raise ValueError(
            "run_review: `novelty_research` configures the novelty check; "
            "pass `check_novelty=True` with it"
        )

    if not confirm_own_draft:
        from .._confidentiality import check_confidentiality

        check_confidentiality(
            markdown if confidentiality_text is None else confidentiality_text
        )

    # Per-run metric counters live in a contextvar; helpers throughout
    # the pipeline bump them after each agent.run(). Read at Finalize time.
    from ._metrics import increment_llm_calls, start_run

    start_run()

    from .reviewers import DEFAULT_PROFILE, may_misfit, resolve

    profile = reviewer if reviewer is not None else DEFAULT_PROFILE

    # Only classify when the answer is actually consumed. Two things consume
    # it: the criterion set, when neither the caller nor the profile supplied
    # one; and the fit check, but only for a reviewer that some document type
    # could fail — a reviewer that fits all six cannot, so asking would be a
    # model call spent on a question whose answer is already known.
    type_decides_criteria = (
        criteria is None and guidelines_text is None and profile.criteria is None
    )
    if document_type == "auto" and (
        type_decides_criteria or may_misfit(profile=profile)
    ):
        from .._document_type import classify

        # sectionize is deterministic; calling it here AND inside
        # the graph's first node is cheap (no LLM).
        titles = [s.title for s in (sections or sectionize(markdown))]
        document_type = await classify(
            model=model,
            section_titles=titles,
            markdown=markdown,
            decision_model=decision_model,
        )
        # Classifier is one LLM call; count it (classify() doesn't
        # touch the metrics module itself to keep _document_type.py
        # decoupled from pipeline internals). A decision model makes no LLM call.
        if decision_model is None:
            increment_llm_calls(1)

    resolved = resolve(
        profile=profile,
        document_type=document_type,
        criteria=criteria,
        guidelines_text=guidelines_text,
        editor_criteria=editor_criteria,
        patch_author=patch_author,
    )

    if resolved.criteria is None:
        # The one answer resolution defers, because it needs a model: the
        # caller gave free-text guidelines and the criteria come out of them.
        from .extract_criteria import extract_criteria_from_guidelines

        assert guidelines_text is not None
        active_criteria = await extract_criteria_from_guidelines(
            guidelines_text, model=model
        )
    else:
        active_criteria = resolved.criteria

    if resolved.fit_warning:
        # The run proceeds. Document type is usually classified rather than
        # stated, the user asked for this reviewer on their own draft, and the
        # failure mode is findings about something the document was never
        # doing — not damage. The same line goes into the report header, so it
        # survives being scrolled off a terminal.
        logger.warning("[review] %s", resolved.fit_warning)

    deps = ReviewDeps(
        agent_model=model,
        cap=cap,
        criteria=active_criteria,
        editor_enabled=editor,
        editor_criteria=resolved.editor_criteria,
        check_novelty=check_novelty,
        flag_novelty=flag_novelty,
        novelty_target_cap=novelty_target_cap,
        novelty_search_depth=novelty_search_depth,
        novelty_research=novelty_research,
        brief=resolved.brief,
        reviewer_header=resolved.header,
        extra_findings=list(extra_findings or []),
        citation_hints=list(citation_hints or []),
    )
    state = ReviewState(source=markdown, sections=list(sections or []))
    return await review_graph.run(state=state, deps=deps, inputs=Sectionize())


async def review_document(
    source: str | ReviewedDocument,
    *,
    model: str,
    cap: int = 3,
    document_type: str = "auto",
    confirm_own_draft: bool = False,
    criteria: list[Criterion] | None = None,
    guidelines_text: str | None = None,
    editor: bool = True,
    editor_criteria: list[str] | None = None,
    check_novelty: bool = False,
    flag_novelty: bool = True,
    novelty_target_cap: int = 8,
    novelty_search_depth: int = 2,
    novelty_research: "NoveltyResearch | None" = None,
    reviewer: "ReviewerProfile | None" = None,
    patch_author: str | None = None,
    extra_findings: list[PublicFinding] | None = None,
    decision_model: DecisionModel | None = None,
) -> ReviewResult:
    """Review a draft: the main entry point of whetstone.

    Reads the draft, checks it criterion by criterion, verifies every finding
    against the text and returns the findings, suggested edits and questions
    for the author. The three renderers (``render_markdown``, ``render_html``,
    ``render_docx``) turn the result into a report or a tracked-changes copy.
    For markdown text you already hold, :func:`run_review` is the same review
    without the file handling.

    Args:
        source: A file path, a URL, markdown text, or a document already
            opened with ``open_source`` (the command line opens it itself so
            the annotated file is provably the one that was reviewed).
        model: The model that reads and reviews the draft. Required.
        cap: Most findings kept per criterion.
        document_type: ``"auto"`` lets a classifier pick one of six types
            (academic, external_communication, essay, tutorial, creative,
            general); name one to skip that call.
        confirm_own_draft: Proceed although the text carries a peer-review
            marker; only for a false positive in your own draft.
        criteria: Your own criteria, used as given.
        guidelines_text: Free-text reviewer guidelines, turned into criteria by
            one model call. Not together with ``criteria``.
        editor: Also propose edits for the findings.
        editor_criteria: Restrict the editor to these criteria.
        check_novelty: Also check the main claims against the literature.
        flag_novelty: With ``check_novelty``, turn contradicted claims into
            findings.
        novelty_target_cap: Most claims the novelty check looks at.
        novelty_search_depth: How deep the novelty search goes per claim.
        novelty_research: Sources and store for the novelty check.
        reviewer: A reviewer profile (stance, criteria, editor criteria).
        patch_author: The author name the tracked changes will carry; recorded
            in the report header.
        extra_findings: Findings you already hold, answered by the editor
            beside the review's own.
        decision_model: An opened decision model that may answer the
            document-type question instead of the classifier.

    Returns:
        A ``ReviewResult``.

    Raises:
        DocumentReviewError: The draft could not be read or reviewed.
        ConfidentialityMarkerError: The text looks like peer-review material
            (checked before any model call).

    Argument details (criterion-set resolution, the confidentiality check) are
    documented on :func:`run_review`.
    """
    from ..reviewed_document import ReviewedDocument as _Opened
    from ..docx.citation_hints import citation_hints as read_citation_hints
    from ..reviewed_document import open_source

    owned = not isinstance(source, _Opened)
    document = await open_source(source) if owned else source
    assert isinstance(document, _Opened)
    try:
        return await run_review(
            document.text,
            model=model,
            cap=cap,
            sections=document.sections,
            confidentiality_text=document.confidentiality_text,
            document_type=document_type,
            confirm_own_draft=confirm_own_draft,
            criteria=criteria,
            guidelines_text=guidelines_text,
            editor=editor,
            editor_criteria=editor_criteria,
            check_novelty=check_novelty,
            flag_novelty=flag_novelty,
            novelty_target_cap=novelty_target_cap,
            novelty_search_depth=novelty_search_depth,
            novelty_research=novelty_research,
            reviewer=reviewer,
            patch_author=patch_author,
            extra_findings=extra_findings,
            # Read from the file the text is the projection of, so a superscript
            # citation counts in the review's citation line as it does in a
            # reference check.
            citation_hints=read_citation_hints(
                document.docx_path, expected_text=document.text
            ),
        )
    finally:
        if owned:
            document.cleanup()
