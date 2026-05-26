"""The v3 review graph — clean pydantic-graph, one job per node.

Linear flow with the gap loop encapsulated in its own node:

    Sectionize(D) → ExtractClaims(A) → BuildModel(D) → ReviewCriteria(A)
      → VerifyFindings(D) → GapLoop(A/D) → Consolidate(A) → Gate(D)
      → Synthesise(A) → CritiqueRevise(A) → Finalize(D) → End[ReviewResult]

Each node wraps a tested phase function; the deterministic/agent split is kept
at the node level. Entry point: `run_review_v3(markdown, *, model=…)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from ..schemas import Edit, ReviewResult
from .consolidate import consolidate
from .criteria import Criterion, criterion_set_for
from .editor import DEFAULT_EDITOR_CRITERIA, run_editor
from .extract import build_claims
from .digest import build_document_model
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


@dataclass
class V3Deps:
    agent_model: str
    cap: int = 2  # gap-loop round cap
    criteria: list[Criterion] = field(
        default_factory=lambda: criterion_set_for("academic")
    )
    # The full markdown is no longer always-passed alongside the digest;
    # instead the criterion-review agents get layer-1 tools
    # (read_section / search_paper) and ask for source content on demand.
    # See docs/plans/2026-05-24-whetstone-v3-layer1-tools-pid.md §3.
    editor_enabled: bool = False
    editor_criteria: list[str] = field(
        default_factory=lambda: list(DEFAULT_EDITOR_CRITERIA)
    )
    check_novelty: bool = False
    novelty_target_cap: int = 8
    novelty_search_depth: int = 2


@dataclass
class V3State:
    source: str
    sections: list[Section] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    document_model: DocumentModel | None = None
    findings: list[Finding] = field(default_factory=list)
    review: StructuredReview | None = None
    edits: list[Edit] = field(default_factory=list)
    novelty_targets: list[NoveltyTarget] = field(default_factory=list)
    novelty_evidence: list[NoveltyEvidence] = field(default_factory=list)


@dataclass
class Sectionize(BaseNode[V3State, V3Deps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "ExtractClaims":
        ctx.state.sections = sectionize(ctx.state.source)
        return ExtractClaims()


@dataclass
class ExtractClaims(BaseNode[V3State, V3Deps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "BuildModel":
        ctx.state.claims = await build_claims(
            ctx.state.sections, ctx.state.source, model=ctx.deps.agent_model
        )
        return BuildModel()


@dataclass
class BuildModel(BaseNode[V3State, V3Deps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "ReviewCriteria":
        ctx.state.document_model = build_document_model(
            ctx.state.source, ctx.state.sections, ctx.state.claims
        )
        return ReviewCriteria()


@dataclass
class ReviewCriteria(BaseNode[V3State, V3Deps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "VerifyFindings":
        model = ctx.state.document_model
        assert model is not None
        ctx.state.findings = await run_criteria(
            ctx.deps.criteria,
            model,
            agent_model=ctx.deps.agent_model,
        )
        return VerifyFindings()


@dataclass
class VerifyFindings(BaseNode[V3State, V3Deps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "GapLoop":
        model = ctx.state.document_model
        assert model is not None
        ctx.state.findings = verify_findings(ctx.state.findings, model)
        return GapLoop()


@dataclass
class GapLoop(BaseNode[V3State, V3Deps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "Consolidate":
        model = ctx.state.document_model
        assert model is not None
        ctx.state.findings = await gap_loop(
            model,
            ctx.state.findings,
            agent_model=ctx.deps.agent_model,
            cap=ctx.deps.cap,
            criterion_names=[c.name for c in ctx.deps.criteria],
        )
        return Consolidate()


@dataclass
class Consolidate(BaseNode[V3State, V3Deps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "FlagNoveltyTargets":
        ctx.state.findings = await consolidate(
            ctx.state.findings, agent_model=ctx.deps.agent_model
        )
        return FlagNoveltyTargets()


@dataclass
class FlagNoveltyTargets(BaseNode[V3State, V3Deps, ReviewResult]):
    """Extract ≤``novelty_target_cap`` novelty claims for verification.
    No-op when ``check_novelty=False`` (default)."""

    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "RunNoveltySearches":
        if not ctx.deps.check_novelty:
            return RunNoveltySearches()
        ctx.state.novelty_targets = await flag_novelty_targets(
            ctx.state.sections,
            ctx.state.source,
            agent_model=ctx.deps.agent_model,
            cap=ctx.deps.novelty_target_cap,
        )
        return RunNoveltySearches()


@dataclass
class RunNoveltySearches(BaseNode[V3State, V3Deps, ReviewResult]):
    """Parallel deep_research per target. Per-target failures are
    captured in NoveltyEvidence.error rather than aborting the loop."""

    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "JudgeNovelty":
        if not ctx.deps.check_novelty or not ctx.state.novelty_targets:
            return JudgeNovelty()
        ctx.state.novelty_evidence = await run_novelty_searches(
            ctx.state.novelty_targets,
            agent_model=ctx.deps.agent_model,
            search_depth=ctx.deps.novelty_search_depth,
        )
        return JudgeNovelty()


@dataclass
class JudgeNovelty(BaseNode[V3State, V3Deps, ReviewResult]):
    """Pure adaptation: evidence → verdict → Finding. Verdicts where
    is_novel=True are silenced (no Finding emitted)."""

    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "Gate":
        if not ctx.deps.check_novelty or not ctx.state.novelty_evidence:
            return Gate()
        verdicts = judge_novelty(ctx.state.novelty_evidence)
        ctx.state.findings.extend(verdicts_to_findings(verdicts))
        return Gate()


@dataclass
class Gate(BaseNode[V3State, V3Deps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "Synthesise":
        ctx.state.findings = gate_and_aggregate(ctx.state.findings)
        return Synthesise()


@dataclass
class Synthesise(BaseNode[V3State, V3Deps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "CritiqueRevise":
        model = ctx.state.document_model
        assert model is not None
        ctx.state.review = await synthesise(
            model, ctx.state.findings, agent_model=ctx.deps.agent_model
        )
        return CritiqueRevise()


@dataclass
class CritiqueRevise(BaseNode[V3State, V3Deps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "EditSections":
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
class EditSections(BaseNode[V3State, V3Deps, ReviewResult]):
    """Optional per-section editor pass. Off by default.

    Gated on ``ctx.deps.editor_enabled``. When enabled, runs the
    editor agent once per section (≤5 concurrent) and populates
    ``ctx.state.edits`` with anchored ``Edit`` objects that the docx
    renderer turns into track-changes.
    """

    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "Finalize":
        if not ctx.deps.editor_enabled:
            return Finalize()
        ctx.state.edits = await run_editor(
            ctx.state.sections,
            criteria=ctx.deps.editor_criteria,
            agent_model=ctx.deps.agent_model,
        )
        return Finalize()


@dataclass
class Finalize(BaseNode[V3State, V3Deps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> End[ReviewResult]:
        model = ctx.state.document_model
        assert model is not None and ctx.state.review is not None
        return End(
            to_review_result(
                model,
                ctx.state.findings,
                ctx.state.review,
                ctx.state.edits,
            )
        )


review_graph_v3 = Graph(
    nodes=[
        Sectionize,
        ExtractClaims,
        BuildModel,
        ReviewCriteria,
        VerifyFindings,
        GapLoop,
        Consolidate,
        FlagNoveltyTargets,
        RunNoveltySearches,
        JudgeNovelty,
        Gate,
        Synthesise,
        CritiqueRevise,
        EditSections,
        Finalize,
    ]
)


async def run_review_v3(
    markdown: str,
    *,
    model: str,
    cap: int = 2,
    document_type: str = "auto",
    confirm_own_draft: bool = False,
    criteria: list[Criterion] | None = None,
    guidelines_text: str | None = None,
    editor: bool = False,
    editor_criteria: list[str] | None = None,
    check_novelty: bool = False,
    novelty_target_cap: int = 8,
    novelty_search_depth: int = 2,
) -> ReviewResult:
    """Run the v3 review over already-harvested markdown. Returns a
    `ReviewResult` the existing renderers consume unchanged.

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

    Refuses to run on text containing confidentiality markers
    (``"Manuscript ID:"``, ``"Reviewer Instructions"``, etc.) — the
    safety gate that protects against accidentally using whetstone on
    peer-review material. Override with ``confirm_own_draft=True`` only
    when the matched marker is a legitimate false positive in the user's
    own draft. The check runs BEFORE any LLM call.
    """
    if criteria is not None and guidelines_text is not None:
        raise ValueError(
            "run_review_v3: pass either `criteria=` (pre-decomposed) or "
            "`guidelines_text=` (free-text to extract from), not both. "
            "These are two different ways to supply the active criterion "
            "set and choosing between them must be explicit."
        )

    if not confirm_own_draft:
        from .._confidentiality import check_confidentiality

        check_confidentiality(markdown)

    if criteria is not None:
        active_criteria = criteria
    elif guidelines_text is not None:
        from .extract_criteria import extract_criteria_from_guidelines

        active_criteria = await extract_criteria_from_guidelines(
            guidelines_text, model=model
        )
    else:
        # Only classify when the result is actually consumed — i.e. we
        # are falling back to the document-type default set.
        if document_type == "auto":
            from .._document_type import classify

            # sectionize is deterministic; calling it here AND inside
            # the graph's first node is cheap (no LLM).
            titles = [s.title for s in sectionize(markdown)]
            document_type = await classify(
                model=model, section_titles=titles, markdown=markdown
            )
        active_criteria = criterion_set_for(document_type)

    deps = V3Deps(
        agent_model=model,
        cap=cap,
        criteria=active_criteria,
        editor_enabled=editor,
        editor_criteria=(
            list(editor_criteria) if editor_criteria else list(DEFAULT_EDITOR_CRITERIA)
        ),
        check_novelty=check_novelty,
        novelty_target_cap=novelty_target_cap,
        novelty_search_depth=novelty_search_depth,
    )
    state = V3State(source=markdown)
    result = await review_graph_v3.run(Sectionize(), state=state, deps=deps)
    return result.output


_PATH_EXTENSIONS: frozenset[str] = frozenset(
    {".md", ".markdown", ".txt", ".html", ".htm", ".pdf", ".docx", ".pptx", ".tex"}
)


def _looks_like_filesystem_path(source: str) -> bool:
    """Heuristic: does this string look like a file path the caller meant
    us to read from disk? Used to distinguish a path that doesn't exist
    (loud-fail) from raw markdown content the caller passed directly.

    True if the string is short, single-line, and starts with a path
    prefix OR ends with a recognised file extension. Conservatively
    biased: a string that clearly looks like a content blob (multi-line,
    long, no slashes) is not treated as a path attempt.
    """
    if not source or "\n" in source:
        return False
    if source.startswith(("/", "./", "../", "~/")):
        return True
    if len(source) > 500:
        return False
    # Heuristic: a single-segment-ish string ending in a recognised
    # extension (e.g. "draft.md") was almost certainly meant as a file.
    from pathlib import Path as _P

    return _P(source).suffix.lower() in _PATH_EXTENSIONS


async def _harvest_or_treat_as_markdown(source: str) -> str:
    """Resolve ``source`` to markdown content. If it points to an existing
    file, harvest it. If it looks like a file path but doesn't exist,
    raise FileNotFoundError loudly — silently treating a missing path as
    raw markdown content (the v3 bug fixed in 2026-05-26) made the LLM
    review the string ``"/tmp/missing.md"`` as if it were the manuscript.
    Only fall back to "treat source as raw markdown" when the string
    clearly is not a path attempt."""
    from pathlib import Path

    from andamentum.harvest import extract

    p = Path(source)
    if p.exists():
        return await extract(p)
    if _looks_like_filesystem_path(source):
        raise FileNotFoundError(
            f"input file not found: {source!r}. The string looks like a "
            "path attempt (starts with /, ./, ../, ~/, or ends in a "
            "recognised extension like .md / .docx / .pdf). To pass raw "
            "markdown content directly, include newlines or omit the "
            "file extension; otherwise check the path."
        )
    return source


async def review_document_v3(
    source: str,
    *,
    model: str,
    cap: int = 2,
    document_type: str = "auto",
    confirm_own_draft: bool = False,
    criteria: list[Criterion] | None = None,
    guidelines_text: str | None = None,
    editor: bool = False,
    editor_criteria: list[str] | None = None,
    check_novelty: bool = False,
    novelty_target_cap: int = 8,
    novelty_search_depth: int = 2,
) -> ReviewResult:
    """v3 entry from a source path/URL: harvest → markdown → review.

    Opt-in alongside the v2 `review_document`; v2 remains the default
    until v3 is validated. Raw markdown can be passed directly to
    `run_review_v3`. See :func:`run_review_v3` for argument semantics
    (criterion-set resolution precedence, confidentiality gate,
    classifier behaviour).
    """
    md = await _harvest_or_treat_as_markdown(source)
    return await run_review_v3(
        md,
        model=model,
        cap=cap,
        document_type=document_type,
        confirm_own_draft=confirm_own_draft,
        criteria=criteria,
        guidelines_text=guidelines_text,
        editor=editor,
        editor_criteria=editor_criteria,
        check_novelty=check_novelty,
        novelty_target_cap=novelty_target_cap,
        novelty_search_depth=novelty_search_depth,
    )
