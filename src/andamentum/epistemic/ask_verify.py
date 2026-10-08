"""Ask a question, or verify a claim, by weighing evidence for and against it.

``ask`` and ``verify`` are the entry points of ``andamentum.epistemic``. Both
run the whole pipeline (preplanning, evidence, claims, scrutiny, integration,
synthesis) and return one typed :class:`AskResult`: the verdict, the claims,
the evidence, the open uncertainties and the report prose.

Files: the run keeps its database at ``database`` when the caller names one
(and writes the HTML report to ``report`` when named). Without ``database``
the run uses a temporary folder that is removed before the function returns.
Nothing is written anywhere else.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

from andamentum.document_store import Store

from .databases import database_path, open_store
from .operations_runner import PipelineResult, ProgressCallback
from .primitives import Claim, ClaimStage, Evidence, Uncertainty
from .result_models import AskResult, RunStats

logger = logging.getLogger(__name__)

#: Called once the pipeline has finished, with its result and the open store.
AfterRun = Callable[[PipelineResult, Store], Awaitable[None]]


async def _get_objective_id_from_db(store: Store) -> Optional[str]:
    """Get the first objective ID from the database using metadata query."""
    results = await store.list(
        where={
            "epistemic_type": "objective",
        },
        limit=1,
    )
    if results:
        return results[0].metadata.get("objective_id")
    return None


async def _gather_primitives_from_db(
    store: Store,
    objective_id: str,
) -> tuple[List[Claim], List[Evidence], List[Uncertainty]]:
    """Gather claims, evidence, and uncertainties from the database.

    Converts document metadata back to proper primitive objects.
    Uses the from_metadata class methods on primitives for proper reconstruction.

    Args:
        store: Store instance
        objective_id: Objective ID to query

    Returns:
        Tuple of (claims, evidence, uncertainties)
    """
    # Gather claims - statement is stored in metadata
    claim_docs = await store.list(
        where={
            "epistemic_type": "claim",
            "objective_id": objective_id,
        },
        limit=None,
    )
    claims: List[Claim] = []
    for doc_meta in claim_docs:
        meta = doc_meta.metadata  # DocumentMetadata.metadata is the nested dict
        if not meta:
            continue
        # Use from_metadata for proper reconstruction
        claims.append(Claim.from_metadata(meta))

    # Sort by stage priority (actionable > robust > provisional > supported > hypothesis)
    stage_priority = {
        ClaimStage.ACTIONABLE: 0,
        ClaimStage.ROBUST: 1,
        ClaimStage.PROVISIONAL: 2,
        ClaimStage.SUPPORTED: 3,
        ClaimStage.HYPOTHESIS: 4,
    }
    claims.sort(key=lambda c: stage_priority.get(c.stage, 5))

    # Gather evidence - need to read full documents for content
    evidence_meta_docs = await store.list(
        where={
            "epistemic_type": "evidence",
            "objective_id": objective_id,
        },
        limit=None,
    )
    evidence: List[Evidence] = []
    for doc_meta in evidence_meta_docs:
        meta = doc_meta.metadata
        if not meta:
            continue
        # Read full document — content is the full entity JSON
        full_doc = await store.get(doc_meta.doc_id)
        content = ""
        if full_doc and full_doc.text:
            import json

            try:
                entity_data = json.loads(full_doc.text)
                content = entity_data.get("extracted_content", "")
            except (json.JSONDecodeError, TypeError):
                content = full_doc.text
        # Use from_metadata for proper reconstruction
        evidence.append(Evidence.from_metadata(meta, content=content))

    # Gather uncertainties - need to read full documents for description
    uncertainty_meta_docs = await store.list(
        where={
            "epistemic_type": "uncertainty",
            "objective_id": objective_id,
        },
        limit=None,
    )
    uncertainties: List[Uncertainty] = []
    for doc_meta in uncertainty_meta_docs:
        meta = doc_meta.metadata
        if not meta:
            continue
        # Read full document — content is the full entity JSON
        full_doc = await store.get(doc_meta.doc_id)
        description = ""
        if full_doc and full_doc.text:
            # Content is JSON — extract the description field
            import json

            try:
                entity_data = json.loads(full_doc.text)
                description = entity_data.get("description", "")
            except (json.JSONDecodeError, TypeError):
                description = full_doc.text
        # Use from_metadata for proper reconstruction
        uncertainties.append(Uncertainty.from_metadata(meta, description=description))

    return claims, evidence, uncertainties


async def _get_synthesis_from_artefact(
    store: Store,
    objective_id: str,
) -> Dict[str, Any]:
    """Get synthesis content from the artefact.

    The artefact is the ONE canonical output of the epistemic system.
    It contains the full research report as markdown.

    Args:
        store: Store instance
        objective_id: Objective ID to query

    Returns:
        Synthesis dict with the full artefact content.
    """
    artefact_docs = await store.list(
        where={
            "epistemic_type": "artefact",
            "objective_id": objective_id,
        },
        limit=None,
    )

    if not artefact_docs:
        return {}

    # Get the first artefact - need full document for content
    doc_meta = artefact_docs[0]
    full_doc = await store.get(doc_meta.doc_id)
    content = ""
    if full_doc and full_doc.text:
        # Content is JSON — extract the markdown content field
        import json

        try:
            entity_data = json.loads(full_doc.text)
            content = entity_data.get("content", "")
        except (json.JSONDecodeError, TypeError):
            content = full_doc.text

    # The "> **Confidence:** HIGH | ..." header this used to scrape has
    # not been emitted by either synthesis operation for some time, so
    # the regex always fell through and reported "medium" regardless of
    # the run. Deleted rather than repaired: confidence is carried by the
    # verdict object (``AskResult.verdict``), not re-parsed out of prose.
    return {
        "summary": content,
    }


async def _run(
    text: str,
    *,
    mode: Literal["verify", "research"],
    model: str,
    embedding_model: str,
    searxng_url: Optional[str],
    browser: bool,
    database: Optional[str | Path],
    report: Optional[str | Path],
    provider: str,
    skip_preplanning: bool,
    verbose: bool,
    progress_callback: Optional[ProgressCallback],
    after_run: Optional[AfterRun],
) -> AskResult:
    from .operations_runner import run_research_question
    from .providers.openalex import OpenAlexQualityScorer

    with tempfile.TemporaryDirectory(prefix="andamentum-epistemic-") as scratch:
        db_file = Path(database) if database is not None else database_path(scratch)
        run_result = await run_research_question(
            question=text,
            database=db_file,
            mode=mode,
            verbose=verbose,
            skip_preplanning=skip_preplanning,
            model=model,
            embedding_model=embedding_model,
            progress_callback=progress_callback,
            provider=provider,
            quality_scorer=OpenAlexQualityScorer(),
            searxng_url=searxng_url,
            browser=browser,
        )
        store = await open_store(db_file)
        try:
            if after_run is not None:
                await after_run(run_result, store)

            objective_id = await _get_objective_id_from_db(store)
            if not objective_id:
                raise ValueError("No objective found in database")
            claims, evidence, uncertainties = await _gather_primitives_from_db(
                store, objective_id
            )
            synthesis = await _get_synthesis_from_artefact(store, objective_id)

            claims_by_stage: dict[str, int] = {}
            for claim in claims:
                stage = claim.stage.value
                claims_by_stage[stage] = claims_by_stage.get(stage, 0) + 1

            if report is not None:
                from .report_generator import ReportGenerator

                generator = ReportGenerator(store, db_file.stem)
                if not await generator.save_html(
                    output_path=Path(report),
                    objective_id=objective_id,
                    model_name=model,
                ):
                    raise OSError(f"could not write the HTML report to {report}")
        finally:
            await store.close()

        return AskResult(
            success=run_result.success,
            question=text,
            database=str(db_file) if database is not None else "",
            claims=claims,
            evidence=evidence,
            uncertainties=uncertainties,
            stats=RunStats(
                workitems_executed_this_run=run_result.iterations,
                total_workitems=run_result.iterations,
                workitems_by_status={
                    "completed": run_result.successful,
                    "failed": run_result.failed,
                },
                claims_by_stage=claims_by_stage,
                evidence_count=len(evidence),
                uncertainty_count=len(uncertainties),
                synthesis=synthesis if synthesis else None,
            ),
            artefact_content=synthesis.get("summary") if synthesis else None,
            verdict=run_result.verdict,
        )


async def ask(
    question: str,
    *,
    model: str,
    embedding_model: str,
    searxng_url: Optional[str] = None,
    browser: bool = False,
    database: Optional[str | Path] = None,
    report: Optional[str | Path] = None,
    provider: Literal["all", "web_search"] = "all",
    skip_preplanning: bool = False,
    verbose: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> AskResult:
    """Answer a research question by weighing the evidence for and against it.

    The question is decomposed into claims when it decomposes; otherwise the
    run proposes claims from the evidence it finds. Every claim is scrutinised,
    tested against rivals and integrated into one verdict.

    Args:
        question: The research question.
        model: LLM model id, for example ``anthropic:claude-haiku-4-5``.
        embedding_model: Ollama embedding model for routing and passage
            extraction.
        searxng_url, browser: Where web search goes: a SearXNG instance, or the
            isolated Chrome in a podman container. One is required.
        database: Path of the ``.db`` file to keep the run in. Without it the
            run uses a temporary folder that is removed afterwards.
        report: Path to write the HTML report to.
        provider: ``"all"`` for the web and the scholarly providers,
            ``"web_search"`` for the web only.
        skip_preplanning: Skip the clarification and conceptual analysis.
        verbose: Log progress.
        progress_callback: Called after each operation with
            ``(operation_type, entity_id, success, message, outputs)``.

    Returns:
        :class:`AskResult` with the verdict, claims, evidence, uncertainties
        and the report prose.

    Raises:
        DegenerateQuestionError: The question cannot be a research question.
        ValueError: A required argument (model, embedding model, search
            address) is missing.
    """
    return await _run(
        question,
        mode="research",
        model=model,
        embedding_model=embedding_model,
        searxng_url=searxng_url,
        browser=browser,
        database=database,
        report=report,
        provider=provider,
        skip_preplanning=skip_preplanning,
        verbose=verbose,
        progress_callback=progress_callback,
        after_run=None,
    )


async def verify(
    claim: str,
    *,
    model: str,
    embedding_model: str,
    searxng_url: Optional[str] = None,
    browser: bool = False,
    database: Optional[str | Path] = None,
    report: Optional[str | Path] = None,
    provider: Literal["all", "web_search"] = "all",
    skip_preplanning: bool = False,
    verbose: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> AskResult:
    """Check one claim by weighing the evidence for and against it.

    Like :func:`ask`, but ``claim`` is a single statement: the run seeds
    exactly one claim from it and skips decomposition.
    """
    return await _run(
        claim,
        mode="verify",
        model=model,
        embedding_model=embedding_model,
        searxng_url=searxng_url,
        browser=browser,
        database=database,
        report=report,
        provider=provider,
        skip_preplanning=skip_preplanning,
        verbose=verbose,
        progress_callback=progress_callback,
        after_run=None,
    )
