
==============================================================================
## andamentum.core  (5 public names)
==============================================================================

  class AgentDefinition(name: 'str', prompt: 'str', output_model: 'type[BaseModel] | None', retries: 'int' = 3, output_retries: 'int' = 5, has_tools: 'bool' = False) -> None
    └ Configuration for a pydantic-ai agent.

  class AgentRunner(*, model: 'Any')
    └ Executes agents with caching and PromptedOutput fallback.
      .clear_cache(self) -> 'None'
        └ Clear the agent cache.
      .run(self, defn: 'AgentDefinition', *, validators: 'list[Callable[..., Any]] | None' = None, **kwargs: 'Any') -> 'Any'
        └ Run an agent with PromptedOutput fallback.

  def resolve_model(model: 'str') -> 'Any'
    └ Resolve a model string to a pydantic-ai model object.

  def resolve_model_from_args(model_arg: 'str | None') -> 'str'
    └ Resolve model from CLI arg or ANDAMENTUM_MAIN_LLM_MODEL env var.

  async def run_agent_with_fallback(model: 'Any', *, instructions: 'str', output_type: 'type[BaseModel]', user_message: 'str', retries: 'int' = 3, output_retries: 'int' = 5, validators: 'list[Callable[..., Any]] | None' = None) -> 'Any'
    └ One-shot agent execution with PromptedOutput fallback.

==============================================================================
## andamentum.deep_research  (30 public names)
==============================================================================

  AGENT_REGISTRY: dict
    └ dict() -> new empty dictionary

  class AgentDefinition(name: 'str', prompt: 'str', output_model: 'type[BaseModel] | None', retries: 'int' = 3, output_retries: 'int' = 5, has_tools: 'bool' = False) -> None
    └ Configuration for a pydantic-ai agent.

  class CircuitBreaker(name: str, failure_threshold: int = 5, recovery_timeout: float = 60.0, half_open_max_calls: int = 1) -> None
    └ Circuit breaker with three states.
      .allow_request(self) -> bool
        └ Check if request should be allowed.
      .record_failure(self) -> None
        └ Record failed request.
      .record_success(self) -> None
        └ Record successful request.
      .reset(self) -> None
        └ Reset circuit breaker to initial state (for testing).

  class CircuitOpenError(breaker_name: str)
    └ Raised when circuit is open and request rejected.

  class EvidenceItem(/, **data: 'Any') -> 'None'
    └ Single piece of evidence extracted from research.

  class EvidenceReport(/, **data: 'Any') -> 'None'
    └ Final research output from Lead Agent.

  class ExtractionError(/, *args, **kwargs)
    └ Raised when content extraction fails or produces no usable content.

  class FetchPlan(/, **data: 'Any') -> 'None'
    └ Simplified output from PageFetcher - just the link IDs to fetch.

  class FetchResults(/, **data: 'Any') -> 'None'
    └ Output from PageFetcher subagent.

  class FetchedPage(/, **data: 'Any') -> 'None'
    └ Content from an opened page.

  class GapAnalysis(/, **data: 'Any') -> 'None'
    └ Output from GapAnalyzer subagent.

  class NoveltyAssessment(/, **data: 'Any') -> 'None'
    └ Structured output from novelty assessment agent.

  class NoveltyReport(claim: str, is_novel: bool, confidence: float, assessment: str, similar_work: List[andamentum.deep_research.novelty.models.SimilarWork] = <factory>, sources: List[str] = <factory>, search_queries_used: List[str] = <factory>) -> None
    └ Result of a novelty check.

  class PageSummary(/, **data: 'Any') -> 'None'
    └ Condensed summary of a fetched page's key points.

  class Relevance(*args, **kwds)
    └ How closely related prior work is to the claim.

  class ResearchErrors(/, **data: 'Any') -> 'None'
    └ Error counts from a research session.

  class ResearchResult(/, **data: 'Any') -> 'None'
    └ Complete result from a research session.

  class ResearchState(query: str, max_iterations: int = 3, max_searches_per_iteration: int = 3, search_history: list[andamentum.deep_research.models.SearchQuery] = <factory>, all_results: dict[str, list[andamentum.deep_research.models.SearchResult]] = <factory>, url_map: dict[int, str] = <factory>, fetched_pages: list[andamentum.deep_research.models.FetchedPage] = <factory>, page_summaries: list[andamentum.deep_research.models.PageSummary] = <factory>, evidence_items: list[andamentum.deep_research.models.EvidenceItem] = <factory>, identified_gaps: list[str] = <factory>, is_complete: bool = False, iteration_count: int = 0, current_phase: Literal['plan', 'search', 'fetch', 'summarize', 'analyze', 'refine', 'synthesize'] = 'plan', total_searches: int = 0, total_pages_fetched: int = 0, searched_urls: set[str] = <factory>, fetched_urls: set[str] = <factory>, search_errors: list[dict[str, str]] = <factory>, fetch_errors: list[dict[str, str]] = <factory>) -> None
    └ Shared state for research workflow.

  class SearchPlan(/, **data: 'Any') -> 'None'
    └ Simplified output from SearchPlanner - just the queries to execute.

  class SearchQuery(/, **data: 'Any') -> 'None'
    └ A search query with metadata.

  class SearchResult(/, **data: 'Any') -> 'None'
    └ Single search result.

  class SearxngManager(container: 'str' = 'mcp-searxng', image: 'str' = 'docker.io/searxng/searxng:latest', host_port: 'int' = 4070, internal_port: 'int' = 8080, bind_host: 'str | None' = None) -> 'None'
    └ Manages a Podman-based SearXNG container.
      .ensure_running(self) -> 'None'
        └ Start the container if it's not already running.
      .health_check(self, timeout: 'float' = 5.0) -> 'dict[str, Any]'
        └ Synchronous health check using urllib (no async dependencies).
      .is_running(self) -> 'bool'
        └ Check if the SearXNG container is running.
      .logs(self, tail: 'int' = 200) -> 'str'
        └ Get container logs.
      .start(self) -> 'None'
        └ Start the SearXNG container.
      .status(self) -> 'str'
        └ Get human-readable status string.
      .stop(self) -> 'None'
        └ Stop and remove the SearXNG container.
      .write_minimal_settings(self) -> 'None'
        └ Write a minimal SearXNG settings.yml to the state directory.

  class SimilarWork(title: str, url: str, relevance: andamentum.deep_research.novelty.models.Relevance, summary: str) -> None
    └ A piece of prior work related to the claim.

  async def check_novelty(claim: str, research_fn: collections.abc.Callable[..., collections.abc.Awaitable[dict[str, typing.Any]]], assess_fn: collections.abc.Callable[[str, str, list[str], list[str]], collections.abc.Awaitable[andamentum.deep_research.novelty.checker.NoveltyAssessment]], search_depth: int = 2, verbose: bool = False) -> andamentum.deep_research.novelty.models.NoveltyReport
    └ Check if a claim is novel by searching for prior work.

  async def check_searxng_health(url: 'str | None' = None, timeout: 'float' = 5.0) -> 'dict[str, Any]'
    └ Async health check for a SearXNG instance.

  def extract_content(data: bytes, content_type: str, url: str) -> str
    └ Route raw bytes to the right extractor based on content-type.

  def extract_html(html: str, url: str) -> str
    └ Extract article content from web page HTML and return clean markdown.

  def extract_pdf(data: bytes, source_name: str = 'document.pdf') -> str
    └ Extract text from PDF bytes and return clean markdown.

  def get_searxng_breaker() -> andamentum.deep_research.circuit_breaker.CircuitBreaker
    └ Get or create the SearXNG circuit breaker.

  def verify_sources(cited_sources: list[str], searched_urls: set[str], fetched_urls: set[str]) -> andamentum.deep_research.verification.VerificationResult
    └ Verify that cited sources were actually accessed during research.

==============================================================================
## andamentum.document_store  (38 public names)
==============================================================================

  class Chunk(text: 'str', section_path: 'str' = '', chunk_index: 'int' = 0, start_char: 'int' = 0, end_char: 'int' = 0) -> None
    └ A chunk of document content with positional metadata.

  class ChunkLLMFields(/, **data: 'Any') -> 'None'
    └ LLM-extracted fields for a chunk.

  class ChunkMetadataFields(/, **data: 'Any') -> 'None'
    └ Structured metadata for a chunk.

  class Document(/, **data: 'Any') -> 'None'
    └ Complete document with content and metadata.

  class DocumentLLMFields(/, **data: 'Any') -> 'None'
    └ LLM-extracted fields for a document.

  class DocumentMetadata(/, **data: 'Any') -> 'None'
    └ Metadata for a document in the store.

  class DocumentMetadataFields(/, **data: 'Any') -> 'None'
    └ Structured metadata for a document.

  class DocumentStore(database_name: 'str', db_dir: 'Optional[str | Path]' = None, embedding_model: 'Optional[str]' = None)
    └ Unified document management with single-tier indexing.
      .add(self, file_path: 'str', content: 'Optional[str]' = None, title: 'Optional[str]' = None, document_type: 'Optional[DocumentType]' = None, metadata: 'Optional[dict]' = None) -> 'str'
        └ Add a document (backward-compatible wrapper around register_document).
      .cluster_summary(self) -> "'ClusterSummary'"
        └ Get high-level summary of clustering state.
      .delete(self, doc_id: 'str') -> 'bool'
        └ Soft-delete a document (sets deleted_at, excluded from search/listing).
      .delete_chunks(self, doc_id: 'str') -> 'int'
        └ Delete all chunks and chunk embeddings for a document.
      .exists_by_hash(self, file_hash: 'str') -> 'bool'
        └ Check if a non-deleted document with the given content hash exists.
      .find_by_metadata(self, filters: 'Mapping[str, Any]', limit: 'int' = 100) -> 'list[DocumentMetadata]'
        └ Find documents by metadata field values.
      .for_database(database_name: 'str', db_dir: 'Optional[str | Path]' = None) -> "'DocumentStore'"
        └ Create DocumentStore instance for a named database.
      .get_cluster(self, cluster_id: 'int', include_docs: 'bool' = True) -> "Optional['ClusterDetail']"
        └ Get detailed information about a specific cluster.
      .get_stats(self) -> 'dict'
        └ Get statistics about this database.
      .hard_delete(self, doc_id: 'str') -> 'bool'
        └ Permanently delete a document and all its chunks. Cannot be undone.
      .initialize(self) -> 'None'
        └ Initialize database tables and metadata.
      .list_clusters(self, sort_by: 'str' = 'last_active_at', include_docs: 'bool' = False) -> 'list'
        └ List all clusters with summary information.
      .list_documents(self, document_type: 'Optional[DocumentType]' = None) -> 'list[DocumentMetadata]'
        └ List all documents, optionally filtered by type.
      .read(self, doc_id: 'str') -> 'Optional[Document]'
        └ Read a document by ID.
      .recluster(self, config: "Optional['DHPConfig']" = None) -> "'ReclusterResult'"
        └ Run full offline DHP re-clustering on all documents.
      .reembed_all(self, embedding_model: 'Optional[str]' = None, batch_size: 'int' = 50) -> 'ReembedResult'
        └ Backfill document-level embeddings for all documents missing them.
      .register_document(self, title: 'str', content: 'str', metadata: 'Optional[dict]' = None) -> 'str'
        └ Register a new document. Content is stored and FTS5-indexed automatically.
      .restore(self, doc_id: 'str') -> 'bool'
        └ Restore a soft-deleted document.
      .search(self, query: 'str', limit: 'int' = 10, query_embedding: 'Optional[list[float]]' = None) -> "list['UnifiedSearchResult']"
        └ Search across all documents with 4-signal RRF fusion.
      .store_chunk(self, doc_id: 'str', text: 'str', embedding: 'list[float]', metadata: 'Optional[dict]' = None, chunk_index: 'int' = 0, start_char: 'int' = 0, end_char: 'int' = 0) -> 'int'
        └ Store a chunk with its embedding for an existing document.
      .store_doc_embedding(self, doc_id: 'str', embedding: 'list[float]') -> 'None'
        └ Store a document-level embedding.
      .update(self, doc_id: 'str', new_content: 'Optional[str]' = None, metadata: 'Optional[dict]' = None, merge_metadata: 'bool' = True) -> 'UpdateResult'
        └ Update document content and/or metadata.

  class DocumentType(*args, **kwds)
    └ Document type classification.

  class DuplicateGroup(doc_ids: 'list[str]' = <factory>, titles: 'list[str]' = <factory>, similarity: 'float' = 0.0) -> None
    └ A group of documents that are near-duplicates based on embedding similarity.

  MetadataFilterValue: UnionType
    └ Represent a PEP 604 union type

  class MultiDatabaseSearchResult(doc_id: 'str', score: 'float', tier: 'str', snippet: 'str' = '', metadata: 'SearchResultMetadata' = <factory>, database_name: 'str' = '') -> None
    └ Search result from multi-database search.

  class ReembedResult(/, **data: 'Any') -> 'None'
    └ Result of a batch re-embedding operation.

  class RepairReport(documents_scanned: 'int' = 0, documents_incomplete: 'int' = 0, documents_repaired: 'int' = 0, documents_failed: 'int' = 0, failures: 'list[str]' = <factory>) -> None
    └ Report from a repair() run.

  class SearchResult(doc_id: 'str', title: 'str', snippet: 'str', score: 'float', metadata: 'dict' = <factory>, match_type: 'str' = '', warning: 'str' = '') -> None
    └ A search result from the knowledge base.

  class SearchResultMetadata(match_type: 'str' = '', entity_matches: 'list[str]' = <factory>, tag_matches: 'list[str]' = <factory>) -> None
    └ Metadata about how a search result was matched.

  class UnifiedSearchResult(doc_id: 'str', score: 'float', tier: 'str', snippet: 'str' = '', metadata: 'SearchResultMetadata' = <factory>) -> None
    └ A single search result from unified search.

  class UpdateResult(/, **data: 'Any') -> 'None'
    └ Result of a document update operation.

  def chunk_markdown(text: 'str', max_tokens: 'int' = 500, overlap_tokens: 'int' = 50, chars_per_token: 'int' = 4) -> 'list[Chunk]'
    └ Split markdown into chunks.

  def database_exists(database_name: str) -> bool
    └ Check if a database exists.

  async def delete(database: 'str', doc_id: 'str') -> 'bool'
    └ Delete a document and all its chunks from the knowledge base.

  def delete_database(database_name: str, ephemeral: bool = False) -> bool
    └ Delete a named database.

  async def extract_chunk_metadata(chunk_text: 'str', model: 'str | None' = None) -> 'ChunkMetadataFields'
    └ Extract LLM fields for a chunk using PydanticAI structured output.

  async def extract_document_metadata(content: 'str', model: 'str | None' = None, max_content_chars: 'int' = 3000) -> 'DocumentMetadataFields'
    └ Extract LLM fields for a document using PydanticAI structured output.

  async def find_by_metadata(database: 'str', filters: 'Mapping[str, MetadataFilterValue]', limit: 'int' = 100) -> 'list[SearchResult]'
    └ Find documents by exact metadata field values.

  async def find_duplicates(database: 'str', threshold: 'float' = 0.92) -> 'list[DuplicateGroup]'
    └ Find groups of near-duplicate documents using embedding similarity.

  def get_databases_dir() -> pathlib.Path
    └ Get the permanent databases directory path.

  def get_db_path(database_name: str, ephemeral: bool = False) -> pathlib.Path
    └ Get database path for a named database.

  async def ingest(database: 'str', content: 'str', title: 'str | None' = None, source: 'str' = 'manual', metadata: 'dict | None' = None, *, model: 'str', embedding_model: 'str') -> 'str'
    └ Add content to the knowledge base. Returns doc_id.

  def list_databases() -> list[str]
    └ List all available databases.

  async def list_deleted(database: 'str', limit: 'int' = 50) -> 'list[SearchResult]'
    └ List soft-deleted documents (for trash view / undo UI).

  async def purge(database: 'str', older_than_days: 'int' = 30) -> 'int'
    └ Permanently delete soft-deleted documents older than N days. Returns count purged.

  async def repair(database: 'str', *, model: 'str', embedding_model: 'str') -> 'RepairReport'
    └ Scan database for incomplete ingestions and re-run phase 2.

  async def restore(database: 'str', doc_id: 'str') -> 'bool'
    └ Restore a soft-deleted document. Returns True if found and restored.

  async def search(database: 'str', query: 'str', limit: 'int' = 10, *, model: 'str', embedding_model: 'str') -> 'list[SearchResult]'
    └ Search the knowledge base with natural language.

  async def search_multi_database(query: 'str', database_names: 'list[str]', limit: 'int' = 10) -> 'list[MultiDatabaseSearchResult]'
    └ Search across multiple named databases with RRF fusion.

  async def search_unified(db_path: 'str', query: 'str', limit: 'int' = 10, query_embedding: 'Optional[list[float]]' = None, doc_uuids: 'Optional[set[str]]' = None, embedding_model: 'Optional[str]' = None) -> 'list[UnifiedSearchResult]'
    └ Unified search across all documents with RRF fusion.

  async def update_metadata(database: 'str', doc_id: 'str', metadata: 'dict', merge: 'bool' = True) -> 'bool'
    └ Update metadata on a document. Changes are recorded in _history.

==============================================================================
## andamentum.epistemic  (44 public names)
==============================================================================

  AGENT_REGISTRY: dict
    └ dict() -> new empty dictionary

  class AgentDefinition(name: 'str', prompt: 'str', output_model: 'type[BaseModel] | None', retries: 'int' = 3, output_retries: 'int' = 5, has_tools: 'bool' = False) -> None
    └ Configuration for a pydantic-ai agent.

  class AgentRunner(*args, **kwargs)
    └ Protocol for running agents.
      .run(self, agent_name: str, **kwargs: Any) -> Any
        └ Run an agent and return its output.

  class Artefact(/, **data: 'Any') -> 'None'
    └ Human-facing output compiled from epistemic state.

  BLOCKING_TYPES: set
    └ set() -> new empty set object

  class BaseOperation(repo: 'EpistemicRepository', agent_runner: Optional[andamentum.epistemic.operations.base.AgentRunner] = None, validator: Optional[andamentum.epistemic.operations.base.OperationValidator] = None, evidence_gatherer: Optional[andamentum.epistemic.operations.base.EvidenceGatherer] = None, quality_scorer: Optional[andamentum.epistemic.operations.base.QualityScorer] = None, embedding_model: Optional[str] = None)
    └ Base class for all epistemic operations.
      .execute(self, work: andamentum.epistemic.operations.base.OperationInput) -> andamentum.epistemic.operations.base.OperationResult
        └ Execute the operation.
      .log_event(self, event_type: str, target_id: str, details: dict[str, typing.Any]) -> None
        └ Log an epistemic event.
      .run_agent(self, agent_name: str, **kwargs: Any) -> Any
        └ Run agent with adapter normalization.

  class CheckResult(name: 'str', status: "Literal['pass', 'fail', 'skip']", message: 'str', elapsed_ms: 'float') -> None
    └ Result of a single health check.

  class Claim(/, **data: 'Any') -> 'None'
    └ Scoped proposition with stage tracking and degeneracy detection.
      .from_metadata(meta: dict[str, typing.Any], statement_override: Optional[str] = None) -> 'Claim'
        └ Reconstruct Claim from metadata dict (legacy API).
      .model_post_init(self, _Claim__context: Any) -> None
        └ Update denormalized fields after initialization.
      .record_demotion(self, target_stage: andamentum.epistemic.entities.claim.ClaimStage, justification: str) -> None
        └ Record a stage demotion with full state cleanup.
      .record_modification(self) -> None
        └ Record a modification for degeneracy detection.
      .record_promotion(self, from_stage: andamentum.epistemic.entities.claim.ClaimStage, to_stage: andamentum.epistemic.entities.claim.ClaimStage, justification: str) -> None
        └ Record a stage promotion in history.

  class ClaimStage(*args, **kwds)
    └ Claim lifecycle stages with increasing confidence requirements.

  class Decision(/, **data: 'Any') -> 'None'
    └ Record of a commitment that changes system behavior.
      .reverse(self, reason: str) -> None
        └ Reverse this decision with a reason.

  class DegeneracyCodes(/, *args, **kwargs)
    └ Degeneracy detection codes from Lakatos methodology.

  ENTITY_CLASSES: dict
    └ dict() -> new empty dictionary

  class EntityNotFoundError(entity_type: str, entity_id: str)
    └ Raised when an entity is not found.

  class EpistemicEntity(/, **data: 'Any') -> 'None'
    └ Base class for all epistemic entities.
      .from_document(content: str, metadata: dict[str, typing.Any]) -> Self
        └ Reconstruct entity from stored document.
      .model_post_init(self, _EpistemicEntity__context: Any) -> None
        └ Hook for subclasses to compute denormalized fields after initialization.
      .to_document(self) -> tuple[str, dict[str, typing.Any]]
        └ Convert entity to (content, metadata) for DocumentStore.
      .to_metadata(self) -> dict[str, typing.Any]
        └ Convert to metadata dict for storage (legacy API compatibility).
      .touch(self) -> None
        └ Update the updated_at timestamp.

  class EpistemicRepository(store: andamentum.document_store.api.DocumentStore)
    └ Single interface for all entity operations.
      .count(self, entity_type: str, **filters: Any) -> int
        └ Count entities matching filters.
      .delete(self, entity_type: str, entity_id: str) -> bool
        └ Delete an entity.
      .exists(self, entity_type: str, entity_id: str) -> bool
        └ Check if entity exists without loading.
      .for_database(name: str, db_dir: pathlib.Path | None = None) -> 'EpistemicRepository'
        └ Create a repository backed by a persistent named database.
      .get(self, entity_type: str, entity_id: str) -> andamentum.epistemic.entities.objective.Objective | andamentum.epistemic.entities.evidence.Evidence | andamentum.epistemic.entities.claim.Claim | andamentum.epistemic.entities.uncertainty.Uncertainty | andamentum.epistemic.entities.decision.Decision | andamentum.epistemic.entities.snapshot.Snapshot | andamentum.epistemic.entities.artefact.Artefact | andamentum.epistemic.entities.base.EpistemicEntity
        └ Load a single entity by ID.
      .get_adversarial_evidence_for_claim(self, claim_id: str) -> Optional[ForwardRef('AdversarialEvidence')]
        └ Get adversarial evidence for a claim.
      .get_artefact(self, artefact_id: str) -> andamentum.epistemic.entities.artefact.Artefact
        └ Get artefact by ID.
      .get_artefacts_for_objective(self, objective_id: str) -> list[andamentum.epistemic.entities.artefact.Artefact]
        └ Get all artefacts for an objective.
      .get_blocking_uncertainties(self, objective_id: str) -> list[andamentum.epistemic.entities.uncertainty.Uncertainty]
        └ Get unresolved blocking uncertainties for an objective.
      .get_blocking_uncertainties_for_claim(self, claim_id: str) -> list[andamentum.epistemic.entities.uncertainty.Uncertainty]
        └ Get unresolved blocking uncertainties affecting a claim.
      .get_claim(self, claim_id: str) -> andamentum.epistemic.entities.claim.Claim
        └ Get claim by ID.
      .get_claims_for_objective(self, objective_id: str, **filters: Any) -> list[andamentum.epistemic.entities.claim.Claim]
        └ Get all claims for an objective with optional filters.
      .get_convergent_evidence_for_claim(self, claim_id: str) -> Optional[ForwardRef('ConvergentEvidence')]
        └ Get convergent evidence for a claim.
      .get_decision(self, decision_id: str) -> andamentum.epistemic.entities.decision.Decision
        └ Get decision by ID.
      .get_decisions_for_objective(self, objective_id: str, include_reversed: bool = False) -> list[andamentum.epistemic.entities.decision.Decision]
        └ Get decisions for an objective, optionally including reversed ones.
      .get_evidence(self, evidence_id: str) -> andamentum.epistemic.entities.evidence.Evidence
        └ Get evidence by ID.
      .get_evidence_for_objective(self, objective_id: str, **filters: Any) -> list[andamentum.epistemic.entities.evidence.Evidence]
        └ Get all evidence for an objective with optional filters.
      .get_objective(self, objective_id: str) -> andamentum.epistemic.entities.objective.Objective
        └ Get objective by ID.
      .get_snapshot(self, snapshot_id: str) -> andamentum.epistemic.entities.snapshot.Snapshot
        └ Get snapshot by ID.
      .get_uncertainties_for_objective(self, objective_id: str, **filters: Any) -> list[andamentum.epistemic.entities.uncertainty.Uncertainty]
        └ Get all uncertainties for an objective with optional filters.
      .get_uncertainty(self, uncertainty_id: str) -> andamentum.epistemic.entities.uncertainty.Uncertainty
        └ Get uncertainty by ID.
      .query(self, entity_type: str, **filters: Any) -> list[andamentum.epistemic.entities.objective.Objective] | list[andamentum.epistemic.entities.evidence.Evidence] | list[andamentum.epistemic.entities.claim.Claim] | list[andamentum.epistemic.entities.uncertainty.Uncertainty] | list[andamentum.epistemic.entities.decision.Decision] | list[andamentum.epistemic.entities.snapshot.Snapshot] | list[andamentum.epistemic.entities.artefact.Artefact] | list[andamentum.epistemic.entities.base.EpistemicEntity]
        └ Find entities matching filters.
      .save(self, entity: andamentum.epistemic.entities.base.EpistemicEntity) -> str
        └ Save entity (create or update).
      .save_adversarial_evidence(self, adv: 'AdversarialEvidence') -> str
        └ Persist AdversarialEvidence so report generation can retrieve it.

  class Evidence(/, **data: 'Any') -> 'None'
    └ Interpreted observation from a source.
      .from_metadata(meta: dict[str, typing.Any], content: str = '', limitations: Optional[list[str]] = None) -> 'Evidence'
        └ Reconstruct Evidence from metadata dict (legacy API).

  class EvidenceGatherer(*args, **kwargs)
    └ Gathers raw evidence from external sources.
      .gather(self, source_type: str, query: str) -> list[andamentum.epistemic.operations.base.GatheredEvidence]
        └ Gather evidence from external sources.

  class GateResult(passed: bool, blocking_reasons: list[str] = <factory>, warnings: list[str] = <factory>, reason: Optional[str] = None) -> None
    └ Result of gate validation.

  class GatheredEvidence(content: str, source_ref: str, source_type: str, evidence_kind: str = 'unknown', identifiers: dict[str, str] = <factory>, structured_data: dict[str, typing.Any] = <factory>, limitations: list[str] = <factory>, quality_score: Optional[float] = None, quality_metadata: Optional[dict[str, Any]] = None) -> None
    └ Raw evidence returned by an EvidenceGatherer.

  class HealthCheckable(*args, **kwargs)
    └ Base class for protocol classes.
      .check_health(self) -> 'CheckResult'

  class Objective(/, **data: 'Any') -> 'None'
    └ Top-level research objective with phase tracking.

  class OperationInput(entity_id: str, entity_type: str, operation: str, metadata: dict[str, typing.Any] = <factory>) -> None
    └ Input for an epistemic operation.

  class OperationResult(success: bool, entity_id: str, message: str = '', created_entities: list[str] = <factory>, validation_errors: list[str] = <factory>) -> None
    └ Result from executing an operation.

  class PipelineResult(objective_id: str, iterations: int, successful: int, failed: int, status: str, errors: Optional[list[str]] = None, posterior: Optional[ForwardRef('PosteriorReport')] = None, quarantined: Optional[list[andamentum.epistemic.graph.quarantine.QuarantineRecord]] = None, retrieval_failed: bool = False)
    └ Result from an epistemic pipeline run.

  class PosteriorReport(/, **data: 'Any') -> 'None'
    └ Posterior probability P(Y) for a yes/no-style research objective.

  class PreflightResult(checks: 'list[CheckResult]' = <factory>) -> None
    └ Aggregate result of all preflight checks.

  class QualityScorer(*args, **kwargs)
    └ Scores evidence source quality via OpenAlex (DOI/PMID lookup).
      .score(self, source_ref: str, source_type: str) -> andamentum.epistemic.operations.base.QualityScore
        └ Score a source's quality.

  class QuarantineRecord(entity_id: 'str', entity_type: 'str', operation: 'str', exception_type: 'str', message: 'str') -> None
    └ Records that an entity was quarantined because an operation raised.

  STAGE_GATES: dict
    └ dict() -> new empty dictionary

  STAGE_HIERARCHY: dict
    └ dict() -> new empty dictionary

  class Snapshot(/, **data: 'Any') -> 'None'
    └ Immutable epistemic state for artefact generation.

  class StageGate(target_stage: andamentum.epistemic.entities.claim.ClaimStage, min_evidence: int, min_quality_sum: float, requires_scrutiny: bool, requires_adversarial: bool, requires_convergence: bool, requires_deductive: bool, requires_computational: bool, blocks_on_uncertainties: bool, min_supporting_sources: int = 0, adversarial_balance_threshold: float = 0.0, custom_check: Optional[Callable[[ForwardRef('Claim'), ForwardRef('EpistemicRepository')], Awaitable[bool]]] = None) -> None
    └ Requirements for advancing to a claim stage.
      .describe(self) -> str
        └ Return human-readable description of gate requirements.

  class Uncertainty(/, **data: 'Any') -> 'None'
    └ First-class uncertainty that blocks or qualifies claims.
      .from_metadata(meta: dict[str, typing.Any], description: str = '') -> 'Uncertainty'
        └ Reconstruct Uncertainty from metadata dict (legacy API).
      .model_post_init(self, _Uncertainty__context: Any) -> None
        └ Update denormalized fields after initialization.
      .resolve(self, resolution: str) -> None
        └ Mark this uncertainty as resolved.

  class UncertaintyScope(*args, **kwds)
    └ Scope of an uncertainty's impact.

  class UncertaintyType(*args, **kwds)
    └ Types of epistemic uncertainty.

  def can_demote(current_stage: andamentum.epistemic.entities.claim.ClaimStage) -> bool
    └ Check if demotion is possible from current stage.

  def check_degeneracy(claim: 'Claim') -> list[str]
    └ Check claim for degenerative research patterns.

  def compute_confidence_score(stage: 'ClaimStage', avg_quality: float, adversarial_balance: Optional[float] = None) -> float
    └ Compute confidence score from stage, evidence quality, and adversarial balance.

  async def compute_posterior(repo: andamentum.epistemic.repository.EpistemicRepository, objective_id: str, *, retrieval_failed: bool = False) -> andamentum.epistemic.confidence.PosteriorReport | None
    └ Compute posterior probability P(Y) by synthesizing counting and integration.

  def get_next_stage(current_stage: andamentum.epistemic.entities.claim.ClaimStage) -> Optional[andamentum.epistemic.entities.claim.ClaimStage]
    └ Get the next stage in the promotion sequence.

  def get_previous_stage(current_stage: andamentum.epistemic.entities.claim.ClaimStage) -> Optional[andamentum.epistemic.entities.claim.ClaimStage]
    └ Get the previous stage for demotion.

  async def preflight(*, model: 'str', providers: 'ProviderRegistry | None' = None, verbose: 'bool' = False) -> 'PreflightResult'
    └ Run preflight checks on LLM, web search, and evidence providers.

  async def quality_weighted_evidence_sum(claim: 'Claim', repo: 'EpistemicRepository') -> float
    └ Sum of quality_score for all scored evidence supporting this claim.

  async def validate_promotion(claim: 'Claim', target_stage: andamentum.epistemic.entities.claim.ClaimStage, repo: 'EpistemicRepository', question_type: Optional[str] = None) -> andamentum.epistemic.gates.GateResult
    └ Check if claim can be promoted to target stage.

==============================================================================
## andamentum.figures  (15 public names)
==============================================================================

  class DataTable(columns: 'dict[str, list[Any]]') -> 'None'
    └ Normalized columnar data representation.
      .from_csv(csv_string: 'str') -> 'DataTable'
        └ Create from CSV string.
      .from_dict(d: 'dict[str, list[Any]]') -> 'DataTable'
        └ Create from columnar dict: {"col": [values]}.
      .from_records(records: 'list[dict[str, Any]]') -> 'DataTable'
        └ Create from row records: [{"col": val, ...}, ...].
      .is_categorical(self, col: 'str') -> 'bool'
        └ Check if a column contains categorical (string) data.
      .is_numeric(self, col: 'str') -> 'bool'
        └ Check if a column contains numeric data.
      .normalize(data: 'dict[str, list[Any]] | list[dict[str, Any]] | str') -> 'DataTable'
        └ Auto-detect format and normalize to DataTable.
      .unique_count(self, col: 'str') -> 'int'
        └ Number of unique values in a column.
      .values_per_category(self, cat_col: 'str', val_col: 'str') -> 'dict[str, list[Any]]'
        └ Group values by category: {category: [values]}.

  class FigureMode(*args, **kwds)
    └ Figure output mode.

  class FigureResult(/, **data: 'Any') -> 'None'
    └ Result returned by figure().

  class PlotKind(*args, **kwds)
    └ Supported plot types.

  def despine(ax: 'Axes', *, left: 'bool' = False, bottom: 'bool' = False) -> 'None'
    └ Remove spines from axes.

  def figure(data: 'dict[str, list[Any]] | list[dict[str, Any]] | str', *, kind: 'str' = 'auto', x: 'str | None' = None, y: 'str | list[str] | None' = None, group: 'str | None' = None, error: 'str | None' = None, error_type: 'str | None' = None, title: 'str | None' = None, x_label: 'str | None' = None, y_label: 'str | None' = None, style: 'str' = 'npg', journal: 'str' = 'default', mode: 'str' = 'publication', width: 'str | float' = 'single', height: 'float | None' = None, dpi: 'int' = 300, log_scale: 'str | None' = None, sort: 'str | None' = None, output: 'str | Path' = 'figure.pdf') -> 'FigureResult'
    └ Generate a publication-quality figure from data.

  def get_palette(name: 'str', n: 'int | None' = None) -> 'list[str]'
    └ Get a color palette by name.

  def get_preset(name: 'str') -> 'JournalPreset'
    └ Get a journal format preset by name.

  def list_palettes() -> 'dict[str, int]'
    └ List available palettes and their color counts.

  def list_presets() -> 'dict[str, str]'
    └ List available presets with descriptions.

  def panel_label(ax: 'Axes', label: 'str', x: 'float' = -0.12, y: 'float' = 1.08) -> 'None'
    └ Add a bold panel label (A, B, C...) to axes.

  def resolve_width(width: 'str | float', preset: 'JournalPreset') -> 'float'
    └ Resolve width specification to inches.

  def savefig(fig: 'Figure', path: 'str | Path', *, dpi: 'int | None' = None, pad: 'float' = 0.05) -> 'str'
    └ Save figure with publication-quality settings.

  def setup_style(journal: 'str' = 'default') -> 'JournalPreset'
    └ Configure matplotlib rcParams for publication-quality figures.

  def shared_legend(fig: 'Figure', labels: 'list[str]', colors: 'list[str]', *, ncol: 'int' = 6, marker: 'str' = 's', y_offset: 'float' = -0.02) -> 'None'
    └ Add a shared legend below all panels.

==============================================================================
## andamentum.scribe  (10 public names)
==============================================================================

  class Block(/, **data: 'Any') -> 'None'
    └ A single block in a scribe document.

  class Document(*, id: 'str', title: 'str', database: 'str', template: 'Optional[str]' = None)
    └ A scribe document — a structured, block-based draft.
      .add_reference(self, *, cite_key: 'str', bibtex: 'Optional[str]' = None, metadata: 'Optional[dict[str, Any]]' = None) -> 'str'
        └ Attach a bibliographic reference to this document.
      .append(self, block_spec: 'dict', *, parent_id: 'Optional[str]' = None) -> 'str'
        └ Append a block to the end of this document. Returns block id.
      .citations(self) -> 'list[str]'
        └ Return all citation keys used in paragraph blocks (deduped).
      .create(*, title: 'str', database: 'str', template: 'Optional[str]' = None, scaffold: 'Optional[str]' = None) -> "'Document'"
        └ Create a new document and return its handle.
      .insert_into_section(self, section_name: 'str', block_spec: 'dict[str, Any]', *, position: 'str' = 'end') -> 'str'
        └ Insert a block at the end of (or beginning of) a named section.
      .list_sections(self) -> 'list[dict]'
        └ Return one entry per top-level (level 1) heading.
      .open(doc_id: 'str', *, database: 'str') -> "'Document'"
        └ Open an existing document by id.
      .query(self, *, type: 'Optional[str]' = None) -> 'list[Block]'
        └ Return blocks for this document, ordered by position.
      .references(self) -> 'list[Reference]'
        └ Return all references attached to this document.
      .render(self, output_path: 'str', *, format: 'str' = 'docx') -> 'None'
        └ Render this document to a file. v1 supports format='docx' only.
      .replace(self, block_id: 'str', new_content: 'str', *, expected_revision: 'int', reason: 'Optional[str]' = None) -> 'None'
        └ Replace a block's content under optimistic locking.
      .replace_section(self, name: 'str', content: 'str', *, reason: 'Optional[str]' = None) -> 'None'
        └ Replace the body blocks of a named section with content parsed from markdown.
      .section(self, name: 'str') -> 'list[Block]'
        └ Return the heading + all blocks belonging to the named section.
      .validate(self) -> 'list[ValidationIssue]'
        └ Run structural validators. See validate.validate_document.

  def Figure(*, path: 'str', caption: 'str', label: 'str', width_in: 'Optional[float]' = None) -> 'dict[str, Any]'

  def Heading(content: 'str', *, level: 'int') -> 'dict[str, Any]'

  def Paragraph(content: 'str') -> 'dict[str, Any]'

  class Reference(/, **data: 'Any') -> 'None'
    └ A bibliographic reference attached to a document.

  class Revision(/, **data: 'Any') -> 'None'
    └ An audit-log entry for a single block edit.

  class StaleRevisionError(block_id: 'str', expected: 'int', actual: 'int')
    └ Raised when Document.replace sees a revision mismatch.

  def Table(*, rows: 'list[list[str]]', header_row: 'bool' = True, caption: 'str' = '', label: 'str' = '') -> 'dict[str, Any]'

  class ValidationIssue(/, **data: 'Any') -> 'None'
    └ One structural issue surfaced by Document.validate().

==============================================================================
## andamentum.typeset  (13 public names)
==============================================================================

  class Report(style: 'str' = 'article', **kwargs: 'Any') -> 'None'
    └ Fluent document builder.
      .aside(self, **kw: 'Any') -> "'Report'"
        └ Append an ``aside`` atom.
      .callout(self, content: 'str', **kw: 'Any') -> "'Report'"
        └ Append a ``callout`` atom.
      .card(self, content: 'str', **kw: 'Any') -> "'Report'"
        └ Append a ``card`` atom.
      .heading(self, content: 'str', **kw: 'Any') -> "'Report'"
        └ Append a ``heading`` atom.
      .items(self, entries: 'list[dict[str, str]]', **kw: 'Any') -> "'Report'"
        └ Append an ``items`` atom.
      .prose(self, content: 'str', **kw: 'Any') -> "'Report'"
        └ Append a ``prose`` atom.
      .reference(self, content: 'str', **kw: 'Any') -> "'Report'"
        └ Append a ``reference`` atom.
      .render(self) -> 'str'
        └ Render to an HTML string.
      .save(self, path: 'str | Path') -> 'Path'
        └ Render and write to an HTML file.
      .save_pdf(self, path: 'str | Path', **kw: 'Any') -> 'Path'
        └ Render and write to a PDF file. Requires WeasyPrint.

  STYLES: dict
    └ dict() -> new empty dictionary

  def aside(**kwargs: 'Any') -> 'dict[str, Any]'
    └ Build an ``aside`` atom dict.

  def callout(content: 'str', **kwargs: 'Any') -> 'dict[str, Any]'
    └ Build a ``callout`` atom dict.

  def card(content: 'str', **kwargs: 'Any') -> 'dict[str, Any]'
    └ Build a ``card`` atom dict.

  def get_style(name: 'str') -> 'str'
    └ Return the CSS string for *name*.

  def heading(content: 'str', **kwargs: 'Any') -> 'dict[str, Any]'
    └ Build a ``heading`` atom dict.

  def items(entries: 'list[dict[str, str]]', **kwargs: 'Any') -> 'dict[str, Any]'
    └ Build an ``items`` atom dict.

  def prose(content: 'str', **kwargs: 'Any') -> 'dict[str, Any]'
    └ Build a ``prose`` atom dict.

  def reference(content: 'str', **kwargs: 'Any') -> 'dict[str, Any]'
    └ Build a ``reference`` atom dict.

  def render(document: 'list[dict[str, object]] | str', *, style: 'str' = 'article', custom_css: 'str | None' = None, title: 'str | None' = None, footer: 'str' = '') -> 'str'
    └ Render *document* to a complete HTML string.

  def render_pdf(document: 'list[dict[str, object]] | str', output: 'str | Path', *, style: 'str' = 'article', custom_css: 'str | None' = None, title: 'str | None' = None, footer: 'str' = '') -> 'Path'
    └ Render *document* to a PDF file using WeasyPrint.

  def render_to_file(document: 'list[dict[str, object]] | str', output: 'str | Path', **kwargs: 'object') -> 'Path'
    └ Render *document* and write the HTML to *output*.

==============================================================================
## andamentum.whetstone  (14 public names)
==============================================================================

  AGENT_REGISTRY: dict
    └ dict() -> new empty dictionary

  class AgentDefinition(name: 'str', prompt: 'str', output_model: 'type[BaseModel] | None', retries: 'int' = 3, output_retries: 'int' = 5, has_tools: 'bool' = False) -> None
    └ Configuration for a pydantic-ai agent.

  class ChecklistItem(/, **data: 'Any') -> 'None'
    └ One pre-submission check, evaluated against a draft.

  class DocumentIssue(/, **data: 'Any') -> 'None'
    └ Represents a specific issue or recommendation found in a document.

  class DocumentPatch(/, **data: 'Any') -> 'None'
    └ Represents a single edit or comment to be applied to a document.
      .validate_patch_fields(self)
        └ Validate that required fields are present based on patch type.

  class PatchApplicationResult(/, **data: 'Any') -> 'None'
    └ Result of applying patches to a document.

  class ReviewResult(/, **data: 'Any') -> 'None'
    └ Complete result from sharpen_document().

  def apply_patches(content: 'str', patches: 'Sequence[DocumentPatch]') -> 'str'
    └ Apply text_edit patches to content via string replacement.

  def convert_fields_to_schema(fields: 'list[AnalysisField]') -> 'dict[str, Any]'
    └ Convert a list of AnalysisField objects into the dict format expected by :func:`create_output_model`.

  def create_output_model(agent_name: 'str', model_spec: 'dict[str, Any]') -> 'type[BaseModel]'
    └ Create a Pydantic model from a field specification dict.

  def render_diff(*, patches: 'Sequence[DocumentPatch]', issues: 'Sequence[DocumentIssue]', original_content: 'str', synthesis_text: 'str | None' = None, checklist: 'list | None' = None) -> 'str'
    └ Render a lightweight markdown diff view.

  def render_docx(*, input_path: pathlib.Path, output_path: pathlib.Path, patches: list[andamentum.whetstone.models.DocumentPatch], review_summary: str = '', critical_issues: Optional[list] = None, expert_reviews: Optional[list] = None, generated_experts: Optional[list] = None, novelty_findings: str = '', author: str = 'Whetstone Review', checklist_items: Optional[list] = None) -> andamentum.whetstone.models.PatchApplicationResult
    └ Render review results as a Word document with track changes.

  def render_html(*, result: 'Any', original_content: 'str', style: 'str' = 'article') -> 'str'
    └ Render a ReviewResult to a standalone HTML string via andamentum.typeset.

  async def sharpen_document(content: 'str', *, task: 'str' = 'review', num_experts: 'int' = 3, criteria: 'Optional[str]' = None, editors: 'Optional[list[str]]' = None, guidelines: 'Optional[str]' = None, model: 'str', verbose: 'bool' = False) -> 'ReviewResult'
    └ Run structured feedback over a draft you wrote yourself.
