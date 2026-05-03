"""Evidence providers for the epistemic system.

Registry-based provider discovery. Each provider implements:
- check_health() → CheckResult
- gather(query) → list[GatheredEvidence]

Usage:
    from andamentum.epistemic.providers import get_all_providers, get_provider

    # Get all providers
    providers = get_all_providers()

    # Get a specific provider
    pubmed = get_provider("pubmed")
    results = await pubmed.gather("BRCA1 breast cancer")
"""

from __future__ import annotations

from typing import Any

from .openalex import OpenAlexProvider, OpenAlexQualityScorer
from .monarch import MonarchProvider
from .pubmed import PubMedProvider
from .biorxiv import BioRxivProvider
from .clinicaltrials import ClinicalTrialsProvider
from .chembl import ChEMBLProvider
from .open_targets import OpenTargetsProvider
from .europepmc import EuropePMCProvider
from .cochrane import CochraneProvider
from .arxiv import ArXivProvider

# ── Provider Registry ────────────────────────────────────────────────────────

PROVIDER_REGISTRY: dict[str, type] = {}
PROVIDER_DESCRIPTIONS: dict[str, str] = {}
PROVIDER_QUERY_GUIDANCE: dict[str, str] = {}


def register_provider(
    name: str,
    cls: type,
    description: str = "",
    query_guidance: str = "",
) -> None:
    """Register a provider class.

    `description` describes content scope and when to pick this provider — read
    by the ranker. `query_guidance` describes the provider's native query
    language with a catalogue of valid styles — read by the formulator.
    """
    PROVIDER_REGISTRY[name] = cls
    if description:
        PROVIDER_DESCRIPTIONS[name] = description
    if query_guidance:
        PROVIDER_QUERY_GUIDANCE[name] = query_guidance


def get_source_catalogue() -> str:
    """Build a formatted catalogue of available providers for the planning agent.

    Returns a markdown-formatted list of provider names and domain descriptions.
    """
    lines = []
    for name in sorted(PROVIDER_REGISTRY):
        desc = PROVIDER_DESCRIPTIONS.get(name, "")
        if desc:
            lines.append(f"- **{name}**: {desc}")
        else:
            lines.append(f"- **{name}**")
    lines.append(
        "- **web_search**: General-purpose web search with evidence synthesis (always available as fallback)"
    )
    return "\n".join(lines)


def get_provider(name: str, **kwargs: Any) -> Any:
    """Get a provider instance by name."""
    cls = PROVIDER_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(PROVIDER_REGISTRY.keys()))
        raise KeyError(f"Unknown provider: {name}. Available: {available}")
    return cls(**kwargs)


def get_all_providers(**kwargs: Any) -> dict[str, Any]:
    """Get instances of all registered providers."""
    return {name: cls(**kwargs) for name, cls in PROVIDER_REGISTRY.items()}


def get_biomedical_providers() -> dict[str, Any]:
    """Get all providers suitable for biomedical research.

    Backward-compatible convenience function.
    """
    return get_all_providers()


# ── Register built-in providers ──────────────────────────────────────────────

register_provider(
    "openalex",
    OpenAlexProvider,
    description=(
        "The default general-purpose academic literature search for any scholarly "
        "question that does not specifically concern human medicine, drug compounds, "
        "or clinical trials. Use this provider whenever the question is about "
        "scientific research, scholarly work, or academic publications in general. "
        "Good default choice for any research question, especially broad or "
        "cross-disciplinary ones. Example queries: 'what do we know about the "
        "Permian-Triassic mass extinction', 'research on transformer attention "
        "mechanisms', 'academic papers about the origin of the Indo-European "
        "languages', 'scholarly work on population genetics and genetic drift'."
    ),
    query_guidance=(
        "The query goes to OpenAlex `/works` as the `search` parameter — "
        'full-text relevance ranking. Phrase quoting ("...") and implicit '
        "AND between tokens are supported.\n"
        "\n"
        "Query styles that all work:\n"
        "- Plain bag of terms: metformin HbA1c diabetes\n"
        '- Phrase-anchored: "GLP-1 receptor agonist" obesity\n'
        "- Topic plus study type: meta-analysis aspirin cardiovascular prevention\n"
        "- Author plus topic: Hinton backpropagation\n"
        "- Cross-disciplinary topic: transformer attention mechanism\n"
        "- Multi-domain: gravitational wave detection LIGO\n"
        "\n"
        "OpenAlex does NOT support PubMed-style [MeSH] field tags. OpenAlex is "
        "the strongest pick for non-biomedical scholarly questions (physics, "
        "history, economics, social sciences) and broad cross-disciplinary "
        "searches; for tight biomedical questions, PubMed and Europe PMC "
        "return less noise. The `site:` operator does not work."
    ),
)
register_provider(
    "pubmed",
    PubMedProvider,
    description=(
        "Peer-reviewed biomedical and life sciences literature from NCBI's MEDLINE. "
        "The default provider for any question about biomedical research, medicine, "
        "biology, disease mechanisms, molecular pathways, genetics, pharmacology, "
        "immunology, neuroscience, epidemiology, public health, or clinical outcomes "
        "as documented in the published peer-reviewed record. Use PubMed whenever a "
        "question is about what biomedical research has established or published, "
        "even if the question also touches on specific drugs, targets, or trials — "
        "other biomedical providers cover those more narrowly. Example queries: "
        "'role of interleukin-6 in rheumatoid arthritis pathogenesis', 'mechanisms "
        "of amyloid beta accumulation in Alzheimer's disease', 'epidemiology of "
        "tuberculosis in sub-Saharan Africa', 'published evidence on ketogenic diet "
        "for refractory epilepsy', 'neurobiology of opioid addiction'."
    ),
    query_guidance=(
        "The query is sent to NCBI esearch as the `term` parameter. The full "
        "PubMed query language is supported: Boolean operators (AND, OR, NOT), "
        'MeSH terms ("X"[MeSH], auto-explodes children unless [Mesh:noexp]), '
        "title/abstract field tags ([tiab], [ti], [ab]), author ([au]), journal "
        "([Journal]), publication date ([pdat]), publication type ([pt]), text "
        'word ([tw]), DOI ([doi]), PMID ([uid]), phrase quoting ("..."), '
        "truncation (brca*), date ranges (2020:2025[pdat]).\n"
        "\n"
        "Query styles that all work — pick whichever best targets the question:\n"
        "- Plain natural-language (uses Best Match relevance ranking): "
        "metformin glycemic control type 2 diabetes\n"
        '- MeSH-anchored Boolean: "Metformin"[MeSH] AND "Diabetes Mellitus, '
        'Type 2"[MeSH]\n'
        '- MeSH plus study-type filter: "Metformin"[MeSH] AND "Diabetes '
        'Mellitus, Type 2"[MeSH] AND ("Randomized Controlled Trial"[pt] OR '
        '"Meta-Analysis"[pt])\n'
        '- Field-tagged with phrases: "glycemic control"[tiab] AND humans[Mesh]\n'
        "- Author plus topic: Madsen KS[au] AND metformin\n"
        "- ID lookup: 35133415[uid]  or  10.1001/jama.2022.0078[doi]\n"
        "- Date-bounded: metformin glycemic 2020:2025[pdat]\n"
        "\n"
        "Length: short and structured beats long and free-text. 3–8 well-chosen "
        "tokens with operators usually outperforms a 12-word natural-language "
        "string. The `site:` operator is silently ignored — do not use it."
    ),
)
register_provider(
    "biorxiv",
    BioRxivProvider,
    description=(
        "Preprint server for unpublished, pre-peer-review biology and medicine "
        "manuscripts. Use this provider ONLY when the question explicitly asks "
        "about preprints, unpublished research, work that has not yet been peer "
        "reviewed, cutting-edge results that have not yet appeared in journals, "
        "or the very latest findings. If the question does not mention preprints "
        "or unpublished work, prefer pubmed or openalex instead. Example queries: "
        "'recent preprints on protein language models', 'unpublished findings on "
        "AlphaFold3 accuracy', 'latest preprint results about CRISPR prime "
        "editing efficiency', 'not-yet-published research on long COVID biomarkers'."
    ),
    query_guidance=(
        "The query is wrapped as `{query} AND (biorxiv[filter] OR "
        "medrxiv[filter])` and sent to NCBI esearch — so all PubMed query "
        "syntax (Boolean, MeSH, [tiab]/[ti]/[ab], [au], [pdat], phrase "
        "quoting, wildcards) is supported, scoped to bioRxiv and medRxiv "
        "preprints indexed by NCBI.\n"
        "\n"
        "Query styles that all work:\n"
        "- Plain text: protein language model AlphaFold\n"
        '- MeSH-anchored: "COVID-19"[MeSH] AND vaccine\n'
        '- Title-restricted: "organoid"[ti] AND brain\n'
        "- Date-bounded: single cell sequencing 2024:2025[pdat]\n"
        "- Author plus topic: Salzberg[au] AND genome assembly\n"
        '- Topic plus study type: "protein structure"[tiab] AND prediction\n'
        "\n"
        "bioRxiv and medRxiv preprint indexing in PubMed is partial — not "
        "every preprint reaches NCBI. Best for 'recent preprints on X' rather "
        "than 'the definitive answer to X'. The `site:` operator does not work."
    ),
)
register_provider(
    "clinicaltrials",
    ClinicalTrialsProvider,
    description=(
        "Registry of FDA-regulated and international clinical trials from "
        "ClinicalTrials.gov. Contains trial protocols, eligibility criteria, primary "
        "and secondary endpoints, enrollment numbers, phase (I/II/III/IV), sponsor "
        "information, recruitment status, and posted results. Best for questions about "
        "ongoing or completed clinical studies in humans, trial design, patient "
        "eligibility, endpoint selection, recruitment, and comparative trial data. "
        "Example queries: 'ongoing phase III trials for semaglutide in heart failure', "
        "'eligibility criteria for CAR-T cell therapy trials in lymphoma', 'primary "
        "endpoints of EMPA-REG OUTCOME study', 'recruiting clinical trials for "
        "pancreatic cancer immunotherapy'."
    ),
    query_guidance=(
        "The query goes to ClinicalTrials.gov v2 as the `query.term` parameter. "
        "Inside `query.term`, the AREA[FieldName]value syntax scopes to specific "
        "fields: Condition, Intervention, BriefTitle, OfficialTitle, Sponsor, "
        "OverallStatus, StudyType, Phase, OutcomeMeasure, LocationCountry, "
        'NCTId. Boolean (AND, OR, NOT), phrase quoting ("...").\n'
        "\n"
        "Query styles that all work:\n"
        "- Plain text: metformin type 2 diabetes\n"
        "- Field-scoped intervention plus condition: AREA[Intervention]metformin "
        'AND AREA[Condition]"type 2 diabetes"\n'
        "- Phase-filtered: AREA[Intervention]semaglutide AND AREA[Phase]Phase3\n"
        '- Outcome-targeted: AREA[OutcomeMeasure]"HbA1c" AND '
        "AREA[Intervention]metformin\n"
        "- Sponsor plus condition: AREA[Sponsor]Pfizer AND AREA[Condition]melanoma\n"
        '- Status filter: AREA[Intervention]"CAR-T" AND '
        "AREA[OverallStatus]Recruiting\n"
        "- NCT ID lookup: AREA[NCTId]NCT04183440\n"
        "\n"
        "This provider returns trial registrations, not literature — it is not "
        "the right place to look for systematic reviews or meta-analyses. The "
        "`site:` operator does not work."
    ),
)
register_provider(
    "chembl",
    ChEMBLProvider,
    description=(
        "Curated database of bioactive drug-like small molecules from EMBL-EBI, with "
        "quantitative bioactivity data, drug mechanisms, ADMET properties, and "
        "compound-target interactions. Contains IC50, EC50, Ki, Kd values, SMILES "
        "structures, ChEMBL IDs, binding assays, and approved drug indications. Best "
        "for questions about specific chemical compounds, drug potency, medicinal "
        "chemistry, quantitative pharmacology, and structure-activity relationships. "
        "Example queries: 'IC50 of imatinib against BCR-ABL kinase', 'mechanism of "
        "action of pembrolizumab', 'SMILES structure and bioactivity of remdesivir', "
        "'EC50 values for ACE inhibitors on angiotensin converting enzyme'."
    ),
    query_guidance=(
        "The query goes to ChEMBL's `/molecule/search.json` `q` parameter. "
        "Accepts: compound generic name, trade name, synonym, ChEMBL ID, drug "
        "INN, IUPAC name. SMILES and InChI substring search uses different "
        "endpoints not exposed by this adapter — do NOT pass SMILES strings.\n"
        "\n"
        "Query styles that all work:\n"
        "- Generic name: imatinib\n"
        "- Trade name: Gleevec\n"
        "- ChEMBL ID: CHEMBL941\n"
        "- Synonym or development code: STI571\n"
        "- Drug class member: pembrolizumab\n"
        "- Compound plus synonym: metformin glucophage\n"
        "\n"
        "This is a compound search, not a literature search — returns "
        "molecular structures, bioactivity (IC50, EC50, Ki), and mechanism. "
        "Use only when the question explicitly asks for compound-level "
        "pharmacology. 1–3 token compound names are optimal; verbose natural-"
        "language descriptions return nothing."
    ),
)
register_provider(
    "monarch",
    MonarchProvider,
    description=(
        "Curated gene–disease and gene–phenotype associations aggregated from "
        "OMIM, HPO (Human Phenotype Ontology), Orphanet, ClinVar, and model organism "
        "databases by the Monarch Initiative. Best for questions about which genes "
        "are linked to which diseases, phenotype-driven rare disease diagnosis, "
        "variant–disease significance, and cross-species orthology of disease genes. "
        "Example queries: 'genes associated with hypertrophic cardiomyopathy', "
        "'phenotypes caused by COL1A1 mutations', 'rare diseases linked to mitochondrial "
        "complex I deficiency', 'clinical significance of BRCA1 c.5266dupC variant'."
    ),
    query_guidance=(
        "The query goes to Monarch's `/search` `q` parameter. Accepts: gene "
        "symbols, disease names, phenotype terms, ontology IDs (HGNC:, MONDO:, "
        "HP:, OMIM:, ORPHA:), and variants.\n"
        "\n"
        "Query styles that all work:\n"
        "- Gene symbol: BRCA1\n"
        "- Gene plus disease: BRCA1 breast cancer\n"
        "- Disease name: cystic fibrosis\n"
        "- HPO phenotype term: intellectual disability\n"
        "- MONDO disease ID: MONDO:0008029\n"
        "- HGNC gene ID: HGNC:1100\n"
        "- HPO phenotype ID: HP:0001263\n"
        "- Variant: c.5266dupC BRCA1\n"
        "\n"
        "Monarch is curated gene-disease-phenotype association data, not "
        "literature. Use only for 'what genes are linked to X' or 'what "
        "phenotypes does mutation in Y cause'. 1–3 token queries are optimal."
    ),
)
register_provider(
    "open_targets",
    OpenTargetsProvider,
    description=(
        "Integrated drug target evidence from the Open Targets Platform, combining "
        "genetic associations (GWAS), somatic mutations (cancer), literature co-mentions, "
        "pathway membership, drug-target interactions, tractability, and "
        "target-disease association scores. Best for questions about which proteins "
        "or genes are therapeutic targets for a given disease, drug repurposing "
        "opportunities, pathway-level target evaluation, and druggability assessment. "
        "Example queries: 'therapeutic targets for Alzheimer's disease with genetic "
        "support', 'druggable targets in KRAS-mutant colorectal cancer', 'pathway "
        "evidence linking TNF signaling to rheumatoid arthritis', 'target tractability "
        "for PCSK9 in cardiovascular disease'."
    ),
    query_guidance=(
        "The query goes to Open Targets GraphQL `search(queryString: $q)`. "
        "Accepts: target names, gene symbols, disease names, pathway names, "
        "ontology IDs (ENSG, MONDO, EFO, HP).\n"
        "\n"
        "Query styles that all work:\n"
        "- Gene symbol: KRAS\n"
        "- Disease name: Alzheimer's disease\n"
        "- Disease plus qualifier: idiopathic pulmonary fibrosis\n"
        "- Ensembl gene ID: ENSG00000133703\n"
        "- EFO disease ID: EFO_0000270\n"
        "- Pathway name: TNF signaling\n"
        "- Drug-disease pair: tofacitinib rheumatoid arthritis\n"
        "\n"
        "Returns target-disease associations, druggability, GWAS signal, and "
        "drug-target interactions. Not for literature. 1–3 token queries are "
        "optimal."
    ),
)
register_provider(
    "europepmc",
    EuropePMCProvider,
    description=(
        "Comprehensive biomedical and life sciences literature from Europe PMC, "
        "covering PubMed, PMC full-text, preprints, and patents. Returns full "
        "abstracts for all results. Use for any biomedical literature search, "
        "especially when full abstracts are needed or when searching across "
        "preprints and published articles simultaneously. Example queries: "
        "'CRISPR-Cas9 gene editing efficiency in vivo', 'single-cell RNA "
        "sequencing methods comparison', 'gut microbiome and immune response'."
    ),
    query_guidance=(
        "The query goes to Europe PMC's `search` endpoint as the `query` "
        "parameter. Native field operators: TITLE:, ABSTRACT:, KW: (keyword), "
        "AUTH: (author), AFF: (affiliation), JOURNAL:, ISSN:, DOI:, PMID:, "
        "PUB_YEAR:, FIRST_AUTH:, OPEN_ACCESS:y, SRC: (MED, PRE for preprints, "
        "AGR for agricultural, CTX, ETH, HIR). Boolean (AND, OR, NOT), phrase "
        'quoting ("..."), wildcards (cancer*), range syntax ([2020 TO 2025]).\n'
        "\n"
        "Query styles that all work:\n"
        "- Plain text: CRISPR Cas9 gene editing efficiency\n"
        '- Title-restricted: TITLE:"metformin" AND TITLE:"HbA1c"\n'
        '- Title OR keyword plus abstract: (TITLE:"metformin" OR '
        'KW:"metformin") AND ABSTRACT:"glycemic"\n'
        "- Date-bounded: metformin HbA1c PUB_YEAR:[2020 TO 2025]\n"
        '- Open-access only: "single cell" AND OPEN_ACCESS:y\n'
        '- Author plus topic: AUTH:"Madsen" AND metformin\n'
        "- Source-filtered (preprints): metformin SRC:PRE\n"
        '- DOI / PMID lookup: DOI:"10.1001/jama.2022.0078"\n'
        "\n"
        "Plain-text queries get heavily diluted by conference abstract "
        "collections (ESICM LIVES, UEG Week, ECTS Congress, etc., which "
        "contain thousands of mentions of any biomedical term) — prefer "
        "field-restricted queries when possible. PubMed-style [MeSH] field "
        "tags do NOT work here. The `site:` operator does not work."
    ),
)
register_provider(
    "cochrane",
    CochraneProvider,
    description=(
        "Cochrane systematic reviews and meta-analyses — the highest level of "
        "clinical evidence. Each review synthesizes findings from multiple "
        "randomized controlled trials on a specific clinical question. Use for "
        "any claim about clinical interventions, treatment effectiveness, drug "
        "safety, or public health interventions where a systematic review may "
        "exist. Example queries: 'exercise interventions for preventing falls "
        "in older adults', 'statins for primary prevention of cardiovascular "
        "disease', 'antibiotics for acute otitis media in children'."
    ),
    query_guidance=(
        'The query goes to NCBI esearch with `AND "Cochrane Database Syst '
        'Rev"[Journal]` appended automatically. Full PubMed query syntax is '
        "supported (Boolean, MeSH, [tiab]/[ti]/[ab], [au], [pdat], phrase "
        "quoting, wildcards), but the corpus is already pre-filtered to "
        "Cochrane systematic reviews — keep queries shorter and more topic-"
        "focused than for PubMed.\n"
        "\n"
        "Query styles that all work:\n"
        "- Short topic phrase: metformin type 2 diabetes\n"
        "- Topic plus intervention: exercise falls older adults\n"
        '- MeSH-anchored: "Metformin"[MeSH] AND "Diabetes Mellitus, Type 2"[MeSH]\n'
        '- Title-tagged: "metformin"[ti] AND "diabetes"[ti]\n'
        "- Topic plus outcome: aspirin myocardial infarction primary prevention\n"
        "- Drug class plus condition: SGLT2 inhibitor heart failure\n"
        "\n"
        "Adding [pt] filters for systematic reviews is redundant — every result "
        "is one. 3–7 tokens is usually optimal. The `site:` operator does not "
        "work."
    ),
)
register_provider(
    "arxiv",
    ArXivProvider,
    description=(
        "Preprint server for physics, mathematics, computer science, "
        "quantitative biology, quantitative finance, statistics, electrical "
        "engineering, and economics. Use for any non-biomedical scientific "
        "claim, especially in physics, AI/ML, mathematics, or computer "
        "science. Also covers quantitative biology preprints not on bioRxiv. "
        "Example queries: 'transformer attention mechanisms', 'quantum error "
        "correction surface codes', 'reinforcement learning from human feedback'."
    ),
    query_guidance=(
        "The query is wrapped as `all:{query}` and sent to the arXiv API's "
        "`search_query` parameter, but a query starting with a field prefix "
        "overrides this. Field prefixes: ti: (title), abs: (abstract), au: "
        "(author), cat: (subject category, e.g., cs.LG, stat.ML, q-bio.PE, "
        "math.ST, hep-ph, cond-mat), all: (all fields), jr: (journal-ref), "
        'id: (arXiv ID). Boolean: AND, OR, ANDNOT. Phrase quoting ("...").\n'
        "\n"
        "Query styles that all work:\n"
        "- Plain bag of terms (auto-prefixed `all:`): transformer attention "
        "mechanism\n"
        '- Title-restricted: ti:"reinforcement learning"\n'
        "- Category plus topic: cat:cs.LG AND ti:transformer\n"
        "- Author plus topic: au:Hinton AND backpropagation\n"
        "- Title plus abstract: ti:diffusion AND abs:image\n"
        "- Multi-category: (cat:cs.LG OR cat:stat.ML) AND ti:scaling\n"
        "- arXiv ID lookup: id:2305.12345\n"
        "\n"
        "Coverage: physics, math, CS, quantitative biology, quantitative "
        "finance, statistics, EE, economics. No clinical or wet-lab biomedical "
        "literature here. Use cat: prefixes to scope to the right subdomain."
    ),
)


# ── Example queries for semantic routing ─────────────────────────────────────
# Each provider has 6-8 short example queries at the same granularity as
# typical user inputs. The semantic router embeds these individually and
# scores each provider by its BEST-matching example (max-sim), solving the
# short-vs-long embedding mismatch that occurs when comparing a terse claim
# against a 200-word description.
#
# When adding a new provider, include 6-8 diverse examples covering the
# provider's core strengths and edge-case subdomains.

PROVIDER_EXAMPLES: dict[str, list[str]] = {
    "openalex": [
        "what do we know about the Permian-Triassic mass extinction",
        "research on transformer attention mechanisms",
        "academic papers about the origin of the Indo-European languages",
        "scholarly work on population genetics and genetic drift",
        "economic effects of monetary policy on inflation",
        "quantum entanglement experiments in superconducting qubits",
        "evidence for dark matter from galaxy rotation curves",
        "sociological research on income inequality and social mobility",
    ],
    "pubmed": [
        "role of interleukin-6 in rheumatoid arthritis pathogenesis",
        "mechanisms of amyloid beta accumulation in Alzheimer's disease",
        "epidemiology of tuberculosis in sub-Saharan Africa",
        "published evidence on ketogenic diet for refractory epilepsy",
        "neurobiology of opioid addiction",
        "cell migration and motility in tissue injury response",
        "molecular mechanisms of apoptosis in cancer cells",
        "renal physiology and glomerular filtration regulation",
    ],
    "biorxiv": [
        "recent preprints on protein language models",
        "unpublished findings on AlphaFold3 accuracy",
        "latest preprint results about CRISPR prime editing efficiency",
        "not-yet-published research on long COVID biomarkers",
        "new preprint data on single-cell RNA sequencing methods",
        "cutting-edge unpublished work on organoid disease models",
    ],
    "clinicaltrials": [
        "ongoing phase III trials for semaglutide in heart failure",
        "eligibility criteria for CAR-T cell therapy trials in lymphoma",
        "primary endpoints of EMPA-REG OUTCOME study",
        "recruiting clinical trials for pancreatic cancer immunotherapy",
        "trial design for GLP-1 receptor agonists in obesity",
        "phase II dose-escalation study results for antibody-drug conjugates",
    ],
    "chembl": [
        "IC50 of imatinib against BCR-ABL kinase",
        "mechanism of action of pembrolizumab",
        "SMILES structure and bioactivity of remdesivir",
        "EC50 values for ACE inhibitors on angiotensin converting enzyme",
        "binding affinity of selective serotonin reuptake inhibitors",
        "structure-activity relationships of benzodiazepine derivatives",
    ],
    "monarch": [
        "genes associated with hypertrophic cardiomyopathy",
        "phenotypes caused by COL1A1 mutations",
        "rare diseases linked to mitochondrial complex I deficiency",
        "clinical significance of BRCA1 c.5266dupC variant",
        "gene-disease associations for hereditary spastic paraplegia",
        "cross-species orthology of Fragile X syndrome gene FMR1",
    ],
    "open_targets": [
        "therapeutic targets for Alzheimer's disease with genetic support",
        "druggable targets in KRAS-mutant colorectal cancer",
        "pathway evidence linking TNF signaling to rheumatoid arthritis",
        "target tractability for PCSK9 in cardiovascular disease",
        "GWAS-supported targets for schizophrenia",
        "drug repurposing candidates for idiopathic pulmonary fibrosis",
    ],
    "europepmc": [
        "CRISPR-Cas9 gene editing efficiency in vivo",
        "single-cell RNA sequencing methods comparison",
        "gut microbiome and immune response",
        "amyloid beta oligomers in Alzheimer's pathology",
        "mRNA vaccine lipid nanoparticle delivery",
        "tumor microenvironment immunotherapy resistance",
    ],
    "cochrane": [
        "exercise interventions for preventing falls in older adults",
        "statins for primary prevention of cardiovascular disease",
        "antibiotics for acute otitis media in children",
        "cognitive behavioral therapy for depression",
        "corticosteroids for preterm birth lung maturation",
        "anticoagulation for atrial fibrillation stroke prevention",
    ],
    "arxiv": [
        "transformer attention mechanisms",
        "quantum error correction surface codes",
        "reinforcement learning from human feedback",
        "neural scaling laws and emergent abilities",
        "gravitational wave detection methods",
        "topological insulators band structure",
    ],
}


__all__ = [
    # Provider classes
    "OpenAlexProvider",
    "OpenAlexQualityScorer",
    "MonarchProvider",
    "PubMedProvider",
    "BioRxivProvider",
    "ClinicalTrialsProvider",
    "ChEMBLProvider",
    "OpenTargetsProvider",
    "EuropePMCProvider",
    "CochraneProvider",
    "ArXivProvider",
    # Registry
    "PROVIDER_REGISTRY",
    "PROVIDER_DESCRIPTIONS",
    "PROVIDER_QUERY_GUIDANCE",
    "PROVIDER_EXAMPLES",
    "register_provider",
    "get_provider",
    "get_all_providers",
    "get_biomedical_providers",
    "get_source_catalogue",
]
