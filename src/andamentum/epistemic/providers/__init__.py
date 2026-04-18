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

# ── Provider Registry ────────────────────────────────────────────────────────

PROVIDER_REGISTRY: dict[str, type] = {}
PROVIDER_DESCRIPTIONS: dict[str, str] = {}


def register_provider(name: str, cls: type, description: str = "") -> None:
    """Register a provider class by name with an optional domain description."""
    PROVIDER_REGISTRY[name] = cls
    if description:
        PROVIDER_DESCRIPTIONS[name] = description


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
    (
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
)
register_provider(
    "pubmed",
    PubMedProvider,
    (
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
)
register_provider(
    "biorxiv",
    BioRxivProvider,
    (
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
)
register_provider(
    "clinicaltrials",
    ClinicalTrialsProvider,
    (
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
)
register_provider(
    "chembl",
    ChEMBLProvider,
    (
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
)
register_provider(
    "monarch",
    MonarchProvider,
    (
        "Curated gene–disease and gene–phenotype associations aggregated from "
        "OMIM, HPO (Human Phenotype Ontology), Orphanet, ClinVar, and model organism "
        "databases by the Monarch Initiative. Best for questions about which genes "
        "are linked to which diseases, phenotype-driven rare disease diagnosis, "
        "variant–disease significance, and cross-species orthology of disease genes. "
        "Example queries: 'genes associated with hypertrophic cardiomyopathy', "
        "'phenotypes caused by COL1A1 mutations', 'rare diseases linked to mitochondrial "
        "complex I deficiency', 'clinical significance of BRCA1 c.5266dupC variant'."
    ),
)
register_provider(
    "open_targets",
    OpenTargetsProvider,
    (
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
    # Registry
    "PROVIDER_REGISTRY",
    "PROVIDER_DESCRIPTIONS",
    "PROVIDER_EXAMPLES",
    "register_provider",
    "get_provider",
    "get_all_providers",
    "get_biomedical_providers",
    "get_source_catalogue",
]
