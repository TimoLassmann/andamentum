"""Ground-truth dataset for the provider-routing benchmark.

200 labeled research queries distributed across the seven built-in evidence
providers plus an ``edge`` category for multi-provider and ambiguous cases.

Schema
------
Each entry is a dict with:

- ``query``: str — the research question as a user might ask it
- ``primary``: str — the single best provider for this query (used for
  top-1 accuracy and MRR)
- ``acceptable``: set[str] — every provider that would be a correct choice
  if it landed in the top-K (used for top-K recall). Always includes
  ``primary``.
- ``category``: str — grouping label for per-category reporting (one of
  ``openalex``, ``pubmed``, ``biorxiv``, ``clinicaltrials``, ``chembl``,
  ``monarch``, ``open_targets``, ``edge``)

Design notes
------------
- ``web_search`` is never a ``primary`` — it is always appended as a
  fallback by :func:`andamentum.epistemic.provider_routing.select_providers`
  and is therefore not a meaningful target of the semantic router.
- The ``acceptable`` sets are deliberately non-trivial for biomedical
  queries because PubMed, Monarch, OpenTargets, and ClinicalTrials genuinely
  overlap for many questions. The benchmark grades top-K recall, not a
  strict one-hot match, because that reflects how :func:`select_providers`
  is actually consumed by ``PlanTaskOperation``.
- Distribution: openalex 30, pubmed 35, clinicaltrials 30, chembl 25,
  monarch 25, open_targets 25, biorxiv 20, edge 10 = 200 total.
"""

from __future__ import annotations

from typing import TypedDict


class BenchmarkQuery(TypedDict):
    query: str
    primary: str
    acceptable: set[str]
    category: str


def _q(
    query: str,
    primary: str,
    also: list[str] | None = None,
    category: str | None = None,
) -> BenchmarkQuery:
    """Helper to build a benchmark entry with less boilerplate."""
    acceptable = {primary}
    if also:
        acceptable.update(also)
    return {
        "query": query,
        "primary": primary,
        "acceptable": acceptable,
        "category": category or primary,
    }


# ── OpenAlex: general, non-biomedical academic literature (30) ───────────────

_OPENALEX: list[BenchmarkQuery] = [
    _q("What caused the Permian-Triassic mass extinction event?", "openalex"),
    _q("Evidence for a ninth planet beyond Neptune", "openalex"),
    _q("Mechanisms of quantum entanglement in superconducting qubits", "openalex"),
    _q("Origin and spread of the Indo-European language family", "openalex"),
    _q("Mathematical proof of the Poincaré conjecture", "openalex"),
    _q("Causes of the collapse of the Bronze Age civilizations", "openalex"),
    _q("Interpretability of transformer attention heads", "openalex"),
    _q("Dark matter halo models in cosmological simulations", "openalex"),
    _q("Isotope evidence for Neanderthal diet and mobility", "openalex", also=["biorxiv"]),
    _q("Influence of Keynesian policy on post-war economic recovery", "openalex"),
    _q("Evolution of bipedalism in early hominins", "openalex", also=["pubmed", "biorxiv"]),
    _q("Plate tectonics theory historical development", "openalex"),
    _q("Formal semantics of modal logic", "openalex"),
    _q("Origins of the Silk Road trade network", "openalex"),
    _q("Gravitational wave detection at LIGO", "openalex"),
    _q("Anthropogenic climate change detection attribution", "openalex"),
    _q("Economic effects of currency unions", "openalex"),
    _q("Roman Empire's fall: economic vs military causes", "openalex"),
    _q("Convolutional neural network architecture innovations", "openalex"),
    _q("Ocean acidification impact on coral reef ecosystems", "openalex", also=["biorxiv"]),
    _q("Fermi paradox and the Drake equation", "openalex"),
    _q("Social contract theory from Hobbes to Rawls", "openalex"),
    _q("Polymer physics of protein folding", "openalex", also=["pubmed", "biorxiv"]),
    _q("Category theory foundations of functional programming", "openalex"),
    _q("Historical linguistics evidence for Proto-Uralic", "openalex"),
    _q("Mantle convection models of Earth's interior", "openalex"),
    _q("Byzantine Empire administrative reforms under Justinian", "openalex"),
    _q("Reinforcement learning from human feedback techniques", "openalex"),
    _q("Population genetics models of genetic drift", "openalex", also=["pubmed", "biorxiv"]),
    _q("Free will debates in contemporary philosophy of mind", "openalex"),
]


# ── PubMed: biomedical literature, clinical + molecular (35) ─────────────────

_PUBMED: list[BenchmarkQuery] = [
    _q(
        "Metformin cardiovascular mortality outcomes in type 2 diabetes",
        "pubmed",
        also=["clinicaltrials", "open_targets"],
    ),
    _q(
        "Efficacy of mRNA vaccines against SARS-CoV-2 Omicron variants",
        "pubmed",
        also=["clinicaltrials"],
    ),
    _q("Association between gut microbiome and Parkinson's disease", "pubmed"),
    _q(
        "Statin therapy for primary prevention of cardiovascular disease",
        "pubmed",
        also=["clinicaltrials", "open_targets"],
    ),
    _q(
        "Mechanisms of amyloid beta accumulation in Alzheimer's disease",
        "pubmed",
        also=["open_targets"],
    ),
    _q(
        "Efficacy of cognitive behavioral therapy for chronic insomnia",
        "pubmed",
        also=["clinicaltrials"],
    ),
    _q(
        "Long-term effects of bariatric surgery on type 2 diabetes remission",
        "pubmed",
        also=["clinicaltrials"],
    ),
    _q(
        "Role of interleukin-6 in rheumatoid arthritis pathogenesis",
        "pubmed",
        also=["open_targets"],
    ),
    _q("Antibiotic resistance mechanisms in Staphylococcus aureus", "pubmed"),
    _q("Impact of sleep deprivation on glucose metabolism", "pubmed"),
    _q(
        "Systematic review of omega-3 fatty acids and cognitive decline",
        "pubmed",
        also=["clinicaltrials"],
    ),
    _q(
        "Meta-analysis of vitamin D supplementation and mortality",
        "pubmed",
        also=["clinicaltrials"],
    ),
    _q(
        "Published evidence on ketogenic diet for refractory epilepsy",
        "pubmed",
        also=["clinicaltrials"],
    ),
    _q(
        "Role of TP53 mutations in colorectal cancer progression",
        "pubmed",
        also=["open_targets", "monarch"],
    ),
    _q(
        "Tau protein pathology in frontotemporal dementia",
        "pubmed",
        also=["open_targets"],
    ),
    _q(
        "Effectiveness of HPV vaccination for cervical cancer prevention",
        "pubmed",
        also=["clinicaltrials"],
    ),
    _q(
        "Neurobiological basis of opioid addiction and withdrawal",
        "pubmed",
        also=["chembl"],
    ),
    _q(
        "Pathophysiology of septic shock in intensive care patients",
        "pubmed",
        also=["clinicaltrials"],
    ),
    _q(
        "Long COVID symptoms and biomarkers in adults",
        "pubmed",
        also=["clinicaltrials"],
    ),
    _q(
        "Mechanisms of immune checkpoint inhibition in melanoma",
        "pubmed",
        also=["open_targets", "chembl"],
    ),
    _q(
        "Clinical features and outcomes of pulmonary hypertension",
        "pubmed",
        also=["clinicaltrials"],
    ),
    _q("Gut-brain axis in inflammatory bowel disease", "pubmed"),
    _q(
        "Molecular mechanisms of insulin resistance in skeletal muscle",
        "pubmed",
        also=["open_targets"],
    ),
    _q("Epidemiology of tuberculosis in sub-Saharan Africa", "pubmed"),
    _q(
        "Role of microglia in neurodegeneration",
        "pubmed",
        also=["open_targets"],
    ),
    _q(
        "Autophagy dysfunction in Huntington's disease",
        "pubmed",
        also=["open_targets", "monarch"],
    ),
    _q(
        "Pharmacokinetics of tacrolimus in kidney transplant patients",
        "pubmed",
        also=["chembl", "clinicaltrials"],
    ),
    _q("Impact of air pollution on asthma exacerbations in children", "pubmed"),
    _q(
        "Review of checkpoint inhibitors in non-small cell lung cancer",
        "pubmed",
        also=["clinicaltrials", "open_targets"],
    ),
    _q(
        "Mitochondrial dysfunction in Parkinson's disease dopamine neurons",
        "pubmed",
        also=["open_targets"],
    ),
    _q("Placental biology and preeclampsia risk", "pubmed"),
    _q(
        "Aspirin for primary prevention of colorectal cancer",
        "pubmed",
        also=["clinicaltrials"],
    ),
    _q("Neuroinflammation in multiple sclerosis lesion progression", "pubmed"),
    _q(
        "Insulin signaling pathway in type 2 diabetes beta cells",
        "pubmed",
        also=["open_targets"],
    ),
    _q(
        "Pathogenesis of idiopathic pulmonary fibrosis",
        "pubmed",
        also=["open_targets"],
    ),
]


# ── ClinicalTrials: trial registry, protocol, endpoints (30) ─────────────────

_CLINICALTRIALS: list[BenchmarkQuery] = [
    _q(
        "Ongoing phase III trials for semaglutide in heart failure",
        "clinicaltrials",
    ),
    _q(
        "Eligibility criteria for CAR-T cell therapy trials in lymphoma",
        "clinicaltrials",
    ),
    _q(
        "Primary endpoints of the EMPA-REG OUTCOME study",
        "clinicaltrials",
        also=["pubmed"],
    ),
    _q(
        "Recruiting clinical trials for pancreatic cancer immunotherapy",
        "clinicaltrials",
    ),
    _q(
        "Enrollment status of phase II trials for ALS therapies",
        "clinicaltrials",
    ),
    _q(
        "Trial design for GLP-1 receptor agonists in obesity",
        "clinicaltrials",
    ),
    _q(
        "Phase IV post-marketing trials of direct oral anticoagulants",
        "clinicaltrials",
    ),
    _q(
        "Sample size calculation for Alzheimer's disease trials",
        "clinicaltrials",
    ),
    _q(
        "Placebo arm design in oncology trials",
        "clinicaltrials",
    ),
    _q(
        "Trials investigating mRNA cancer vaccines",
        "clinicaltrials",
    ),
    _q(
        "Inclusion criteria for pediatric epilepsy drug trials",
        "clinicaltrials",
    ),
    _q(
        "Adaptive trial designs in platform studies for COVID-19",
        "clinicaltrials",
    ),
    _q(
        "Completed trials of GLP-1 agonists for NASH",
        "clinicaltrials",
    ),
    _q(
        "Primary completion dates for ongoing sickle cell disease trials",
        "clinicaltrials",
    ),
    _q(
        "Comparator arms in immunotherapy combination trials",
        "clinicaltrials",
    ),
    _q(
        "Withdrawn phase III trials for Alzheimer's disease 2020-2024",
        "clinicaltrials",
    ),
    _q(
        "Trial protocols for gene therapy in Duchenne muscular dystrophy",
        "clinicaltrials",
    ),
    _q(
        "Dosing regimens tested in phase I oncology trials",
        "clinicaltrials",
    ),
    _q(
        "Trial endpoints for heart failure with preserved ejection fraction",
        "clinicaltrials",
    ),
    _q(
        "Recruitment locations for multi-center rare disease trials",
        "clinicaltrials",
    ),
    _q(
        "Active surveillance arms in prostate cancer clinical trials",
        "clinicaltrials",
    ),
    _q(
        "Ongoing interventional studies for fibromyalgia",
        "clinicaltrials",
    ),
    _q(
        "Cross-over trial designs in chronic pain management",
        "clinicaltrials",
    ),
    _q(
        "Phase II/III trials combining PD-1 and CTLA-4 inhibitors",
        "clinicaltrials",
    ),
    _q(
        "Recruitment status of tirzepatide cardiovascular outcome trials",
        "clinicaltrials",
    ),
    _q(
        "Biomarker stratification in precision oncology trials",
        "clinicaltrials",
    ),
    _q(
        "Randomization procedures in pragmatic clinical trials",
        "clinicaltrials",
    ),
    _q(
        "Trials comparing SGLT2 inhibitors across diabetic populations",
        "clinicaltrials",
    ),
    _q(
        "Phase I dose-escalation trial results for antibody-drug conjugates",
        "clinicaltrials",
    ),
    _q(
        "Eligibility criteria for Parkinson's disease deep brain stimulation trials",
        "clinicaltrials",
    ),
]


# ── ChEMBL: compounds, bioactivity, medicinal chemistry (25) ─────────────────

_CHEMBL: list[BenchmarkQuery] = [
    _q("IC50 of imatinib against BCR-ABL kinase", "chembl"),
    _q("Mechanism of action of pembrolizumab", "chembl", also=["open_targets"]),
    _q("SMILES structure and bioactivity of remdesivir", "chembl"),
    _q("EC50 values for ACE inhibitors on angiotensin converting enzyme", "chembl"),
    _q("Binding affinity of trastuzumab for HER2", "chembl", also=["open_targets"]),
    _q("Ki values for selective serotonin reuptake inhibitors", "chembl"),
    _q("ADMET properties of orally available kinase inhibitors", "chembl"),
    _q("Compound library for JAK2 inhibitors", "chembl"),
    _q("Dissociation constant of atorvastatin for HMG-CoA reductase", "chembl"),
    _q("Structure-activity relationships of benzodiazepines", "chembl"),
    _q("Bioactivity data for caffeine on adenosine receptors", "chembl"),
    _q("Pharmacophore models for GPCR ligands", "chembl"),
    _q("Quantitative pharmacology of opioid mu receptor agonists", "chembl"),
    _q("Binding constants of warfarin to vitamin K epoxide reductase", "chembl"),
    _q("IC50 measurements for COX-2 selective inhibitors", "chembl"),
    _q("Compound screening data against SARS-CoV-2 main protease", "chembl"),
    _q("Potency data for FXa direct oral anticoagulants", "chembl"),
    _q("Molecular properties of Lipinski rule-of-five compliant drugs", "chembl"),
    _q("Selectivity profile of gefitinib across EGFR variants", "chembl"),
    _q("Bioactivity of natural products in ChEMBL database", "chembl"),
    _q("Chemical scaffold diversity of approved antibiotics", "chembl"),
    _q("IC50 of palbociclib against CDK4 and CDK6", "chembl"),
    _q("EC50 of GLP-1 receptor agonists in HEK293 cells", "chembl"),
    _q("Binding affinity ratios for beta-1 vs beta-2 adrenergic antagonists", "chembl"),
    _q("In vitro potency data for HIV protease inhibitors", "chembl"),
]


# ── Monarch: gene-disease and gene-phenotype associations (25) ───────────────

_MONARCH: list[BenchmarkQuery] = [
    _q(
        "Genes associated with hypertrophic cardiomyopathy",
        "monarch",
        also=["pubmed", "open_targets"],
    ),
    _q("Phenotypes caused by COL1A1 mutations", "monarch"),
    _q(
        "Rare diseases linked to mitochondrial complex I deficiency",
        "monarch",
        also=["pubmed"],
    ),
    _q(
        "Clinical significance of BRCA1 c.5266dupC variant",
        "monarch",
        also=["pubmed", "open_targets"],
    ),
    _q("Gene-phenotype associations for Noonan syndrome", "monarch"),
    _q("HPO terms matching Kabuki syndrome presentation", "monarch"),
    _q("Known gene associations with Charcot-Marie-Tooth disease", "monarch"),
    _q("Monogenic causes of early-onset Parkinson's disease", "monarch", also=["pubmed"]),
    _q("Orphanet-classified rare causes of sensorineural hearing loss", "monarch"),
    _q("Phenotype ontology for 22q11.2 deletion syndrome", "monarch"),
    _q("Genes linked to Bardet-Biedl syndrome", "monarch"),
    _q("Rare immune deficiency syndromes by causal gene", "monarch"),
    _q("Gene associations for hereditary spastic paraplegia", "monarch"),
    _q("Variants in SCN1A causing Dravet syndrome", "monarch", also=["pubmed"]),
    _q("Known gene panels for mitochondrial encephalopathies", "monarch"),
    _q("Cross-species orthology of Fragile X syndrome gene FMR1", "monarch"),
    _q("Phenotype-driven differential diagnosis for craniofacial syndromes", "monarch"),
    _q("Rare diseases caused by peroxisomal disorders", "monarch"),
    _q("Genes linked to congenital disorders of glycosylation", "monarch"),
    _q("HPO terms for skeletal dysplasias in achondroplasia family", "monarch"),
    _q("Clinical presentations of Gaucher disease by subtype", "monarch"),
    _q("Gene-disease links for Li-Fraumeni syndrome", "monarch", also=["pubmed"]),
    _q("Rare hereditary forms of amyotrophic lateral sclerosis", "monarch", also=["pubmed"]),
    _q("Known genetic causes of primary ciliary dyskinesia", "monarch"),
    _q("Phenotype overlap between Usher syndrome subtypes", "monarch"),
]


# ── OpenTargets: drug target evidence, druggability, pathways (25) ───────────

_OPEN_TARGETS: list[BenchmarkQuery] = [
    _q(
        "Therapeutic targets for Alzheimer's disease with genetic support",
        "open_targets",
        also=["pubmed"],
    ),
    _q("Druggable targets in KRAS-mutant colorectal cancer", "open_targets"),
    _q(
        "Pathway evidence linking TNF signaling to rheumatoid arthritis",
        "open_targets",
        also=["pubmed"],
    ),
    _q("Target tractability for PCSK9 in cardiovascular disease", "open_targets"),
    _q("Disease-target association score for APOE in Alzheimer's", "open_targets"),
    _q("Known drugs targeting EGFR in non-small cell lung cancer", "open_targets"),
    _q("Genetic GWAS targets for ulcerative colitis", "open_targets"),
    _q("Drug repurposing candidates for idiopathic pulmonary fibrosis", "open_targets"),
    _q("Tractable targets in triple-negative breast cancer", "open_targets"),
    _q("Open Targets evidence for IL-17 in psoriasis", "open_targets", also=["pubmed"]),
    _q("Target-disease association for SOD1 in ALS", "open_targets", also=["monarch"]),
    _q("Druggability of intrinsically disordered proteins in cancer", "open_targets"),
    _q("Pathway-level target evaluation for Crohn's disease", "open_targets"),
    _q("Somatic mutation evidence for targets in melanoma", "open_targets"),
    _q("Genetic validation of HMGCR as a lipid-lowering target", "open_targets"),
    _q("Target tractability assessment for tau in Alzheimer's", "open_targets"),
    _q("Disease-target associations from Open Targets for Type 2 diabetes", "open_targets"),
    _q("Literature co-mention evidence for CFTR in cystic fibrosis", "open_targets"),
    _q("Mendelian randomization support for IL6R in cardiovascular disease", "open_targets"),
    _q("GWAS-supported targets for schizophrenia", "open_targets"),
    _q("Drug-target interaction network for multiple sclerosis", "open_targets"),
    _q("Known drugs targeting JAK1/2 in myeloproliferative disorders", "open_targets"),
    _q("Target prioritization for Parkinson's disease based on genetic evidence", "open_targets"),
    _q("Cancer driver genes with tractable small-molecule inhibitors", "open_targets"),
    _q("Open Targets score for BTK in chronic lymphocytic leukemia", "open_targets"),
]


# ── bioRxiv: preprints, unpublished, cutting-edge (20) ───────────────────────

_BIORXIV: list[BenchmarkQuery] = [
    _q("Recent preprints on protein language models for structure prediction", "biorxiv"),
    _q("Latest unpublished results on AlphaFold3 accuracy", "biorxiv"),
    _q("Preprints discussing CRISPR prime editing efficiency", "biorxiv"),
    _q("New preprint findings on long COVID biomarkers", "biorxiv", also=["pubmed"]),
    _q("bioRxiv preprints on single-cell multi-omics methods", "biorxiv"),
    _q("Unreviewed findings on mRNA vaccine durability", "biorxiv", also=["pubmed"]),
    _q("Recent medRxiv preprints on gut microbiome and depression", "biorxiv", also=["pubmed"]),
    _q("Cutting-edge preprints on organoid disease modeling", "biorxiv"),
    _q("Preprint reports of novel Alzheimer's disease biomarkers", "biorxiv", also=["pubmed"]),
    _q("Latest preprint evidence on CAR-T cell persistence", "biorxiv"),
    _q("Unpublished findings on ancient DNA population genetics", "biorxiv"),
    _q("Recent preprints on climate change effects on infectious diseases", "biorxiv", also=["pubmed"]),
    _q("New preprint data on psychedelic-assisted psychotherapy", "biorxiv", also=["pubmed"]),
    _q("bioRxiv preprints discussing GPT models for protein design", "biorxiv"),
    _q("Preprint findings on mRNA therapeutics for rare diseases", "biorxiv"),
    _q("Latest unpublished cryo-EM structures of GPCRs", "biorxiv"),
    _q("Preprints on base editing for sickle cell disease", "biorxiv", also=["pubmed"]),
    _q("Recent medRxiv preprints on pandemic preparedness", "biorxiv", also=["pubmed"]),
    _q("Unreviewed findings on gene drive systems in mosquitoes", "biorxiv"),
    _q("Latest preprint data on tumor microenvironment immunotherapy", "biorxiv", also=["pubmed"]),
]


# ── Edge cases: genuinely ambiguous or multi-provider queries (10) ───────────

_EDGE: list[BenchmarkQuery] = [
    _q(
        "BRCA1 breast cancer risk and clinical management",
        "pubmed",
        also=["monarch", "open_targets", "clinicaltrials"],
        category="edge",
    ),
    _q(
        "Metformin as a potential anti-aging drug",
        "pubmed",
        also=["clinicaltrials", "open_targets", "chembl"],
        category="edge",
    ),
    _q(
        "Genetic variants influencing response to warfarin",
        "pubmed",
        also=["monarch", "open_targets", "chembl"],
        category="edge",
    ),
    _q(
        "KRAS G12C inhibitor sotorasib in lung cancer",
        "clinicaltrials",
        also=["pubmed", "chembl", "open_targets"],
        category="edge",
    ),
    _q(
        "Drug targets and trial evidence for obesity",
        "open_targets",
        also=["pubmed", "clinicaltrials", "chembl"],
        category="edge",
    ),
    _q(
        "BRAF V600E mutations and targeted therapies in melanoma",
        "open_targets",
        also=["pubmed", "chembl", "clinicaltrials"],
        category="edge",
    ),
    _q(
        "CFTR modulators for cystic fibrosis treatment",
        "pubmed",
        also=["clinicaltrials", "chembl", "open_targets", "monarch"],
        category="edge",
    ),
    _q(
        "Tau-targeting therapies for Alzheimer's disease",
        "open_targets",
        also=["pubmed", "clinicaltrials", "chembl"],
        category="edge",
    ),
    _q(
        "Role of methylation in cancer and potential epigenetic drugs",
        "pubmed",
        also=["open_targets", "chembl", "biorxiv"],
        category="edge",
    ),
    _q(
        "Repurposing rapamycin for neurodegenerative disease",
        "pubmed",
        also=["clinicaltrials", "chembl", "open_targets", "biorxiv"],
        category="edge",
    ),
]


# ── Full dataset ─────────────────────────────────────────────────────────────

QUERIES: list[BenchmarkQuery] = (
    _OPENALEX
    + _PUBMED
    + _CLINICALTRIALS
    + _CHEMBL
    + _MONARCH
    + _OPEN_TARGETS
    + _BIORXIV
    + _EDGE
)


# Runtime invariants — these catch copy-paste errors when adding queries.
assert len(_OPENALEX) == 30, f"openalex: {len(_OPENALEX)} != 30"
assert len(_PUBMED) == 35, f"pubmed: {len(_PUBMED)} != 35"
assert len(_CLINICALTRIALS) == 30, f"clinicaltrials: {len(_CLINICALTRIALS)} != 30"
assert len(_CHEMBL) == 25, f"chembl: {len(_CHEMBL)} != 25"
assert len(_MONARCH) == 25, f"monarch: {len(_MONARCH)} != 25"
assert len(_OPEN_TARGETS) == 25, f"open_targets: {len(_OPEN_TARGETS)} != 25"
assert len(_BIORXIV) == 20, f"biorxiv: {len(_BIORXIV)} != 20"
assert len(_EDGE) == 10, f"edge: {len(_EDGE)} != 10"
assert len(QUERIES) == 200, f"total: {len(QUERIES)} != 200"


__all__ = ["BenchmarkQuery", "QUERIES"]
