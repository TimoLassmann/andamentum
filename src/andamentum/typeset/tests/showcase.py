"""Showcase — generate sample documents for all three styles in HTML + PDF.

Run:
    uv run python -m andamentum.typeset.tests.showcase

Output:
    /tmp/typeset_showcase/
        article.html, article.pdf
        cv.html, cv.pdf
        report.html, report.pdf
"""

from pathlib import Path

from andamentum.typeset import render_to_file

OUTPUT_DIR = Path("/tmp/typeset_showcase")


# ── Article: epistemic-style research report ─────────────────────────────────

ARTICLE_DOC = [
    {
        "kind": "heading",
        "content": "What does the published literature say about metformin and cardiovascular mortality in type 2 diabetes?",
        "meta": {"date": "2026-04-15", "model": "gemma4:26b", "project": "metformin_smoke_test"},
    },
    {
        "kind": "callout",
        "content": "The evidence is mixed, showing reduced mortality in some populations but no significant effect on major cardiovascular events in others.",
    },
    {
        "kind": "items",
        "entries": [
            {"label": "What was studied?", "body": "Metformin's effect on cardiovascular mortality in type 2 diabetes."},
            {"label": "What did we find?", "body": "Mixed evidence — reduced mortality in some populations, no significant effect on MACE in others. 1 challenged by counter-evidence."},
            {"label": "How confident?", "body": "High (0.88). Posterior P(Y) = 0.0474"},
            {"label": "How thorough?", "body": "37 evidence sources examined, 4/6 verification checks passed."},
        ],
    },
    {
        "kind": "prose",
        "heading": "Summary",
        "content": (
            "**Research Question:** *What does the published literature say about metformin and "
            "cardiovascular mortality in type 2 diabetes?*\n\n"
            "**Evidence Sources:** 15 | **Claims Established:** 0 of 2\n\n"
            "The literature provides conflicting evidence regarding metformin's impact on "
            "cardiovascular mortality. Some meta-analyses and large-scale observational studies "
            "suggest that metformin reduces both all-cause and cardiovascular mortality, "
            "particularly in patients with coronary artery disease. For example, a nationwide "
            "cohort study in Taiwan reported a lower incidence of acute myocardial infarction "
            "among metformin users.\n\n"
            "Other high-quality meta-analyses report that metformin does not significantly "
            "reduce the risk of major adverse cardiovascular events, including stroke or AMI, "
            "in the general type 2 diabetes population."
        ),
    },
    {
        "kind": "prose",
        "heading": "Findings",
        "content": "The investigation produced 2 distinct findings, each traced to specific evidence sources. Findings are ordered by strength of support.",
    },
    {
        "kind": "card",
        "content": "Metformin use is significantly associated with a reduction in both cardiovascular-related mortality and all-cause mortality in individuals with type 2 diabetes or coronary artery disease.",
        "badge": "supported",
        "refs": ["2", "10"],
        "details": "**Scope:** Patients with type 2 diabetes or coronary artery disease.\n\n**Verification:** Scrutiny: pass.",
    },
    {
        "kind": "card",
        "content": "Metformin use is associated with significantly lower levels of the cardiac biomarker CK-MB compared to non-metformin use.",
        "badge": "challenged",
        "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12028114/",
        "details": "**Scope:** In clinical trials comparing metformin use to non-metformin use.\n\n**Verification:** Scrutiny: pass. Adversarial: counter-evidence search found strong opposition (balance: 0.02). This claim was demoted after adversarial challenge.",
    },
    {
        "kind": "prose",
        "heading": "Sources",
        "content": "#### Supporting",
    },
    {
        "kind": "reference",
        "number": 1,
        "content": "The meta-analysis reports a statistically significant reduction in pooled CK-MB levels (SMD -0.15, P = 0.04) for metformin use compared to non-metformin use.",
        "source": "https://link.springer.com/article/10.1186/s12933-019-0900-7",
        "badge": "supports",
    },
    {
        "kind": "reference",
        "number": 2,
        "content": "The evidence explicitly states that metformin treatment is associated with a significant reduction in CVD-related mortality, which directly supports a key component of the claim.",
        "source": "https://the.evidencejournals.com/index.php/j/article/view/3",
        "badge": "supports",
    },
    {
        "kind": "reference",
        "number": 3,
        "content": "The meta-analysis found that metformin use was not significantly associated with a reduced risk of major cardiovascular events (such as MACE, stroke, or AMI) in patients with type 2 diabetes.",
        "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9539433/",
        "badge": "contradicts",
    },
    {
        "kind": "aside",
        "groups": {
            "Investigation": {"Evidence items": "37", "Claims": "2", "Uncertainties": "9"},
            "Confidence": {"Score": "0.88 HIGH", "Posterior P(Y)": "0.047"},
            "Model": {"LLM": "gemma4:26b", "Embeddings": "embeddinggemma"},
        },
    },
]


# ── CV: academic curriculum vitae ────────────────────────────────────────────

CV_DOC = [
    {
        "kind": "heading",
        "content": "Timo Lassmann",
        "subtitle": "Telethon Kids Institute | Perth, Australia",
        "meta": "Publications: 85 | Citations: 4,200 | h-index: 32",
    },
    {"kind": "prose", "heading": "Education", "content": ""},
    {
        "kind": "items",
        "variant": "right",
        "entries": [
            {"label": "2006", "body": "*PhD, Bioinformatics*\nStockholm University, Sweden"},
            {"label": "2001", "body": "*MSc, Applied Mathematics*\nUniversity of Adelaide, Australia"},
            {"label": "1999", "body": "*BSc (Hons), Biochemistry*\nUniversity of Adelaide, Australia"},
        ],
    },
    {"kind": "prose", "heading": "Positions", "content": ""},
    {
        "kind": "items",
        "variant": "right",
        "entries": [
            {"label": "2019\u2013present", "body": "**Head, Computational Biology**\nTelethon Kids Institute, Perth, Australia"},
            {"label": "2014\u20132019", "body": "**Senior Research Fellow**\nTelethon Kids Institute, Perth, Australia"},
            {"label": "2009\u20132014", "body": "**Research Scientist**\nOMICS Science Center, RIKEN, Yokohama, Japan"},
            {"label": "2006\u20132009", "body": "**Postdoctoral Fellow**\nKarolinska Institutet, Stockholm, Sweden"},
        ],
    },
    {"kind": "prose", "heading": "Selected Publications", "content": ""},
    {"kind": "reference", "number": 85, "group": "2025", "content": "**Lassmann T**, et al. Andamentum: composable agentic systems for scientific automation. *Nature Methods* (in preparation)."},
    {"kind": "reference", "number": 84, "group": "2025", "content": "Smith J, **Lassmann T**. Deep epistemic verification of biomedical claims. *Bioinformatics* **41**(2), 234\u2013241."},
    {"kind": "reference", "number": 83, "group": "2024", "content": "**Lassmann T**, Jones A. MAP-Elites for antibody optimization: a computational framework. *PLOS Computational Biology* **20**(5), e1012345."},
    {"kind": "reference", "number": 82, "group": "2024", "content": "Chen L, **Lassmann T**, et al. Single-cell multi-omics reveals immune dysregulation in rare disease. *Cell Reports* **43**(3), 113987."},
    {"kind": "reference", "number": 81, "group": "2023", "content": "**Lassmann T**. Kalign 3: multiple sequence alignment of large datasets. *Bioinformatics* **36**(6), 1928\u20131929.", "badge": "cited 340"},
    {"kind": "reference", "number": 80, "group": "2023", "content": "Park S, **Lassmann T**, et al. Promoter-level transcription atlas of the developing human brain. *Science* **370**(6520), eabc5765."},
    {
        "kind": "prose",
        "heading": "Grants",
        "content": (
            "| Grant | Year | Amount |\n"
            "|-------|------|--------|\n"
            "| ARC Discovery: Computational epistemic frameworks for scientific automation | 2024\u20132027 | A$450,000 |\n"
            "| NHMRC Investigator Grant: Genomics of rare childhood disease | 2021\u20132026 | A$1,200,000 |\n"
            "| Telethon\u2013Perth Children's Hospital Research Fund | 2020\u20132023 | A$380,000 |\n"
            "| Channel 7 Telethon Trust: Bioinformatics platform for rare disease | 2019\u20132022 | A$250,000 |"
        ),
    },
    {"kind": "prose", "heading": "Supervision", "content": ""},
    {
        "kind": "items",
        "variant": "right",
        "entries": [
            {"label": "2022\u2013present", "body": "**Dr. Jane Chen** (Postdoc) Single-cell multi-omics in rare disease"},
            {"label": "2021\u20132024", "body": "**Alex Kumar** (PhD) Machine learning for variant pathogenicity prediction"},
            {"label": "2020\u20132023", "body": "**Sarah Park** (PhD) Promoter-level transcription in neurodevelopment"},
        ],
    },
    {"kind": "prose", "heading": "Teaching", "content": ""},
    {
        "kind": "items",
        "variant": "left",
        "entries": [
            {"label": "2024", "body": "Deep Learning in Genomics. Australian Bioinformatics Conference."},
            {"label": "2023", "body": "Introduction to Bioinformatics. University of Western Australia."},
            {"label": "2022", "body": "Computational Biology Workshop. Telethon Kids Institute."},
            {"label": "2021", "body": "Advanced Sequence Analysis. Perth Biomedical Summer School."},
        ],
    },
    {"kind": "prose", "heading": "Awards", "content": ""},
    {
        "kind": "items",
        "variant": "right",
        "entries": [
            {"label": "2023", "body": "WA Premier's Science Award \u2014 Early Career Scientist"},
            {"label": "2020", "body": "Telethon Kids Institute Director's Prize for Research Excellence"},
            {"label": "2006", "body": "Best PhD Thesis, Stockholm University Faculty of Science"},
        ],
    },
    {
        "kind": "prose",
        "heading": "Software",
        "content": (
            "- **[Kalign](https://github.com/TimoLassmann/kalign)** \u2014 Fast multiple sequence alignment (C, 340+ citations)\n"
            "- **[Andamentum](https://github.com/andamentum)** \u2014 Composable agentic systems for scientific automation (Python)\n"
            "- **[TagDust](https://github.com/TimoLassmann/tagdust)** \u2014 HTS read demultiplexer with error correction (C)"
        ),
    },
]


# ── Report: blue technical report ────────────────────────────────────────────

REPORT_DOC = [
    {
        "kind": "heading",
        "content": "Semantic Provider Routing Benchmark",
        "subtitle": "Andamentum epistemic module \u2014 routing accuracy evaluation",
        "meta": {"date": "2026-04-15", "version": "v1.0", "author": "Timo Lassmann"},
    },
    {
        "kind": "callout",
        "content": "Top-3 recall reached **97.5%** across 200 labeled queries, replacing a brittle 16-keyword lookup table with embedding-based semantic routing.",
        "tone": "success",
    },
    {
        "kind": "items",
        "entries": [
            {"label": "Objective", "body": "Replace keyword-based provider selection with semantic similarity routing using embedding cosine distance."},
            {"label": "Method", "body": "200 labeled research queries across 7 biomedical and general-academic evidence providers, evaluated with top-1/top-3/MRR metrics."},
            {"label": "Result", "body": "97.5% top-3 recall, 84.5% permissive top-1, MRR 0.820."},
        ],
    },
    {
        "kind": "prose",
        "heading": "Background",
        "content": (
            "The epistemic module routes research questions to evidence providers (PubMed, "
            "ClinicalTrials, ChEMBL, Monarch, OpenTargets, bioRxiv, OpenAlex). The previous "
            "implementation used a hand-maintained keyword table (`DOMAIN_PROVIDER_MAP`) with "
            "16 keyword\u2192provider entries. This was brittle, hard to extend, and could not "
            "handle queries outside its vocabulary.\n\n"
            "The new semantic router embeds provider descriptions and ranks them by cosine "
            "similarity to the query embedding, using the same Ollama backend already present "
            "in the pipeline."
        ),
    },
    {
        "kind": "prose",
        "heading": "Results",
        "content": (
            "| Category | n | Top-1 strict | Top-1 permissive | Top-3 recall | MRR |\n"
            "|----------|---|-------------|-----------------|-------------|-----|\n"
            "| bioRxiv | 20 | 95.0% | 95.0% | 100.0% | 0.975 |\n"
            "| ChEMBL | 25 | 96.0% | 96.0% | 100.0% | 0.980 |\n"
            "| ClinicalTrials | 30 | 100.0% | 100.0% | 100.0% | 1.000 |\n"
            "| Edge cases | 10 | 40.0% | 100.0% | 100.0% | 0.524 |\n"
            "| Monarch | 25 | 96.0% | 96.0% | 100.0% | 0.980 |\n"
            "| OpenTargets | 25 | 84.0% | 84.0% | 100.0% | 0.920 |\n"
            "| OpenAlex | 30 | 63.3% | 63.3% | 93.3% | 0.787 |\n"
            "| PubMed | 35 | 17.1% | 62.9% | 91.4% | 0.390 |\n"
            "| **Total** | **200** | **73.5%** | **84.5%** | **97.5%** | **0.820** |"
        ),
    },
    {
        "kind": "callout",
        "content": "ClinicalTrials achieved **100% accuracy** across all metrics \u2014 the router perfectly distinguishes trial-specific queries from other biomedical questions.",
        "tone": "info",
    },
    {
        "kind": "prose",
        "heading": "Confusion Analysis",
        "content": (
            "The remaining 5% of top-3 misses are concentrated in two categories:\n\n"
            "- **PubMed** (91.4% top-3): PubMed's broad biomedical scope overlaps with "
            "every specialized provider. The router correctly picks the more specific "
            "provider (OpenTargets for target questions, ChEMBL for compound questions) "
            "but PubMed falls out of top-3 in 3 cases.\n"
            "- **OpenAlex** (93.3% top-3): Two general-academic queries about biology-adjacent "
            "topics (coral reef ecosystems, population genetics) route to bioRxiv instead. "
            "These are legitimately multi-provider queries.\n\n"
            "> The `min_score` threshold was calibrated at 0.15 based on OpenAlex's score "
            "distribution (\u03bc=0.178, \u03c3=0.057). This ensures general-academic queries "
            "clear the gate while off-topic queries fall back to web search."
        ),
    },
    {
        "kind": "card",
        "content": "Semantic routing achieves 97.5% top-3 recall and 84.5% permissive top-1 accuracy, replacing a brittle keyword table with zero manual maintenance.",
        "badge": "approved",
        "details": "**Method:** 200 labeled queries, 7 providers, embedding cosine similarity via embeddinggemma:latest.\n\n**Gate:** `min_score=0.15`, `top_k=3`, `web_search` always appended.\n\n**Regression guard:** `PASS_THRESHOLD_TOP_3_RECALL = 0.95` in pytest benchmark.",
    },
    {
        "kind": "callout",
        "content": "**Next steps:** Per-provider query formulation tuning \u2014 the biomedical APIs (Monarch, ClinicalTrials, OpenTargets) returned only 1 item each despite correct routing, suggesting the `epistemic_formulate_query` agent needs API-specific guidance.",
        "tone": "warning",
    },
    {
        "kind": "aside",
        "groups": {
            "Benchmark": {"Queries": "200", "Providers": "7", "Runs": "4"},
            "Metrics": {"Top-3 recall": "97.5%", "Top-1 permissive": "84.5%", "MRR": "0.820"},
            "Config": {"Embedding model": "embeddinggemma:latest", "min_score": "0.15", "top_k": "3"},
        },
    },
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    docs = {
        "article": (ARTICLE_DOC, "article", ""),
        "cv": (CV_DOC, "cv", "April 2026"),
        "report": (REPORT_DOC, "report", "Andamentum \u2014 Confidential"),
    }

    for name, (doc, style, footer) in docs.items():
        # HTML
        html_path = OUTPUT_DIR / f"{name}.html"
        render_to_file(doc, html_path, style=style, footer=footer)
        print(f"  HTML: {html_path}")

        # PDF
        try:
            from andamentum.typeset import render_pdf

            pdf_path = OUTPUT_DIR / f"{name}.pdf"
            render_pdf(doc, pdf_path, style=style, footer=footer)
            print(f"  PDF:  {pdf_path}")
        except ImportError:
            print("  PDF:  skipped (weasyprint not installed)")

    print(f"\nOpen all HTML: open {OUTPUT_DIR}/*.html")
    print(f"Open all PDF:  open {OUTPUT_DIR}/*.pdf")


if __name__ == "__main__":
    main()
