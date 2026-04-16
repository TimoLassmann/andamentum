"""Visual smoke test — generates sample HTML files for manual inspection.

Run: uv run python -m andamentum.typeset.tests.visual_smoke
"""

from pathlib import Path

from andamentum.typeset import render

SAMPLE_DOC = [
    {"kind": "heading", "content": "Sample Report", "subtitle": "Typeset visual test", "meta": {"date": "2026-04-16", "author": "Test"}},
    {"kind": "callout", "content": "This is the key finding of the research.", "tone": "note"},
    {"kind": "items", "heading": "Key Facts", "entries": [
        {"label": "What was studied?", "body": "The effect of **metformin** on cardiovascular mortality."},
        {"label": "What did we find?", "body": "Mixed evidence across populations."},
        {"label": "Confidence", "body": "High (0.88)"},
    ]},
    {"kind": "prose", "content": "## Summary\n\nThe literature provides conflicting evidence regarding the effect. Some studies show benefit, others do not.\n\n### Sub-section\n\nAdditional detail with a [link](https://example.com) and some `inline code`.\n\n> A blockquote for emphasis.\n\n| Column A | Column B |\n|----------|----------|\n| Data 1   | Data 2   |\n| Data 3   | Data 4   |"},
    {"kind": "card", "content": "Metformin reduces cardiovascular mortality in T2D patients.", "badge": "supported", "refs": ["1", "2"], "details": "**Scope:** patients with T2D or coronary artery disease.\n\n**Verification:** scrutiny passed."},
    {"kind": "card", "content": "Metformin lowers CK-MB biomarker levels.", "badge": "challenged", "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12028114/"},
    {"kind": "reference", "content": "Taiwan nationwide cohort study shows reduced AMI incidence among metformin users.", "number": 1, "source": "https://www.nature.com/articles/s41598-025-13211-z", "badge": "supports"},
    {"kind": "reference", "content": "Meta-analysis of cardiac biomarkers found significant CK-MB reduction.", "number": 2, "source": "https://link.springer.com/article/10.1186/s12933-019-0900-7", "badge": "supports"},
    {"kind": "reference", "content": "Systematic review found no significant MACE reduction.", "number": 3, "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9539433/", "badge": "contradicts"},
    {"kind": "items", "variant": "right", "heading": "Education", "entries": [
        {"label": "2006", "body": "*PhD, Bioinformatics*\nStockholm University, Sweden"},
        {"label": "2001", "body": "*MSc, Applied Mathematics*\nUniversity of Adelaide"},
    ]},
    {"kind": "items", "variant": "left", "heading": "Teaching", "entries": [
        {"label": "2024", "body": "Deep Learning in Genomics. Australian Bioinformatics Conference."},
        {"label": "2023", "body": "Introduction to Bioinformatics. University of Western Australia."},
    ]},
    {"kind": "callout", "content": "**Warning:** This data has not been peer-reviewed.", "tone": "warning"},
    {"kind": "callout", "content": "All systems operational.", "tone": "success"},
    {"kind": "aside", "groups": {
        "Investigation": {"Evidence items": "37", "Claims": "2", "Uncertainties": "9"},
        "Confidence": {"Score": "0.88 HIGH", "Posterior P(Y)": "0.047"},
        "Model": {"LLM": "gemma4:26b", "Embeddings": "embeddinggemma"},
    }},
]

if __name__ == "__main__":
    out_dir = Path("/tmp/typeset_smoke")
    out_dir.mkdir(exist_ok=True)
    for style in ["article", "cv", "report"]:
        path = out_dir / f"sample_{style}.html"
        path.write_text(render(SAMPLE_DOC, style=style))
        print(f"Written: {path}")
    print(f"\nOpen in browser: open {out_dir}/sample_article.html")
