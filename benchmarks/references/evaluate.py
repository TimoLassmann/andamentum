"""Score the reference checker on the style test set (see build_corpus.py).

For each document: was the style read correctly; what share of the real
citations were accepted (recall) and what share of the accepted ones are real
(precision); were the three planted errors found; and how many other problems
were reported (false alarms). No Crossref lookup — this measures reading.

    uv run python benchmarks/references/evaluate.py CORPUS_DIR [--model MODEL]

With no --model the checker decides from evidence alone and reports what it
cannot settle. With a model (e.g. openai:gpt-5.4-nano, ollama:gemma4:26b-nvfp4)
the undecided cases go to it. Model calls are strictly sequential.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from andamentum.references import check_references
from andamentum.whetstone.docx.citation_hints import citation_hints
from andamentum.whetstone.reviewed_document import open_source


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


async def score(case: dict, directory: Path, model: str | None) -> dict:
    document = await open_source(str(directory / case["file"]))
    assert document.text == case["text"], f"{case['style']}: projection differs from the manifest"
    report = await check_references(
        document.text,
        sections=[(s.title, s.start, s.end) for s in document.sections],
        hints=citation_hints(document.docx_path, expected_text=document.text),
        model=model,
        lookup=False,
    )
    truth = [tuple(span) for span in case["citations"]]
    accepted = [(c.start, c.end) for c in report.citations]
    found = sum(1 for t in truth if any(_overlaps(t, a) for a in accepted))
    real = sum(1 for a in accepted if any(_overlaps(a, t) for t in truth))
    planted = case["planted"]
    hits: dict[str, bool] = {}
    explained: set[int] = set()
    for index, problem in enumerate(report.problems):
        span = (problem.start, problem.end)
        if "missing_citation" in planted and problem.kind == "cited_not_listed" and _overlaps(span, tuple(planted["missing_citation"])):
            hits["missing"] = True
            explained.add(index)
        if "uncited_entry_start" in planted and problem.kind == "listed_not_cited" and problem.start <= planted["uncited_entry_start"] + 8 and problem.end > planted["uncited_entry_start"]:
            hits["uncited"] = True
            explained.add(index)
        if "year_mismatch_citation" in planted and problem.kind == "year_mismatch":
            hits["year"] = True
            explained.add(index)
    expected = ["missing", "uncited"] + (["year"] if "year_mismatch_citation" in planted else [])
    by_model = sum(1 for j in report.judged if j.reason == "model")
    unjudged = sum(1 for j in report.judged if j.reason.startswith("not judged"))
    return {
        "style": case["style"],
        "expected": case["expected_style"],
        "read_as": report.style.style if report.style else "none",
        "decided_by": report.style.decided_by if report.style else "none",
        "recall": f"{found}/{len(truth)}",
        "precision": f"{real}/{len(accepted)}",
        "planted": f"{sum(hits.get(k, False) for k in expected)}/{len(expected)}",
        "false_alarms": [p.kind for i, p in enumerate(report.problems) if i not in explained],
        "model_judged": by_model,
        "unjudged": unjudged,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    cases = json.loads((args.corpus / "manifest.json").read_text())
    rows = []
    for case in cases:  # sequential: never two model calls at once
        rows.append(await score(case, args.corpus, args.model))
    for row in rows:
        ok = "OK " if row["read_as"] == row["expected"] or (row["expected"] == "unsupported" and row["read_as"] == "undecided") else "BAD"
        print(
            f"{ok} {row['style']:<30} {row['expected']:<20} read {row['read_as']:<20} "
            f"by {row['decided_by']:<8} recall {row['recall']:<5} precision {row['precision']:<5} "
            f"planted {row['planted']}  model {row['model_judged']}  unjudged {row['unjudged']}  "
            f"false {row['false_alarms']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
