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
    # Each planted problem is found when a problem of its kind lands on its
    # span; every other problem is a false alarm.
    found_planted = 0
    explained: set[int] = set()
    for expected in case["planted"]:
        span = (expected["start"], expected["end"])
        for index, problem in enumerate(report.problems):
            if index not in explained and problem.kind == expected["kind"] and _overlaps((problem.start, problem.end), span):
                explained.add(index)
                found_planted += 1
                break
    by_model = sum(1 for j in report.judged if j.reason == "model")
    unjudged = sum(1 for j in report.judged if j.reason.startswith("not judged"))
    return {
        "style": case["style"],
        "expected": case["expected_style"],
        "read_as": report.style.style if report.style else "none",
        "decided_by": report.style.decided_by if report.style else "none",
        "recall": f"{found}/{len(truth)}",
        "precision": f"{real}/{len(accepted)}",
        "planted": f"{found_planted}/{len(case['planted'])}",
        "false_alarms": [p.kind for i, p in enumerate(report.problems) if i not in explained],
        "model_judged": by_model,
        "unjudged": unjudged,
    }


async def score_pdf(case: dict, directory: Path, model: str | None, *, soffice: str) -> dict:
    """The document as a PDF, read through whetstone's real path."""
    import subprocess

    pdf_dir = directory / "pdf"
    pdf_dir.mkdir(exist_ok=True)
    pdf = pdf_dir / case["file"].replace(".docx", ".pdf")
    if not pdf.exists():
        subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(pdf_dir), str(directory / case["file"])],
            check=True,
            capture_output=True,
        )
    document = await open_source(str(pdf))
    report = await check_references(
        document.text,
        sections=[(s.title, s.start, s.end) for s in document.sections],
        hints=citation_hints(document.docx_path, expected_text=document.text),
        model=model,
        lookup=False,
    )
    expected = sorted(p["kind"] for p in case["planted"])
    found = sorted(p.kind for p in report.problems)
    remaining = list(found)
    hits = 0
    for kind in expected:
        if kind in remaining:
            remaining.remove(kind)
            hits += 1
    read_as = report.style.style if report.style else "none"
    wanted = case["expected_style"]
    ok = (read_as == wanted or (wanted == "unsupported" and read_as == "undecided")) and not remaining
    return {
        "style": case["style"],
        "read_as": read_as,
        "planted": f"{hits}/{len(expected)}",
        "other": remaining,
        "unjudged": sum(1 for j in report.judged if j.reason.startswith("not judged")),
        "ok": ok,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--only", nargs="*", default=None, help="Style names to score.")
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Convert each document to PDF (LibreOffice) and read it through "
        "harvest, as a PDF submission would be. Scores the style read and the "
        "planted problems by kind; citation spans do not carry over.",
    )
    parser.add_argument("--soffice", default="soffice", help="LibreOffice binary for --pdf.")
    args = parser.parse_args()
    cases = json.loads((args.corpus / "manifest.json").read_text())
    if args.only:
        cases = [c for c in cases if c["style"] in args.only]
    if args.pdf:
        for case in cases:
            row = await score_pdf(case, args.corpus, args.model, soffice=args.soffice)
            print(
                f"{'OK ' if row['ok'] else 'BAD'} {row['style']:<34} read {row['read_as']:<20} "
                f"planted {row['planted']}  other problems {row['other']}  unjudged {row['unjudged']}"
            )
        return
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
