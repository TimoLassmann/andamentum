"""Benchmark runner: the harness arm against a one-call baseline arm.

    uv run python -m andamentum.compose.harness.bench.run --offline --seed 0
    uv run python -m andamentum.compose.harness.bench.run --offline --baseline-model M

Both arms read the same planted manuscripts (``bench/planted.py``) and are
scored against the same ground truth.

- **Harness arm**: ``run_check`` on each file. No model. ``--offline`` answers
  every Crossref lookup and URL probe from the recorded fixtures
  (``bench/fixture_http.py``); ``--live`` makes the real requests.
- **Baseline arm** (only with ``--baseline-model``): one model call per file,
  "find all errors in this manuscript", given the paragraph-numbered text and
  asked to label each finding with one of the planted kinds. The model has no
  network and no document structure, which is what the arm stands for.

Scoring. A diagnostic or finding is mapped to a defect kind (harness: by check
id and the diagnostic's ``data["kind"]``; baseline: by the model's label). A
defect is **found** when some diagnostic of its kind lies in one of its
paragraphs (the reference entry and the in-text citation both count for a
bad DOI). Per-kind **recall** is found over planted. A **false positive** is an
error or warning (info is advice, not a finding) that matches no defect: any
on a clean file, and on a defective file any whose kind and paragraph match no
planted defect. Cost is the tokens the arm's model calls report and wall-clock
seconds; the harness arm adds the network calls it made.

Not a statement about real manuscripts: the documents are synthetic and the
defects are the eight planted kinds, so recall here measures whether the
harness finds what it was built to find, not how often a real error is one of
those.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
import time
from pathlib import Path
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from andamentum.core.agents import run_agent_with_usage

from ..check import CheckParams, run_check
from ..docx.reader import open_docx
from ..kernel import Diagnostic, DocumentView
from .fixture_http import fixture_transport, scheme_only_url_check
from .planted import KINDS, generate

Kind = Literal[
    "fabricated_doi",
    "retracted_doi",
    "broken_url",
    "dangling_crossref",
    "malformed_citation_json",
    "abstract_number_missing",
    "inconsistent_statistic",
    "missing_figure",
]

#: (check id, diagnostic ``data["kind"]``) -> the defect kind it reports. A
#: check id with ``None`` maps every diagnostic of that check.
HARNESS_KINDS: dict[tuple[str, str | None], Kind] = {
    ("citations", "unresolved"): "fabricated_doi",
    ("citations", "details_differ"): "fabricated_doi",
    ("citations", "retraction"): "retracted_doi",
    ("citations", "url"): "broken_url",
    ("crossrefs", "dangling_target"): "dangling_crossref",
    ("crossrefs", "unparseable_field"): "malformed_citation_json",
    ("crossrefs", "missing_caption"): "missing_figure",
    ("abstract_body", None): "abstract_number_missing",
    ("statistics", None): "inconsistent_statistic",
}


class _Loose(BaseModel):
    model_config = ConfigDict(extra="ignore")


class TruthLocation(_Loose):
    paragraph_index: int


class TruthDefect(_Loose):
    id: str
    kind: str
    locations: list[TruthLocation]


class Truth(_Loose):
    file: str
    clean: bool
    defects: list[TruthDefect]


class Finding(BaseModel):
    """One diagnostic or baseline finding, reduced to what scoring needs."""

    kind: str | None  # None: maps to no defect kind
    paragraph: int | None
    severity: str


class KindScore(BaseModel):
    planted: int = 0
    found: int = 0

    @property
    def recall(self) -> float | None:
        return self.found / self.planted if self.planted else None


class ArmScore(BaseModel):
    name: str
    per_kind: dict[str, KindScore] = Field(default_factory=dict)
    false_positives_clean: int = 0
    false_positives_defective: int = 0
    findings: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    network_calls: int = 0
    seconds: float = 0.0

    @property
    def planted(self) -> int:
        return sum(k.planted for k in self.per_kind.values())

    @property
    def found(self) -> int:
        return sum(k.found for k in self.per_kind.values())

    @property
    def recall(self) -> float:
        return self.found / self.planted if self.planted else 0.0

    @property
    def false_positives(self) -> int:
        return self.false_positives_clean + self.false_positives_defective


class BenchResult(BaseModel):
    mode: Literal["offline", "live"]
    seed: int
    files: int
    clean_files: int
    defects: int
    harness: ArmScore
    baseline: ArmScore | None


# ------------------------------------------------------------------- scoring


def harness_findings(
    diagnostics: list[Diagnostic], view: DocumentView
) -> list[Finding]:
    index = {p.id: p.index for p in view.paragraphs}
    found: list[Finding] = []
    for diagnostic in diagnostics:
        if diagnostic.severity == "info":
            continue
        data_kind = diagnostic.data.get("kind")
        kind = HARNESS_KINDS.get(
            (diagnostic.check, data_kind if isinstance(data_kind, str) else None)
        ) or HARNESS_KINDS.get((diagnostic.check, None))
        found.append(
            Finding(
                kind=kind,
                paragraph=index.get(diagnostic.paragraph_id or ""),
                severity=diagnostic.severity,
            )
        )
    return found


def score_file(truth: Truth, findings: list[Finding], score: ArmScore) -> None:
    """Add one file's result to *score*."""
    for defect in truth.defects:
        entry = score.per_kind.setdefault(defect.kind, KindScore())
        entry.planted += 1
        places = {loc.paragraph_index for loc in defect.locations}
        if any(f.kind == defect.kind and f.paragraph in places for f in findings):
            entry.found += 1
    planted = {
        (d.kind, loc.paragraph_index) for d in truth.defects for loc in d.locations
    }
    wrong = sum(1 for f in findings if (f.kind, f.paragraph) not in planted)
    score.findings += len(findings)
    if truth.clean:
        score.false_positives_clean += wrong
    else:
        score.false_positives_defective += wrong


def _empty_score(name: str) -> ArmScore:
    return ArmScore(name=name, per_kind={k: KindScore() for k in KINDS})


# ----------------------------------------------------------------- the arms


async def harness_arm(
    files: list[tuple[Path, Truth]],
    *,
    work: Path,
    offline: bool,
    mailto: str | None,
) -> ArmScore:
    score = _empty_score("harness")
    for docx, truth in files:
        copy = work / docx.name
        shutil.copy(docx, copy)
        params = CheckParams(file=copy, mailto=mailto)
        started = time.monotonic()
        run = await run_check(
            params,
            transport=fixture_transport() if offline else None,
            url_check=scheme_only_url_check if offline else None,
        )
        score.seconds += time.monotonic() - started
        score.network_calls += run.report.network_calls
        score.input_tokens += run.report.input_tokens
        score.output_tokens += run.report.output_tokens
        score_file(
            truth, harness_findings(run.report.diagnostics, open_docx(copy)), score
        )
    return score


class BaselineFinding(BaseModel):
    paragraph: int
    category: Literal[
        "fabricated_doi",
        "retracted_doi",
        "broken_url",
        "dangling_crossref",
        "malformed_citation_json",
        "abstract_number_missing",
        "inconsistent_statistic",
        "missing_figure",
        "other",
    ]
    quote: str
    problem: str


class BaselineFindings(BaseModel):
    findings: list[BaselineFinding]


BASELINE_INSTRUCTIONS = (
    "You are checking a research manuscript for errors. Find every error you "
    "can. Each paragraph starts with its number in square brackets. For each "
    "error give the paragraph number, a category, the exact quote and the "
    "problem. Categories: fabricated_doi (a reference or DOI that does not "
    "exist), retracted_doi (a reference to a retracted work), broken_url (a "
    "URL that does not work), dangling_crossref (a cross-reference whose "
    "target is missing, e.g. 'Error! Reference source not found'), "
    "malformed_citation_json (citation field data that is corrupted), "
    "abstract_number_missing (a number in the abstract that the body does not "
    "contain), inconsistent_statistic (a reported p value that does not "
    "follow from the test statistic and degrees of freedom), missing_figure "
    "(a figure or table mentioned in the text that does not exist), other."
)


def numbered_text(view: DocumentView) -> str:
    return "\n".join(f"[{p.index}] {p.text}" for p in view.paragraphs)


async def baseline_arm(files: list[tuple[Path, Truth]], *, model: str) -> ArmScore:
    score = _empty_score("baseline")
    for docx, truth in files:
        view = open_docx(docx)
        started = time.monotonic()
        output, usage = await run_agent_with_usage(
            model,
            instructions=BASELINE_INSTRUCTIONS,
            output_type=BaselineFindings,
            user_message=numbered_text(view),
        )
        score.seconds += time.monotonic() - started
        score.input_tokens += usage.input_tokens
        score.output_tokens += usage.output_tokens
        findings = [
            Finding(
                kind=None if f.category == "other" else f.category,
                paragraph=f.paragraph,
                severity="warning",
            )
            for f in output.findings
        ]
        score_file(truth, findings, score)
    return score


# --------------------------------------------------------------------- table


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def table(result: BenchResult) -> str:
    base = result.baseline
    header = "| defect kind | planted | harness found | harness recall |"
    rule = "|---|---:|---:|---:|"
    if base is not None:
        header += " baseline found | baseline recall |"
        rule += "---:|---:|"
    lines = [header, rule]
    for kind in KINDS:
        h = result.harness.per_kind[kind]
        row = f"| {kind} | {h.planted} | {h.found} | {_pct(h.recall)} |"
        if base is not None:
            b = base.per_kind[kind]
            row += f" {b.found} | {_pct(b.recall)} |"
        lines.append(row)
    total = f"| **all** | {result.harness.planted} | {result.harness.found} | {_pct(result.harness.recall)} |"
    if base is not None:
        total += f" {base.found} | {_pct(base.recall)} |"
    lines.append(total)
    lines += ["", "| cost and noise | harness |" + (" baseline |" if base else "")]
    lines.append("|---|---:|" + ("---:|" if base else ""))

    def cost_row(label: str, get: Callable[[ArmScore], object]) -> str:
        cells = [str(get(result.harness))]
        if base is not None:
            cells.append(str(get(base)))
        return f"| {label} | " + " | ".join(cells) + " |"

    lines += [
        cost_row("findings (error or warning)", lambda a: a.findings),
        cost_row("false positives on clean files", lambda a: a.false_positives_clean),
        cost_row("false positives elsewhere", lambda a: a.false_positives_defective),
        cost_row("input tokens", lambda a: a.input_tokens),
        cost_row("output tokens", lambda a: a.output_tokens),
        cost_row("network calls", lambda a: a.network_calls),
        cost_row("seconds", lambda a: f"{a.seconds:.1f}"),
    ]
    head = (
        f"mode: {result.mode}; seed {result.seed}; {result.files} files "
        f"({result.clean_files} clean), {result.defects} planted defects\n\n"
    )
    return head + "\n".join(lines)


# ---------------------------------------------------------------------- main


def load_set(directory: Path) -> list[tuple[Path, Truth]]:
    pairs = []
    for docx in sorted(directory.glob("*.docx")):
        truth = Truth.model_validate_json(
            docx.with_suffix(".truth.json").read_text(encoding="utf-8")
        )
        pairs.append((docx, truth))
    if not pairs:
        raise FileNotFoundError(f"no planted .docx files in {directory}")
    return pairs


async def run_benchmark(
    files: list[tuple[Path, Truth]],
    *,
    work: Path,
    offline: bool,
    seed: int,
    baseline_model: str | None,
    mailto: str | None = None,
) -> BenchResult:
    harness = await harness_arm(files, work=work, offline=offline, mailto=mailto)
    baseline = (
        await baseline_arm(files, model=baseline_model) if baseline_model else None
    )
    return BenchResult(
        mode="offline" if offline else "live",
        seed=seed,
        files=len(files),
        clean_files=sum(t.clean for _, t in files),
        defects=sum(len(t.defects) for _, t in files),
        harness=harness,
        baseline=baseline,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m andamentum.compose.harness.bench.run",
        description="Harness arm against a one-call baseline on planted manuscripts.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--offline", action="store_true", help="answer Crossref and URLs from fixtures"
    )
    mode.add_argument("--live", action="store_true", help="make the real requests")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dir", type=Path, help="planted set to use (default: generate from --seed)"
    )
    parser.add_argument(
        "--work", type=Path, help="scratch folder (default: a temporary one)"
    )
    parser.add_argument(
        "--baseline-model", help="model id; omit to skip the baseline arm"
    )
    parser.add_argument("--mailto", help="contact address for Crossref (live mode)")
    parser.add_argument(
        "--json", type=Path, dest="json_out", help="write the result JSON"
    )
    args = parser.parse_args(argv)

    def go(root: Path) -> BenchResult:
        planted_dir = args.dir
        if planted_dir is None:
            planted_dir = root / "planted"
            generate(planted_dir, seed=args.seed)
        work = args.work or root / "work"
        work.mkdir(parents=True, exist_ok=True)
        return asyncio.run(
            run_benchmark(
                load_set(planted_dir),
                work=work,
                offline=args.offline,
                seed=args.seed,
                baseline_model=args.baseline_model,
                mailto=args.mailto,
            )
        )

    if args.work is not None:
        args.work.mkdir(parents=True, exist_ok=True)
        result = go(args.work)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            result = go(Path(tmp))
    print(table(result))
    if args.json_out is not None:
        args.json_out.write_text(result.model_dump_json(indent=2) + "\n", "utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
