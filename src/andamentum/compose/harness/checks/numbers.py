"""Numbers in the text against the project's result files (spec section 4, Tier 1).

Optional: it needs ``--results DIR`` and a model, and is skipped without them.
The split of work is the point: a model only *extracts* the numeric claims of a
paragraph (sentence, value as written, what it refers to); code does the
comparing. A claim is matched when some number in a result file rounds to the
written value at the written precision (``0.234`` matches ``23.4%`` and
``0.23``; ``23.4%`` also matches a stored ``23.4``). Unmatched claims are
warnings that list the nearest numbers found, so a person can see whether the
text or the file is out of date.

Result files read: ``.csv`` and ``.tsv`` (every cell that parses as a number)
and ``.json`` (every number in the tree). Derived values (a difference the text
computes from two stored numbers) are not found; that is a coverage limit, and
the warning is advisory.

An extracted sentence or value that does not occur in the paragraph is dropped
and counted in one info diagnostic: the model's output is never trusted for
placement. At most ``MAX_PARAGRAPHS`` paragraphs are examined per run (one
model call each, sequential); the info diagnostic says how many were left.
Paragraphs after a References or Bibliography heading are not examined.
Per-paragraph extractions are cached in the project store, keyed by the
paragraph text and the content of the result files.

The effective tier follows the kernel's promotion policy: advisory (tier 2)
until a measured precision record for this check says otherwise.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from andamentum.core.agents import run_agent_with_usage

from ..kernel import (
    CheckContext,
    CheckSpec,
    Diagnostic,
    DocumentView,
    Paragraph,
    register,
)
from ..kernel.checks import cache_key

MAX_PARAGRAPHS = 80
MAX_CANDIDATES = 3
_NAMESPACE = "numbers"
_SUFFIXES = {".csv": ",", ".tsv": "\t", ".json": None}
_REFERENCE_HEADINGS = {"references", "bibliography", "works cited", "reference list"}
_NUMBER = re.compile(r"[-−]?\d+(?:,\d{3})*(?:\.\d+)?|[-−]?\.\d+")

_INSTRUCTIONS = (
    "You read one paragraph of a research manuscript. List each numeric claim "
    "the authors make about their OWN results: means, percentages, counts, "
    "rates, p-values, effect sizes, confidence limits. Skip years, dates, "
    "citation numbers, figure and table numbers, and numbers attributed to "
    "other work. For each claim give the full sentence exactly as written, the "
    "number exactly as written (keep a % sign, decimals and a minus), and a "
    "few words saying what the number refers to. If there are none, return an "
    "empty list."
)


class NumericClaim(BaseModel):
    sentence: str
    value: str
    refers_to: str


class NumericClaims(BaseModel):
    claims: list[NumericClaim]


@dataclass(frozen=True)
class ResultNumber:
    value: float
    where: str  # file and position


def _to_float(text: str) -> float | None:
    cleaned = text.strip().replace("−", "-").replace(",", "").rstrip("%")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _walk_json(node: object, path: str, source: str, out: list[ResultNumber]) -> None:
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        out.append(ResultNumber(float(node), f"{source} {path or '$'}"))
    elif isinstance(node, dict):
        for key, child in node.items():
            _walk_json(child, f"{path}.{key}" if path else str(key), source, out)
    elif isinstance(node, list):
        for index, child in enumerate(node):
            _walk_json(child, f"{path}[{index}]", source, out)


def load_results(directory: Path) -> tuple[list[ResultNumber], list[str]]:
    """Every number in the result files under *directory*, and the digest parts."""
    if not directory.is_dir():
        raise FileNotFoundError(f"results directory not found: {directory}")
    results: list[ResultNumber] = []
    digests: list[str] = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in _SUFFIXES or not path.is_file():
            continue
        raw = path.read_bytes()
        digests.append(hashlib.sha256(raw).hexdigest())
        label = str(path.relative_to(directory))
        if path.suffix.lower() == ".json":
            try:
                _walk_json(json.loads(raw), "", label, results)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: not valid JSON ({exc})") from exc
            continue
        reader = csv.reader(
            raw.decode("utf-8").splitlines(), delimiter=_SUFFIXES[path.suffix.lower()]
        )
        header: list[str] = []
        for row_number, row in enumerate(reader, 1):
            if row_number == 1:
                header = row
            for column, cell in enumerate(row):
                number = _to_float(cell)
                if number is not None:
                    name = header[column] if column < len(header) else str(column)
                    results.append(
                        ResultNumber(
                            number, f"{label} row {row_number} column {name!r}"
                        )
                    )
    return results, digests


def _numeric_part(written: str) -> str:
    """The number of a value as the model extracted it, without a trailing unit.

    The model is told to give the number as written but often adds the unit
    (``3,310 mg``); code compares the number alone, keeping a ``%`` sign.
    """
    found = _NUMBER.search(written)
    if found is None:
        return written
    number = found.group(0)
    return number + "%" if written[found.end() :].lstrip().startswith("%") else number


def _decimals(written: str) -> int:
    text = written.strip().rstrip("%")
    return len(text.split(".")[1]) if "." in text else 0


def _matches(written: str, results: list[ResultNumber]) -> list[ResultNumber]:
    written = _numeric_part(written)
    claimed = _to_float(written)
    if claimed is None:
        return []
    tolerance = 0.5 * 10 ** -_decimals(written) + 1e-9
    percent = written.strip().endswith("%")
    found = []
    for result in results:
        values = [result.value] + ([result.value * 100] if percent else [])
        if any(abs(v - claimed) <= tolerance for v in values):
            found.append(result)
    return found


def _nearest(written: str, results: list[ResultNumber]) -> list[ResultNumber]:
    written = _numeric_part(written)
    claimed = _to_float(written)
    if claimed is None:
        return []
    percent = written.strip().endswith("%")

    def gap(result: ResultNumber) -> float:
        values = [result.value] + ([result.value * 100] if percent else [])
        return min(abs(v - claimed) for v in values)

    return sorted(results, key=gap)[:MAX_CANDIDATES]


def _examined(view: DocumentView) -> list[Paragraph]:
    chosen: list[Paragraph] = []
    for paragraph in view.paragraphs:
        if paragraph.heading_level is not None:
            if paragraph.text.strip().strip(":").lower() in _REFERENCE_HEADINGS:
                break
            continue
        if len(paragraph.text.split()) >= 5 and _NUMBER.search(paragraph.text):
            chosen.append(paragraph)
    return chosen


class NumbersCheck:
    spec = CheckSpec(
        id="numbers", tier=1, scope="paragraph", needs=["results", "model"]
    )
    version = "1"

    async def run(self, view: DocumentView, ctx: CheckContext) -> list[Diagnostic]:
        if ctx.results_dir is None or "model" not in ctx.models:
            raise ValueError("the numbers check needs --results and a model")
        results, digests = load_results(ctx.results_dir)
        if not results:
            raise ValueError(
                f"no numbers found in the result files under {ctx.results_dir}"
            )
        results_digest = hashlib.sha256("".join(digests).encode()).hexdigest()
        paragraphs = _examined(view)
        examined, left = paragraphs[:MAX_PARAGRAPHS], paragraphs[MAX_PARAGRAPHS:]
        out: list[Diagnostic] = []
        unlocated = 0
        for paragraph in examined:
            claims = await self._claims(paragraph, ctx, results_digest)
            for claim in claims:
                where = self._locate(paragraph, claim)
                if where is None:
                    unlocated += 1
                    continue
                if _matches(claim.value, results):
                    continue
                out.append(self._unmatched(paragraph, claim, where, results))
        if unlocated:
            out.append(
                self._note(
                    f"{unlocated} extracted numeric claim(s) did not occur in the text and were ignored."
                )
            )
        if left:
            out.append(
                self._note(
                    f"{len(left)} paragraph(s) with numbers were not examined "
                    f"(limit {MAX_PARAGRAPHS} per run)."
                )
            )
        return out

    async def _claims(
        self, paragraph: Paragraph, ctx: CheckContext, results_digest: str
    ) -> list[NumericClaim]:
        key = cache_key(
            self, version=self.version, inputs=[paragraph.text, results_digest]
        )
        if ctx.store is not None:
            cached = ctx.store.cache_get(_NAMESPACE, key)
            if cached is not None:
                return NumericClaims.model_validate({"claims": cached}).claims
        output, usage = await run_agent_with_usage(
            ctx.models["model"],
            instructions=_INSTRUCTIONS,
            output_type=NumericClaims,
            user_message=paragraph.text,
        )
        ctx.stats.add_usage(
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens
        )
        if ctx.store is not None:
            ctx.store.cache_put(
                _NAMESPACE, key, [c.model_dump() for c in output.claims]
            )
        return output.claims

    @staticmethod
    def _locate(paragraph: Paragraph, claim: NumericClaim) -> tuple[int, int] | None:
        """Offsets of the value inside its sentence, or None when either is absent."""
        sentence_at = paragraph.text.find(claim.sentence)
        if sentence_at < 0 or not claim.value.strip():
            return None
        inside = claim.sentence.find(claim.value)
        if inside < 0:
            return None
        start = sentence_at + inside
        return start, start + len(claim.value)

    def _unmatched(
        self,
        paragraph: Paragraph,
        claim: NumericClaim,
        where: tuple[int, int],
        results: list[ResultNumber],
    ) -> Diagnostic:
        nearest = _nearest(claim.value, results)
        listed = [f"{r.value:g} ({r.where})" for r in nearest]
        return Diagnostic(
            check=self.spec.id,
            tier=self.spec.tier,
            severity="warning",
            paragraph_id=paragraph.id,
            start=where[0],
            end=where[1],
            message=(
                f"{claim.value!r} ({claim.refers_to}) matches no number in the "
                "result files."
            ),
            evidence=[claim.sentence, *(f"nearest: {item}" for item in listed)],
            data={"value": claim.value, "refers_to": claim.refers_to},
        )

    def _note(self, message: str) -> Diagnostic:
        return Diagnostic(
            check=self.spec.id,
            tier=self.spec.tier,
            severity="info",
            paragraph_id=None,
            start=None,
            end=None,
            message=message,
            evidence=[],
        )


register(NumbersCheck())
