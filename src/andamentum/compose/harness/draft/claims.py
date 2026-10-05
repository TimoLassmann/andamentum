"""Claims: what each drafted sentence rests on, and the ledger that records it.

A section body is cut into sentences. For each one the facts its markers name
decide most of what is known without a model:

- a marker on a fact from a result file makes it a ``result`` claim, and the
  numbers in the sentence that match a value of that file (at the written
  precision) are bound to it;
- a marker on a fact from a verified literature work makes it ``background``
  and the work is what gets cited;
- every other sentence (no marker, or markers on the user's own notes) gets its
  kind from one model call per section, with four answers: ``result``,
  ``method``, ``background`` or ``interpretation``. A sentence the model leaves
  unlabelled is treated as interpretation: the safe side is to ask the author.

An ``interpretation`` claim is a judgement the sources do not carry; it becomes
``needs-author`` in the ledger and an anchored comment in the document.

``claims.yaml`` in the project store is the ledger. It is shared with the
bindings code, which keeps entries it did not write untouched; this module does
the same for the entries it adds (a ``section`` field marks them; the bindings
code marks its own with a ``binding`` block).
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from andamentum.compose._call import call_model
from andamentum.compose.log import ActionLog, sha
from andamentum.compose.schemas import Fact, Source

from ..bindings.ledger import CLAIMS_FILE, NumberBinding
from ..bindings.results import Catalogue, ResultEntry
from ..bindings.written import parse_written, scale_for
from ..kernel import ProjectStore
from ..kernel.csl import typed_citations
from .body import TO_FILL, BodyParagraph
from .works import CitableWork

ClaimKind = Literal["result", "method", "background", "interpretation"]
NUMBER_TOKEN = re.compile(r"[-−+]?\d+(?:,\d{3})*(?:\.\d+)?%?|[-−+]?\.\d+%?")
_LEDGER_ID = re.compile(r"^d(\d+)$")

CLASSIFY_INSTRUCTIONS = """\
You label the sentences of one section of a document by what kind of claim each \
makes. Answer with one label per sentence, using its number. The labels:
- result: reports what the authors' own data or analysis showed.
- method: describes what was done or how it was done.
- background: states established knowledge or context from outside the \
authors' own work.
- interpretation: draws an inference, explains why, evaluates, recommends or \
predicts; a judgement the authors must own.
When a sentence mixes kinds, pick interpretation.
"""


class ClaimLabel(BaseModel):
    number: int
    kind: ClaimKind


class ClaimLabels(BaseModel):
    labels: list[ClaimLabel]


@dataclass(frozen=True)
class SourceInfo:
    """What the draft knows about one source: a literature work, a result file, or neither."""

    source: Source
    work: CitableWork | None = None
    results_file: str | None = None  # path relative to the results folder


@dataclass(frozen=True)
class NumberSpot:
    paragraph: int  # index into the section's paragraphs
    start: int
    end: int
    binding: NumberBinding


@dataclass(frozen=True)
class TypedCitation:
    """Citation-shaped text the drafter wrote itself, in the cleaned paragraph text.

    The harness inserts a citation only from a ``[[fN]]`` marker on a verified
    work; this text came from nowhere the harness checked.
    """

    paragraph: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Sentence:
    paragraph: int
    start: int
    end: int
    text: str
    fact_ids: tuple[str, ...]
    kind: ClaimKind
    numbers: tuple[NumberSpot, ...] = ()
    typed: tuple[str, ...] = ()


@dataclass
class SectionAnalysis:
    """Everything known about a section body's sentences."""

    sentences: list[Sentence] = field(default_factory=list)
    unknown_markers: list[str] = field(default_factory=list)
    ambiguous_numbers: list[str] = field(default_factory=list)
    typed: list[TypedCitation] = field(default_factory=list)

    def number_spots(self) -> list[NumberSpot]:
        return [spot for s in self.sentences for spot in s.numbers]


@dataclass(frozen=True)
class RawSentence:
    paragraph: int
    start: int
    end: int
    text: str
    fact_ids: tuple[str, ...]


LabelCache = dict[str, dict[int, ClaimKind]]


def raw_sentences(
    paragraphs: list[BodyParagraph], *, known: Mapping[str, Fact]
) -> tuple[list[RawSentence], list[str]]:
    """Sentences with the fact ids of the markers inside them, and unknown marker ids.

    Table rows are data, not claims, and are skipped.
    """
    found: list[RawSentence] = []
    unknown: list[str] = []
    for index, paragraph in enumerate(paragraphs):
        if paragraph.table_row:
            continue
        spans = paragraph.sentences()
        owners: dict[int, list[str]] = {i: [] for i in range(len(spans))}
        for cite in paragraph.cites:
            ids = [i for i in cite.fact_ids if i in known]
            unknown += [i for i in cite.fact_ids if i not in known and i not in unknown]
            target = next(
                (i for i, (s, e) in enumerate(spans) if s <= cite.at <= e), None
            )
            if target is None and spans:
                target = max(
                    (i for i, (s, _) in enumerate(spans) if s <= cite.at), default=0
                )
            if target is not None:
                owners[target] += ids
        for i, (start, end) in enumerate(spans):
            if TO_FILL.fullmatch(paragraph.text[start:end].strip()):
                continue  # a placeholder is a question, not a claim
            found.append(
                RawSentence(
                    paragraph=index,
                    start=start,
                    end=end,
                    text=paragraph.text[start:end],
                    fact_ids=tuple(dict.fromkeys(owners[i])),
                )
            )
    return found, unknown


def _fixed_kind(
    fact_ids: tuple[str, ...],
    *,
    facts: Mapping[str, Fact],
    infos: Mapping[str, SourceInfo],
) -> ClaimKind | None:
    """The kind the markers settle (result before background), or None."""
    cited = [infos[facts[i].source_id] for i in fact_ids if facts[i].source_id in infos]
    if any(info.results_file for info in cited):
        return "result"
    if any(info.work is not None for info in cited):
        return "background"
    return None


async def label_sentences(
    sentences: list[RawSentence],
    *,
    facts: Mapping[str, Fact],
    infos: Mapping[str, SourceInfo],
    model: str,
    cache: LabelCache,
    log: ActionLog | None,
) -> dict[int, ClaimKind]:
    """The kind of every sentence, by position in *sentences*.

    Sentences the markers settle never reach the model; the others are labelled
    in one call per distinct text (cached by the section's sentences).
    """
    kinds: dict[int, ClaimKind] = {}
    open_positions: list[int] = []
    for position, sentence in enumerate(sentences):
        fixed = _fixed_kind(sentence.fact_ids, facts=facts, infos=infos)
        if fixed is None:
            open_positions.append(position)
        else:
            kinds[position] = fixed
    if not open_positions:
        return kinds
    texts = [sentences[p].text for p in open_positions]
    key = sha("\n".join(texts))
    labelled: dict[int, ClaimKind] | None = cache.get(key)
    if labelled is None:
        message = "\n".join(f"{n}: {t}" for n, t in enumerate(texts, start=1))
        out = await call_model(
            model=model,
            instructions=CLASSIFY_INSTRUCTIONS,
            output_type=ClaimLabels,
            user_message=message,
            node="claims",
            log=log,
            role="judge",
        )
        labelled = {item.number: item.kind for item in out.labels}
        cache[key] = labelled
    for n, position in enumerate(open_positions, start=1):
        kind = labelled.get(n)
        if kind is None and log is not None:
            log.event(node="claims", event="label_missing", sentence=texts[n - 1][:80])
        kinds[position] = kind or "interpretation"
    return kinds


def _anchor_id(entry: ResultEntry, sentence: str, taken: set[str]) -> str:
    base = (
        "n-"
        + hashlib.sha1(f"{entry.file}|{entry.key}|{sentence}".encode()).hexdigest()[:8]
    )
    anchor = base
    count = 1
    while anchor in taken:
        count += 1
        anchor = f"{base}-{count}"
    taken.add(anchor)
    return anchor


def bind_numbers(
    sentence: RawSentence,
    *,
    paragraph: BodyParagraph,
    files: set[str],
    quotes: list[str],
    catalogue: Catalogue,
    taken: set[str],
) -> tuple[list[NumberSpot], list[str]]:
    """Bindings for the numbers of a result sentence, and notes on the ones left out.

    A number binds only to a value of a result file one of the sentence's facts
    came from, equal at the written precision. When several values of that file
    match, those whose stored text occurs in a fact's quote are preferred; if
    that still leaves more than one, the number is left unbound and reported:
    a wrong binding would be worse than none.
    """
    spots: list[NumberSpot] = []
    notes: list[str] = []
    fill = paragraph.to_fill()
    for token in NUMBER_TOKEN.finditer(sentence.text):
        start = sentence.start + token.start()
        end = sentence.start + token.end()
        if any(s <= start < e for s, e in fill):
            continue
        written = parse_written(token.group(0))
        if written is None:
            continue
        candidates = [
            (entry, scale)
            for entry in catalogue.entries
            if entry.file in files
            for scale in [scale_for(written, entry.value)]
            if scale is not None
        ]
        if len(candidates) > 1:
            quoted = [
                c for c in candidates if any(f"{c[0].value:g}" in q for q in quotes)
            ]
            candidates = quoted or candidates
        if not candidates:
            continue
        if len({entry.key for entry, _ in candidates}) > 1:
            notes.append(
                f"{token.group(0)!r} in {sentence.text[:60]!r} matches "
                f"{len(candidates)} values of the result files; not bound"
            )
            continue
        entry, scale = candidates[0]
        spots.append(
            NumberSpot(
                paragraph=sentence.paragraph,
                start=start,
                end=end,
                binding=NumberBinding(
                    anchor_id=_anchor_id(entry, sentence.text, taken),
                    sentence=sentence.text,
                    file=entry.file,
                    key=entry.key,
                    value=entry.value,
                    written=token.group(0),
                    scale=scale,
                    context=entry.context,
                ),
            )
        )
    return spots, notes


async def analyse_section(
    paragraphs: list[BodyParagraph],
    *,
    facts: Mapping[str, Fact],
    infos: Mapping[str, SourceInfo],
    catalogue: Catalogue | None,
    taken_anchors: set[str],
    model: str,
    cache: LabelCache,
    log: ActionLog | None,
) -> SectionAnalysis:
    """Sentences, their kinds, and the number bindings of one section body."""
    raw, unknown = raw_sentences(paragraphs, known=facts)
    kinds = await label_sentences(
        raw, facts=facts, infos=infos, model=model, cache=cache, log=log
    )
    analysis = SectionAnalysis(unknown_markers=unknown)
    for index, paragraph in enumerate(paragraphs):
        analysis.typed += [
            TypedCitation(index, start, end, paragraph.text[start:end])
            for start, end in typed_citations(paragraph.text)
        ]
    for position, sentence in enumerate(raw):
        spots: list[NumberSpot] = []
        if catalogue is not None and kinds[position] == "result":
            cited = [infos[facts[i].source_id] for i in sentence.fact_ids]
            files = {c.results_file for c in cited if c.results_file}
            quotes = [facts[i].quote for i in sentence.fact_ids]
            spots, notes = bind_numbers(
                sentence,
                paragraph=paragraphs[sentence.paragraph],
                files=files,
                quotes=quotes,
                catalogue=catalogue,
                taken=taken_anchors,
            )
            analysis.ambiguous_numbers += notes
        analysis.sentences.append(
            Sentence(
                paragraph=sentence.paragraph,
                start=sentence.start,
                end=sentence.end,
                text=sentence.text,
                fact_ids=sentence.fact_ids,
                kind=kinds[position],
                numbers=tuple(spots),
                typed=tuple(
                    t.text
                    for t in analysis.typed
                    if t.paragraph == sentence.paragraph
                    and sentence.start <= t.start < sentence.end
                ),
            )
        )
    return analysis


# ------------------------------------------------------------------- ledger


def evidence_for(
    fact_ids: tuple[str, ...],
    *,
    facts: Mapping[str, Fact],
    infos: Mapping[str, SourceInfo],
) -> list[dict[str, object]]:
    """Evidence pointers of a sentence: the facts, with source and quote."""
    pointers: list[dict[str, object]] = []
    for fact_id in fact_ids:
        fact = facts[fact_id]
        info = infos.get(fact.source_id)
        pointer: dict[str, object] = {
            "type": "fact",
            "fact_id": fact.id,
            "source": info.source.label if info else fact.source_id,
            "quote": fact.quote,
        }
        if info is not None and info.work is not None:
            pointer["work"] = info.work.key
        if info is not None and info.results_file:
            pointer["file"] = info.results_file
        pointers.append(pointer)
    return pointers


def _strength(*, has_fact: bool, has_number: bool, typed: bool = False) -> str:
    if typed:
        return "typed citation, not verified"
    if has_fact and has_number:
        return "quoted source, number bound to result file"
    if has_fact:
        return "quoted source"
    if has_number:
        return "number bound to result file"
    return "none"


def claim_entries(
    section: str,
    analysis: SectionAnalysis,
    *,
    facts: Mapping[str, Fact],
    infos: Mapping[str, SourceInfo],
    bound: Collection[str],
) -> list[dict[str, object]]:
    """Ledger entries (without ids) for the sentences of one section.

    *bound* holds the anchor ids that really made it into the document; a
    number the writer refused is not listed as bound.
    """
    entries: list[dict[str, object]] = []
    for sentence in analysis.sentences:
        evidence = evidence_for(sentence.fact_ids, facts=facts, infos=infos)
        anchors = [
            n.binding.anchor_id
            for n in sentence.numbers
            if n.binding.anchor_id in bound
        ]
        if sentence.typed:
            # A citation the drafter typed was not verified; the sentence is not
            # supported by the work it names, whatever else it quotes.
            status = "unsupported"
        elif sentence.kind == "interpretation":
            status = "needs-author"
        else:
            status = "supported" if evidence or anchors else "unsupported"
        entry: dict[str, object] = {
            "text": sentence.text,
            "kind": sentence.kind,
            "section": section,
            "evidence": evidence,
            "strength": _strength(
                has_fact=bool(evidence),
                has_number=bool(anchors),
                typed=bool(sentence.typed),
            ),
            "status": status,
            "approved_by": None,
            "anchors": anchors,
        }
        entries.append(entry)
    return entries


def record_claims(store: ProjectStore, entries: list[dict[str, object]]) -> int:
    """Append *entries* to ``claims.yaml`` with fresh ids ``d1``, ``d2``, ...

    Every entry already in the file (this module's from an earlier run or the
    bindings code's) is kept as it is. Returns the number written.
    """
    path = store.path / CLAIMS_FILE
    existing: list[dict[str, object]] = []
    if path.is_file():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        claims = data.get("claims") if isinstance(data, dict) else None
        if not isinstance(claims, list):
            raise ValueError(f"{path} must hold a mapping with a 'claims' list")
        existing = claims
    taken = [
        int(m.group(1))
        for claim in existing
        if (m := _LEDGER_ID.match(str(claim.get("id", ""))))
    ]
    next_id = max(taken, default=0) + 1
    added = [{"id": f"d{next_id + i}", **entry} for i, entry in enumerate(entries)]
    text = yaml.safe_dump(
        {"claims": existing + added}, sort_keys=False, allow_unicode=True
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return len(added)
