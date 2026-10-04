"""Evidence: ingest sources, extract quoted facts, verify the quotes.

``ingest_sources`` and ``check_quotes`` are deterministic; ``extract_facts``
is one model call per source. A fact survives only if its quote is found in
the source text, so every ``[[fN]]`` marker in a draft traces to real words.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from pydantic import BaseModel

from andamentum.harvest import extract
from andamentum.whetstone import check_confidentiality

from ._call import call_model
from .log import ActionLog
from .schemas import Brief, Fact, Source

MAX_FACTS_PER_SOURCE = 25
MAX_SOURCE_CHARS = 60000
TABULAR_SUFFIXES = (".csv", ".tsv")
SOURCES_SUBDIR = Path("evidence") / "sources"

EXTRACT_INSTRUCTIONS = """\
You extract facts from one source for a writer who will use them in a document.

Return up to 25 facts that matter for the stated purpose. For each fact give:
- statement: the fact in one plain sentence.
- quote: a verbatim span copied character for character from the source that \
shows the fact. Keep it short (one sentence or less). Never paraphrase the quote.

Skip anything you cannot back with a verbatim quote. Do not add outside \
knowledge. Prefer specifics (numbers, names, dates, decisions) over generalities.
"""


class RawFact(BaseModel):
    statement: str
    quote: str


class RawFacts(BaseModel):
    facts: list[RawFact]


def is_tabular(path: str) -> bool:
    return path.lower().endswith(TABULAR_SUFFIXES)


def stored_source_path(bundle_dir: str | Path, source: Source) -> Path:
    """Where ``ingest_sources`` keeps this source inside the bundle."""
    suffix = Path(source.path).suffix.lower() if is_tabular(source.path) else ".md"
    return Path(bundle_dir) / SOURCES_SUBDIR / f"{source.sha256[:8]}{suffix}"


def _title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
    return fallback


async def ingest_sources(
    paths: list[str],
    *,
    bundle_dir: str | Path,
    confirm_own_sources: bool = False,
) -> list[Source]:
    """Bring each file or URL into ``<bundle>/evidence/sources/`` as markdown.

    Tabular files (.csv, .tsv) are copied as-is. Each text goes through the
    confidentiality check before anything is stored; a hit raises unless
    ``confirm_own_sources`` is true. Idempotent: a file already stored under
    the same digest is not rewritten. Ids are ``src1``, ``src2`` ... in order.
    """
    target_dir = Path(bundle_dir) / SOURCES_SUBDIR
    sources: list[Source] = []
    for index, raw in enumerate(paths, start=1):
        is_url = raw.startswith(("http://", "https://"))
        if is_tabular(raw) and not is_url:
            data = Path(raw).read_bytes()
            text = data.decode("utf-8")
            digest = hashlib.sha256(data).hexdigest()
        else:
            text = await extract(raw)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if not confirm_own_sources:
            check_confidentiality(text)
        fallback = raw.rstrip("/").rsplit("/", 1)[-1] or raw
        source = Source(
            id=f"src{index}",
            path=raw,
            title=fallback if is_tabular(raw) else _title(text, fallback),
            sha256=digest,
            chars=len(text),
            kind="url" if is_url else "file",
            label=raw if is_url else Path(raw).name,
        )
        stored = stored_source_path(bundle_dir, source)
        if not stored.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            if is_tabular(raw) and not is_url:
                shutil.copyfile(raw, stored)
            else:
                tmp = stored.with_suffix(stored.suffix + ".tmp")
                tmp.write_text(text, encoding="utf-8")
                tmp.replace(stored)
        sources.append(source)
    return sources


def _normalise(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace and lowercase; map each output char to its source index."""
    out: list[str] = []
    index: list[int] = []
    pending_space = False
    for pos, ch in enumerate(text):
        if ch.isspace():
            pending_space = bool(out)
            continue
        if pending_space:
            out.append(" ")
            index.append(pos)
            pending_space = False
        for lowered in ch.lower():
            out.append(lowered)
            index.append(pos)
    return "".join(out), index


def find_quote(text: str, quote: str) -> tuple[int, int] | None:
    """Locate ``quote`` in ``text``: exact first, then whitespace- and case-normalised.

    Returns ``(start, end)`` offsets into ``text`` or ``None``.
    """
    if not quote.strip():
        return None
    start = text.find(quote)
    if start >= 0:
        return start, start + len(quote)
    norm_text, mapping = _normalise(text)
    norm_quote, _ = _normalise(quote)
    if not norm_quote:
        return None
    pos = norm_text.find(norm_quote)
    if pos < 0:
        return None
    start = mapping[pos]
    end = mapping[pos + len(norm_quote) - 1] + 1
    return start, end


def check_quotes(
    raw_facts: list[RawFact],
    *,
    source_id: str,
    text: str,
    first_number: int,
) -> tuple[list[Fact], int]:
    """Keep the facts whose quote is in ``text``; return them and the dropped count."""
    facts: list[Fact] = []
    dropped = 0
    number = first_number
    for raw in raw_facts:
        span = find_quote(text, raw.quote)
        if span is None or not raw.statement.strip():
            dropped += 1
            continue
        facts.append(
            Fact(
                id=f"f{number}",
                source_id=source_id,
                statement=raw.statement.strip(),
                quote=text[span[0] : span[1]],
                start=span[0],
                end=span[1],
            )
        )
        number += 1
    return facts, dropped


async def extract_facts(
    source: Source,
    text: str,
    *,
    brief: Brief,
    first_number: int,
    model: str,
    log: ActionLog | None = None,
) -> list[Fact]:
    """One call for one source, then the quote check.

    ``first_number`` makes ids global: pass ``len(facts_so_far) + 1``. Callers
    run sources sequentially. Facts whose quote is absent are dropped and the
    count is logged.
    """
    sent = text[:MAX_SOURCE_CHARS]
    message = (
        f"PURPOSE: {brief.purpose}\n"
        f"KEY MESSAGES: {'; '.join(brief.key_messages)}\n"
        f"MUST INCLUDE: {'; '.join(brief.must_include) or 'nothing specific'}\n\n"
        f"SOURCE {source.id} ({source.title}):\n{sent}"
    )
    raw = await call_model(
        model=model,
        instructions=EXTRACT_INSTRUCTIONS,
        output_type=RawFacts,
        user_message=message,
        node="evidence",
        log=log,
    )
    facts, dropped = check_quotes(
        raw.facts[:MAX_FACTS_PER_SOURCE],
        source_id=source.id,
        text=text,
        first_number=first_number,
    )
    if log is not None:
        if dropped:
            log.event(
                node="evidence",
                event="facts_dropped",
                source_id=source.id,
                dropped=dropped,
                kept=len(facts),
            )
        if len(text) > MAX_SOURCE_CHARS:
            log.event(
                node="evidence",
                event="source_truncated",
                source_id=source.id,
                sent_chars=MAX_SOURCE_CHARS,
                chars=len(text),
            )
    return facts
