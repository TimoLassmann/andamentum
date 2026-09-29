"""Deterministic checks over a slide deck: ``analyze_slides(slides) -> SlidesResult``.

A slide is a headline, the body text frames under it, the source line, and the
speaker notes. The checks are the ones a scientific talk fails most often and a
machine can see without judgement:

- ``headline_length``   — over 62 characters wraps; over 90 is two messages.
- ``deixis``            — a headline that opens on "This", "These", "Here", ...
- ``banned_word``       — words from a fixed list, in the headline or the body.
- ``acronym``           — an acronym on the slide with no expansion on that slide.
- ``two_names``         — an acronym and a capitalised phrase whose initials fit
                          it, both in the deck: probably one entity, two names.
- ``restatement``       — the body repeats two or more of the headline's content
                          words. A heuristic: a direct label of the entity the
                          headline names also trips it, so every hit is read.

Pure function, no I/O. Reading a ``.pptx`` into ``Slide`` objects lives in
``slides_io`` and is used only by the CLI.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, Field

ONE_LINE = 62
MAX_HEADLINE = 90

DEFAULT_BANNED = frozenset(
    {
        "leverage",
        "leverages",
        "leveraging",
        "utilize",
        "utilise",
        "utilizes",
        "utilises",
        "delve",
        "delves",
        "foster",
        "fosters",
        "underscore",
        "underscores",
        "nuanced",
        "multifaceted",
        "intricate",
        "meticulous",
        "cutting-edge",
        "underpin",
        "underpins",
        "realm",
        "tapestry",
        "beacon",
    }
)

# Acronyms that need no expansion in any scientific room. Callers extend it.
DEFAULT_ALLOWED_ACRONYMS = frozenset({"DNA", "RNA", "PCR", "USA", "UK", "EU", "OK"})

_DEIXIS = re.compile(r"^\s*(this|these|that|those|here|it|they)\b", re.IGNORECASE)
_ACRONYM = re.compile(r"(?<![\w-])([A-Z]{2,6})(?![\w-])")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_PHRASE = re.compile(r"\b([A-Z][a-z]+(?:\s+(?:of\s+|and\s+|for\s+)?[A-Z][a-z]+)+)")

_STOP = frozenset(
    """a an the and or but if of to in on at by for from with without into onto
    over under about than then so as is are was were be been being have has had
    do does did not no nor only also even now just very more most less least
    can could will would should may might must shall this these that those it
    its they them their we our us you your he she his her i me my who whom which
    what when where why how all any each every some such own same other one two
    three four five first second new""".split()
)

_IRREGULAR = {
    "children": "child",
    "people": "person",
    "women": "woman",
    "men": "man",
    "mice": "mouse",
    "data": "datum",
}
_SUFFIXES = (
    "isation",
    "ization",
    "ations",
    "ation",
    "ions",
    "ion",
    "isers",
    "izers",
    "iser",
    "izer",
    "ises",
    "izes",
    "ised",
    "ized",
    "ise",
    "ize",
    "ings",
    "ing",
    "ers",
    "er",
    "ed",
    "es",
    "s",
)


class Slide(BaseModel):
    """One slide as the checks see it."""

    number: int
    headline: str
    body: list[str] = Field(
        default_factory=list,
        description="Text frames other than the headline and source line.",
    )
    source: str = ""
    notes: str = ""


class SlideFinding(BaseModel):
    """One problem on one slide. ``slide`` is the 1-based slide number."""

    slide: int
    check: str
    detail: str


class SlidesResult(BaseModel):
    """Result of ``analyze_slides``. Deterministic; same input, same output."""

    slide_count: int
    findings: list[SlideFinding]
    summary: str


def stem(word: str) -> str:
    """Crude suffix stripper: enough to match "summarises" with "summariser"."""
    w = word.lower().strip("'’-")
    w = re.sub(r"['’]s$", "", w)
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            w = w[: -len(suf)]
            break
    # "reduce" and "reducing" both end as "reduc".
    return w[:-1] if w.endswith("e") and len(w) > 4 else w


def content_stems(text: str) -> set[str]:
    """Stems of the lower-case content words in ``text``.

    Capitalised words, acronyms and numbers are left out: they name entities,
    and a body that labels the entity its headline names is doing its job.
    """
    out: set[str] = set()
    for m in _WORD.finditer(text):
        w = m.group(0)
        # The first word of a sentence or a label is capitalised by sentence
        # case, not because it names something: "Reducing", "Results".
        # A capitalised word followed by another ("Apollo Kinase") is a name.
        opens = (
            not text[: m.start()].strip(" \t\"'“‘(")
            or text[: m.start()].rstrip()[-1:] in ".!?\n"
        )
        nxt = _WORD.match(text, m.end() + 1)
        in_name = (
            nxt is not None and nxt.start() == m.end() + 1 and nxt.group(0)[0].isupper()
        )
        if not (w[0].islower() or (opens and w[1:].islower() and not in_name)):
            continue
        if w.lower() in _STOP or len(w) < 3:
            continue
        out.add(stem(w))
    return out


def _name(phrase: str) -> str:
    """Drop a sentence-initial article: "The Apollo Centre" is "Apollo Centre"."""
    words = phrase.split()
    while len(words) > 2 and words[0].lower() in _STOP:
        words = words[1:]
    return " ".join(words)


def _initials(phrase: str) -> str:
    return "".join(w[0] for w in _name(phrase).split() if w[0].isupper())


def _is_subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return all(ch in it for ch in short)


def _expanded(acronym: str, text: str) -> bool:
    """True when ``text`` expands ``acronym``: "(ABC)", or a phrase whose initials are ABC."""
    if f"({acronym})" in text:
        return True
    return any(_initials(p) == acronym for p in _PHRASE.findall(text))


def analyze_slides(
    slides: Iterable[Slide],
    *,
    banned: frozenset[str] = DEFAULT_BANNED,
    allowed_acronyms: frozenset[str] = DEFAULT_ALLOWED_ACRONYMS,
) -> SlidesResult:
    """Run every slide check. Findings are ordered by slide, then by check."""
    deck = list(slides)
    findings: list[SlideFinding] = []

    for s in deck:
        head = " ".join(s.headline.split())
        body = " ".join(s.body)
        visible = f"{head} {body}"

        if len(head) > MAX_HEADLINE:
            findings.append(
                SlideFinding(
                    slide=s.number,
                    check="headline_length",
                    detail=f"{len(head)} characters; over {MAX_HEADLINE} is two messages, split the slide",
                )
            )
        elif len(head) > ONE_LINE:
            findings.append(
                SlideFinding(
                    slide=s.number,
                    check="headline_length",
                    detail=f"{len(head)} characters; over {ONE_LINE} wraps, tighten or break by hand and flag it",
                )
            )

        if _DEIXIS.match(head):
            findings.append(
                SlideFinding(
                    slide=s.number,
                    check="deixis",
                    detail=f"headline opens on '{head.split()[0]}'; name the subject",
                )
            )

        for w in sorted({m.group(0).lower() for m in _WORD.finditer(visible)} & banned):
            findings.append(
                SlideFinding(slide=s.number, check="banned_word", detail=f"'{w}'")
            )

        for acr in sorted(set(_ACRONYM.findall(visible)) - allowed_acronyms):
            if not _expanded(acr, visible):
                findings.append(
                    SlideFinding(
                        slide=s.number,
                        check="acronym",
                        detail=f"{acr} is not expanded on this slide",
                    )
                )

        body_stems: set[str] = set().union(*(content_stems(b) for b in s.body))
        shared = content_stems(head) & body_stems
        if len(shared) >= 2:
            findings.append(
                SlideFinding(
                    slide=s.number,
                    check="restatement",
                    detail="body repeats headline words: " + ", ".join(sorted(shared)),
                )
            )

    findings.extend(_two_names(deck))
    findings.sort(key=lambda f: (f.slide, f.check))
    return SlidesResult(
        slide_count=len(deck),
        findings=findings,
        summary=_summarise(len(deck), findings),
    )


def _two_names(deck: list[Slide]) -> list[SlideFinding]:
    """An acronym and a capitalised phrase whose initials fit inside it.

    "Rare Care Clinic" (RCC) beside "RCCC" is flagged: the initials match all but
    one letter, which is how a long name and a short form of it usually differ.
    """
    acronyms: dict[str, int] = {}
    phrases: dict[str, int] = {}
    for s in deck:
        visible = " ".join([s.headline, *s.body])
        for a in _ACRONYM.findall(visible):
            acronyms.setdefault(a, s.number)
        for p in _PHRASE.findall(visible):
            phrases.setdefault(_name(p), s.number)

    out: list[SlideFinding] = []
    for p, p_slide in phrases.items():
        ini = _initials(p)
        for a, a_slide in acronyms.items():
            if ini == a or (
                len(a) - len(ini) == 1 and ini[0] == a[0] and _is_subsequence(ini, a)
            ):
                if ini == a and a_slide == p_slide:
                    continue  # expanded in place, which is the fix
                out.append(
                    SlideFinding(
                        slide=max(p_slide, a_slide),
                        check="two_names",
                        detail=f"'{p}' (slide {p_slide}) and {a} (slide {a_slide}) may be one entity; use one name",
                    )
                )
    return out


def _summarise(count: int, findings: list[SlideFinding]) -> str:
    if not findings:
        return f"{count} slides. No slide flags."
    by: dict[str, int] = {}
    for f in findings:
        by[f.check] = by.get(f.check, 0) + 1
    parts = ", ".join(f"{n} {k}" for k, n in sorted(by.items()))
    return f"{count} slides. Flags: {parts}."
