"""The synthetic web question set: questions with short, checkable answers.

Each question was written for this benchmark. Its answer is a plain fact that a
stable public page states (Wikipedia, mostly), so a deterministic check can
decide correctness without a judge model: an answer is correct when it contains
every one of ``all_of``'s groups, and a group holds when any of its phrases
occurs as whole words. Matching is case-, accent- and punctuation-insensitive
and reads ``1,000`` as ``1000``.

The check is lenient on purpose: it does not read negation, so an answer that
names the right fact next to a wrong one passes. It is a floor for correctness,
not a grader of prose. ``verify_sources`` is the drift check: it fetches each
question's page and confirms the page still states the answer.
"""

from __future__ import annotations

import asyncio
import html
import re
import unicodedata
import urllib.request
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_NON_WORD = re.compile(r"[^0-9a-z]+")


def normalise_for_match(text: str) -> str:
    """Case-, accent- and punctuation-folded words joined by single spaces."""
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in folded if not unicodedata.combining(c)).casefold()
    folded = _THOUSANDS.sub("", folded)
    return " ".join(_NON_WORD.sub(" ", folded).split())


def contains_phrase(text: str, phrase: str) -> bool:
    """Does *phrase* occur in *text* as whole words (both normalised)?"""
    wanted = normalise_for_match(phrase)
    if not wanted:
        raise ValueError(f"the phrase {phrase!r} has no letters or digits")
    return f" {wanted} " in f" {normalise_for_match(text)} "


class AnswerSpec(BaseModel):
    """What a correct answer contains: every group, any phrase of a group."""

    model_config = ConfigDict(frozen=True)

    all_of: list[list[str]] = Field(min_length=1)


def answer_correct(text: str, spec: AnswerSpec) -> bool:
    """True when *text* satisfies every group of *spec*."""
    return all(
        any(contains_phrase(text, phrase) for phrase in group) for group in spec.all_of
    )


class WebQuestion(BaseModel):
    """One question, the answer to check for, and the page that states it."""

    model_config = ConfigDict(frozen=True)

    question_id: str = Field(pattern=r"^W\d{2}$")
    question: str = Field(min_length=1)
    answer: AnswerSpec
    source_url: str = Field(pattern=r"^https://")
    expected: str = Field(min_length=1, description="The answer in words, for the log.")


def _q(
    number: int,
    question: str,
    all_of: list[list[str]],
    source_url: str,
    expected: str,
) -> WebQuestion:
    return WebQuestion(
        question_id=f"W{number:02d}",
        question=question,
        answer=AnswerSpec(all_of=all_of),
        source_url=source_url,
        expected=expected,
    )


_WIKI = "https://en.wikipedia.org/wiki/"

WEB_QUESTIONS: tuple[WebQuestion, ...] = (
    _q(
        1,
        "What is the highest mountain in Africa, and in which country is its "
        "summit located?",
        [["Kilimanjaro"], ["Tanzania"]],
        _WIKI + "Mount_Kilimanjaro",
        "Mount Kilimanjaro, Tanzania",
    ),
    _q(
        2,
        "In which year was the HTTP/1.1 specification first published as RFC 2068?",
        [["1997"]],
        "https://www.rfc-editor.org/info/rfc2068",
        "1997",
    ),
    _q(
        3,
        "The chemical symbol of tungsten comes from an alternative name for the "
        "element. What is that name?",
        [["wolfram"]],
        _WIKI + "Tungsten",
        "wolfram",
    ),
    _q(
        4,
        "Who created the Python programming language, and in which year was "
        "Python 1.0 released?",
        [["van Rossum"], ["1994"]],
        _WIKI + "History_of_Python",
        "Guido van Rossum, 1994",
    ),
    _q(
        5,
        "Approximately how many Earth days does Mercury take to orbit the Sun once?",
        [["88", "87.97"]],
        _WIKI + "Mercury_(planet)",
        "about 88 days (87.97)",
    ),
    _q(
        6,
        "Which Apollo mission first landed humans on the Moon, and in which year?",
        [["Apollo 11"], ["1969"]],
        _WIKI + "Apollo_11",
        "Apollo 11, 1969",
    ),
    _q(
        7,
        "What is the capital of Australia, and what is the name of the artificial "
        "lake at the centre of that city?",
        [["Canberra"], ["Burley Griffin"]],
        _WIKI + "Canberra",
        "Canberra, Lake Burley Griffin",
    ),
    _q(
        8,
        "Which mathematician proved Fermat's Last Theorem, and in which year was "
        "the proof published in the Annals of Mathematics?",
        [["Wiles"], ["1995"]],
        _WIKI + "Fermat%27s_Last_Theorem",
        "Andrew Wiles, 1995",
    ),
    _q(
        9,
        "How many bits long is an IPv6 address?",
        [["128"]],
        _WIKI + "IPv6_address",
        "128",
    ),
    _q(
        10,
        "What is the name of the deepest known point of Earth's oceans, and in "
        "which ocean trench does it lie?",
        [["Challenger Deep"], ["Mariana"]],
        _WIKI + "Challenger_Deep",
        "Challenger Deep, Mariana Trench",
    ),
    _q(
        11,
        "Under which version of the GNU General Public License is the Linux "
        "kernel released?",
        [
            [
                "GPL-2.0",
                "GPLv2",
                "GPL v2",
                "GPL version 2",
                "GPL 2",
                "GNU General Public License version 2",
                "GNU General Public License v2",
                "GNU GPL v2",
            ]
        ],
        _WIKI + "Linux_kernel",
        "version 2 (GPL-2.0)",
    ),
    _q(
        12,
        "What is the SI unit of electric capacitance, and after which scientist "
        "is it named?",
        [["farad"], ["Faraday"]],
        _WIKI + "Farad",
        "the farad, after Michael Faraday",
    ),
    _q(
        13,
        "Which is the longest bone in the human body, and in which part of the "
        "body is it found?",
        [["femur", "thigh bone"], ["thigh", "leg"]],
        _WIKI + "Femur",
        "the femur, in the thigh",
    ),
    _q(
        14,
        "What is the longest river in Europe?",
        [["Volga"]],
        _WIKI + "Volga",
        "the Volga",
    ),
    _q(
        15,
        "In which year was the Hubble Space Telescope launched, and on which "
        "Space Shuttle?",
        [["1990"], ["Discovery"]],
        _WIKI + "Hubble_Space_Telescope",
        "1990, Space Shuttle Discovery",
    ),
)

Fetch = Callable[[str], Awaitable[str]]

PAGE_USER_AGENT = "andamentum-bench/0.1"
PAGE_TIMEOUT_S = 30.0
_TAGS = re.compile(r"<(script|style)\b.*?</\1>|<[^>]+>", re.DOTALL | re.IGNORECASE)


def page_text(markup: str) -> str:
    """The text of an HTML page: scripts, styles and tags dropped, entities read."""
    return html.unescape(_TAGS.sub(" ", markup))


def _get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": PAGE_USER_AGENT})
    with urllib.request.urlopen(request, timeout=PAGE_TIMEOUT_S) as response:
        return page_text(response.read().decode("utf-8", errors="replace"))


async def fetch_page(url: str) -> str:
    """The text of one fixed public page (the question set's own sources).

    ``urllib`` rather than ``httpx``: Wikipedia answers the ``httpx`` client
    with 403 and the standard library with 200. The URLs are constants of this
    module, so no SSRF gate is needed; the call runs off the event loop.
    """
    return await asyncio.to_thread(_get, url)


class SourceCheck(BaseModel):
    """Whether one question's source page still states its answer."""

    question_id: str
    source_url: str
    states_answer: bool
    error: str = ""


async def verify_sources(
    fetch: Fetch, questions: tuple[WebQuestion, ...] = WEB_QUESTIONS
) -> list[SourceCheck]:
    """Fetch each question's page, one after the other, and check the answer.

    *fetch* returns the page text and raises on any failure; a failure is
    recorded on that question's check, never skipped.
    """
    checks: list[SourceCheck] = []
    for question in questions:
        try:
            page = await fetch(question.source_url)
        except Exception as error:  # a failed fetch is a result, not a crash
            checks.append(
                SourceCheck(
                    question_id=question.question_id,
                    source_url=question.source_url,
                    states_answer=False,
                    error=f"{type(error).__name__}: {error}",
                )
            )
            continue
        checks.append(
            SourceCheck(
                question_id=question.question_id,
                source_url=question.source_url,
                states_answer=answer_correct(page, question.answer),
            )
        )
    return checks
