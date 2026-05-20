"""Surface-3-style calibration test for the page_summarizer prompt.

Real ``page_summarizer`` agent against ``openai:gpt-5.4-nano`` with a
curated corpus of synthetic (page_text, query, expected_band) tuples.
Synthetic content is used so the test is deterministic and runs offline
(no flaky web fetches), while still exercising the real LLM scoring on
realistic prose.

The corpus is intentionally domain-mixed (biomedical + software +
historical + economic) so the prompt's score scale is calibrated to
work across research fields, not just biomedical literature.

Bands:
- high   (>= 0.6) — page directly addresses the question or contains
                    substantial relevant evidence
- medium (0.3-0.6) — tangential or partial coverage
- low    (< 0.3)  — incidental mentions only

Pass thresholds (loose, first-run; tighten as the prompt is tuned):
- per-band: at most 1 mis-classification per 4 cases
- overall: ≥ 75% accuracy

A failure prints the full confusion matrix + offending cases for
diagnosis.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field

import pytest
from dotenv import load_dotenv

from andamentum.core.agents import build_pydantic_ai_agent
from andamentum.core.models import resolve_model
from andamentum.deep_research.agents import get_agent

load_dotenv()
pytestmark = pytest.mark.cloud


CLOUD_MODEL = "openai:gpt-5.4-nano"


# ── Synthetic page fixtures ────────────────────────────────────────────
#
# Each fixture is a short (300-700 word) synthetic page. Keep them
# realistic enough that the LLM treats them like real prose, but
# concrete enough that we can predict the expected score band.


PAGE_KALIGN_PRIMARY = """
Kalign — a novel and accurate multiple sequence alignment program.

Background: Multiple sequence alignment is one of the most fundamental
problems in computational molecular biology. We present a new MSA
program, Kalign, designed for large-scale alignments.

Methods: We compared Kalign against MAFFT (version 3.85), MUSCLE
(version 3.0), ClustalW (version 1.83), DIALIGN (version 2.2.1), and
T-Coffee (version 1.37) using three benchmark suites: Balibase 2.01,
PREFAB 3.0, and a new large simulated test set.

Results: On Balibase, Kalign performed comparably to MAFFT and MUSCLE,
slightly worse on average due to the small sample sizes (around 10
sequences per alignment). On PREFAB 3.0 (1,932 alignments), Kalign was
approximately as accurate as MUSCLE and MAFFT but 4-7 times faster
overall. For example, aligning 500 sequences took Kalign 5 minutes;
the same alignment took MUSCLE 90 minutes.

On the large simulated set, T-Coffee and DIALIGN had to be excluded
because they could not handle inputs of more than 200 sequences within
practical time limits. Kalign maintained accuracy comparable to MAFFT
and MUSCLE while being substantially faster.

Conclusion: Kalign is suitable for very large multiple sequence
alignments and produces accuracy comparable to the best progressive
aligners while being substantially more efficient.
"""

PAGE_RUST_OWNERSHIP = """
The Rust Programming Language — Ownership

Ownership is Rust's most unique feature, and it enables Rust to make
memory safety guarantees without needing a garbage collector. Every
value in Rust has a single owner, and when the owner goes out of scope
the value is dropped. References borrow values without taking ownership.

The borrow checker enforces three rules at compile time: at any given
time you can have either one mutable reference or any number of
immutable references; references must always be valid; and there can be
exactly one owner of each value. These rules eliminate use-after-free,
double-free, and data-race bugs at compile time.

Compared to garbage-collected languages like Java or Go, Rust avoids
runtime overhead from garbage collection while still preventing the
manual-memory bugs common in C and C++. The trade-off is a steeper
learning curve: programmers must reason about ownership explicitly.

Lifetimes annotate how long references live and let the compiler verify
references don't outlive the data they point to. Most lifetimes are
elided automatically, but in complex generic code they must be written
explicitly.
"""

PAGE_PYTHON_GENERAL = """
Python is a high-level, interpreted programming language with dynamic
typing and garbage collection. Python's design philosophy emphasises
readability with significant indentation. It supports multiple
programming paradigms including procedural, object-oriented, and
functional programming.

Python's standard library is large; the language has a vibrant
ecosystem of third-party packages distributed via PyPI. Python is
frequently used for web development, data analysis, scientific
computing, and machine learning.

Python uses reference counting plus a cycle-detecting garbage collector
for memory management. Programmers do not manage memory manually;
allocation and deallocation are handled by the runtime. This makes
Python easier to learn than systems languages but introduces runtime
overhead that limits performance in CPU-bound workloads.
"""

PAGE_2008_FINANCIAL_CRISIS = """
The 2008 financial crisis was triggered by the collapse of the United
States housing bubble, which had been inflated by aggressive lending
practices in the subprime mortgage market. As default rates rose,
mortgage-backed securities lost value rapidly, triggering a chain
reaction across leveraged financial institutions.

Key causal factors documented by economists include: (1) the
proliferation of subprime adjustable-rate mortgages issued to
borrowers with limited ability to repay; (2) the use of complex
derivatives — particularly collateralised debt obligations (CDOs) and
credit default swaps (CDS) — that obscured systemic risk; (3) credit
rating agencies assigning AAA ratings to mortgage-backed securities
that contained substantial subprime exposure; (4) excessive leverage
at investment banks, with several institutions running 30-to-1 debt-
to-equity ratios; (5) regulatory failures, including the 1999
repeal of Glass-Steagall and the 2004 SEC rule change permitting
voluntary regulation of broker-dealer net capital.

Lehman Brothers' bankruptcy on September 15, 2008 marked the most
acute phase of the crisis, freezing inter-bank lending and prompting
unprecedented government interventions including the Troubled Asset
Relief Program ($700 billion) and emergency Federal Reserve liquidity
facilities.
"""

PAGE_GOLDEN_RETRIEVERS = """
The Golden Retriever is a medium-large breed of retrieving dog. It is
characterised by a dense, water-repellent outer coat with a thick
undercoat. The breed was developed in Scotland in the mid-19th century
by Lord Tweedmouth, who crossed a Yellow Retriever with a now-extinct
Tweed Water Spaniel.

Golden Retrievers are commonly used as guide dogs, search and rescue
dogs, and family companions. They are known for their friendly
temperament and trainability. The American Kennel Club recognised the
breed in 1925.

Common health concerns include hip dysplasia, certain cancers, and
heart conditions. Average lifespan is 10-12 years.
"""

PAGE_KALIGN_SERVER = """
Kalign — Web Server

This server provides access to the Kalign multiple sequence alignment
algorithm via a web interface. Users submit sequences in FASTA format
and receive aligned output. The server also includes Kalignvu (an
alignment viewer) and Mumsa (a quality scoring tool).

On average, Kalign takes less than a second to align one hundred
protein sequences of length 500 on the server hardware. The server
exposes three gap-penalty parameters (gap open, internal extension,
terminal extension) and an optional bonus score for substitution
matrix fields.

Mumsa computes alignment quality using AOS (average overlap score)
and MOS (multiple overlap score), each ranging from 0 to 1. An AOS
score above 0.8 indicates good agreement among input alignments;
below 0.5 indicates the sequences are very difficult to align.
Alignments with a MOS score above 0.8 may be considered reliable.

References to other alignment tools (MUSCLE, MAFFT, T-Coffee) appear
only in the citation section of the paper this server accompanies.
"""


# ── Calibration corpus ─────────────────────────────────────────────────
#
# (page_text, query, expected_band, category) tuples.
# Bands: "high" (≥ 0.6), "medium" (0.3-0.6), "low" (< 0.3)


CORPUS: list[tuple[str, str, str, str]] = [
    # Direct competitor data → high
    (
        PAGE_KALIGN_PRIMARY,
        "What are Kalign's MSA competitors and how does it compare?",
        "high",
        "biomedical-direct",
    ),
    # Page about server features for the SAME tool but the question is
    # about competitors → low (the server page only references other
    # tools in citations).
    (
        PAGE_KALIGN_SERVER,
        "What are Kalign's MSA competitors and how does it compare?",
        "low",
        "biomedical-passing-only",
    ),
    # Same page, different question → server page IS direct evidence for
    # "how do you use Kalign" / "what does the Kalign server do".
    (
        PAGE_KALIGN_SERVER,
        "How do I use the Kalign web server and assess alignment quality?",
        "high",
        "biomedical-direct",
    ),
    # Software language with explicit memory model → high
    (
        PAGE_RUST_OWNERSHIP,
        "How does Rust handle memory safety?",
        "high",
        "software-direct",
    ),
    # Different language; mentions the comparison briefly → low
    (
        PAGE_PYTHON_GENERAL,
        "How does Rust handle memory safety?",
        "low",
        "software-different-subject",
    ),
    # Python page asked about Python memory → high
    (
        PAGE_PYTHON_GENERAL,
        "How does Python manage memory?",
        "high",
        "software-direct",
    ),
    # Historical / economic causal analysis → high for the matching question
    (
        PAGE_2008_FINANCIAL_CRISIS,
        "What caused the 2008 financial crisis?",
        "high",
        "economic-direct",
    ),
    # Off-topic page → low for the unrelated question
    (
        PAGE_GOLDEN_RETRIEVERS,
        "What caused the 2008 financial crisis?",
        "low",
        "economic-off-topic",
    ),
    # Tangential — the Kalign primary paper does NOT discuss server use
    (
        PAGE_KALIGN_PRIMARY,
        "How do I install and use the Kalign command-line tool on Linux?",
        "medium",
        "biomedical-tangential",
    ),
    # Tangential — Rust memory page on a question about Rust ecosystem
    (
        PAGE_RUST_OWNERSHIP,
        "What are the most popular crates in the Rust ecosystem?",
        "low",
        "software-different-aspect",
    ),
]


def _band_for(score: float) -> str:
    if score >= 0.6:
        return "high"
    if score >= 0.3:
        return "medium"
    return "low"


@dataclass
class BandStats:
    n: int = 0
    correct: int = 0
    mistakes: list[tuple[str, str, str, float]] = field(
        default_factory=list
    )  # (category, expected, predicted, score)


async def _summarize_one(query: str, page_text: str, model) -> float:
    """Call the page_summarizer agent on one page and return its score."""
    agent = build_pydantic_ai_agent(get_agent("page_summarizer"), model)
    word_count = len(page_text.split())
    prompt = f"""Question: {query}

Page Content ({word_count} words):
{page_text}

Follow the process in your instructions: extract usable facts first,
then derive a relevance score from the scale."""
    result = await agent.run(prompt)
    return float(result.output.relevance_score)


async def test_summarizer_calibration_against_synthetic_corpus():
    """Score each (page, query) pair and check it lands in the expected band.

    Pass thresholds (loose first-run):
    - overall accuracy ≥ 75%
    - no single band misclassifies > 50% of its cases

    A failure prints the confusion matrix + offending cases for diagnosis.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    model = resolve_model(CLOUD_MODEL)

    by_band: dict[str, BandStats] = defaultdict(BandStats)
    rows: list[tuple[str, str, str, str, float]] = []
    # rows entries: (category, expected, predicted, query, score)

    for page_text, query, expected, category in CORPUS:
        score = await _summarize_one(query, page_text, model)
        predicted = _band_for(score)
        rows.append((category, expected, predicted, query, score))

        s = by_band[expected]
        s.n += 1
        if predicted == expected:
            s.correct += 1
        else:
            s.mistakes.append((category, expected, predicted, score))

    # ── Print confusion matrix ────────────────────────────────────────
    print("\n\n========== PAGE_SUMMARIZER CALIBRATION ==========")
    print(f"Model: {CLOUD_MODEL}")
    print(f"Corpus size: {len(CORPUS)}\n")

    print(f"{'expected band':<10} {'n':>3} {'correct':>8} {'acc':>5}")
    print("-" * 35)
    overall_n = 0
    overall_correct = 0
    for band in ("high", "medium", "low"):
        s = by_band.get(band, BandStats())
        if s.n == 0:
            continue
        acc = (s.correct / s.n) * 100
        print(f"{band:<10} {s.n:>3} {s.correct:>8} {acc:>4.0f}%")
        overall_n += s.n
        overall_correct += s.correct
    print("-" * 35)
    overall_acc = (overall_correct / overall_n) * 100 if overall_n else 0
    print(f"{'OVERALL':<10} {overall_n:>3} {overall_correct:>8} {overall_acc:>4.0f}%")

    # ── Print mistakes ────────────────────────────────────────────────
    mistakes = [r for r in rows if r[1] != r[2]]
    if mistakes:
        print(f"\n========== MISTAKES ({len(mistakes)}) ==========")
        for category, expected, predicted, query, score in mistakes:
            print(
                f"\n[{category}] expected={expected}  got={predicted}  "
                f"score={score:.2f}\n  query: {query}"
            )

    # ── Soft assertions ───────────────────────────────────────────────
    failures: list[str] = []
    if overall_acc < 75:
        failures.append(
            f"overall accuracy {overall_acc:.0f}% < 75% threshold"
        )
    for band, s in by_band.items():
        if s.n >= 2 and s.correct / s.n < 0.5:
            failures.append(
                f"{band}-band accuracy {s.correct}/{s.n} < 50%"
            )
    assert not failures, "Calibration regressions:\n  " + "\n  ".join(failures)
