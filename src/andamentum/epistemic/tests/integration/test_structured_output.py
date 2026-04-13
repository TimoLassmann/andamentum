"""Test structured output reliability across models and output complexities.

Diagnoses where small models fail at producing structured output:
- Simple output (2 fields)
- Medium output (4-5 fields)
- Complex output (nested fields, lists)
- The actual epistemic agent outputs that fail in production

Usage:
    uv run python packages/epistemic/integration_tests/test_structured_output.py --model ollama:gemma4:e2b
    uv run python packages/epistemic/integration_tests/test_structured_output.py --model openai:gpt-5.4-mini
    uv run python packages/epistemic/integration_tests/test_structured_output.py --model ollama:gemma4:e2b --verbose
"""

import argparse
import asyncio
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field


# ── Output models: graduated complexity ──────────────────────────────────


class SimpleOutput(BaseModel):
    """2 fields — minimal."""

    answer: str = Field(description="A short answer")
    confidence: float = Field(description="Confidence between 0.0 and 1.0")


class MediumOutput(BaseModel):
    """4 fields — typical epistemic agent."""

    question_type: str = Field(
        description="One of: verificatory, explanatory, exploratory, comparative"
    )
    reasoning: str = Field(description="One sentence explaining the classification")
    key_terms: list[str] = Field(description="2-3 key terms from the question")
    confidence: float = Field(description="Confidence between 0.0 and 1.0")


class ComplexOutput(BaseModel):
    """6+ fields with optional and nested — where small models struggle."""

    title: str = Field(description="A concise title")
    verdict: str = Field(default="", description="One sentence bottom line")
    answer: str = Field(description="2-3 paragraph answer")
    has_issues: bool = Field(description="Whether issues were found")
    issues: list[str] = Field(default_factory=list, description="List of issues found")
    confidence: float = Field(description="Confidence between 0.0 and 1.0")


class WriteAnswerOutput(BaseModel):
    """The actual output model that fails in production."""

    title: str = Field(description="A concise title for the research report")
    verdict: str = Field(
        default="", description="One sentence answering the research question"
    )
    answer: str = Field(description="A direct answer to the research question")


class ClarifyOutput(BaseModel):
    """The clarify_question output model."""

    clarified_question: str = Field(
        description="Clarified version of the research question"
    )
    key_terms: list[str] = Field(description="Key terms identified")
    reasoning: str = Field(description="Why this clarification improves the question")


# ── Test definitions ─────────────────────────────────────────────────────


@dataclass
class StructuredOutputTest:
    name: str
    system_prompt: str
    user_message: str
    output_model: type[BaseModel]
    output_retries: int = 3
    output_mode: str = "auto"  # "auto", "prompted", "tool"


TESTS: list[StructuredOutputTest] = [
    StructuredOutputTest(
        name="simple_2_fields_auto",
        system_prompt="You answer questions concisely.",
        user_message="What is the capital of France?",
        output_model=SimpleOutput,
        output_mode="auto",
    ),
    StructuredOutputTest(
        name="simple_2_fields_prompted",
        system_prompt="You answer questions concisely.",
        user_message="What is the capital of France?",
        output_model=SimpleOutput,
        output_mode="prompted",
    ),
    StructuredOutputTest(
        name="medium_4_fields",
        system_prompt="You classify research questions by type.",
        user_message="question: Does homeopathy have therapeutic effects beyond placebo?",
        output_model=MediumOutput,
    ),
    StructuredOutputTest(
        name="complex_6_fields",
        system_prompt="You review claims and identify issues.",
        user_message="claim: Homeopathy cures cancer\nevidence: A single uncontrolled case study from 1998",
        output_model=ComplexOutput,
    ),
    StructuredOutputTest(
        name="write_answer_production",
        system_prompt=(
            "You directly answer a research question based on evidence.\n\n"
            "Output:\n"
            "1. title: A concise title\n"
            "2. verdict: One sentence answering the question\n"
            "3. answer: 2-3 paragraphs synthesizing the evidence"
        ),
        user_message=(
            "research_question: Does homeopathy have therapeutic effects beyond placebo?\n"
            "claims: [HYPOTHESIS] The evidence does not support homeopathy having effects beyond placebo.\n"
            "evidence: [1] Cochrane review found no convincing evidence. [2] NHMRC 2015 found no reliable evidence.\n"
            "adversarial_results: Strong counter-evidence found (balance: 0.07)"
        ),
        output_model=WriteAnswerOutput,
    ),
    StructuredOutputTest(
        name="clarify_production",
        system_prompt=(
            "You clarify research questions to make them more precise.\n"
            "Output the clarified question, key terms, and reasoning."
        ),
        user_message="question: Does homeopathy have therapeutic effects beyond placebo?",
        output_model=ClarifyOutput,
    ),
]


# ── Runner ───────────────────────────────────────────────────────────────


@dataclass
class TestResult:
    name: str
    passed: bool
    output: Optional[BaseModel] = None
    error: Optional[str] = None
    retries_needed: int = 0
    duration_ms: int = 0
    messages: list[dict] = None  # type: ignore[assignment]


async def run_test(
    test: StructuredOutputTest,
    model_str: str,
    verbose: bool = False,
) -> TestResult:
    """Run a single structured output test and capture all details."""
    from andamentum.epistemic.runner import _resolve_model

    # Late import to avoid loading pydantic-ai at module level
    from pydantic_ai import Agent, PromptedOutput, ToolOutput

    model = _resolve_model(model_str)

    # Choose output mode
    if test.output_mode == "prompted":
        output_type = PromptedOutput(test.output_model)
    elif test.output_mode == "tool":
        output_type = ToolOutput(test.output_model)
    else:
        output_type = test.output_model  # auto (default)

    agent = Agent(
        model,
        system_prompt=test.system_prompt,
        output_type=output_type,
        retries=1,
        output_retries=test.output_retries,
    )

    start = time.monotonic()
    messages = []

    try:
        result = await agent.run(test.user_message)
        duration = int((time.monotonic() - start) * 1000)

        # Capture message history for debugging
        messages = []
        for msg in result.all_messages():
            msg_dict = {
                "kind": msg.kind,
            }
            if hasattr(msg, "content"):
                content = getattr(msg, "content")  # type: ignore[union-attr]
                if isinstance(content, str):
                    msg_dict["content"] = content[:500]
                elif isinstance(content, list):
                    # Tool call parts, text parts, etc.
                    parts_summary = []
                    for part in content:
                        if hasattr(part, "content"):
                            parts_summary.append(
                                f"{part.part_kind}: {str(part.content)[:200]}"
                            )
                        elif hasattr(part, "args"):
                            parts_summary.append(
                                f"{part.part_kind}: {str(part.args)[:200]}"
                            )
                        else:
                            parts_summary.append(str(part)[:200])
                    msg_dict["parts"] = parts_summary  # type: ignore[assignment]
            messages.append(msg_dict)

        return TestResult(
            name=test.name,
            passed=True,
            output=result.output,
            duration_ms=duration,
            messages=messages,
        )
    except Exception as e:
        duration = int((time.monotonic() - start) * 1000)
        error_detail = f"{type(e).__name__}: {e}"
        if verbose:
            error_detail += f"\n{traceback.format_exc()}"

        return TestResult(
            name=test.name,
            passed=False,
            error=error_detail,
            duration_ms=duration,
            messages=messages,
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test structured output reliability")
    parser.add_argument(
        "--model", required=True, help="Model string (e.g. ollama:gemma4:e2b)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show full message history and errors"
    )
    parser.add_argument("--test", default=None, help="Run only this test (by name)")
    args = parser.parse_args()

    tests = TESTS
    if args.test:
        tests = [t for t in TESTS if t.name == args.test]
        if not tests:
            print(f"Unknown test: {args.test}")
            print(f"Available: {', '.join(t.name for t in TESTS)}")
            sys.exit(1)

    print(f"Model: {args.model}")
    print(f"Tests: {len(tests)}")
    print()

    results: list[TestResult] = []
    for test in tests:
        print(
            f"  {test.name} ({test.output_model.__name__}, {len(test.output_model.model_fields)} fields)..."
        )
        result = await run_test(test, args.model, verbose=args.verbose)
        results.append(result)

        if result.passed:
            print(f"    PASS ({result.duration_ms}ms)")
            if args.verbose and result.output:
                for field_name, field_value in result.output.model_dump().items():
                    val_str = str(field_value)
                    if len(val_str) > 100:
                        val_str = val_str[:100] + "..."
                    print(f"      {field_name}: {val_str}")
        else:
            print(f"    FAIL ({result.duration_ms}ms)")
            print(f"      Error: {result.error}")

        if args.verbose and result.messages:
            print(f"    Message history ({len(result.messages)} messages):")
            for i, msg in enumerate(result.messages):
                print(f"      [{i}] {msg['kind']}", end="")
                if "content" in msg:
                    content_preview = msg["content"][:150]
                    print(f": {content_preview}")
                elif "parts" in msg:
                    print(":")
                    for part in msg["parts"][:5]:
                        print(f"          {part[:150]}")
                else:
                    print()
        print()

    # Summary
    print("=" * 60)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    print(f"Results: {passed} passed, {failed} failed")
    print()

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name:30s} {r.duration_ms:>6d}ms")

    if failed > 0:
        print()
        print("Failed tests:")
        for r in results:
            if not r.passed:
                error_first_line = (r.error or "").split("\n")[0]
                print(f"  {r.name}: {error_first_line}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
