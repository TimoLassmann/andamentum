"""Drift detection — alerts when tested error handling code changes.

This module checksums specific code sections (exception handlers, fallback chains,
gate validation paths) that are covered by error path tests. When a checksum changes,
the test fails with a message identifying which section changed and which test file
to review.

This is NOT a correctness test — it's a maintenance signal. When it fails:
1. Read the changed section in the source file
2. Update the corresponding error path tests if behavior changed
3. Update the checksum in this file

Run with: uv run pytest packages/epistemic/tests/test_drift_detection.py -v
"""

import hashlib
from pathlib import Path

import pytest

# Root of epistemic source
_SRC = Path(__file__).parent.parent


def _extract_section(
    filepath: Path,
    start_pattern: str,
    end_pattern: str | None = None,
    num_lines: int | None = None,
) -> str:
    """Extract a code section by finding start pattern and reading until end pattern or N lines."""
    text = filepath.read_text()
    lines = text.splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        if start_pattern in line:
            start_idx = i
            break

    if start_idx is None:
        return ""

    if end_pattern:
        end_idx = None
        for i in range(start_idx + 1, len(lines)):
            if end_pattern in lines[i]:
                end_idx = i + 1
                break
        if end_idx is None:
            end_idx = min(start_idx + 50, len(lines))
    elif num_lines:
        end_idx = min(start_idx + num_lines, len(lines))
    else:
        end_idx = min(start_idx + 30, len(lines))

    return "\n".join(lines[start_idx:end_idx])


def _sha256(text: str) -> str:
    """Compute SHA-256 of text, stripping trailing whitespace per line."""
    normalized = "\n".join(line.rstrip() for line in text.splitlines())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


# ── Section definitions ──────────────────────────────────────────────────────
# Each entry maps a source code section to the test file that covers it.

TRACKED_SECTIONS = [
    # operations/evidence.py: _score_evidence chain (no fallback default — raises on miss)
    {
        "file": _SRC / "operations" / "evidence.py",
        "start": "async def _score_evidence",
        "num_lines": 100,
        "description": "Evidence scoring 3-path chain (no fallback)",
        "test_file": "test_no_silent_fallbacks.py",
    },
    # operations/scrutiny.py: Scrutiny evidence loading loop
    {
        "file": _SRC / "operations" / "scrutiny.py",
        "start": "evidence_summaries: list[str] = []",
        "num_lines": 15,
        "description": "Scrutiny evidence loading loop",
        "test_file": "test_operations_failure.py::TestScrutinyOperationFailure",
    },
    # operations/verification.py: Counterargument evaluation (no fallback)
    {
        "file": _SRC / "operations" / "verification.py",
        "start": "async def _evaluate_one(",
        "end": "# Step 5: Compute adversarial balance (deterministic)",
        "description": "Counterargument evaluation (no fallback)",
        "test_file": "test_no_silent_fallbacks.py::test_adversarial_check_propagates_counterarg_eval_failure",
    },
    # operations/synthesis.py: Writer-validator loop
    {
        "file": _SRC / "operations" / "synthesis.py",
        "start": "async def _writer_validator_loop",
        "end": "return title, answer",
        "description": "Writer-validator loop",
        "test_file": "test_operations_failure.py::TestWriterValidatorLoop",
    },
    # operations/investigation.py: Prediction generation per-aspect error handling
    {
        "file": _SRC / "operations" / "investigation.py",
        "start": "# Steps 2-4: For each aspect, specify",
        "num_lines": 40,
        "description": "Prediction generation per-aspect error handling",
        "test_file": "test_operations_failure.py::TestPredictionClassificationFailure",
    },
    # gates.py: validate_promotion — full function including exception handlers
    {
        "file": _SRC / "gates.py",
        "start": "async def validate_promotion",
        "end": "# TMS: CURRENT STAGE VALIDATION",
        "description": "Gate validate_promotion logic",
        "test_file": "test_gates_failure.py::TestValidatePromotionFailure",
    },
    # gates.py: validate_current_stage
    {
        "file": _SRC / "gates.py",
        "start": "async def validate_current_stage",
        "end": "return GateResult(",
        "description": "Gate validate_current_stage (TMS) logic",
        "test_file": "test_gates_failure.py::TestValidateCurrentStageFailure",
    },
    # evidence_gathering.py: CompositeGatherer error handling
    {
        "file": _SRC / "evidence_gathering.py",
        "start": "class CompositeGatherer",
        "end": "class WebSearchGatherer",
        "description": "CompositeGatherer provider error handling",
        "test_file": "test_providers.py::TestCompositeGathererErrorPaths",
    },
    # adapters.py: full adapter registry
    {
        "file": _SRC / "adapters.py",
        "start": "ADAPTERS",
        "num_lines": 30,
        "description": "Adapter registry (agent name → adapter function mapping)",
        "test_file": "test_adapters.py",
    },
]

# ── Expected checksums ────────────────────────────────────────────────────────
# Update these when you intentionally change the tracked source sections.
# The test failure message tells you the new hash to use.

EXPECTED_CHECKSUMS: dict[str, str] = {
    "Evidence scoring 3-path chain (no fallback)": "b97e4f9a7012e8d5",
    "Scrutiny evidence loading loop": "cd12fd7cdd66ac9d",
    "Counterargument evaluation (no fallback)": "7242b31b99fd2632",
    "Writer-validator loop": "565af4046d169c0d",
    "Prediction generation per-aspect error handling": "fb5787ea93a0df4b",
    "Gate validate_promotion logic": "e87d8901f731e130",
    "Gate validate_current_stage (TMS) logic": "0fac458779974ec0",
    "CompositeGatherer provider error handling": "418013e2910f0263",
    "Adapter registry (agent name → adapter function mapping)": "e780e336e9788f27",
}


class TestDriftDetection:
    """Detect when tested error handling code changes.

    These tests fail when the source code sections covered by error path tests
    are modified. This is a maintenance signal, not a bug.

    When a test fails:
    1. Read the changed section in the source file
    2. Check if the error handling behavior changed
    3. Update the corresponding error path tests if needed
    4. Run: uv run pytest packages/epistemic/tests/ -v
    5. Update the checksum below
    """

    @pytest.mark.parametrize(
        "section",
        TRACKED_SECTIONS,
        ids=[s["description"] for s in TRACKED_SECTIONS],
    )
    def test_section_unchanged(self, section: dict) -> None:
        filepath = section["file"]
        assert filepath.exists(), f"Source file not found: {filepath}"

        extracted = _extract_section(
            filepath,
            section["start"],
            section.get("end"),
            section.get("num_lines"),
        )
        assert extracted, (
            f"Could not find section starting with '{section['start']}' in {filepath.name}"
        )

        current_hash = _sha256(extracted)

        expected = EXPECTED_CHECKSUMS.get(section["description"])
        if expected is None:
            pytest.fail(
                f"No checksum recorded for '{section['description']}'. "
                f"Current hash: {current_hash}\n"
                f"Add to EXPECTED_CHECKSUMS dict."
            )

        assert current_hash == expected, (
            f"\n{'=' * 60}\n"
            f"DRIFT DETECTED: {section['description']}\n"
            f"{'=' * 60}\n"
            f"File: {filepath.name}\n"
            f"Expected hash: {expected}\n"
            f"Current hash:  {current_hash}\n"
            f"\n"
            f"The error handling code has changed. Please:\n"
            f"1. Review the change in {filepath.name}\n"
            f"2. Update tests in {section['test_file']}\n"
            f"3. Update EXPECTED_CHECKSUMS['{section['description']}'] = '{current_hash}'\n"
            f"{'=' * 60}"
        )
