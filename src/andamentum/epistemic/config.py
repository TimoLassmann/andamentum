"""Research configuration for the epistemic system.

All tunable parameters in one place. No config files - CLI-first design.

Usage:
    # Use a preset
    config = ResearchConfig.light()
    config = ResearchConfig.comprehensive()

    # Override individual values
    config = ResearchConfig.light(adversarial_queries=4)

    # From CLI args (preset + overrides)
    config = ResearchConfig.from_args(args)
"""

from dataclasses import dataclass
from typing import Optional, Literal

# Preset names
PresetName = Literal["light", "comprehensive"]


@dataclass
class ResearchConfig:
    """All tunable parameters for epistemic research.

    These control the depth/breadth tradeoff between speed and thoroughness.

    Attributes:
        # Deep Research (per search query)
        max_iterations: Research loop iterations (search→fetch→analyze)
        max_results_per_search: Search results returned per query
        max_pages_to_fetch: Pages fetched and summarized per iteration
        relevance_threshold: Minimum relevance score to keep page summaries (0.0-1.0)

        # Evidence Collection
        evidence_depth: Depth preset for evidence gathering ("quick", "standard", "thorough")

        # Adversarial Search (the slow part!)
        adversarial_queries: Number of queries to challenge each claim
        adversarial_iterations: Deep research iterations per adversarial query

        # Claim Verification
        quote_overlap_ratio: Fraction of quote words that must appear in evidence (0.0-1.0)
        adversarial_balance_threshold: Below this, creates "strong counterevidence" uncertainty
        convergence_strength_threshold: Below this, creates "weak convergence" uncertainty

        # Workflow Control
        max_retries: Maximum retries per failed workitem
        max_workitems: Maximum workitems to execute (None = unlimited)
    """

    # === Deep Research (per query) ===
    max_iterations: int = 2
    max_results_per_search: int = 10
    max_pages_to_fetch: int = 5
    relevance_threshold: float = 0.3

    # === Evidence Collection ===
    evidence_depth: Literal["quick", "standard", "thorough"] = "standard"

    # === Adversarial Search ===
    adversarial_queries: int = 4
    adversarial_iterations: int = 1

    # === Claim Verification ===
    quote_overlap_ratio: float = 0.6
    adversarial_balance_threshold: float = 0.3
    convergence_strength_threshold: float = 0.5

    # === Workflow Control ===
    max_retries: int = 3
    max_workitems: Optional[int] = None

    # === Scope Control (principled budget derivation) ===
    # All operation budgets derive from these two parameters.
    # target_claims: how many claims to fully verify
    # The evidence budget is derived from stage gate requirements:
    #   SUPPORTED needs ≥1 evidence → 2 per claim (with buffer)
    #   PROVISIONAL needs ≥2, quality_sum ≥0.5 → 3 per claim
    #   ROBUST needs ≥3 from independent domains → 4 per claim
    target_claims: int = 5
    evidence_per_claim: int = 3

    @classmethod
    def light(cls, **overrides) -> "ResearchConfig":
        """Fast research preset - good for quick exploration.

        ~5-10 minutes per question. Less thorough but much faster.
        Good for: initial exploration, simple questions, testing.
        """
        defaults = {
            "max_iterations": 1,
            "max_results_per_search": 5,
            "max_pages_to_fetch": 3,
            "relevance_threshold": 0.4,
            "evidence_depth": "quick",
            "adversarial_queries": 2,
            "adversarial_iterations": 1,
            "max_retries": 2,
            "target_claims": 3,
            "evidence_per_claim": 2,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def comprehensive(cls, **overrides) -> "ResearchConfig":
        """Thorough research preset - for important questions.

        ~30-60 minutes per question. More thorough but slower.
        Good for: critical decisions, complex topics, publication-quality research.
        """
        defaults = {
            "max_iterations": 3,
            "max_results_per_search": 10,
            "max_pages_to_fetch": 5,
            "relevance_threshold": 0.25,
            "evidence_depth": "thorough",
            "adversarial_queries": 8,
            "adversarial_iterations": 1,
            "max_retries": 3,
            "target_claims": 8,
            "evidence_per_claim": 4,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def from_preset(cls, preset: PresetName, **overrides) -> "ResearchConfig":
        """Create config from a named preset with optional overrides."""
        if preset == "light":
            return cls.light(**overrides)
        elif preset == "comprehensive":
            return cls.comprehensive(**overrides)
        else:
            raise ValueError(
                f"Unknown preset: {preset}. Use 'light' or 'comprehensive'."
            )

    @classmethod
    def from_args(cls, args) -> "ResearchConfig":
        """Create config from argparse namespace.

        Supports both preset selection and individual overrides.
        Individual args override preset values.
        """
        # Start with preset (default: light for speed)
        preset_raw = getattr(args, "preset", "light") or "light"
        # Validate preset name (default to light if invalid)
        preset: PresetName = (
            "comprehensive" if preset_raw == "comprehensive" else "light"
        )

        # Collect explicit overrides (non-None values)
        overrides = {}

        param_mappings = {
            "max_iterations": "max_iterations",
            "max_results": "max_results_per_search",
            "max_pages": "max_pages_to_fetch",
            "relevance_threshold": "relevance_threshold",
            "evidence_depth": "evidence_depth",
            "adversarial_queries": "adversarial_queries",
            "adversarial_iterations": "adversarial_iterations",
            "quote_overlap": "quote_overlap_ratio",
            "adversarial_threshold": "adversarial_balance_threshold",
            "convergence_threshold": "convergence_strength_threshold",
            "max_retries": "max_retries",
            "max_items": "max_workitems",
        }

        for arg_name, config_name in param_mappings.items():
            value = getattr(args, arg_name, None)
            if value is not None:
                overrides[config_name] = value

        return cls.from_preset(preset, **overrides)

    @classmethod
    def add_arguments(cls, parser) -> None:
        """Add research config arguments to an argparse parser.

        Usage:
            parser = argparse.ArgumentParser()
            ResearchConfig.add_arguments(parser)
            args = parser.parse_args()
            config = ResearchConfig.from_args(args)
        """
        group = parser.add_argument_group(
            "Research Configuration",
            "Control research depth and speed. Use --preset for quick selection, "
            "or override individual parameters.",
        )

        # Preset selection
        group.add_argument(
            "--preset",
            "-p",
            choices=["light", "comprehensive"],
            default="light",
            help="Research preset: 'light' (~5-10 min, fast) or 'comprehensive' (~30-60 min, thorough). Default: light",
        )

        # Deep research params
        group.add_argument(
            "--max-iterations",
            type=int,
            metavar="N",
            help="Research loop iterations per query (light=1, comprehensive=3)",
        )
        group.add_argument(
            "--max-results",
            type=int,
            metavar="N",
            help="Search results per query (light=5, comprehensive=10)",
        )
        group.add_argument(
            "--max-pages",
            type=int,
            metavar="N",
            help="Pages to fetch per iteration (light=3, comprehensive=5)",
        )
        group.add_argument(
            "--relevance-threshold",
            type=float,
            metavar="F",
            help="Min relevance to keep summaries, 0.0-1.0 (light=0.4, comprehensive=0.25)",
        )

        # Evidence depth
        group.add_argument(
            "--evidence-depth",
            choices=["quick", "standard", "thorough"],
            help="Evidence collection depth (light=quick, comprehensive=thorough)",
        )

        # Adversarial search params
        group.add_argument(
            "--adversarial-queries",
            type=int,
            metavar="N",
            help="Queries to challenge each claim (light=2, comprehensive=8). Major speed impact!",
        )
        group.add_argument(
            "--adversarial-iterations",
            type=int,
            metavar="N",
            help="Deep research iterations per adversarial query (default=1)",
        )

        # Verification thresholds
        group.add_argument(
            "--quote-overlap",
            type=float,
            metavar="F",
            help="Quote-evidence word overlap ratio, 0.0-1.0 (default=0.6)",
        )
        group.add_argument(
            "--adversarial-threshold",
            type=float,
            metavar="F",
            help="Adversarial balance threshold for uncertainty (default=0.3)",
        )
        group.add_argument(
            "--convergence-threshold",
            type=float,
            metavar="F",
            help="Convergence strength threshold for uncertainty (default=0.5)",
        )

    def operation_budgets(self) -> dict[str, int]:
        """Derive per-operation budgets for scope-creating operations only.

        Design principle: budgets gate SCOPE (how many entities are created),
        not EXECUTION (whether existing entities are fully processed).

        Processing operations (scrutinise_claim, adversarial_search, etc.) are
        NOT budgeted here. Pattern filters already guarantee idempotency — each
        fires at most once per entity based on entity state. Capping them causes
        claims to be stranded mid-pipeline, producing incomplete data.

        The scope is controlled by:
        - plan_task: told to create target_claims × evidence_per_claim evidence stubs
        - propose_claims: told to propose target_claims claims
        Both are LLM-driven and may produce slightly more than target — that's fine,
        those extra entities will be fully processed.
        """
        k = self.target_claims
        return {
            # One-time bootstrap — safety caps only
            "clarify_question": 2,
            "conceptual_analysis": 2,
            "plan_task": 2,
            "propose_claims": 2,
            # Investigation cycling — also capped by pattern filter (investigation_count < 3)
            "investigate_claim": k * 3 + 2,
        }

    def summary(self) -> str:
        """Return a human-readable summary of key settings."""
        depth_emoji = {"quick": "🚀", "standard": "⚖️", "thorough": "🔬"}
        emoji = depth_emoji.get(self.evidence_depth, "")

        return (
            f"{emoji} Research Config: "
            f"iterations={self.max_iterations}, "
            f"pages={self.max_pages_to_fetch}, "
            f"adversarial={self.adversarial_queries}q, "
            f"depth={self.evidence_depth}"
        )

    def estimated_time_per_claim(self) -> str:
        """Rough estimate of time per claim based on settings."""
        # Each adversarial query takes ~2-3 min with full deep research
        base_per_query = 2.0  # minutes
        adversarial_time = (
            self.adversarial_queries * self.adversarial_iterations * base_per_query
        )

        # Evidence collection
        evidence_times = {"quick": 2, "standard": 5, "thorough": 10}
        evidence_time = evidence_times.get(self.evidence_depth, 5)

        total = adversarial_time + evidence_time

        if total < 10:
            return f"~{int(total)} min/claim"
        else:
            return f"~{int(total)} min/claim"
