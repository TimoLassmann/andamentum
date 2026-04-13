"""Plan validators for epistemic workflow planning.

Validates plan structure and dependencies before expensive execution:
- validate_plan_dependencies: Check workitem dependency graph is valid

Architecture: Layer 1 (framework-agnostic, no model calls)
"""

from typing import Dict, List, Any

from .types import ValidationResult


class PlanValidators:
    """Validates plan structure and dependency ordering.

    Fail-fast validation - better to reject a bad plan upfront than
    discover traceability violations after expensive evidence collection.
    """

    def validate_plan_dependencies(
        self, workitems: List[Dict[str, Any]]
    ) -> ValidationResult:
        """Validate that plan workitems have valid dependency ordering.

        Called after PLAN_TASK to catch ordering errors before wasting LLM computation.
        This is fail-fast validation - better to reject a bad plan upfront than
        discover traceability violations after expensive evidence collection.

        Checks:
        1. No dangling references - all dependencies must exist in the plan
        2. No cycles - dependency graph must be acyclic
        3. Logical ordering - evidence collection before claim proposals

        Args:
            workitems: List of workitem dicts from planner output

        Returns:
            ValidationResult with errors if plan is invalid
        """
        result = ValidationResult(valid=True)

        # Build ID lookup
        workitem_ids = {
            wi.get("workitem_id") for wi in workitems if wi.get("workitem_id")
        }

        # Check 1: No dangling dependency references
        for wi in workitems:
            wi_id = wi.get("workitem_id", "unknown")
            for dep_id in wi.get("dependencies", []):
                if dep_id not in workitem_ids:
                    result.add_error(
                        "PLAN_001",
                        f"WorkItem {wi_id[:8]} depends on unknown workitem: {dep_id[:8]}",
                        workitem_id=wi_id,
                        missing_dependency=dep_id,
                    )

        # Check 2: No cycles (topological sort attempt)
        # Build adjacency list: workitem -> list of workitems that depend on it
        dependents: Dict[str, List[str]] = {
            wi.get("workitem_id", ""): [] for wi in workitems
        }
        in_degree: Dict[str, int] = {wi.get("workitem_id", ""): 0 for wi in workitems}

        for wi in workitems:
            wi_id = wi.get("workitem_id", "")
            for dep_id in wi.get("dependencies", []):
                if dep_id in dependents:  # Only count valid dependencies
                    dependents[dep_id].append(wi_id)
                    in_degree[wi_id] = in_degree.get(wi_id, 0) + 1

        # Kahn's algorithm for cycle detection
        queue = [wid for wid, deg in in_degree.items() if deg == 0]
        processed = 0

        while queue:
            current = queue.pop(0)
            processed += 1
            for dependent in dependents.get(current, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if processed < len(workitems):
            result.add_error(
                "PLAN_002",
                f"Dependency cycle detected - {len(workitems) - processed} workitems unreachable",
            )

        # Check 3: Logical ordering - evidence before claims
        # Find all COLLECT_EVIDENCE and PROPOSE_CLAIMS workitems
        evidence_workitem_ids: set[str] = set()
        claim_workitem_ids: set[str] = set()

        for wi in workitems:
            op_type = wi.get("operation_type", "")
            wi_id = str(wi.get("workitem_id", ""))
            if op_type in ("collect_evidence", "extract_evidence"):
                evidence_workitem_ids.add(wi_id)
            elif op_type == "propose_claims":
                claim_workitem_ids.add(wi_id)

        # For each PROPOSE_CLAIMS, check it has evidence collection in its dependency chain
        if evidence_workitem_ids and claim_workitem_ids:
            # Build reachability: what can each workitem reach via dependencies?
            def get_all_dependencies(wi_id: str, visited: set[str]) -> set[str]:
                """Get transitive closure of dependencies."""
                if wi_id in visited:
                    return set()
                visited.add(wi_id)
                deps: set[str] = set()
                for wi in workitems:
                    if wi.get("workitem_id") == wi_id:
                        for dep in wi.get("dependencies", []):
                            deps.add(dep)
                            deps.update(get_all_dependencies(dep, visited))
                        break
                return deps

            for wi in workitems:
                if wi.get("operation_type") == "propose_claims":
                    wi_id = wi.get("workitem_id", "")
                    visited: set[str] = set()
                    all_deps: set[str] = get_all_dependencies(wi_id, visited)

                    # Check if any evidence workitem is in the dependency chain
                    if not all_deps.intersection(evidence_workitem_ids):
                        result.add_warning(
                            "PLAN_010",
                            f"PROPOSE_CLAIMS {wi_id[:8]} has no evidence collection in dependency chain - "
                            f"claims may lack traceability",
                            workitem_id=wi_id,
                        )

        return result
