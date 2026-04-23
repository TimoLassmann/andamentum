#!/usr/bin/env python3
"""
Document Issue Models for structured document analysis.

This module provides models for representing document issues, similar to
code review comments or editorial feedback. Issues are categorized by
severity (major/minor/suggestion) and domain (structure/novelty/clarity).
"""

import uuid
import logging
from typing import List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ValidationInfo

# Logger for this module
logger = logging.getLogger(__name__)


class DocumentIssue(BaseModel):
    """
    Represents a specific issue or recommendation found in a document.

    Similar to code review comments, each issue has a type, location,
    and detailed description to help authors improve their document.
    """

    issue_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Issue classification
    issue_type: Literal["major", "minor", "suggestion", "strength"] = Field(
        ...,
        description="Severity/type of the issue: major (critical problems), minor (improvements needed), suggestion (optional enhancements), strength (positive aspects)",
    )

    category: str = Field(
        ..., description="Domain category like 'structure', 'novelty', 'methodology', 'clarity', 'style', 'technical'"
    )

    # Issue content
    title: str = Field(..., description="Brief, clear title summarizing the issue (like a commit message)")

    description: str = Field(..., description="Detailed explanation of the issue and why it matters")

    recommendation: Optional[str] = Field(None, description="Specific actionable advice for addressing the issue")

    # Location and context
    location: Optional[str] = Field(
        None, description="Where the issue occurs: 'Introduction', 'Section 2.3', 'Overall structure', 'Abstract', etc."
    )

    # Quality metadata
    confidence: float = Field(0.8, ge=0.0, le=1.0, description="Confidence in this issue assessment (0.0-1.0)")

    priority: Literal["high", "medium", "low"] = Field("medium", description="Priority level for addressing this issue")

    # Attribution
    agent_type: str = Field(
        ..., description="Type of agent that identified this issue (novelty, structure, scientific_review, etc.)"
    )

    timestamp: datetime = Field(default_factory=datetime.now)

    # Clustering metadata fields
    cluster_id: Optional[int] = Field(
        None,
        description="Cluster ID if this issue is part of a cluster (-1 for noise/singletons, None if not clustered)",
    )

    is_cluster_representative: bool = Field(
        False, description="Whether this issue is the representative for its cluster"
    )

    cluster_size: Optional[int] = Field(None, description="Number of similar issues in the same cluster")

    related_agents: List[str] = Field(
        default_factory=list, description="Other agent types that identified similar issues in the same cluster"
    )

    def __str__(self) -> str:
        """Human-readable representation of the issue."""
        priority_icon = {"high": "🔴", "medium": "🟡", "low": "🔵"}
        type_icon = {"major": "❌", "minor": "⚠️", "suggestion": "💡", "strength": "✅"}

        icon = f"{priority_icon.get(self.priority, '⚪')} {type_icon.get(self.issue_type, '📝')}"
        location_str = f" ({self.location})" if self.location else ""

        return f"{icon} {self.title}{location_str}"


class DocumentIssueCollection(BaseModel):
    """
    Collection of issues for a document, with metadata and organization.

    Provides methods to group, filter, and format issues for presentation.
    """

    issues: List[DocumentIssue] = Field(default_factory=list)
    document_title: Optional[str] = Field(None)
    analysis_timestamp: datetime = Field(default_factory=datetime.now)

    def add_issue(self, issue: DocumentIssue):
        """Add an issue to the collection."""
        self.issues.append(issue)

    def get_issues_by_type(self, issue_type: str) -> List[DocumentIssue]:
        """Get all issues of a specific type."""
        return [issue for issue in self.issues if issue.issue_type == issue_type]

    def get_issues_by_category(self, category: str) -> List[DocumentIssue]:
        """Get all issues in a specific category."""
        return [issue for issue in self.issues if issue.category.lower() == category.lower()]

    def get_issues_by_agent(self, agent_type: str) -> List[DocumentIssue]:
        """Get all issues from a specific agent."""
        return [issue for issue in self.issues if issue.agent_type == agent_type]

    def get_issues_by_priority(self, priority: str) -> List[DocumentIssue]:
        """Get all issues of a specific priority."""
        return [issue for issue in self.issues if issue.priority == priority]

    def get_high_priority_issues(self) -> List[DocumentIssue]:
        """Get all major issues and high-priority items."""
        return [issue for issue in self.issues if issue.issue_type == "major" or issue.priority == "high"]

    def get_summary_stats(self) -> dict:
        """Get summary statistics about the issues."""
        if not self.issues:
            return {"total": 0}

        stats = {
            "total": len(self.issues),
            "by_type": {},
            "by_priority": {},
            "by_category": {},
            "by_agent": {},
            "avg_confidence": sum(issue.confidence for issue in self.issues) / len(self.issues),
        }

        for issue in self.issues:
            # Count by type
            stats["by_type"][issue.issue_type] = stats["by_type"].get(issue.issue_type, 0) + 1

            # Count by priority
            stats["by_priority"][issue.priority] = stats["by_priority"].get(issue.priority, 0) + 1

            # Count by category
            stats["by_category"][issue.category] = stats["by_category"].get(issue.category, 0) + 1

            # Count by agent
            stats["by_agent"][issue.agent_type] = stats["by_agent"].get(issue.agent_type, 0) + 1

        return stats

    def format_as_markdown(self) -> str:
        """
        Format all issues as a structured markdown document.

        Returns:
            Formatted markdown string ready for prepending to document
        """
        if not self.issues:
            return "# Document Review\n\nNo issues identified.\n"

        lines = []

        # Header
        title = self.document_title or "Document"
        lines.append(f"# {title} - Review Issues")
        lines.append("")
        lines.append(f"*Generated on {self.analysis_timestamp.strftime('%Y-%m-%d %H:%M')}*")
        lines.append("")

        # AI Transparency Disclaimer
        lines.append("## ⚠️ AI-Generated Analysis Notice")
        lines.append("")
        lines.append("This review was automatically generated by AI and has important limitations:")
        lines.append("")
        lines.append(
            "- **No Internet Access**: The AI cannot check current research, recent publications, or verify facts online"
        )
        lines.append(
            "- **Limited Knowledge**: The AI's knowledge may be outdated and doesn't include the latest developments in your field"
        )
        lines.append(
            "- **Human Review Required**: All suggestions should be checked by experts and validated against current literature"
        )
        lines.append(
            "- **Starting Point Only**: This analysis is meant to help guide your review process, not replace human judgment"
        )
        lines.append("")
        lines.append(
            "*Please treat these suggestions as preliminary feedback that requires your professional verification.*"
        )
        lines.append("")
        lines.append("---")
        lines.append("")

        # Show simple issue count if there are issues
        if self.issues:
            stats = self.get_summary_stats()
            total = stats["total"]
            major = stats["by_type"].get("major", 0)
            if major > 0:
                lines.append(f"*Found {total} issues including {major} major issues requiring attention*")
            else:
                lines.append(f"*Found {total} issues and suggestions for improvement*")
            lines.append("")
            lines.append("---")
            lines.append("")

        # Check if issues are organized by clusters
        has_clusters = any(hasattr(issue, "cluster_id") and issue.cluster_id is not None for issue in self.issues)

        if has_clusters:
            self._format_clustered_issues(lines)
        else:
            self._format_traditional_issues(lines)

        # Footer
        lines.append("## 📄 Document Analysis Complete")
        lines.append("")
        lines.append(
            "*The following pages contain the original document with tracked changes and comments from the editing agents.*"
        )

        return "\n".join(lines)

    def _format_clustered_issues(self, lines: list[str]) -> None:
        """Format issues organized by clusters with visual hierarchy."""
        # Group issues by type and cluster
        issue_order = ["major", "minor", "suggestion", "strength"]
        type_headers = {
            "major": "## Major Issues (Requires Attention)",
            "minor": "## Minor Issues (Recommended Fixes)",
            "suggestion": "## Suggestions (Optional Improvements)",
            "strength": "## Document Strengths",
        }

        for issue_type in issue_order:
            type_issues = self.get_issues_by_type(issue_type)
            if not type_issues:
                continue

            lines.append(type_headers[issue_type])
            lines.append("")

            # Group by clusters and singletons
            clusters = {}
            singletons = []

            for issue in type_issues:
                if hasattr(issue, "cluster_id") and issue.cluster_id is not None:
                    cluster_id = issue.cluster_id
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    clusters[cluster_id].append(issue)
                else:
                    singletons.append(issue)

            # Format clustered issues
            for cluster_id, cluster_issues in clusters.items():
                if len(cluster_issues) <= 1:
                    # Single issue in cluster, treat as singleton
                    singletons.extend(cluster_issues)
                    continue

                # Determine cluster theme from representative
                representative = next(
                    (
                        issue
                        for issue in cluster_issues
                        if hasattr(issue, "is_cluster_representative") and issue.is_cluster_representative
                    ),
                    cluster_issues[0],
                )

                # Get related agents
                all_agents = {issue.agent_type for issue in cluster_issues}
                agent_display_names = {
                    "core_scientific_merit": "Scientific Merit",
                    "methodology": "Methodology",
                    "results_interpretation": "Results Analysis",
                    "clarity_accessibility": "Clarity & Structure",
                    "academic_writing": "Academic Writing",
                    "style": "Style & Flow",
                    "grammar": "Grammar",
                    "technical": "Technical",
                    "polish": "Polish",
                }
                agent_names = [agent_display_names.get(agent, agent.replace("_", " ").title()) for agent in all_agents]

                # Clean cluster header
                cluster_theme = self._extract_cluster_theme(representative.title)
                lines.append(f"#### Cluster {cluster_id + 1}: {cluster_theme} ({len(cluster_issues)} related issues)")
                lines.append(f"*Multiple agents: {', '.join(sorted(n for n in agent_names if n))}*")
                lines.append("")

                # Format representative issue first
                self._format_single_issue(lines, representative, is_representative=True)

                # Format related issues in cluster
                related_issues = [
                    issue
                    for issue in cluster_issues
                    if not (hasattr(issue, "is_cluster_representative") and issue.is_cluster_representative)
                ]

                if related_issues:
                    lines.append("  **Related Issues in this cluster:**")
                    lines.append("")

                    # Sort related by confidence
                    related_issues.sort(key=lambda x: -x.confidence)

                    for issue in related_issues:
                        self._format_single_issue(lines, issue, is_representative=False, is_related=True)

                # Add horizontal line after cluster with extra spacing
                lines.append("***")
                lines.append("")
                lines.append("")

            # Format singleton issues
            if singletons:
                if clusters:  # Only show header if there were clusters above
                    lines.append(f"#### Individual {issue_type.title()} Issues")
                    lines.append("")

                # Sort singletons by priority and confidence
                priority_order = {"high": 0, "medium": 1, "low": 2}
                singletons.sort(key=lambda x: (priority_order.get(x.priority, 3), -x.confidence, x.title))

                for issue in singletons:
                    self._format_single_issue(lines, issue, is_representative=False)

    def _format_traditional_issues(self, lines: list[str]) -> None:
        """Format issues using traditional grouping by type."""
        # Group issues by type for presentation
        issue_order = ["major", "minor", "suggestion", "strength"]
        type_headers = {
            "major": "## Major Issues (Requires Attention)",
            "minor": "## Minor Issues (Recommended Fixes)",
            "suggestion": "## Suggestions (Optional Improvements)",
            "strength": "## Document Strengths",
        }

        for issue_type in issue_order:
            type_issues = self.get_issues_by_type(issue_type)
            if not type_issues:
                continue

            lines.append(type_headers[issue_type])
            lines.append("")

            # Sort by priority within type
            priority_order = {"high": 0, "medium": 1, "low": 2}
            type_issues.sort(key=lambda x: (priority_order.get(x.priority, 3), x.title))

            for issue in type_issues:
                self._format_single_issue(lines, issue, is_representative=False)

    def _format_single_issue(
        self, lines: list[str], issue: DocumentIssue, is_representative: bool = False, is_related: bool = False
    ) -> None:
        """Format a single issue with improved navigation hierarchy."""

        # Clean agent display names (no emojis)
        agent_display_names = {
            "core_scientific_merit": "Scientific Merit",
            "methodology": "Methodology",
            "results_interpretation": "Results Analysis",
            "clarity_accessibility": "Clarity & Structure",
            "academic_writing": "Academic Writing",
            "style": "Style & Flow",
            "grammar": "Grammar",
            "technical": "Technical",
            "polish": "Polish",
        }
        agent_name = agent_display_names.get(issue.agent_type, issue.agent_type.replace("_", " ").title())

        # Clean confidence display
        confidence_pct = issue.confidence * 100 if issue.confidence <= 1.0 else issue.confidence
        confidence_text = f"{confidence_pct:.0f}%"

        # Location and priority info for title
        location_str = f" *({issue.location})*" if issue.location else ""
        priority_suffix = " — High Priority" if issue.priority == "high" else ""

        # Issue title with appropriate heading level - CONTENT FIRST
        if is_representative:
            heading_level = "###"
        elif is_related:
            heading_level = "####"
        else:
            heading_level = "###"

        # Indentation for related issues in clusters
        indent = "  " if is_related else ""

        # 1. TITLE (most important - immediate scanning)
        lines.append(f"{indent}{heading_level} {issue.title}{location_str}{priority_suffix}")
        lines.append("")

        # 2. DESCRIPTION (core content - what the user needs to know) - INDENTED slightly (0.5cm = ~0.2in)
        lines.append(f"{indent}  {issue.description}")
        lines.append("")

        # 3. RECOMMENDATION (actionable - what to do) - Same level as title
        if issue.recommendation:
            lines.append(f"{indent}**Recommendation:**")
            lines.append(f"{indent}  {issue.recommendation}")
            lines.append("")

        # 4. METADATA (reference info) - Right-aligned, grey, non-bold
        metadata_parts = [
            f"Source: {agent_name}",
            f"Confidence: {confidence_text}",
            f"Category: {issue.category.title()}",
        ]

        # Add related agents info if available
        if hasattr(issue, "related_agents") and issue.related_agents:
            clean_related_names = [
                agent_display_names.get(agent, agent.replace("_", " ").title()) for agent in issue.related_agents
            ]
            if clean_related_names:
                metadata_parts.append(f"Also identified by: {', '.join(clean_related_names)}")

        metadata_line = " | ".join(metadata_parts)
        # Use special marker for right-aligned grey text
        lines.append(f"{indent}~~~Analysis Details:~~~")
        lines.append(f"{indent}~~~{metadata_line}~~~")
        lines.append("")

        # Add extra vertical spacing between issues using spacing marker
        lines.append("~~~SPACING~~~")
        lines.append("~~~SPACING~~~")

    def _extract_cluster_theme(self, representative_title: str) -> str:
        """Extract a thematic name for the cluster from the representative title."""
        # Simple heuristic to create cluster themes
        title_lower = representative_title.lower()

        # Common themes in academic/document review
        if any(word in title_lower for word in ["method", "approach", "design"]):
            return "Methodology Concerns"
        elif any(word in title_lower for word in ["sample", "data", "statistical"]):
            return "Data Analysis Issues"
        elif any(word in title_lower for word in ["result", "finding", "conclusion"]):
            return "Results & Conclusions"
        elif any(word in title_lower for word in ["clarity", "structure", "organization"]):
            return "Document Structure"
        elif any(word in title_lower for word in ["reference", "citation", "literature"]):
            return "Literature & Citations"
        elif any(word in title_lower for word in ["novelty", "contribution", "significance"]):
            return "Scientific Merit"
        elif any(word in title_lower for word in ["grammar", "style", "writing"]):
            return "Writing Quality"
        else:
            # Default: use first few words of title
            words = representative_title.split()[:3]
            return " ".join(words) if words else "General Issues"

    def clear(self):
        """Clear all issues."""
        self.issues.clear()

    def __len__(self) -> int:
        """Return number of issues."""
        return len(self.issues)

    def __bool__(self) -> bool:
        """Return True if there are issues."""
        return len(self.issues) > 0


def validate_issue_content(issue: DocumentIssue) -> tuple[bool, str]:
    """
    Validate that an issue has appropriate content.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not issue.title or len(issue.title.strip()) == 0:
        return False, "Issue title cannot be empty"

    if not issue.description or len(issue.description.strip()) == 0:
        return False, "Issue description cannot be empty"

    if not issue.category or len(issue.category.strip()) == 0:
        return False, "Issue category cannot be empty"

    if len(issue.title) > 200:
        return False, "Issue title too long (max 200 characters)"

    if len(issue.description) < 20:
        return False, "Issue description too short (min 20 characters)"

    if not (0.0 <= issue.confidence <= 1.0):
        return False, f"Confidence must be between 0.0 and 1.0, got {issue.confidence}"

    return True, "Valid"


class LimitedDocumentIssueList(BaseModel):
    """
    Container for document issues with quantity limits and smart truncation.

    This model ensures that agents don't generate excessive numbers of issues
    while preserving a balanced distribution across issue types.
    """

    max_items: int = Field(default=15, ge=5, le=30, description="Maximum number of issues allowed")
    issues: List[DocumentIssue] = Field(default_factory=list)

    @field_validator("issues", mode="after")
    def validate_issue_count_and_quality(cls, v: List[DocumentIssue], info: ValidationInfo) -> List[DocumentIssue]:
        """Validate and limit the number of issues, preserving quality and distribution."""
        if not v:
            return v

        # Get max_items from the model (default to 15 if not set)
        max_items = info.data.get("max_items", 15) if info.data else 15

        if len(v) <= max_items:
            return v

        # Need to truncate - prioritize by importance and maintain distribution
        return cls._smart_truncate_issues(v, max_items)

    @classmethod
    def _smart_truncate_issues(cls, issues: List[DocumentIssue], max_items: int) -> List[DocumentIssue]:
        """
        Smart truncation that maintains issue type distribution and selects highest quality issues.
        """
        # Group by issue type
        by_type = {
            "major": [i for i in issues if i.issue_type == "major"],
            "minor": [i for i in issues if i.issue_type == "minor"],
            "suggestion": [i for i in issues if i.issue_type == "suggestion"],
            "strength": [i for i in issues if i.issue_type == "strength"],
        }

        # Calculate ideal distribution (preserving relative proportions)
        total_original = len(issues)
        target_distribution = {
            "major": max(1, int((len(by_type["major"]) / total_original) * max_items)),
            "minor": max(1, int((len(by_type["minor"]) / total_original) * max_items)),
            "suggestion": max(1, int((len(by_type["suggestion"]) / total_original) * max_items)),
            "strength": max(1, int((len(by_type["strength"]) / total_original) * max_items)),
        }

        # Adjust if totals don't match (rounding issues)
        current_total = sum(target_distribution.values())
        if current_total > max_items:
            # Reduce largest category first
            largest_type = max(target_distribution.keys(), key=lambda k: target_distribution[k])
            target_distribution[largest_type] -= current_total - max_items
        elif current_total < max_items:
            # Add to major issues first (most important)
            if by_type["major"]:
                target_distribution["major"] += max_items - current_total

        # Select best issues from each type
        selected = []
        for issue_type, target_count in target_distribution.items():
            type_issues = by_type[issue_type]
            if not type_issues:
                continue

            # Sort by priority and confidence (high priority, high confidence first)
            priority_order = {"high": 3, "medium": 2, "low": 1}
            type_issues.sort(
                key=lambda x: (
                    priority_order.get(x.priority, 1),  # Priority first
                    x.confidence,  # Then confidence
                    -len(x.description or ""),  # Then content richness (desc)
                ),
                reverse=True,
            )

            # Take top N for this type
            selected.extend(type_issues[:target_count])

        logger.info(f"Issue truncation: {len(issues)} → {len(selected)} issues (max: {max_items})")
        logger.info(
            f"Distribution: {len([i for i in selected if i.issue_type == 'major'])} major, "
            f"{len([i for i in selected if i.issue_type == 'minor'])} minor, "
            f"{len([i for i in selected if i.issue_type == 'suggestion'])} suggestions, "
            f"{len([i for i in selected if i.issue_type == 'strength'])} strengths"
        )

        return selected

    def add_issue(self, issue: DocumentIssue) -> None:
        """Add an issue, automatically managing limits."""
        self.issues.append(issue)
        # Re-validate to trigger truncation if needed
        mock_info = type("MockInfo", (), {"data": {"max_items": self.max_items}})()
        self.issues = self.__class__.validate_issue_count_and_quality(
            self.issues,
            mock_info,  # type: ignore[arg-type]
        )

    def extend_issues(self, new_issues: List[DocumentIssue]) -> None:
        """Add multiple issues, automatically managing limits."""
        self.issues.extend(new_issues)
        # Re-validate to trigger truncation if needed
        mock_info = type("MockInfo", (), {"data": {"max_items": self.max_items}})()
        self.issues = self.__class__.validate_issue_count_and_quality(
            self.issues,
            mock_info,  # type: ignore[arg-type]
        )

    def get_summary_stats(self) -> dict:
        """Get summary statistics about the issues."""
        if not self.issues:
            return {"total_issues": 0, "by_type": {}, "truncated": False}

        by_type = {}
        for issue in self.issues:
            by_type[issue.issue_type] = by_type.get(issue.issue_type, 0) + 1

        return {
            "total_issues": len(self.issues),
            "by_type": by_type,
            "max_allowed": self.max_items,
            "truncated": len(self.issues) == self.max_items,
        }
