#!/usr/bin/env python3
"""
Change attribution tracking for multi-agent document editing.

This module provides token-level attribution tracking to preserve which agent
made which specific changes, preventing attribution loss when multiple agents
edit the same paragraph.
"""

import re
import difflib
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from .token_processor import TokenProcessor


class ChangeType(Enum):
    """Types of changes that can be tracked."""

    UNCHANGED = "unchanged"
    INSERTION = "ins"
    DELETION = "del"


@dataclass
class TokenAttribution:
    """Attribution information for a single token."""

    token: str
    author: str
    change_type: ChangeType
    confidence: float = 1.0
    position: int = 0  # Position in the text

    def __str__(self) -> str:
        """Return a debug-friendly representation of the attribution.

        Format: ``token[author:change_type]`` — useful for printing
        attribution traces during debugging of multi-agent edits.
        """
        return f"{self.token}[{self.author}:{self.change_type.value}]"


class ChangeAttributionTracker:
    """
    Tracks attribution of changes at the token level for multi-agent editing.

    This class maintains a mapping of which agent made which specific token-level
    changes, allowing for proper attribution in track changes even when multiple
    agents edit the same paragraph.
    """

    # Use centralized token processor

    def __init__(self, initial_text: str = "", initial_author: str = "Original"):
        """
        Initialize the attribution tracker.

        Args:
            initial_text: Initial text content
            initial_author: Author of the initial text
        """
        self.initial_author = initial_author
        self.attributions: List[TokenAttribution] = []
        self.change_history: List[Dict] = []

        if initial_text:
            self._initialize_from_text(initial_text, initial_author)

    def _initialize_from_text(self, text: str, author: str):
        """Initialize attributions from initial text."""
        tokens = TokenProcessor.tokenize(text)
        self.attributions = [
            TokenAttribution(token=token, author=author, change_type=ChangeType.UNCHANGED, position=i)
            for i, token in enumerate(tokens)
        ]

    def track_change(self, original_text: str, new_text: str, author: str) -> None:
        """
        Track changes made by an agent, updating token-level attributions.

        Args:
            original_text: Text before the change
            new_text: Text after the change
            author: Agent who made the change
        """
        original_tokens = TokenProcessor.tokenize(original_text)
        new_tokens = TokenProcessor.tokenize(new_text)

        # Record the change in history
        change_record = {
            "author": author,
            "original_tokens": len(original_tokens),
            "new_tokens": len(new_tokens),
            "timestamp": len(self.change_history),
        }
        self.change_history.append(change_record)

        # Generate new attributions with unified patch attribution
        new_attributions = self._generate_unified_patch_attributions(original_tokens, new_tokens, author)

        self.attributions = new_attributions

    def _generate_attributions_from_diff(
        self, original_tokens: List[str], new_tokens: List[str], author: str
    ) -> List[TokenAttribution]:
        """
        Generate new attributions based on token-level diff.

        Args:
            original_tokens: Original tokens
            new_tokens: New tokens after change
            author: Agent making the change

        Returns:
            Updated list of token attributions
        """
        new_attributions = []
        original_idx = 0
        new_idx = 0

        # Use difflib to find the optimal sequence of operations
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, original_tokens, new_tokens).get_opcodes():
            if op == "equal":
                # Unchanged tokens - preserve original attributions
                for k in range(i1, i2):
                    if original_idx < len(self.attributions):
                        # Preserve existing attribution (including agent attributions)
                        existing = self.attributions[original_idx]
                        new_attributions.append(
                            TokenAttribution(
                                token=existing.token,
                                author=existing.author,  # Keep existing agent attribution
                                change_type=existing.change_type,
                                position=len(new_attributions),
                            )
                        )
                    else:
                        # New unchanged token - attribute to current author if this is an agent edit
                        # This handles cases where the document has grown
                        token_author = author if "Specialist" in author else self.initial_author
                        new_attributions.append(
                            TokenAttribution(
                                token=original_tokens[k],
                                author=token_author,
                                change_type=ChangeType.UNCHANGED,
                                position=len(new_attributions),
                            )
                        )
                    original_idx += 1

            elif op == "delete":
                # Deleted tokens - mark them but don't include in new attributions
                # (they'll appear as deletions in track changes)
                original_idx += i2 - i1

            elif op == "insert":
                # Inserted tokens - attribute to current author
                for k in range(j1, j2):
                    new_attributions.append(
                        TokenAttribution(
                            token=new_tokens[k],
                            author=author,
                            change_type=ChangeType.INSERTION,
                            position=len(new_attributions),
                        )
                    )

            elif op == "replace":
                # Replaced tokens - deletions and insertions
                # Skip the deleted tokens (original_idx advances)
                original_idx += i2 - i1

                # Add the replacement tokens with current author
                for k in range(j1, j2):
                    new_attributions.append(
                        TokenAttribution(
                            token=new_tokens[k],
                            author=author,
                            change_type=ChangeType.INSERTION,
                            position=len(new_attributions),
                        )
                    )

        return new_attributions

    def _generate_unified_patch_attributions(
        self, original_tokens: List[str], new_tokens: List[str], author: str
    ) -> List[TokenAttribution]:
        """
        Generate new attributions with unified patch attribution.

        All tokens that are part of the same logical change get attributed to the same agent.
        This ensures consistent attribution for track changes and comments.

        Args:
            original_tokens: Original tokens
            new_tokens: New tokens after change
            author: Agent making the change

        Returns:
            Updated list of token attributions with unified patch attribution
        """
        new_attributions = []
        original_idx = 0

        # Use difflib to find the optimal sequence of operations
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, original_tokens, new_tokens).get_opcodes():
            if op == "equal":
                # Unchanged tokens - preserve original attributions
                for k in range(i1, i2):
                    if original_idx < len(self.attributions):
                        # Preserve existing attribution (including agent attributions)
                        existing = self.attributions[original_idx]
                        new_attributions.append(
                            TokenAttribution(
                                token=existing.token,
                                author=existing.author,  # Keep existing agent attribution
                                change_type=existing.change_type,
                                position=len(new_attributions),
                            )
                        )
                    else:
                        # New unchanged token - attribute to current author if this is an agent edit
                        token_author = author if "Specialist" in author else "Original"
                        new_attributions.append(
                            TokenAttribution(
                                token=original_tokens[k],
                                author=token_author,
                                change_type=ChangeType.UNCHANGED,
                                position=len(new_attributions),
                            )
                        )
                    original_idx += 1

            elif op == "delete":
                # Deleted tokens - mark them but don't include in new attributions
                # All deletions in this patch get attributed to the current agent
                original_idx += i2 - i1

            elif op == "insert":
                # Inserted tokens - ALL attributed to current agent (unified attribution)
                for k in range(j1, j2):
                    new_attributions.append(
                        TokenAttribution(
                            token=new_tokens[k],
                            author=author,  # Unified attribution: all insertions in this patch to same agent
                            change_type=ChangeType.INSERTION,
                            position=len(new_attributions),
                        )
                    )

            elif op == "replace":
                # Replaced tokens - ALL attributed to current agent (unified attribution)
                # Skip the deleted tokens (original_idx advances)
                original_idx += i2 - i1

                # Add ALL replacement tokens with current agent (unified attribution)
                for k in range(j1, j2):
                    new_attributions.append(
                        TokenAttribution(
                            token=new_tokens[k],
                            author=author,  # Unified attribution: all replacements in this patch to same agent
                            change_type=ChangeType.INSERTION,
                            position=len(new_attributions),
                        )
                    )

        return new_attributions

    def get_attribution_for_token(self, token_index: int) -> Optional[TokenAttribution]:
        """
        Get attribution for a specific token by index.

        Args:
            token_index: Index of the token

        Returns:
            TokenAttribution if found, None otherwise
        """
        if 0 <= token_index < len(self.attributions):
            return self.attributions[token_index]
        return None

    def get_attribution_for_diff_operation(self, original_tokens: List[str], new_tokens: List[str]) -> Dict[int, str]:
        """
        Get author attribution for each diff operation during track changes creation.

        This method is called during DocxEditor.write() to determine which author
        should be used for each insertion/deletion in the track changes.

        Args:
            original_tokens: Original paragraph tokens
            new_tokens: New paragraph tokens

        Returns:
            Dictionary mapping token indices to author names
        """
        author_map = {}

        # Build a mapping from current tokens back to attributions
        current_tokens = [attr.token for attr in self.attributions]

        # For each token in the new text, find its attribution
        for i, token in enumerate(new_tokens):
            author = self._find_token_author(token, i)
            author_map[i] = author

        return author_map

    def _find_token_author(self, token: str, position: int) -> str:
        """
        Find the author of a specific token at a given position.

        Args:
            token: Token to find author for
            position: Position in the text

        Returns:
            Author name
        """
        # Try to find exact match at this position
        if position < len(self.attributions):
            attr = self.attributions[position]
            if attr.token == token:
                return attr.author

        # Find first matching token with same content
        for attr in self.attributions:
            if attr.token == token:
                return attr.author

        # Default to most recent author if not found
        if self.attributions:
            # Find the most recent insertion author
            for attr in reversed(self.attributions):
                if attr.change_type == ChangeType.INSERTION:
                    return attr.author

            # Fall back to last author in history
            if self.change_history:
                return self.change_history[-1]["author"]

        return "Unknown"

    def get_current_text(self) -> str:
        """Get the current text based on attributions."""
        return "".join(attr.token for attr in self.attributions)

    def get_attribution_summary(self) -> Dict[str, Dict]:
        """
        Get summary of attributions by author.

        Returns:
            Dictionary with author statistics
        """
        summary = {}

        for attr in self.attributions:
            if attr.author not in summary:
                summary[attr.author] = {"tokens": 0, "insertions": 0, "unchanged": 0, "total_chars": 0}

            summary[attr.author]["tokens"] += 1
            summary[attr.author]["total_chars"] += len(attr.token)

            if attr.change_type == ChangeType.INSERTION:
                summary[attr.author]["insertions"] += 1
            else:
                summary[attr.author]["unchanged"] += 1

        return summary

    def merge_attributions(self, other: "ChangeAttributionTracker") -> None:
        """
        Merge attributions from another tracker.

        Args:
            other: Another ChangeAttributionTracker to merge
        """
        # For now, use simple append strategy
        # In the future, this could be more sophisticated
        self.change_history.extend(other.change_history)

        if other.attributions:
            # Use the other tracker's attributions as they're more recent
            self.attributions = other.attributions.copy()

    def debug_print(self) -> str:
        """
        Generate debug representation of current attributions.

        Returns:
            Human-readable debug string
        """
        lines = ["Attribution Tracker Debug:"]
        lines.append(f"Total tokens: {len(self.attributions)}")
        lines.append(f"Change history: {len(self.change_history)} changes")

        # Group by author
        summary = self.get_attribution_summary()
        for author, stats in summary.items():
            lines.append(f"  {author}: {stats['tokens']} tokens, {stats['insertions']} insertions")

        # Show first few tokens with attribution
        lines.append("\nFirst 10 tokens:")
        for i, attr in enumerate(self.attributions[:10]):
            lines.append(f"  {i}: {attr}")

        if len(self.attributions) > 10:
            lines.append(f"  ... and {len(self.attributions) - 10} more")

        return "\n".join(lines)

    def validate_consistency(self) -> Tuple[bool, List[str]]:
        """
        Validate the consistency of attributions.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check for missing attributions
        if not self.attributions:
            issues.append("No attributions found")

        # Check for valid authors
        for i, attr in enumerate(self.attributions):
            if not attr.author or attr.author.strip() == "":
                issues.append(f"Token {i} has empty author")

            if not attr.token:
                issues.append(f"Token {i} has empty content")

        # Check position consistency
        for i, attr in enumerate(self.attributions):
            if attr.position != i:
                issues.append(f"Token {i} has incorrect position {attr.position}")

        return len(issues) == 0, issues
