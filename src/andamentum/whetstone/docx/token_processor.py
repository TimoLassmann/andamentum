#!/usr/bin/env python3
"""
Token processing utilities for consistent tokenization across the editor.

This module eliminates the duplication of token processing logic found
in docxeditor.py, change_attribution_tracker.py, and other components.
"""

import re
import copy
from typing import List, Optional, Dict, Iterator, Tuple
from dataclasses import dataclass
from lxml import etree  # type: ignore


@dataclass
class TokenData:
    """
    Represents a token with its associated properties.

    Consolidates token representation across different components.
    """

    text: str
    rPr: Optional[etree.Element] = None  # Run properties for DOCX
    position: int = 0  # Position in original text
    author: str = "Unknown"  # For attribution tracking
    change_type: str = "unchanged"  # For change tracking

    def copy_with_new_text(self, new_text: str) -> "TokenData":
        """Create a copy with new text but same properties."""
        return TokenData(
            text=new_text,
            rPr=copy.deepcopy(self.rPr) if self.rPr is not None else None,
            position=self.position,
            author=self.author,
            change_type=self.change_type,
        )


class TokenProcessor:
    """
    Unified token processing for consistent tokenization across the editor.

    This class eliminates the duplication of the TOKEN_REGEX pattern and
    associated processing logic found in multiple files.
    """

    # Unified token regex - matches the pattern used throughout the codebase
    TOKEN_REGEX = re.compile(r"\s+|\S+")

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """
        Tokenize text using the standard editor pattern.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens (whitespace and non-whitespace sequences)
        """
        return cls.TOKEN_REGEX.findall(text)

    @classmethod
    def tokenize_with_positions(cls, text: str) -> List[Tuple[str, int, int]]:
        """
        Tokenize text with start and end positions.

        Args:
            text: Text to tokenize

        Returns:
            List of (token, start_pos, end_pos) tuples
        """
        tokens = []
        for match in cls.TOKEN_REGEX.finditer(text):
            tokens.append((match.group(0), match.start(), match.end()))
        return tokens

    @classmethod
    def create_token_data_from_runs(cls, paragraph_element: etree.Element, text: str) -> List[TokenData]:
        """
        Create TokenData objects from a paragraph element and its text.

        This consolidates the token creation logic from docxeditor.py ParagraphData.

        Args:
            paragraph_element: XML paragraph element
            text: Full paragraph text

        Returns:
            List of TokenData objects with run properties
        """
        # Build character-to-rPr mapping (from docxeditor.py logic)
        char_rprs = cls._build_char_rpr_mapping(paragraph_element)

        # Create tokens with properties
        tokens = []
        for match in cls.TOKEN_REGEX.finditer(text):
            token_text = match.group(0)
            start_pos = match.start()

            # Get rPr for this token (use first character's rPr)
            rPr = None
            if start_pos < len(char_rprs):
                rPr = char_rprs[start_pos]
            elif char_rprs:
                # Fall back to first non-None rPr
                rPr = next((rpr for rpr in char_rprs if rpr is not None), None)

            tokens.append(TokenData(text=token_text, rPr=rPr, position=len(tokens)))

        return tokens

    @classmethod
    def _build_char_rpr_mapping(cls, paragraph_element: etree.Element) -> List[Optional[etree.Element]]:
        """
        Build mapping from character position to run properties.

        This consolidates the char_rprs logic from docxeditor.py ParagraphData.

        Args:
            paragraph_element: XML paragraph element

        Returns:
            List where index is character position, value is rPr element
        """
        from .low_level import NS

        char_rprs = []
        for run in paragraph_element.findall(".//w:r", namespaces=NS):
            rpr = run.find("w:rPr", namespaces=NS)
            run_rpr = copy.deepcopy(rpr) if rpr is not None else None

            # Get text from this run
            run_text = "".join(t.text or "" for t in run.findall(".//w:t", namespaces=NS))

            # Each character in run_text gets this run_rpr
            char_rprs.extend([run_rpr] * len(run_text))

        return char_rprs

    @classmethod
    def create_token_data_list(cls, text: str, author: str = "Unknown") -> List[TokenData]:
        """
        Create a simple list of TokenData objects from text.

        For use in attribution tracking and other contexts.

        Args:
            text: Text to tokenize
            author: Author to assign to tokens

        Returns:
            List of TokenData objects
        """
        tokens = cls.tokenize(text)
        return [TokenData(text=token, position=i, author=author) for i, token in enumerate(tokens)]

    @classmethod
    def extract_text_from_tokens(cls, tokens: List[TokenData]) -> str:
        """
        Reconstruct text from TokenData objects.

        Args:
            tokens: List of TokenData objects

        Returns:
            Reconstructed text
        """
        return "".join(token.text for token in tokens)

    @classmethod
    def filter_tokens(cls, tokens: List[TokenData], include_whitespace: bool = True) -> List[TokenData]:
        """
        Filter tokens based on criteria.

        Args:
            tokens: List of TokenData objects
            include_whitespace: Whether to include whitespace tokens

        Returns:
            Filtered list of tokens
        """
        if include_whitespace:
            return tokens

        return [token for token in tokens if not token.text.isspace()]

    @classmethod
    def find_token_by_content(cls, tokens: List[TokenData], search_text: str) -> Optional[int]:
        """
        Find the index of a token by its text content.

        Args:
            tokens: List of TokenData objects
            search_text: Text to search for

        Returns:
            Index of first matching token or None
        """
        for i, token in enumerate(tokens):
            if token.text == search_text:
                return i
        return None

    @classmethod
    def find_tokens_by_pattern(cls, tokens: List[TokenData], pattern: str) -> List[int]:
        """
        Find tokens matching a regex pattern.

        Args:
            tokens: List of TokenData objects
            pattern: Regex pattern to match

        Returns:
            List of indices of matching tokens
        """
        regex = re.compile(pattern)
        matches = []
        for i, token in enumerate(tokens):
            if regex.search(token.text):
                matches.append(i)
        return matches

    @classmethod
    def merge_adjacent_tokens(cls, tokens: List[TokenData], merge_whitespace: bool = True) -> List[TokenData]:
        """
        Merge adjacent tokens with same properties.

        Args:
            tokens: List of TokenData objects
            merge_whitespace: Whether to merge whitespace tokens

        Returns:
            List with merged tokens
        """
        if not tokens:
            return []

        merged = [tokens[0]]

        for token in tokens[1:]:
            last = merged[-1]

            # Check if tokens can be merged
            can_merge = (
                last.author == token.author
                and last.change_type == token.change_type
                and (merge_whitespace or not (last.text.isspace() and token.text.isspace()))
            )

            if can_merge:
                # Merge with previous token
                merged_text = last.text + token.text
                merged[-1] = last.copy_with_new_text(merged_text)
            else:
                # Add as separate token
                merged.append(token)

        return merged

    @classmethod
    def split_token_at_position(cls, token: TokenData, position: int) -> Tuple[TokenData, TokenData]:
        """
        Split a token at a specific character position.

        Args:
            token: TokenData to split
            position: Character position to split at

        Returns:
            Tuple of (left_token, right_token)
        """
        if position <= 0:
            return TokenData(text="", rPr=token.rPr, author=token.author), token
        if position >= len(token.text):
            return token, TokenData(text="", rPr=token.rPr, author=token.author)

        left_text = token.text[:position]
        right_text = token.text[position:]

        left_token = token.copy_with_new_text(left_text)
        right_token = token.copy_with_new_text(right_text)

        return left_token, right_token

    @classmethod
    def estimate_token_count(cls, text: str) -> int:
        """
        Estimate token count using the standard pattern.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0
        return len(cls.tokenize(text))

    @classmethod
    def get_token_statistics(cls, tokens: List[TokenData]) -> Dict[str, int]:
        """
        Get statistics about a token list.

        Args:
            tokens: List of TokenData objects

        Returns:
            Dictionary with token statistics
        """
        if not tokens:
            return {"total_tokens": 0, "whitespace_tokens": 0, "word_tokens": 0, "total_chars": 0, "unique_authors": 0}

        total_tokens = len(tokens)
        whitespace_tokens = sum(1 for t in tokens if t.text.isspace())
        word_tokens = total_tokens - whitespace_tokens
        total_chars = sum(len(t.text) for t in tokens)
        unique_authors = len(set(t.author for t in tokens))

        return {
            "total_tokens": total_tokens,
            "whitespace_tokens": whitespace_tokens,
            "word_tokens": word_tokens,
            "total_chars": total_chars,
            "unique_authors": unique_authors,
        }


class TokenIterator:
    """
    Iterator for processing tokens with context.

    Provides convenient iteration with lookahead/lookback capabilities.
    """

    def __init__(self, tokens: List[TokenData]):
        """
        Initialize the iterator.

        Args:
            tokens: List of TokenData objects to iterate over
        """
        self.tokens = tokens
        self.position = 0

    def __iter__(self) -> Iterator[TokenData]:
        """Return self as iterator."""
        return self

    def __next__(self) -> TokenData:
        """Get next token."""
        if self.position >= len(self.tokens):
            raise StopIteration

        token = self.tokens[self.position]
        self.position += 1
        return token

    def peek(self, offset: int = 0) -> Optional[TokenData]:
        """
        Peek at a token without advancing position.

        Args:
            offset: Offset from current position (0 = current, 1 = next, -1 = previous)

        Returns:
            TokenData at offset position or None if out of bounds
        """
        target_pos = self.position + offset
        if 0 <= target_pos < len(self.tokens):
            return self.tokens[target_pos]
        return None

    def has_next(self) -> bool:
        """Check if there are more tokens."""
        return self.position < len(self.tokens)

    def current_index(self) -> int:
        """Get current position index."""
        return self.position

    def reset(self) -> None:
        """Reset iterator to beginning."""
        self.position = 0
