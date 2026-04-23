#!/usr/bin/env python3
"""
Text processing utilities for consistent text operations across the editor.

This module consolidates text processing, similarity calculation, and pattern
matching logic found across multiple components.
"""

import re
import difflib
from typing import List, Optional, Tuple, Dict, Set
from dataclasses import dataclass
from enum import Enum


class SimilarityMethod(Enum):
    """Methods for calculating text similarity."""

    SEQUENCE_MATCHER = "sequence_matcher"
    WORD_OVERLAP = "word_overlap"
    CHARACTER_NGRAM = "character_ngram"
    COMBINED = "combined"


@dataclass
class TextMatch:
    """Represents a text match with confidence and location information."""

    text: str
    start_pos: int
    end_pos: int
    confidence: float
    method: SimilarityMethod
    metadata: Optional[Dict] = None


@dataclass
class SimilarityResult:
    """Result of text similarity calculation."""

    similarity: float
    method: SimilarityMethod
    details: Optional[Dict] = None


@dataclass
class FuzzyMatchResult:
    """Result of fuzzy text matching."""

    found: bool
    best_match: Optional[str] = None
    similarity: float = 0.0
    start_pos: int = -1
    end_pos: int = -1
    context: Optional[str] = None
    method: SimilarityMethod = SimilarityMethod.SEQUENCE_MATCHER
    alternatives: Optional[List[Tuple[str, float]]] = None


class TextProcessor:
    """
    Unified text processing utilities for the editor.

    This class consolidates text similarity, pattern matching, and processing
    logic found across patch_docx_editor.py and other components.
    """

    # Common regex patterns
    WHITESPACE_PATTERN = re.compile(r"\s+")
    WORD_PATTERN = re.compile(r"\b\w+\b")
    PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")

    @classmethod
    def calculate_similarity(
        cls, text1: str, text2: str, method: SimilarityMethod = SimilarityMethod.SEQUENCE_MATCHER
    ) -> SimilarityResult:
        """
        Calculate similarity between two text strings.

        Consolidates similarity calculation from patch_docx_editor.py and other components.

        Args:
            text1: First text string
            text2: Second text string
            method: Similarity calculation method

        Returns:
            SimilarityResult with score and details
        """
        if method == SimilarityMethod.SEQUENCE_MATCHER:
            return cls._sequence_similarity(text1, text2)
        elif method == SimilarityMethod.WORD_OVERLAP:
            return cls._word_overlap_similarity(text1, text2)
        elif method == SimilarityMethod.CHARACTER_NGRAM:
            return cls._character_ngram_similarity(text1, text2)
        elif method == SimilarityMethod.COMBINED:
            return cls._combined_similarity(text1, text2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")

    @classmethod
    def _sequence_similarity(cls, text1: str, text2: str) -> SimilarityResult:
        """Calculate similarity using SequenceMatcher."""
        matcher = difflib.SequenceMatcher(None, text1.lower(), text2.lower())
        similarity = matcher.ratio()

        return SimilarityResult(
            similarity=similarity,
            method=SimilarityMethod.SEQUENCE_MATCHER,
            details={"matching_blocks": matcher.get_matching_blocks(), "opcodes": matcher.get_opcodes()},
        )

    @classmethod
    def _word_overlap_similarity(cls, text1: str, text2: str) -> SimilarityResult:
        """Calculate similarity based on word overlap."""
        words1 = set(cls.WORD_PATTERN.findall(text1.lower()))
        words2 = set(cls.WORD_PATTERN.findall(text2.lower()))

        if not words1 and not words2:
            return SimilarityResult(1.0, SimilarityMethod.WORD_OVERLAP)
        if not words1 or not words2:
            return SimilarityResult(0.0, SimilarityMethod.WORD_OVERLAP)

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        similarity = len(intersection) / len(union) if union else 0.0

        return SimilarityResult(
            similarity=similarity,
            method=SimilarityMethod.WORD_OVERLAP,
            details={
                "common_words": list(intersection),
                "unique_words1": list(words1 - words2),
                "unique_words2": list(words2 - words1),
            },
        )

    @classmethod
    def _character_ngram_similarity(cls, text1: str, text2: str, n: int = 3) -> SimilarityResult:
        """Calculate similarity using character n-grams."""

        def get_ngrams(text: str, n: int) -> Set[str]:
            """Return the set of character n-grams for a text string.

            Lowercases the input and strips spaces before extraction
            so n-grams compare content, not formatting.
            """
            text = text.lower().replace(" ", "")
            return set(text[i : i + n] for i in range(len(text) - n + 1))

        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)

        if not ngrams1 and not ngrams2:
            return SimilarityResult(1.0, SimilarityMethod.CHARACTER_NGRAM)
        if not ngrams1 or not ngrams2:
            return SimilarityResult(0.0, SimilarityMethod.CHARACTER_NGRAM)

        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)

        similarity = len(intersection) / len(union) if union else 0.0

        return SimilarityResult(
            similarity=similarity,
            method=SimilarityMethod.CHARACTER_NGRAM,
            details={"ngram_size": n, "common_ngrams": len(intersection), "total_ngrams": len(union)},
        )

    @classmethod
    def _combined_similarity(cls, text1: str, text2: str) -> SimilarityResult:
        """Calculate combined similarity using multiple methods."""
        seq_result = cls._sequence_similarity(text1, text2)
        word_result = cls._word_overlap_similarity(text1, text2)
        ngram_result = cls._character_ngram_similarity(text1, text2)

        # Weighted combination (sequence matcher gets highest weight)
        combined_score = seq_result.similarity * 0.5 + word_result.similarity * 0.3 + ngram_result.similarity * 0.2

        return SimilarityResult(
            similarity=combined_score,
            method=SimilarityMethod.COMBINED,
            details={
                "sequence_score": seq_result.similarity,
                "word_score": word_result.similarity,
                "ngram_score": ngram_result.similarity,
                "weights": {"sequence": 0.5, "word": 0.3, "ngram": 0.2},
            },
        )

    @classmethod
    def find_text_pattern(
        cls,
        pattern: str,
        texts: List[str],
        threshold: float = 0.8,
        method: SimilarityMethod = SimilarityMethod.SEQUENCE_MATCHER,
    ) -> Optional[int]:
        """
        Find text pattern in a list of texts.

        Consolidates pattern finding logic from patch_docx_editor.py.

        Args:
            pattern: Text pattern to find
            texts: List of texts to search in
            threshold: Similarity threshold for fuzzy matching
            method: Similarity calculation method

        Returns:
            Index of best match or None if not found
        """
        pattern_lower = pattern.lower().strip()

        # First try exact matching
        for i, text in enumerate(texts):
            if pattern_lower in text.lower():
                return i

        # Fall back to fuzzy matching
        best_match_idx = None
        best_similarity = threshold

        for i, text in enumerate(texts):
            result = cls.calculate_similarity(pattern_lower, text.lower(), method)
            if result.similarity > best_similarity:
                best_similarity = result.similarity
                best_match_idx = i

        return best_match_idx

    @classmethod
    def find_all_matches(cls, pattern: str, texts: List[str], threshold: float = 0.6) -> List[Tuple[int, float]]:
        """
        Find all matches of a pattern in texts above threshold.

        Args:
            pattern: Text pattern to find
            texts: List of texts to search in
            threshold: Minimum similarity threshold

        Returns:
            List of (index, similarity) tuples sorted by similarity
        """
        matches = []
        pattern_lower = pattern.lower().strip()

        for i, text in enumerate(texts):
            result = cls.calculate_similarity(pattern_lower, text.lower())
            if result.similarity >= threshold:
                matches.append((i, result.similarity))

        # Sort by similarity, highest first
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    @classmethod
    def extract_context(cls, text: str, pattern: str, context_chars: int = 100) -> Optional[str]:
        """
        Extract context around a pattern match in text.

        Args:
            text: Text to search in
            pattern: Pattern to find
            context_chars: Number of characters of context on each side

        Returns:
            Context string or None if pattern not found
        """
        text_lower = text.lower()
        pattern_lower = pattern.lower()

        start = text_lower.find(pattern_lower)
        if start == -1:
            return None

        end = start + len(pattern)

        # Expand context
        context_start = max(0, start - context_chars)
        context_end = min(len(text), end + context_chars)

        context = text[context_start:context_end]

        # Add ellipsis if truncated
        if context_start > 0:
            context = "..." + context
        if context_end < len(text):
            context = context + "..."

        return context

    @classmethod
    def normalize_text(
        cls, text: str, remove_punctuation: bool = False, normalize_whitespace: bool = True, lowercase: bool = True
    ) -> str:
        """
        Normalize text for comparison.

        Args:
            text: Text to normalize
            remove_punctuation: Whether to remove punctuation
            normalize_whitespace: Whether to normalize whitespace
            lowercase: Whether to convert to lowercase

        Returns:
            Normalized text
        """
        normalized = text

        if lowercase:
            normalized = normalized.lower()

        if remove_punctuation:
            normalized = cls.PUNCTUATION_PATTERN.sub(" ", normalized)

        if normalize_whitespace:
            normalized = cls.WHITESPACE_PATTERN.sub(" ", normalized).strip()

        return normalized

    @classmethod
    def find_common_prefix(cls, text1: str, text2: str) -> str:
        """
        Find common prefix between two texts.

        Consolidates prefix finding logic from docxeditor.py.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Common prefix string
        """
        min_len = min(len(text1), len(text2))
        i = 0
        while i < min_len and text1[i] == text2[i]:
            i += 1
        return text1[:i]

    @classmethod
    def find_common_suffix(cls, text1: str, text2: str) -> str:
        """
        Find common suffix between two texts.

        Consolidates suffix finding logic from docxeditor.py.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Common suffix string
        """
        # Work backwards from the end
        len1, len2 = len(text1), len(text2)
        min_len = min(len1, len2)
        i = 0

        while i < min_len and text1[len1 - 1 - i] == text2[len2 - 1 - i]:
            i += 1

        return text1[len1 - i :] if i > 0 else ""

    @classmethod
    def split_on_affixes(cls, old_text: str, new_text: str, min_affix_length: int = 3) -> Dict[str, str]:
        """
        Split texts into prefix, middle, and suffix parts.

        Consolidates affix splitting logic from docxeditor.py.

        Args:
            old_text: Original text
            new_text: New text
            min_affix_length: Minimum length for affixes to be considered

        Returns:
            Dictionary with prefix, old_middle, new_middle, suffix
        """
        prefix = cls.find_common_prefix(old_text, new_text)
        suffix = cls.find_common_suffix(old_text, new_text)

        # Ensure prefix + suffix don't overlap
        if len(prefix) + len(suffix) > min(len(old_text), len(new_text)):
            suffix = ""  # Prioritize prefix

        # Only use affixes if they meet minimum length
        if len(prefix) < min_affix_length:
            prefix = ""
        if len(suffix) < min_affix_length:
            suffix = ""

        # Extract middle parts
        prefix_len = len(prefix)
        suffix_len = len(suffix) if suffix else 0

        old_end = len(old_text) - suffix_len if suffix_len > 0 else len(old_text)
        new_end = len(new_text) - suffix_len if suffix_len > 0 else len(new_text)

        old_middle = old_text[prefix_len:old_end]
        new_middle = new_text[prefix_len:new_end]

        return {"prefix": prefix, "old_middle": old_middle, "new_middle": new_middle, "suffix": suffix}

    @classmethod
    def is_whitespace_only_change(cls, old_text: str, new_text: str) -> bool:
        """
        Check if the difference between texts is only whitespace.

        Consolidates whitespace change detection from docxeditor.py.

        Args:
            old_text: Original text
            new_text: New text

        Returns:
            True if only whitespace differs
        """
        # Remove all whitespace and compare
        old_normalized = re.sub(r"\s+", "", old_text)
        new_normalized = re.sub(r"\s+", "", new_text)

        return old_normalized == new_normalized

    @classmethod
    def get_text_statistics(cls, text: str) -> Dict[str, int]:
        """
        Get basic statistics about text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with text statistics
        """
        words = cls.WORD_PATTERN.findall(text)
        sentences = text.count(".") + text.count("!") + text.count("?")
        paragraphs = len([p for p in text.split("\n\n") if p.strip()])

        return {
            "characters": len(text),
            "characters_no_spaces": len(text.replace(" ", "")),
            "words": len(words),
            "unique_words": len(set(word.lower() for word in words)),
            "sentences": max(1, sentences),  # At least 1
            "paragraphs": max(1, paragraphs),  # At least 1
            "whitespace_chars": len(text) - len(text.replace(" ", "")),
            "lines": len(text.split("\n")),
        }

    @classmethod
    def clean_text(
        cls, text: str, remove_extra_whitespace: bool = True, remove_empty_lines: bool = True, strip_lines: bool = True
    ) -> str:
        """
        Clean text by removing unwanted whitespace and formatting.

        Args:
            text: Text to clean
            remove_extra_whitespace: Remove multiple consecutive spaces
            remove_empty_lines: Remove empty lines
            strip_lines: Strip whitespace from line ends

        Returns:
            Cleaned text
        """
        lines = text.split("\n")

        if strip_lines:
            lines = [line.strip() for line in lines]

        if remove_empty_lines:
            lines = [line for line in lines if line]

        cleaned = "\n".join(lines)

        if remove_extra_whitespace:
            # Replace multiple spaces with single space
            cleaned = re.sub(r" {2,}", " ", cleaned)
            # Replace multiple tabs with single space
            cleaned = re.sub(r"\t+", " ", cleaned)

        return cleaned

    @classmethod
    def detect_language_hints(cls, text: str) -> Dict[str, bool]:
        """
        Detect language and formatting hints in text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with detected features
        """
        features = {}

        # Basic language detection hints
        features["has_english_words"] = bool(
            re.search(r"\b(the|and|or|but|in|on|at|to|for|of|with|by)\b", text.lower())
        )
        features["has_code_markers"] = bool(re.search(r"[{}();\[\]]|def |class |import |function\s*\(", text))
        features["has_markdown"] = bool(re.search(r"^#+\s|^\*\s|^\-\s|`[^`]+`|\*\*[^*]+\*\*", text, re.MULTILINE))
        features["has_citations"] = bool(re.search(r"\[[0-9]+\]|\([^)]*[0-9]{4}[^)]*\)", text))
        features["has_urls"] = bool(re.search(r"https?://[^\s]+", text))
        features["has_emails"] = bool(re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text))
        features["has_numbers"] = bool(re.search(r"\b\d+\b", text))
        features["has_special_chars"] = bool(re.search(r'[^\w\s.,!?;:()"\'-]', text))

        return features

    @classmethod
    def find_fuzzy_match(
        cls,
        pattern: str,
        text: str,
        threshold: float = 0.85,
        method: SimilarityMethod = SimilarityMethod.SEQUENCE_MATCHER,
        max_alternatives: int = 3,
        context_chars: int = 100,
    ) -> FuzzyMatchResult:
        """
        Find the best fuzzy match for a pattern in text.

        Args:
            pattern: Text pattern to find
            text: Text to search in
            threshold: Minimum similarity threshold (0.0-1.0)
            method: Similarity calculation method
            max_alternatives: Maximum number of alternative matches to return
            context_chars: Number of characters of context around match

        Returns:
            FuzzyMatchResult with match information
        """
        pattern_normalized = pattern.lower().strip()

        # First try exact matching
        if pattern_normalized in text.lower():
            start_pos = text.lower().find(pattern_normalized)
            end_pos = start_pos + len(pattern)
            context = cls.extract_context(text, pattern, context_chars)

            return FuzzyMatchResult(
                found=True,
                best_match=text[start_pos:end_pos],
                similarity=1.0,
                start_pos=start_pos,
                end_pos=end_pos,
                context=context,
                method=method,
            )

        # Use sliding window for fuzzy matching
        pattern_len = len(pattern)
        best_similarity = 0.0
        best_match = None
        best_start = -1
        best_end = -1
        alternatives = []

        # Try different window sizes around the pattern length
        for window_factor in [1.0, 1.2, 0.8, 1.5, 0.6]:
            window_size = max(1, int(pattern_len * window_factor))

            # Slide window across text
            for i in range(len(text) - window_size + 1):
                window_text = text[i : i + window_size]
                result = cls.calculate_similarity(pattern_normalized, window_text.lower(), method)

                if result.similarity >= threshold:
                    alternatives.append((window_text, result.similarity))

                    if result.similarity > best_similarity:
                        best_similarity = result.similarity
                        best_match = window_text
                        best_start = i
                        best_end = i + window_size

        # Sort alternatives by similarity and limit
        alternatives.sort(key=lambda x: x[1], reverse=True)
        alternatives = alternatives[:max_alternatives]

        if best_match:
            context = cls.extract_context(text, best_match, context_chars)
            return FuzzyMatchResult(
                found=True,
                best_match=best_match,
                similarity=best_similarity,
                start_pos=best_start,
                end_pos=best_end,
                context=context,
                method=method,
                alternatives=alternatives,
            )

        return FuzzyMatchResult(found=False, method=method, alternatives=alternatives if alternatives else None)

    @classmethod
    def suggest_correction(
        cls, pattern: str, text: str, threshold: float = 0.7, max_suggestions: int = 3
    ) -> Optional[str]:
        """
        Suggest the correct text pattern to use based on fuzzy matching.

        Args:
            pattern: Original (failed) pattern
            text: Text to search in for alternatives
            threshold: Minimum similarity for suggestions
            max_suggestions: Maximum number of suggestions

        Returns:
            Formatted suggestion string or None if no good matches
        """
        fuzzy_result = cls.find_fuzzy_match(pattern, text, threshold=threshold)

        if not fuzzy_result.found and not fuzzy_result.alternatives:
            return None

        suggestions = []

        # Add the best match if found
        if fuzzy_result.found:
            suggestions.append((fuzzy_result.best_match, fuzzy_result.similarity))

        # Add alternatives
        if fuzzy_result.alternatives:
            for alt_text, similarity in fuzzy_result.alternatives:
                if (alt_text, similarity) not in suggestions:
                    suggestions.append((alt_text, similarity))

        # Limit suggestions
        suggestions = suggestions[:max_suggestions]

        if not suggestions:
            return None

        # Format suggestion message
        if len(suggestions) == 1:
            return f"Did you mean: '{suggestions[0][0]}' (similarity: {suggestions[0][1]:.2f})?"
        else:
            suggestion_list = []
            for i, (text, sim) in enumerate(suggestions, 1):
                suggestion_list.append(f"{i}. '{text}' (similarity: {sim:.2f})")

            return "Did you mean one of these?\n" + "\n".join(suggestion_list)

    @classmethod
    def find_best_pattern_match(
        cls, pattern: str, text: str, threshold: float = 0.85
    ) -> Tuple[bool, Optional[str], float]:
        """
        Find the best matching pattern in text for correction purposes.

        Args:
            pattern: Pattern that failed to match exactly
            text: Text to search in
            threshold: Minimum similarity threshold

        Returns:
            Tuple of (found, best_match_text, similarity)
        """
        fuzzy_result = cls.find_fuzzy_match(pattern, text, threshold=threshold)

        if fuzzy_result.found:
            return True, fuzzy_result.best_match, fuzzy_result.similarity
        elif fuzzy_result.alternatives:
            # Return the best alternative
            best_alt = fuzzy_result.alternatives[0]
            return False, best_alt[0], best_alt[1]
        else:
            return False, None, 0.0


# Convenience functions for backward compatibility


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Legacy function for simple similarity calculation.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0.0 and 1.0
    """
    result = TextProcessor.calculate_similarity(text1, text2)
    return result.similarity


def find_text_in_paragraphs(pattern: str, paragraphs: List[str], threshold: float = 0.8) -> Optional[int]:
    """
    Legacy function for finding text patterns.

    Args:
        pattern: Text pattern to find
        paragraphs: List of paragraph texts
        threshold: Similarity threshold

    Returns:
        Index of best match or None
    """
    return TextProcessor.find_text_pattern(pattern, paragraphs, threshold)
