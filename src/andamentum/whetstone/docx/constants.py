"""Constants for document review system."""

# Skip fields (metadata that shouldn't appear in formatted output)
SKIP_FIELDS = {"doc_id", "status", "error", "proposal_title"}

# Field naming patterns
SCORE_SUFFIX = "_score"
JUSTIFICATION_SUFFIX = "_justification"

# Structured field names (known fields with special formatting)
STRUCTURED_FIELDS = {"strengths", "weaknesses", "recommendation", "overall_assessment"}

# Report section headers
REPORT_TITLE = "# DOCUMENT REVIEW REPORT"
SECTION_SEPARATOR = "---"
EXECUTIVE_SUMMARY_HEADER = "## Executive Summary"
CRITICAL_ISSUES_HEADER = "## Critical Issues Identified"
EXPERT_REVIEWS_HEADER = "## Individual Expert Reviews"
NOVELTY_FINDINGS_HEADER = "## Novelty Assessment"

# Field labels
FIELD_LABEL_LOCATION = "Location"
FIELD_LABEL_DISCIPLINE = "Discipline"
FIELD_LABEL_POSITION = "Position"
FIELD_LABEL_EDUCATION = "Education"
FIELD_LABEL_SCORES = "Scores"
FIELD_LABEL_ASSESSMENTS = "Detailed Assessments"
FIELD_LABEL_ADDITIONAL = "Additional Review Details"

# Default values
DEFAULT_EXPERT_NAME = "Expert"
DEFAULT_DISCIPLINE = "Not specified"
DEFAULT_DESCRIPTION = "No description"
DEFAULT_SEVERITY = "medium"
DEFAULT_ISSUE_TITLE = "Issue"

# Inline markdown patterns
MARKDOWN_BOLD_PATTERN = r"\*\*(.*?)\*\*"
MARKDOWN_ITALIC_PATTERN = r"\*(.*?)\*"
