"""Errors raised by andamentum.vision_critique."""

from __future__ import annotations


class VisionCritiqueError(Exception):
    """Base class for vision_critique errors."""


class UnsupportedImageError(VisionCritiqueError, ValueError):
    """The image is not a PNG, JPEG, WEBP or GIF (for example a PDF or SVG)."""
