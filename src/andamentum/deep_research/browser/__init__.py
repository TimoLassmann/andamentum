"""The isolated Chrome of deep_research: a podman container per research session,
and one isolated page (a new browser context) per page read."""

from __future__ import annotations

from .cdp import CdpError, IsolatedPage, isolated_page
from .container import (
    LABEL,
    LABEL_KEY,
    LABEL_VALUE,
    BrowserSession,
    BrowserUnavailable,
    browser_session,
    ensure_image,
    image_tag,
    require_podman,
)

__all__ = [
    "LABEL",
    "LABEL_KEY",
    "LABEL_VALUE",
    "BrowserSession",
    "BrowserUnavailable",
    "CdpError",
    "IsolatedPage",
    "browser_session",
    "ensure_image",
    "image_tag",
    "isolated_page",
    "require_podman",
]
