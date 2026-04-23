"""andamentum.whetstone — a workshop for sharpening your own drafts.

Whetstone runs structured, multi-lens feedback over documents you wrote
yourself: grammar and style editing, specialist critique, and multi-expert
panel review. Output formats: Word track changes, HTML report (via
andamentum.typeset), or a lightweight markdown diff.

**This is not a peer-review tool.** Do not use it on manuscripts you have
been asked to review confidentially — that would violate journal policy.
Use it on *your own* drafts, before submission.

Quick start::

    import asyncio
    from andamentum.whetstone import sharpen_document, render_html, apply_patches

    async def main():
        text = open("my_draft.md").read()
        result = await sharpen_document(text, task="edit")
        html = render_html(result=result, original_content=text)
        open("review.html", "w").write(html)

        # Or apply the edits directly to the text:
        revised = apply_patches(text, result.patches)
        open("my_draft.revised.md", "w").write(revised)

    asyncio.run(main())
"""

from .agents import AGENT_REGISTRY, AgentDefinition
from .dynamic_models import convert_fields_to_schema, create_output_model
from .issues import DocumentIssue
from .models import DocumentPatch, PatchApplicationResult
from .orchestrator import ReviewResult, sharpen_document
from .renderers import apply_patches, render_diff, render_docx, render_html

__version__ = "0.1.0"

__all__ = [
    # Public entry point
    "sharpen_document",
    "ReviewResult",
    # Data models
    "DocumentPatch",
    "DocumentIssue",
    "PatchApplicationResult",
    # Renderers
    "render_docx",
    "render_html",
    "render_diff",
    "apply_patches",
    # Agents (for introspection / extension)
    "AgentDefinition",
    "AGENT_REGISTRY",
    # Dynamic schema helpers
    "convert_fields_to_schema",
    "create_output_model",
]
