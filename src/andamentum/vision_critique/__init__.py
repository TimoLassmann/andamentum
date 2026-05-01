"""andamentum.vision_critique — bounded vision critique of rendered figures.

Single async public API: ``await critique_figure(image, *, model)`` →
:class:`FigureCritique`. Image is bytes, a Path, or an http(s) URL;
``model`` is any pydantic-ai multimodal model id (Ollama, Anthropic,
OpenAI, etc.) — required, no hidden default.

The default critique schema (:class:`FigureCritique`) is tightly bounded
on purpose: small local vision models reliably fill close-set enums,
and the receiver gets a predictable action surface for mapping flagged
issues to render-parameter changes. Callers wanting a different shape
pass their own ``BaseModel`` subclass via ``schema=``.

Example::

    from andamentum.vision_critique import critique_figure

    critique = await critique_figure(
        "fig.png",
        model="ollama:gemma4:e4b-it-q4_K_M",
    )
    if critique.has_issues:
        print(critique.one_line_summary)
        print("Try:", critique.suggested_fixes)
"""

from .api import critique_figure
from .schemas import AspectIssue, FigureCritique, SuggestedFix

__all__ = [
    "critique_figure",
    "FigureCritique",
    "AspectIssue",
    "SuggestedFix",
]
