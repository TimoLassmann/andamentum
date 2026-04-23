# andamentum.whetstone

> Sharpen your own drafts. Not someone else's.

A workshop for running structured, multi-lens feedback over documents **you**
wrote — grammar and style editing, specialist critique, or a multi-expert
panel review. Output as Word track changes, HTML report, or a markdown diff.

## What this is for

- Improving your own manuscripts before you submit them.
- Drafting grant proposals, thesis chapters, reports, policy briefs, cover
  letters — anything you authored.
- Self-review before a supervisor's or colleague's eyes see it.

## What this is **not** for

- Peer-reviewing manuscripts that another author has sent you confidentially.
  Most journals explicitly forbid uploading such material to an LLM. Do not
  use whetstone for that purpose.

## Quick start

```python
import asyncio
from andamentum.whetstone import sharpen_document, render_html, apply_patches

async def main():
    text = open("my_draft.md").read()
    result = await sharpen_document(text, task="edit", model="openai:gpt-4o")

    # Write an HTML report with track-change-style diffs:
    html = render_html(result=result, original_content=text)
    open("review.html", "w").write(html)

    # Or apply the accepted edits directly:
    revised = apply_patches(text, result.patches)
    open("my_draft.revised.md", "w").write(revised)

asyncio.run(main())
```

## Tasks

| Task | What it does | Agents |
|---|---|---|
| `edit` | Grammar, style, polish as structured patches | 1 unified editor (or N parallel editors via `editors=[...]`) |
| `review` | Specialist critique with prioritised synthesis | 4 reviewers (clarity, merit, methodology, results) + synthesizer |
| `panel` | Multi-expert panel review | N generated expert biosketches, parallel reviews, panel synthesizer |

Custom criteria (`criteria="..."`) generates a runtime schema and replaces
the standard review with one tailored to your brief.

## CLI

```bash
andamentum-whetstone my_draft.md --task review -o report.html
andamentum-whetstone my_draft.docx --task edit -o reviewed.docx
andamentum-whetstone my_draft.md --task panel --num-experts 5 -o panel.html
andamentum-whetstone agents            # list registered agents
```

Set `ANDAMENTUM_MAIN_LLM_MODEL` to avoid passing `--model` every time.

## Output formats

- **DOCX** — Word track changes + a prepended executive-summary page.
- **HTML** — standalone report styled by `andamentum.typeset`; opens in
  any browser, no external assets.
- **Markdown diff** — lightweight diff view, default when no output file is
  specified.

## Dependencies

Python 3.10+, python-docx, lxml, pydantic, pydantic-ai, andamentum (core +
typeset). All installed by `pip install andamentum`.
