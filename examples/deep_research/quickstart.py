"""Quick start for andamentum.deep_research: one question in, a cited report out.

Needs a language model, podman and internet access. The search and the page
reads go through an isolated Chrome in a podman container that the run starts and
removes (the image is built on first use, about two minutes).

Run:

    uv run python examples/deep_research/quickstart.py \
        --model ollama:gemma4:31b-cloud \
        "Who wrote the novel Middlemarch?"

Every finding in the report is a verbatim quote that was found in a fetched
page; the report cites findings by id ([F1]).
"""

import argparse
import asyncio

from andamentum.deep_research import SourceConfig, research


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research one question over the web and print the report."
    )
    parser.add_argument("question")
    parser.add_argument("--model", required=True, help="a pydantic-ai model id")
    args = parser.parse_args()

    result = await research(
        args.question,
        mode="web",
        model=args.model,
        sources=SourceConfig(browser=True),
        progress=print,
    )
    print(result.report_markdown)
    print(
        f"\nstop: {result.stop_reason}; {len(result.findings)} findings from "
        f"{len(result.works)} pages"
    )


if __name__ == "__main__":
    asyncio.run(main())
