"""Quick start for andamentum.deep_research: one question in, a cited report out.

Needs a language model and a web search. The search goes through a Chrome you
start yourself with its debugging port open:

    chrome --remote-debugging-port=9222 --user-data-dir=/path/to/a/profile/dir

Run:

    uv run python examples/deep_research/quickstart.py \
        --model ollama:gemma4:31b-cloud \
        --browser-cdp-url http://127.0.0.1:9222 \
        "Who wrote the novel Middlemarch?"

Every finding in the report is a verbatim quote that was found in a fetched
page; the report cites findings by id ([F1]).
"""

import argparse
import asyncio

from andamentum.deep_research.v2 import SourceConfig, research


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research one question over the web and print the report."
    )
    parser.add_argument("question")
    parser.add_argument("--model", required=True, help="a pydantic-ai model id")
    parser.add_argument("--browser-cdp-url", required=True)
    args = parser.parse_args()

    result = await research(
        args.question,
        mode="web",
        model=args.model,
        sources=SourceConfig(browser_cdp_url=args.browser_cdp_url),
        progress=print,
    )
    print(result.report_markdown)
    print(
        f"\nstop: {result.stop_reason}; {len(result.findings)} findings from "
        f"{len(result.works)} pages"
    )


if __name__ == "__main__":
    asyncio.run(main())
