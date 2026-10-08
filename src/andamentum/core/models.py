"""Model resolution for pydantic-ai.

Handles model string prefixes:
- ollama:model_name -> OllamaModel with OllamaProvider
- bedrock:model_id -> BedrockConverseModel with regional inference profiles
  (model_id is a Bedrock model ID, optionally aliased via BEDROCK_MODEL_MAP)
- anything else -> passthrough (pydantic-ai resolves via infer_model)

Environment variables:
- OLLAMA_BASE_URL: Ollama API endpoint (default http://localhost:11434/v1)
- AWS_PROFILE: boto3 session profile for Bedrock
- AWS_DEFAULT_REGION / AWS_REGION: Bedrock region
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Optional Bedrock friendly-name → model-ID alias table.
#
# Empty by default: ``bedrock:<id>`` passes ``<id>`` straight through to
# Bedrock (see ``BEDROCK_MODEL_MAP.get(friendly, friendly)`` below), so the
# canonical usage is ``bedrock:anthropic.claude-haiku-4-5-20251001-v1:0``.
#
# The exact friendly→ID mappings are account- and region-specific (model
# availability, provisioned-throughput ARNs, foundation-vs-inference-profile
# IDs all vary by deployment), so they are NOT shipped. Populate this dict in
# your own deployment if you want short aliases, e.g.::
#
#     from andamentum.core.models import BEDROCK_MODEL_MAP
#     BEDROCK_MODEL_MAP["haiku"] = "anthropic.claude-haiku-4-5-20251001-v1:0"
BEDROCK_MODEL_MAP: dict[str, str] = {}

REGION_PREFIX_MAP: dict[str, str] = {
    "ap-southeast-2": "au",
    "eu-west-1": "eu",
    "eu-central-1": "eu",
    "ap-northeast-1": "ap",
}


def resolve_model(model: str) -> Any:
    """Resolve a model string to a pydantic-ai model object.

    Handles:
    - "ollama:llama3" -> OllamaModel with OllamaProvider
    - "bedrock:anthropic.claude-haiku-4-5-20251001-v1:0" -> BedrockConverseModel
    - "openai:gpt-4o" -> passthrough string (pydantic-ai resolves)
    - "anthropic:claude-haiku-4-5" -> passthrough string
    """
    if model.startswith("ollama:"):
        from pydantic_ai.models.ollama import OllamaModel
        from pydantic_ai.providers.ollama import OllamaProvider

        model_name = model.split(":", 1)[1]
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return OllamaModel(
            model_name=model_name,
            provider=OllamaProvider(base_url=base_url),
        )

    if model.startswith("bedrock:"):
        import boto3
        from pydantic_ai.models.bedrock import BedrockConverseModel
        from pydantic_ai.providers.bedrock import BedrockProvider

        friendly = model.split(":", 1)[1]
        model_id = BEDROCK_MODEL_MAP.get(friendly, friendly)

        region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
        if region:
            prefix = REGION_PREFIX_MAP.get(region, "")
            if prefix:
                model_id = f"{prefix}.{model_id}"

        profile = os.environ.get("AWS_PROFILE")
        session = boto3.Session(profile_name=profile, region_name=region)
        client = session.client("bedrock-runtime", region_name=region)

        return BedrockConverseModel(
            model_name=model_id,
            provider=BedrockProvider(bedrock_client=client),
        )

    # Passthrough — pydantic-ai resolves via infer_model()
    return model


def resolve_model_from_args(model_arg: str | None) -> str:
    """Return ``model_arg``, or raise ``ValueError`` when it is missing.

    There is no environment fallback and no default model: a command takes
    ``--model`` on the command line (argparse ``required=True``), and library
    code passes ``model=`` explicitly.
    """
    if not model_arg:
        raise ValueError("a model is required: pass --model (for example ollama:gemma4:31b)")
    return model_arg
