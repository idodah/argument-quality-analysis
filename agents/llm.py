"""Thin wrappers around Amazon Bedrock (Nova 2 Lite) used by every chain.

The rest of the codebase only ever touches the three names exported here
(`creative_llm`, `deterministic_llm`, `chat`), so the underlying provider is an
implementation detail confined to this module. We migrated from OpenAI
(gpt-5.4-nano) to Bedrock Nova 2 Lite here; nothing else changed.
"""

from __future__ import annotations

import os

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage

# Bedrock model id. Overridable via env so a different region / model can be
# pinned without a code change (e.g. an inference-profile ARN).
BEDROCK_MODEL = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-2-lite-v1:0")

# Region the Bedrock runtime client targets. boto3 falls back to AWS_REGION /
# AWS_DEFAULT_REGION when this is unset; we read it explicitly so a missing
# region surfaces as our own error rather than a deep boto stack trace.
BEDROCK_REGION = os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION")


def _bedrock(temperature: float) -> ChatBedrockConverse:
    """Build a Bedrock chat client. Credentials/region come from the standard
    AWS chain (task role on Fargate, env/profile locally) — we never pass keys
    explicitly, so a misconfigured environment raises boto's own clear error."""
    return ChatBedrockConverse(
        model=BEDROCK_MODEL,
        temperature=temperature,
        region_name=BEDROCK_REGION,
    )


def creative_llm() -> ChatBedrockConverse:
    return _bedrock(temperature=0.7)


def deterministic_llm() -> ChatBedrockConverse:
    return _bedrock(temperature=0.0)


def _as_text(content) -> str:
    """Flatten a Bedrock Converse response to plain text.

    `ChatBedrockConverse` returns `.content` as a string for simple replies, but
    as a list of content blocks (dicts with a `text` field, plus possible
    reasoning/tool blocks) in other cases. Every caller here wants the text, so
    we normalize both shapes and drop non-text blocks."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
    return "".join(parts)


def chat(llm: ChatBedrockConverse, system: str, user: str) -> str:
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return _as_text(resp.content).strip()
