"""Thin wrappers around the chat LLM used by every chain.

The rest of the codebase only ever touches the three names exported here
(`creative_llm`, `deterministic_llm`, `chat`), so the underlying provider is an
implementation detail confined to this module.

Two backends, selected by the `LLM_BACKEND` env var:

  - **bedrock** (default) — Amazon Bedrock (Nova 2 Lite). Credentials/region come
    from the standard AWS chain; this is what the cloud deployment uses.
  - **openai** — OpenAI (gpt-5.4-nano) via `langchain_openai`, needing only
    `OPENAI_API_KEY`. This is the no-AWS path for local / cluster runs.

Set `LLM_BACKEND=openai` in `.env` to run without AWS. The default stays
`bedrock` so the cloud deployment is unaffected. Each backend's client is
imported lazily inside its branch, so the unused provider is never a hard import
dependency at module load.
"""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage

# Which provider to use. Default "bedrock" preserves the cloud deployment's
# behavior; set LLM_BACKEND=openai for the local/cluster (no-AWS) path.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "bedrock").strip().lower()

# Per-backend model id, each overridable via env so a different region / model
# can be pinned without a code change.
BEDROCK_MODEL = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-2-lite-v1:0")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL_ID", "gpt-5.4-nano")

# Region the Bedrock runtime client targets. boto3 falls back to AWS_REGION /
# AWS_DEFAULT_REGION when this is unset; we read it explicitly so a missing
# region surfaces as our own error rather than a deep boto stack trace.
BEDROCK_REGION = os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION")


def _build(temperature: float):
    """Build the chat client for the configured backend at the given temperature.
    The provider client is imported lazily so the unused backend's package is not
    required to import this module."""
    if LLM_BACKEND == "openai":
        from langchain_openai import ChatOpenAI

        # API key comes from OPENAI_API_KEY in the environment; langchain_openai
        # reads it itself, so we never pass it explicitly.
        return ChatOpenAI(model=OPENAI_MODEL, temperature=temperature)
    if LLM_BACKEND == "bedrock":
        from langchain_aws import ChatBedrockConverse

        # Credentials/region come from the standard AWS chain (task role on
        # Fargate, env/profile locally) — we never pass keys explicitly, so a
        # misconfigured environment raises boto's own clear error.
        return ChatBedrockConverse(
            model=BEDROCK_MODEL,
            temperature=temperature,
            region_name=BEDROCK_REGION,
        )
    raise ValueError(
        f"Unknown LLM_BACKEND={LLM_BACKEND!r}; expected 'bedrock' or 'openai'."
    )


def creative_llm():
    return _build(temperature=0.7)


def deterministic_llm():
    return _build(temperature=0.0)


def _as_text(content) -> str:
    """Flatten a chat response to plain text.

    Both `ChatBedrockConverse` and `ChatOpenAI` return `.content` as a string for
    simple replies, but as a list of content blocks (dicts with a `text` field,
    plus possible reasoning/tool blocks) in other cases. Every caller here wants
    the text, so we normalize both shapes and drop non-text blocks."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
    return "".join(parts)


def chat(llm, system: str, user: str) -> str:
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return _as_text(resp.content).strip()
