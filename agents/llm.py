"""Thin wrappers around ChatOpenAI used by every chain."""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

GPT_MODEL = "gpt-5.4-nano"


def creative_llm() -> ChatOpenAI:
    return ChatOpenAI(model=GPT_MODEL, temperature=0.7, api_key=os.environ.get("OPENAI_API_KEY"))


def deterministic_llm() -> ChatOpenAI:
    return ChatOpenAI(model=GPT_MODEL, temperature=0.0, api_key=os.environ.get("OPENAI_API_KEY"))


def chat(llm: ChatOpenAI, system: str, user: str) -> str:
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content.strip()