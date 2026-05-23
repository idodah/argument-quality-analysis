"""Adaptive-RAG router chain: chooses local | web | none."""

from __future__ import annotations

from agents import prompts
from agents.graph.llm import chat, deterministic_llm
from agents.graph.state import RetrievalMode


def route_retrieval(post: str, draft: str, critique: str = "") -> RetrievalMode:
    """Return one of 'local', 'web', 'none' for the current draft."""
    llm = deterministic_llm()
    user = prompts.ROUTER_USER.format(post=post, draft=draft, critique=critique or "(none yet)")
    decision = chat(llm, prompts.ROUTER_SYSTEM, user).lower().strip()
    if decision.startswith("local"):
        return "local"
    if decision.startswith("web"):
        return "web"
    return "none"