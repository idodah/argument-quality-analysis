"""Reflexion-style critic chain: produces verbal self-reflection on a draft."""

from __future__ import annotations

from agents import prompts
from agents.graph.llm import chat, creative_llm


def reflect_on_draft(post: str, draft: str, evidence_chunks: list[str]) -> str:
    """Produce a verbal critique of `draft` given the post and retrieved evidence."""
    evidence = "\n\n---\n\n".join(evidence_chunks) if evidence_chunks else "(no retrieval this iteration)"
    llm = creative_llm()
    user = prompts.REFLECT_USER.format(post=post, draft=draft, evidence=evidence)
    return chat(llm, prompts.REFLECT_SYSTEM, user)