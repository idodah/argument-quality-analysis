"""Refinement chain: rewrites a draft using critique + retrieved evidence."""

from __future__ import annotations

from agents import prompts
from agents.graph.llm import chat, creative_llm


def refine_draft(topic: str, post: str, draft: str, critique: str, evidence_chunks: list[str]) -> str:
    evidence = "\n\n---\n\n".join(evidence_chunks) if evidence_chunks else "(no retrieval this iteration)"
    llm = creative_llm()
    user = prompts.REFINE_USER.format(
        topic=topic, post=post, draft=draft, critique=critique, evidence=evidence
    )
    return chat(llm, prompts.REFINE_SYSTEM, user)