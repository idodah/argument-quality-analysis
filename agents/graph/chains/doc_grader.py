"""Self-RAG relevance grader: per-chunk yes/no."""

from __future__ import annotations

from agents import prompts
from agents.graph.llm import chat, deterministic_llm


def grade_chunks(draft: str, chunks: list[str]) -> list[str]:
    """Filter chunks down to those the grader marks relevant for refining `draft`."""
    if not chunks:
        return []
    llm = deterministic_llm()
    kept: list[str] = []
    for chunk in chunks:
        user = prompts.GRADER_USER.format(draft=draft[:4000], chunk=chunk[:4000])
        verdict = chat(llm, prompts.GRADER_SYSTEM, user).lower().strip()
        if verdict.startswith("yes"):
            kept.append(chunk)
    return kept