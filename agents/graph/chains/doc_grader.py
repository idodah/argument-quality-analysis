"""Self-RAG relevance grader: batched keep-list over all chunks in one call."""

from __future__ import annotations

import json
import re

from agents import prompts
from agents.graph.llm import chat, deterministic_llm

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_keep(text: str, n_chunks: int) -> list[int]:
    """Parse the grader's JSON; on failure, keep everything (conservative)."""
    match = _JSON_OBJ_RE.search(text)
    if not match:
        return list(range(n_chunks))
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return list(range(n_chunks))
    raw = obj.get("keep")
    if not isinstance(raw, list):
        return list(range(n_chunks))
    out: list[int] = []
    for x in raw:
        try:
            i = int(x) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < n_chunks and i not in out:
            out.append(i)
    return out


def grade_chunks(draft: str, chunks: list[str]) -> list[str]:
    """Filter chunks down to those the grader marks relevant for refining `draft`."""
    if not chunks:
        return []
    numbered = "\n\n".join(f"[{i + 1}]\n{c[:4000]}" for i, c in enumerate(chunks))
    llm = deterministic_llm()
    user = prompts.GRADER_USER.format(draft=draft[:4000], chunks=numbered)
    text = chat(llm, prompts.GRADER_SYSTEM, user)
    keep_idx = _parse_keep(text, len(chunks))
    return [chunks[i] for i in keep_idx]