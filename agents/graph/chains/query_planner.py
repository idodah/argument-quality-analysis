"""Web-query planner chain: turns post + draft + critique into search queries.

The web retrieval arm used to search with `topic + draft[:200]`, which echoes the
draft itself and ignores the critique. This chain instead asks the LLM to plan a
handful of targeted queries aimed at the evidence gaps the critique identified.
"""

from __future__ import annotations

import re

from agents import prompts
from agents.graph.llm import chat, deterministic_llm
from agents.graph.state import WEB_QUERIES


def _parse_queries(text: str, n: int) -> list[str]:
    """Pull numbered/bulleted lines out of the LLM response."""
    queries: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip a leading "1.", "1)", "-", "*" enumerator if present.
        cleaned = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip()
        if cleaned:
            queries.append(cleaned)
    return queries[:n]


def plan_web_queries(post: str, draft: str, critique: str, n: int = WEB_QUERIES) -> list[str]:
    """Return up to `n` web search queries targeting the critique's evidence gaps."""
    llm = deterministic_llm()
    system = prompts.WEB_QUERY_PLANNER_SYSTEM.format(n=n)
    user = prompts.WEB_QUERY_PLANNER_USER.format(
        post=post, draft=draft, critique=critique or "(no critique yet)", n=n
    )
    text = chat(llm, system, user)
    queries = _parse_queries(text, n)
    return queries