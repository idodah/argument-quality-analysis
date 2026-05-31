"""Adaptive-RAG router + web-query planner: one call returns mode and queries.

The router picks between two retrieval sources, 'local' or 'web'. When mode='web'
the same LLM call also plans the web search queries that target the critique's
evidence gaps, avoiding a second round-trip.
"""

from __future__ import annotations

import json
import re

from agents import prompts
from agents.llm import chat, deterministic_llm
from agents.graph.state import WEB_QUERIES, RetrievalMode

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_decision(text: str, n: int) -> tuple[RetrievalMode, list[str]]:
    # Default to 'local': the curated corpus is always available, so it is the
    # safe fallback when the router output is missing or unparseable.
    match = _JSON_OBJ_RE.search(text)
    if not match:
        return "local", []
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "local", []
    raw_mode = str(obj.get("mode", "")).lower().strip()
    mode: RetrievalMode = "web" if raw_mode.startswith("web") else "local"
    raw_queries = obj.get("queries") or []
    if not isinstance(raw_queries, list):
        raw_queries = []
    queries = [str(q).strip() for q in raw_queries if str(q).strip()]
    if mode != "web":
        queries = []
    return mode, queries[:n]


def route_retrieval(post: str, draft: str, critique: str = "") -> tuple[RetrievalMode, list[str]]:
    """Return (mode, web_queries). Queries are empty unless mode='web'."""
    llm = deterministic_llm()
    system = prompts.ROUTER_SYSTEM.format(n=WEB_QUERIES)
    user = prompts.ROUTER_USER.format(
        post=post, draft=draft, critique=critique or "(none yet)"
    )
    text = chat(llm, system, user)
    return _parse_decision(text, WEB_QUERIES)


def _parse_queries(text: str, n: int) -> list[str]:
    match = _JSON_OBJ_RE.search(text)
    if not match:
        return []
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    raw = obj.get("queries") or []
    if not isinstance(raw, list):
        return []
    return [str(q).strip() for q in raw if str(q).strip()][:n]


def plan_web_queries(post: str, draft: str, critique: str = "") -> list[str]:
    """Plan web search queries from the post/draft/critique.

    Used by the web_search fallback node when the router did not pre-plan
    queries (e.g. it chose 'local' but grade_docs then found the docs irrelevant).
    """
    llm = deterministic_llm()
    system = prompts.WEB_QUERY_PLANNER_SYSTEM.format(n=WEB_QUERIES)
    user = prompts.WEB_QUERY_PLANNER_USER.format(
        post=post, draft=draft, critique=critique or "(none yet)"
    )
    text = chat(llm, system, user)
    return _parse_queries(text, WEB_QUERIES)