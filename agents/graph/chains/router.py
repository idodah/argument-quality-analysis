"""Adaptive-RAG router + web-query planner: one call returns mode and queries.

The router picks one of three retrieval modes: 'local' (curated CMV corpus),
'web' (Tavily), or 'none' (skip retrieval; the current draft is already strong
enough to refine without new evidence). When mode='web' the same LLM call also
plans the web search queries that target the critique's evidence gaps,
avoiding a second round-trip.
"""

from __future__ import annotations

import json
import re

from agents import prompts
from agents.llm import chat, deterministic_llm
from agents.graph.state import WEB_QUERIES, RetrievalMode

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_obj(text: str) -> dict | None:
    """Extract the first JSON object from `text`, or None if absent/malformed."""
    match = _JSON_OBJ_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _clean_queries(obj: dict, n: int) -> list[str]:
    """Pull `obj["queries"]` as up to `n` non-blank strings (in order)."""
    raw = obj.get("queries") or []
    if not isinstance(raw, list):
        return []
    return [str(q).strip() for q in raw if str(q).strip()][:n]


def _parse_decision(text: str, n: int) -> tuple[RetrievalMode, list[str]]:
    # Default to 'local': the curated corpus is always available, so it is the
    # safe fallback when the router output is missing or unparseable.
    obj = _parse_json_obj(text)
    if obj is None:
        return "local", []
    raw_mode = str(obj.get("mode", "")).lower().strip()
    if raw_mode.startswith("web"):
        mode: RetrievalMode = "web"
    elif raw_mode.startswith("none") or raw_mode == "skip":
        mode = "none"
    else:
        mode = "local"
    # Queries are only meaningful for the web arm.
    queries = _clean_queries(obj, n) if mode == "web" else []
    return mode, queries


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
    obj = _parse_json_obj(text)
    if obj is None:
        return []
    return _clean_queries(obj, n)


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