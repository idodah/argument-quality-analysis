"""Adaptive-RAG router + web-query planner: one call returns mode and queries.

When mode='web' the same LLM call also plans the web search queries that target
the critique's evidence gaps, avoiding a second round-trip.
"""

from __future__ import annotations

import json
import re

from agents import prompts
from agents.graph.llm import chat, deterministic_llm
from agents.graph.state import WEB_QUERIES, RetrievalMode

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_decision(text: str, n: int) -> tuple[RetrievalMode, list[str]]:
    match = _JSON_OBJ_RE.search(text)
    if not match:
        return "none", []
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "none", []
    raw_mode = str(obj.get("mode", "")).lower().strip()
    if raw_mode.startswith("local"):
        mode: RetrievalMode = "local"
    elif raw_mode.startswith("web"):
        mode = "web"
    else:
        mode = "none"
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