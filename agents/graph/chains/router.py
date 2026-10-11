"""Adaptive-RAG router + web-query planner, via native LLM tool calling.

Each retrieval option is exposed to the LLM as a tool, and the router is forced
(`tool_choice="any"`) to call exactly one of them:

  - search_local_corpus  -> 'local' (curated CMV corpus)
  - search_web(queries)  -> 'web'   (Tavily), with the queries as tool arguments
  - skip_retrieval       -> 'none'  (the current draft is already strong enough
                                     to refine without new evidence)

The tools are never executed by the model: the chosen call is translated into
(mode, queries) and the graph's conditional edges run the matching retriever
node. Mode and queries therefore still come from a single LLM call, but the
arguments are schema-validated by the provider instead of regex-parsed JSON.
"""

from __future__ import annotations

from agents import prompts
from agents.llm import call_tools, deterministic_llm
from agents.graph.state import WEB_QUERIES, RetrievalMode

SEARCH_LOCAL_CORPUS = {
    "type": "function",
    "function": {
        "name": "search_local_corpus",
        "description": (
            "Search the curated local document store of persuasive CMV arguments. "
            "Best for historical, legal, or well-documented factual claims."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

SEARCH_WEB = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Run live web searches. Best for recent events or current statistics. "
            f"Provide exactly {WEB_QUERIES} short, search-friendly queries, each "
            "covering a different angle of the gaps the critique identified."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Exactly {WEB_QUERIES} distinct search queries.",
                }
            },
            "required": ["queries"],
        },
    },
}

SKIP_RETRIEVAL = {
    "type": "function",
    "function": {
        "name": "skip_retrieval",
        "description": (
            "Fetch no new evidence this pass: the draft only needs structure / "
            "framing / rhetoric work, or prior evidence already covers its claims. "
            "Do not pick this on the first pass, before any retrieval has happened."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

ROUTER_TOOLS = [SEARCH_LOCAL_CORPUS, SEARCH_WEB, SKIP_RETRIEVAL]

_TOOL_MODES: dict[str, RetrievalMode] = {
    "search_local_corpus": "local",
    "search_web": "web",
    "skip_retrieval": "none",
}


def _clean_queries(args: dict, n: int) -> list[str]:
    """Pull `args["queries"]` as up to `n` non-blank strings (in order)."""
    raw = args.get("queries") or []
    if not isinstance(raw, list):
        return []
    return [str(q).strip() for q in raw if str(q).strip()][:n]


def _decision_from_tool_calls(tool_calls: list[dict], n: int) -> tuple[RetrievalMode, list[str]]:
    # Default to 'local': the curated corpus is always available, so it is the
    # safe fallback when the model made no (or an unknown) tool call.
    if not tool_calls:
        return "local", []
    call = tool_calls[0]
    mode = _TOOL_MODES.get(call.get("name", ""), "local")
    # Queries are only meaningful for the web arm.
    queries = _clean_queries(call.get("args") or {}, n) if mode == "web" else []
    return mode, queries


def route_retrieval(post: str, draft: str, critique: str = "") -> tuple[RetrievalMode, list[str]]:
    """Return (mode, web_queries). Queries are empty unless mode='web'."""
    user = prompts.ROUTER_USER.format(
        post=post, draft=draft, critique=critique or "(none yet)"
    )
    calls = call_tools(deterministic_llm(), prompts.ROUTER_SYSTEM, user, ROUTER_TOOLS)
    return _decision_from_tool_calls(calls, WEB_QUERIES)


def _queries_from_tool_calls(tool_calls: list[dict], n: int) -> list[str]:
    for call in tool_calls:
        if call.get("name") == "search_web":
            return _clean_queries(call.get("args") or {}, n)
    return []


def plan_web_queries(post: str, draft: str, critique: str = "") -> list[str]:
    """Plan web search queries from the post/draft/critique.

    Used by the web_search fallback node when the router did not pre-plan
    queries (e.g. it chose 'local' but grade_docs then found the docs irrelevant).
    Forces a `search_web` call and returns its query arguments.
    """
    user = prompts.WEB_QUERY_PLANNER_USER.format(
        post=post, draft=draft, critique=critique or "(none yet)"
    )
    calls = call_tools(
        deterministic_llm(),
        prompts.WEB_QUERY_PLANNER_SYSTEM.format(n=WEB_QUERIES),
        user,
        [SEARCH_WEB],
        tool_choice="search_web",
    )
    return _queries_from_tool_calls(calls, WEB_QUERIES)
