"""Nodes: local retrieval, web retrieval, and a no-op skip node."""

from __future__ import annotations

from agents.graph.state import GraphState, LOCAL_K, WEB_K, active_view
from agents.retrieval import LocalRetriever, WebRetriever

_LOCAL: LocalRetriever | None = None
_WEB: WebRetriever | None = None


def _local() -> LocalRetriever:
    global _LOCAL
    if _LOCAL is None:
        _LOCAL = LocalRetriever(k=LOCAL_K)
    return _LOCAL


def _web() -> WebRetriever:
    global _WEB
    if _WEB is None:
        _WEB = WebRetriever(k=WEB_K)
    return _WEB


def retrieve_local(state: GraphState) -> GraphState:
    # Query with the topic + the post we're arguing against, so we retrieve
    # CMV arguments that successfully countered a similar opposing position
    # (rather than arguments that merely resemble our own current draft).
    _side, current, _prev, _crit = active_view(state)
    post = state.get("original_post", "") or ""
    query = f"{state['topic']}\n\n{post[:600]}"
    # Restrict to the curated CMV-Israel corpus (skips any loose docs/ chunks).
    chunks = _local().retrieve(query, k=LOCAL_K, where={"source": "cmv_israel"})
    print(f"[retrieve_local] {len(chunks)} chunks for side {_side}")
    return {**state, "retrieved": chunks}


def retrieve_web(state: GraphState) -> GraphState:
    # Queries were planned by the router in the same LLM call that picked 'web'.
    # Querying by the critique (not the draft itself) steers search toward what
    # is MISSING rather than what the draft already says.
    _side, current, _prev, _critique = active_view(state)
    queries = state.get("web_queries") or []
    if not queries:
        # Router returned nothing usable; fall back to the old topic+draft query.
        queries = [f"{state['topic']} {current[:200]}"]

    chunks: list[str] = []
    for q in queries:
        chunks.extend(_web().retrieve(q, k=WEB_K))
    # Dedupe while preserving order. Prefer the [url] header (one entry per page)
    # over full-content match, since the same page can surface for multiple
    # queries with slightly different snippets.
    seen: set[str] = set()
    deduped: list[str] = []
    for c in chunks:
        if not c:
            continue
        key = c
        first_line = c.split("\n", 1)[0]
        if first_line.startswith("[url] "):
            key = first_line[len("[url] "):].strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    print(f"[retrieve_web] {len(queries)} queries -> {len(deduped)} unique chunks")
    return {**state, "retrieved": deduped}


def skip_retrieval(state: GraphState) -> GraphState:
    return {**state, "retrieved": []}