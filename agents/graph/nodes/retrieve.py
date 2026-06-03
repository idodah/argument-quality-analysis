"""Nodes: local retrieval, web retrieval (router arm), and the grade-triggered
web_search node.

Retrieved chunks are written to both `retrieved` (the working set the grader and
reflect/refine read this iteration) and `documents` (the running citation pool).
"""

from __future__ import annotations

from agents.graph.chains.router import plan_web_queries
from agents.graph.state import GraphState, LOCAL_K, WEB_ALLOWED_DOMAINS, WEB_K, active_view
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
        _WEB = WebRetriever(k=WEB_K, allowed_domains=WEB_ALLOWED_DOMAINS)
    return _WEB


def _dedupe(chunks: list[str]) -> list[str]:
    """Dedupe by [url] header when present, else by full content; keep order."""
    seen: set[str] = set()
    out: list[str] = []
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
        out.append(c)
    return out


def retrieve_local(state: GraphState) -> GraphState:
    # Query with the topic + the post we're arguing against, so we retrieve
    # CMV arguments that successfully countered a similar opposing position.
    _side, _current, _prev, _crit = active_view(state)
    post = state.get("original_post", "") or ""
    query = f"{state['topic']}\n\n{post[:600]}"
    chunks = _local().retrieve(query, k=LOCAL_K, where={"source": "cmv_israel"})
    print(f"[retrieve_local] {len(chunks)} chunks for side {_side}")
    return {**state, "retrieved": chunks, "documents": chunks}


def retrieve_web(state: GraphState) -> GraphState:
    # Queries were planned by the router in the same LLM call that picked 'web'
    # (3-5 search terms targeting the critique's evidence gaps).
    _side, current, _prev, _critique = active_view(state)
    queries = state.get("web_queries") or []
    if not queries:
        queries = [f"{state['topic']} {current[:200]}"]

    chunks: list[str] = []
    for q in queries:
        chunks.extend(_web().retrieve(q, k=WEB_K))
    deduped = _dedupe(chunks)
    print(f"[retrieve_web] {len(queries)} queries -> {len(deduped)} unique chunks")
    return {**state, "retrieved": deduped, "documents": deduped}


def skip_retrieval(state: GraphState) -> GraphState:
    """Router 'none' arm: skip retrieval this pass.

    Clears `retrieved` so reflect/refine see no new evidence this iteration,
    but PRESERVES `documents` (the running citation pool from earlier passes)
    so refine can still cite previously-grounded facts. The downstream
    hallucination_check skips on retrieval_mode=='none' for the same reason it
    skips on 'local': there is no fresh factual evidence to ground against.
    """
    _side, _current, _prev, _critique = active_view(state)
    print(f"[skip_retrieval] side={_side} router chose 'none' — no new retrieval this pass")
    return {**state, "retrieved": []}


def web_search(state: GraphState) -> GraphState:
    """Self-RAG web-search fallback, triggered when grade_docs finds the local
    documents irrelevant (and drops them). Runs a Tavily search (max_results=5)
    and merges the results into `documents` (deduped). Since grade_docs clears
    irrelevant docs first, `documents` is normally empty here, so this fills it
    with web results."""
    print("---WEB SEARCH---")
    _side, current, _prev, critique = active_view(state)
    # Prefer the router's pre-planned queries. If none exist (router chose
    # 'local' but the docs were graded irrelevant), plan fresh queries from the
    # critique rather than searching the raw topic+draft.
    queries = state.get("web_queries") or []
    if not queries:
        queries = plan_web_queries(state["original_post"], current, critique)
        print(f"[web_search] router had no queries -> planned {len(queries)} from critique")
    if not queries:
        queries = [f"{state['topic']} {current[:200]}"]

    new_chunks: list[str] = []
    for q in queries:
        new_chunks.extend(_web().retrieve(q, k=WEB_K))

    documents = _dedupe((state.get("documents") or []) + new_chunks)
    print(f"[web_search] {len(queries)} queries -> documents now {len(documents)}")
    return {**state, "retrieved": documents, "documents": documents, "web_search": False}
