"""Node: Adaptive-RAG router (local | web) + web-query planner.

The router chain returns both the retrieval mode AND, when mode='web', the
search queries to use — both in one LLM call.

DIAGNOSTIC OVERRIDE: set FORCE_RETRIEVAL_MODE={local|web} in the env to pin the
route regardless of what the router LLM decides. Used to exercise the web path
on demand. Unset it to restore normal adaptive routing.
"""

from __future__ import annotations

import os

from agents.graph.chains.router import route_retrieval
from agents.graph.state import GraphState, active_view


def router(state: GraphState) -> GraphState:
    side, current, _prev, critique = active_view(state)
    mode, queries = route_retrieval(state["original_post"], current, critique)

    forced = os.environ.get("FORCE_RETRIEVAL_MODE", "").lower().strip()
    if forced in ("local", "web"):
        if forced != mode:
            print(f"[router] FORCE_RETRIEVAL_MODE={forced} overriding LLM choice ({mode})")
        mode = forced
        # If we forced web but the LLM didn't plan queries, the retrieve_web
        # node falls back to a topic+draft query, so an empty list is safe.
        if mode != "web":
            queries = []

    # Each refinement pass starts fresh: reset the grounding-retry budget and
    # clear any grounding verdict left over from the previous pass, so stale
    # ungrounded issues don't leak into this pass's refine before it has been
    # re-graded.
    regen = state.get("regen_reason") or ""
    if regen:
        print(f"[router] refine pass {state.get('refine_iter', 0)}: stance fix (reason: {regen})")
    print(f"[router] side={side} refine_iter={state.get('refine_iter', 0)} -> {mode} (queries={len(queries)})")
    return {
        **state,
        "retrieval_mode": mode,
        "web_queries": queries,
        "ground_retries": 0,
        "grounded": True,
        "hallucination_issues": [],
    }