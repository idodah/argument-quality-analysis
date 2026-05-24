"""Node: Adaptive-RAG router (local | web | none) + web-query planner.

The router chain returns both the retrieval mode AND, when mode='web', the
search queries to use — both in one LLM call.
"""

from __future__ import annotations

from agents.graph.chains.router import route_retrieval
from agents.graph.state import GraphState, active_view


def router(state: GraphState) -> GraphState:
    side, current, _prev, critique = active_view(state)
    mode, queries = route_retrieval(state["original_post"], current, critique)
    print(f"[router] side={side} iter={state.get(f'iter_{side.lower()}', 0)} -> {mode} (queries={len(queries)})")
    return {**state, "retrieval_mode": mode, "web_queries": queries}