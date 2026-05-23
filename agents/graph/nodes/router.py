"""Node: Adaptive-RAG router (local | web | none)."""

from __future__ import annotations

from agents.graph.chains.router import route_retrieval
from agents.graph.state import GraphState, active_view


def router(state: GraphState) -> GraphState:
    side, current, _prev, critique = active_view(state)
    mode = route_retrieval(state["original_post"], current, critique)
    print(f"[router] side={side} iter={state.get(f'iter_{side.lower()}', 0)} -> {mode}")
    return {**state, "retrieval_mode": mode}