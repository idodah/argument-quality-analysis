"""Conditional-edge routing functions for the graph."""

from __future__ import annotations

from agents.graph.state import GraphState, MAX_ITERS


def route_after_router(state: GraphState) -> str:
    return {"local": "retrieve_local", "web": "retrieve_web", "none": "skip_retrieval"}[state["retrieval_mode"]]


def route_after_rank(state: GraphState) -> str:
    a_done = state.get("converged_a", False) or state.get("iter_a", 0) >= MAX_ITERS
    b_done = state.get("converged_b", False) or state.get("iter_b", 0) >= MAX_ITERS
    if a_done and b_done:
        return "final_compare"
    return "switch_side"