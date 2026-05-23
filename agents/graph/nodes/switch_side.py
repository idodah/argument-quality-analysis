"""Node: flip active_side, also enforcing max-iter convergence."""

from __future__ import annotations

from agents.graph.state import GraphState, MAX_ITERS


def switch_side(state: GraphState) -> GraphState:
    a_done = state.get("converged_a", False) or state.get("iter_a", 0) >= MAX_ITERS
    b_done = state.get("converged_b", False) or state.get("iter_b", 0) >= MAX_ITERS
    state = {**state, "converged_a": a_done, "converged_b": b_done}
    if state["active_side"] == "A" and not b_done:
        return {**state, "active_side": "B"}
    if state["active_side"] == "B" and not a_done:
        return {**state, "active_side": "A"}
    return state