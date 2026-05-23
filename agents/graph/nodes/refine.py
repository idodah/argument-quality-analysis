"""Node: rewrite the active side's draft using critique + evidence."""

from __future__ import annotations

from agents.graph.chains.refiner import refine_draft
from agents.graph.state import GraphState, active_view


def refine(state: GraphState) -> GraphState:
    side, current, _prev, critique = active_view(state)
    improved = refine_draft(
        state["topic"],
        state["original_post"],
        current,
        critique,
        state.get("retrieved", []) or [],
    )
    if side == "A":
        return {**state, "arg_a_prev": current, "arg_a": improved, "iter_a": state.get("iter_a", 0) + 1}
    return {**state, "arg_b_prev": current, "arg_b": improved, "iter_b": state.get("iter_b", 0) + 1}