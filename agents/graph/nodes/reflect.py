"""Node: Reflexion-style critique persisted per side."""

from __future__ import annotations

from agents.graph.chains.reflector import reflect_on_draft
from agents.graph.state import GraphState, active_view


def reflect(state: GraphState) -> GraphState:
    side, current, _prev, _crit = active_view(state)
    critique = reflect_on_draft(state["original_post"], current, state.get("retrieved", []) or [])
    print(f"[reflect] side={side} critique={len(critique)} chars")
    if side == "A":
        return {**state, "critique": critique, "critique_a": critique}
    return {**state, "critique": critique, "critique_b": critique}