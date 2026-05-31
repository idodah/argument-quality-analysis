"""Node: Reflexion-style critic that accumulates critiques across passes.

Each pass's critique is appended to `critique_history` (labeled by outer pass)
so the refiner sees the full memory of prior mistakes, not just the latest note.
"""

from __future__ import annotations

from agents.graph.chains.reflector import reflect_on_draft
from agents.graph.state import GraphState, active_view


def reflect(state: GraphState) -> GraphState:
    side, current, _prev, _crit = active_view(state)
    evidence = state.get("documents") or state.get("retrieved") or []
    critique = reflect_on_draft(state["original_post"], current, evidence)

    pass_n = state.get("outer_iter", 0) + 1
    labeled = f"[critique from pass {pass_n}]\n{critique}"
    history = state.get("critique_history", []) + [labeled]
    print(f"[reflect] side={side} pass={pass_n} critique={len(critique)} chars "
          f"(history now {len(history)} critiques)")

    out = {**state, "critique": critique, "critique_history": history}
    out["critique_a" if side == "A" else "critique_b"] = critique
    return out