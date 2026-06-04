"""Node: Reflexion-style critic that accumulates critiques across passes.

Each pass's critique is appended to `critique_history` (labeled by refinement
pass) so the refiner sees the full memory of prior mistakes, not just the
latest note.
"""

from __future__ import annotations

from agents.graph.chains.reflector import reflect_on_draft
from agents.graph.state import GraphState, current_argument


def reflect(state: GraphState) -> GraphState:
    current = current_argument(state)
    evidence = state.get("documents") or []
    critique = reflect_on_draft(state["original_post"], current, evidence)

    pass_n = state.get("refine_iter", 0) + 1
    labeled = f"[critique from pass {pass_n}]\n{critique}"
    history = state.get("critique_history", []) + [labeled]
    print(f"[reflect] pass={pass_n} critique={len(critique)} chars "
          f"(history now {len(history)} critiques)")

    return {"critique": critique, "critique_history": history}