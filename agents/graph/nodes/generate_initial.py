"""Node: produce the two initial pro-Israel drafts."""

from __future__ import annotations

from agents.graph.chains.initial_generator import generate_initial_pair
from agents.graph.state import GraphState


def generate_initial(state: GraphState) -> GraphState:
    print("[generate_initial] drafting two pro-Israel responses...")
    arg_a, arg_b = generate_initial_pair(state["topic"], state["original_post"])
    return {
        **state,
        "post": state["original_post"],
        "arg_a": arg_a,
        "arg_b": arg_b,
        "arg_a_prev": "",
        "arg_b_prev": "",
        "iter_a": 0,
        "iter_b": 0,
        "converged_a": False,
        "converged_b": False,
        "active_side": "A",
        "documents": [],
        "web_search": False,
        "critique_history": [],
        "outer_iter": 0,
        "ground_retries": 0,
        "regen_reason": "",
        "history": [],
    }