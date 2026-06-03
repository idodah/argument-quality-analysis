"""Node: produce the two initial pro-Israel drafts.

Runs once at the start of the graph AND again whenever stance_check sends an
off-topic / anti-Israel draft back for regeneration. On the first call all
counters and history start at zero; on a regeneration we reset the
per-generation state (drafts, refinement counter, retrieval / critique
memory) but preserve the cross-run state (`regen_iter`, `history`,
`regen_reason`) so the traceability survives the restart and the regeneration
budget keeps counting.
"""

from __future__ import annotations

from agents.graph.chains.initial_generator import generate_initial_pair
from agents.graph.state import GraphState


def generate_initial(state: GraphState) -> GraphState:
    regen_iter = state.get("regen_iter", 0)
    regen_reason = state.get("regen_reason", "")
    if regen_iter > 0:
        print(f"[generate_initial] REGENERATING (regen pass {regen_iter}) — reason: {regen_reason}")
    else:
        print("[generate_initial] drafting two pro-Israel responses...")

    arg_a, arg_b = generate_initial_pair(state["topic"], state["original_post"])

    history = state.get("history", [])
    if regen_iter > 0:
        history = history + [{
            "stage": "generate_initial",
            "regen_iter": regen_iter,
            "regen_reason": regen_reason,
        }]

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
        # Reset per-generation memory: retrieval pool, critique trail, refinement
        # and grounding counters. regen_iter and history are preserved.
        "documents": [],
        "web_search": False,
        "critique_history": [],
        "refine_iter": 0,
        "ground_retries": 0,
        # Clear stance state so the new pair gets re-evaluated cleanly.
        "stance": None,
        "pro_israel_reply": True,
        "stance_reason": "",
        "gave_up": False,
        "give_up_reason": "",
        "history": history,
    }