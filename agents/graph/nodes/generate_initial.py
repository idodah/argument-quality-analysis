"""Node: produce the two initial candidate refutations.

Runs once at the start of the graph AND again whenever either stance gate
sends an off-topic / drifting draft back for regeneration. On the first
call all counters and history start at zero; on a regeneration we reset the
per-generation state (drafts, refinement counter, retrieval / critique
memory) but preserve the cross-run state (`early_regen_iter`,
`late_regen_iter`, `history`, `regen_reason`) so the traceability survives
the restart and each gate's regeneration budget keeps counting independently.
"""

from __future__ import annotations

from agents.graph.chains.initial_generator import generate_initial_pair
from agents.graph.state import GraphState


def generate_initial(state: GraphState) -> GraphState:
    early_regen_iter = state.get("early_regen_iter", 0)
    late_regen_iter = state.get("late_regen_iter", 0)
    regen_reason = state.get("regen_reason", "")
    total_regens = early_regen_iter + late_regen_iter
    if total_regens > 0:
        print(f"[generate_initial] REGENERATING "
              f"(early={early_regen_iter}, late={late_regen_iter}) — reason: {regen_reason}")
    else:
        print("[generate_initial] drafting two candidate refutations...")

    arg_a, arg_b = generate_initial_pair(
        state["topic"], state["original_post"], regen_reason=regen_reason,
    )

    history = state.get("history", [])
    if total_regens > 0:
        history = history + [{
            "stage": "generate_initial",
            "early_regen_iter": early_regen_iter,
            "late_regen_iter": late_regen_iter,
            "regen_reason": regen_reason,
        }]

    return {
        "post": state["original_post"],
        # Pair handed off to eliminate_loser; downstream of that node only
        # `argument` is read.
        "arg_a": arg_a,
        "arg_b": arg_b,
        "argument": "",
        "iter": 0,
        # Reset per-generation memory: retrieval pool, critique trail, refinement
        # and grounding counters. Regen counters and history are preserved.
        "documents": [],
        "web_search": False,
        "critique_history": [],
        "refine_iter": 0,
        "ground_retries": 0,
        "consecutive_noop_refines": 0,
        "noop_streak_pass": -1,
        # Clear stance state so the new pair gets re-evaluated cleanly.
        "stance": None,
        "refutes_trope": True,
        "stance_reason": "",
        "gave_up": False,
        "give_up_reason": "",
        "history": history,
    }