"""Node: pairwise rank new vs prev version of the active side using Llama."""

from __future__ import annotations

from agents.graph.ranker import get_ranker
from agents.graph.state import GraphState, strip_citations


def rank_against_prev(state: GraphState) -> GraphState:
    side = state["active_side"]
    if side == "A":
        new_arg, prev_arg, iter_n = state["arg_a"], state["arg_a_prev"], state["iter_a"]
    else:
        new_arg, prev_arg, iter_n = state["arg_b"], state["arg_b_prev"], state["iter_b"]

    if not prev_arg:
        history = state.get("history", []) + [{
            "side": side, "iter": iter_n, "kept": True, "reason": "first_refine",
        }]
        return {**state, "history": history}

    ranker = get_ranker()
    result = ranker.score_pair(
        state["topic"],
        state["original_post"],
        arg_a=strip_citations(new_arg),
        arg_b=strip_citations(prev_arg),
    )
    new_won = result["winner"] == "A"
    margin = result["score_a"] - result["score_b"]
    history = state.get("history", []) + [{
        "side": side, "iter": iter_n,
        "score_new": result["score_a"], "score_prev": result["score_b"],
        "margin": margin, "kept": new_won,
    }]
    print(f"[rank] side={side} iter={iter_n} new={result['score_a']:.3f} prev={result['score_b']:.3f} margin={margin:+.3f} kept={new_won}")

    if new_won:
        return {**state, "history": history}

    if side == "A":
        return {**state, "arg_a": prev_arg, "converged_a": True, "history": history}
    return {**state, "arg_b": prev_arg, "converged_b": True, "history": history}