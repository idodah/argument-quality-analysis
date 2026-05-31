"""Node: A-vs-B compare on the INITIAL drafts; eliminate the loser.

Runs immediately after `generate_initial`. Compares the two raw drafts on the
fine-tuned Qwen ranker, keeps the winner as the active side, and marks the
other side converged so the rest of the graph never touches it. This moves
the only A-vs-B decision to the moment when both arguments are at the same
level of polish (raw initial), which is the fair comparison.
"""

from __future__ import annotations

from agents.ranker import get_ranker
from agents.graph.state import GraphState, strip_citations


def eliminate_loser(state: GraphState) -> GraphState:
    ranker = get_ranker()
    result = ranker.score_pair(
        state["topic"],
        state["original_post"],
        arg_a=strip_citations(state["arg_a"]),
        arg_b=strip_citations(state["arg_b"]),
    )
    winner = result["winner"]
    print(
        f"[eliminate_loser] initial-draft compare: "
        f"score_a={result['score_a']:.3f} score_b={result['score_b']:.3f} "
        f"-> survivor={winner}"
    )

    history = state.get("history", []) + [{
        "stage": "eliminate_loser",
        "score_a": result["score_a"],
        "score_b": result["score_b"],
        "survivor": winner,
    }]

    if winner == "A":
        return {
            **state,
            "active_side": "A",
            "converged_a": False,
            "converged_b": True,  # B is out; never touch again
            "generation": state["arg_a"],
            "history": history,
        }
    return {
        **state,
        "active_side": "B",
        "converged_a": True,  # A is out; never touch again
        "converged_b": False,
        "generation": state["arg_b"],
        "history": history,
    }
