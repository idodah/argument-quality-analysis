"""Node: A-vs-B compare on the INITIAL drafts; eliminate the loser.

Runs immediately after `generate_initial`. Compares the two raw drafts on the
fine-tuned Qwen ranker and writes the winner to `argument`; the loser is simply
dropped and nothing downstream reads it again. This is the only A-vs-B decision
in the graph, made when both drafts are equally unpolished (raw initial), which
is the fair point to compare them.

The ranker only decides which of the two drafts is *better* — it does not
verify that the winner actually refutes the trope. That job belongs to the stance
gate downstream: if the chosen survivor is off-topic or drifting, stance
check will trigger a regeneration (and both drafts get replaced). So the stance
gate effectively overrides the ranker on bad initial pairs.

CAVEAT — elimination is permanent and not stance-aware. The loser is frozen
(only the survivor iterates), and regeneration discards BOTH drafts, not just
the survivor. So if the ranker picks the weaker of two similar-quality drafts,
the better raw material is unrecoverable except by a full regeneration. We
accept this: a per-iteration A-vs-B race would cost a ranker call every pass,
and the cheap stance gate (early_stance_check) plus regeneration already
catch the case where the survivor is on the wrong track.
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

    chosen = state["arg_a"] if winner == "A" else state["arg_b"]
    return {
        "winner": winner,
        "argument": chosen,
        "generation": chosen,
        "history": history,
    }
