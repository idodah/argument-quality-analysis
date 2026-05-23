"""Node: final pairwise Llama comparison between arg_a and arg_b."""

from __future__ import annotations

from agents.graph.ranker import get_ranker
from agents.graph.state import GraphState, strip_citations


def final_compare(state: GraphState) -> GraphState:
    ranker = get_ranker()
    result = ranker.score_pair(
        state["topic"],
        state["original_post"],
        arg_a=strip_citations(state["arg_a"]),
        arg_b=strip_citations(state["arg_b"]),
    )
    print(f"[final] score_a={result['score_a']:.3f} score_b={result['score_b']:.3f} winner={result['winner']}")
    return {**state, "winner": result["winner"], "final_scores": result}