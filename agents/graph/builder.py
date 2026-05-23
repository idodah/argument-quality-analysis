"""LangGraph wiring: builds the refinement graph and exposes a run entrypoint.

The graph fuses four patterns:

  - Adaptive RAG : the router node picks per-iteration between local docs,
                   web search, or no retrieval.
  - Self-RAG     : the grade_docs node filters retrieved chunks for relevance
                   before they feed the critic/refiner.
  - Reflective RAG: the reflect node grounds its critique in retrieved evidence
                   and the current draft.
  - Reflexion    : the verbal critique persists across iterations and conditions
                   the next refinement; the fine-tuned Qwen pairwise ranker
                   (models.qwen.score_pair) supplies the scalar reward that
                   decides whether to keep or revert each new version.

The two arguments are refined in alternation. A side is marked "converged"
when its new version fails to beat the previous version on the ranker, or
when MAX_ITERS is reached. The graph exits when both sides have converged.
"""

from __future__ import annotations

import argparse

from langgraph.graph import END, StateGraph

from agents.graph.nodes import (
    final_compare,
    generate_initial,
    grade_docs,
    hallucination_check,
    rank_against_prev,
    refine,
    reflect,
    retrieve_local,
    retrieve_web,
    route_after_hallucination_check,
    route_after_rank,
    route_after_router,
    router,
    skip_retrieval,
    switch_side,
)
from agents.graph.state import GraphState


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("generate_initial", generate_initial)
    g.add_node("router", router)
    g.add_node("retrieve_local", retrieve_local)
    g.add_node("retrieve_web", retrieve_web)
    g.add_node("skip_retrieval", skip_retrieval)
    g.add_node("grade_docs", grade_docs)
    g.add_node("reflect", reflect)
    g.add_node("refine", refine)
    g.add_node("hallucination_check", hallucination_check)
    g.add_node("rank_against_prev", rank_against_prev)
    g.add_node("switch_side", switch_side)
    g.add_node("final_compare", final_compare)

    g.set_entry_point("generate_initial")
    g.add_edge("generate_initial", "router")

    g.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retrieve_local": "retrieve_local",
            "retrieve_web": "retrieve_web",
            "skip_retrieval": "skip_retrieval",
        },
    )
    g.add_edge("retrieve_local", "grade_docs")
    g.add_edge("retrieve_web", "grade_docs")
    g.add_edge("skip_retrieval", "reflect")
    g.add_edge("grade_docs", "reflect")
    g.add_edge("reflect", "refine")
    g.add_edge("refine", "hallucination_check")

    g.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination_check,
        {"refine": "refine", "rank_against_prev": "rank_against_prev"},
    )

    g.add_conditional_edges(
        "rank_against_prev",
        route_after_rank,
        {"switch_side": "switch_side", "final_compare": "final_compare"},
    )
    g.add_edge("switch_side", "router")
    g.add_edge("final_compare", END)

    return g.compile()


def run_refinement(topic: str, original_post: str) -> dict:
    """Run the full graph end-to-end and return the final state."""
    graph = build_graph()
    return graph.invoke(
        {"topic": topic, "original_post": original_post},
        config={"recursion_limit": 100},
    )


def _main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Refine two pro-Israel arguments against an anti-Israel CMV post.")
    parser.add_argument("--topic", required=True, help="Topic / CMV title.")
    parser.add_argument("--post", required=True, help="The anti-Israel original post body.")
    args = parser.parse_args()

    out = run_refinement(topic=args.topic, original_post=args.post)
    print("\n========== FINAL ==========")
    print(f"Winner: {out['winner']}")
    print(f"Final scores: {out.get('final_scores')}")
    print(f"\n--- Argument A (iters={out.get('iter_a', 0)}) ---\n{out['arg_a']}\n")
    print(f"--- Argument B (iters={out.get('iter_b', 0)}) ---\n{out['arg_b']}\n")
    print("--- History ---")
    for h in out.get("history", []):
        print(h)


if __name__ == "__main__":
    _main()