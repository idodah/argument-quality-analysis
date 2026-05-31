"""LangGraph wiring: builds the merged self-RAG refinement graph.

Flow:
  generate_initial  -> two distinct pro-Israel drafts
  eliminate_loser   -> Qwen pairwise compare (once); surviving side iterates
  router            -> local | web retrieval
  grade_docs        -> binary relevance; if irrelevant, web_search (Tavily) first
  reflect           -> critique (missing / superfluous vs the original post)
  refine            -> revise the comment + add citations
  hallucination_check (binary):
       not grounded -> refine again (GROUNDING loop, up to MAX_GROUND_RETRIES
                       per outer pass; reset by the router each pass)
       grounded     -> stance_check
  stance_check (pro-Israel?):
       yes -> final_compare -> END
       no  -> router again (OUTER loop, up to MAX_OUTER_ITERS passes, each
              carrying a regen_reason so refine fixes the stance)
       no & outer cap hit -> force_regenerate -> final_compare -> END

Two independent caps: MAX_OUTER_ITERS outer passes x MAX_GROUND_RETRIES grounding
re-refines each (3 x 3 -> up to 9 grounding refines). Reflexion memory: reflect
accumulates every critique into critique_history, which refine consumes in full.
Qwen is used only at eliminate_loser (no per-iteration ranking).
"""

from __future__ import annotations

import argparse

from langgraph.graph import END, StateGraph

from agents.graph.nodes import (
    eliminate_loser,
    final_compare,
    force_regenerate,
    generate_initial,
    grade_docs,
    hallucination_check,
    refine,
    reflect,
    retrieve_local,
    retrieve_web,
    route_after_grade,
    route_after_hallucination,
    route_after_router,
    route_after_stance,
    router,
    stance_check,
    web_search,
)
from agents.graph.state import GraphState


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("generate_initial", generate_initial)
    g.add_node("eliminate_loser", eliminate_loser)
    g.add_node("router", router)
    g.add_node("retrieve_local", retrieve_local)
    g.add_node("retrieve_web", retrieve_web)
    g.add_node("grade_docs", grade_docs)
    g.add_node("web_search", web_search)
    g.add_node("reflect", reflect)
    g.add_node("refine", refine)
    g.add_node("hallucination_check", hallucination_check)
    g.add_node("stance_check", stance_check)
    g.add_node("force_regenerate", force_regenerate)
    g.add_node("final_compare", final_compare)

    g.set_entry_point("generate_initial")
    g.add_edge("generate_initial", "eliminate_loser")
    g.add_edge("eliminate_loser", "router")

    # Router -> retrieval arm (local or web).
    g.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retrieve_local": "retrieve_local",
            "retrieve_web": "retrieve_web",
        },
    )
    g.add_edge("retrieve_local", "grade_docs")
    g.add_edge("retrieve_web", "grade_docs")

    # Grade docs -> web_search fallback (if irrelevant) or straight to reflect.
    g.add_conditional_edges(
        "grade_docs",
        route_after_grade,
        {"web_search": "web_search", "reflect": "reflect"},
    )
    g.add_edge("web_search", "reflect")

    g.add_edge("reflect", "refine")
    g.add_edge("refine", "hallucination_check")

    # Hallucination check: ungrounded -> re-refine (capped); grounded -> stance.
    g.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination,
        {"refine": "refine", "stance_check": "stance_check"},
    )

    # Stance check: pro-Israel -> done; not -> reroute to the router for another
    # pro-Israel refinement pass; force a rewrite once the outer cap is hit.
    g.add_conditional_edges(
        "stance_check",
        route_after_stance,
        {
            "final_compare": "final_compare",
            "router": "router",
            "force_regenerate": "force_regenerate",
        },
    )
    g.add_edge("force_regenerate", "final_compare")
    g.add_edge("final_compare", END)

    return g.compile()


def run_refinement(topic: str, original_post: str) -> dict:
    """Run the full graph end-to-end and return the final state."""
    from agents.tracing import setup_tracing

    setup_tracing()  # no-op unless LangSmith is configured in the env
    graph = build_graph()
    return graph.invoke(
        {"topic": topic, "original_post": original_post},
        config={"recursion_limit": 100},
    )


def _main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Refine a pro-Israel argument against an anti-Israel CMV post.")
    parser.add_argument("--topic", required=True, help="Topic / CMV title.")
    parser.add_argument("--post", required=True, help="The anti-Israel original post body.")
    args = parser.parse_args()

    out = run_refinement(topic=args.topic, original_post=args.post)
    winner = out["winner"]
    winning_arg = out["arg_a"] if winner == "A" else out["arg_b"]
    iters = out.get(f"iter_{winner.lower()}", 0)
    print("\n========== FINAL ==========")
    print(f"Winner: {winner} (iters={iters})")
    print(f"Final scores: {out.get('final_scores')}")
    print(f"\n--- Winning argument ---\n{winning_arg}\n")
    print("--- History ---")
    for h in out.get("history", []):
        print(h)


if __name__ == "__main__":
    _main()
