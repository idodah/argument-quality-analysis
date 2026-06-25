"""LangGraph wiring: builds the merged self-RAG refinement graph.

Flow:
  generate_initial    -> two distinct pro-Israel drafts
  eliminate_loser     -> Qwen pairwise compare (once per generation)
  early_stance_check  -> cheap stance gate on the RAW survivor: off-topic /
                         anti-Israel survivors regenerate immediately (no
                         refinement pass wasted); on-topic survivors fall
                         through to the router.
  router              -> local | web | none retrieval (LLM-decided)
  retrieve_*        -> arm-specific retrieval (skipped on 'none')
  grade_docs        -> binary relevance; if irrelevant, web_search (Tavily) first
  reflect           -> critique (missing / superfluous vs the original post)
  refine            -> revise the comment + add citations
  hallucination_check (binary, skipped on local/none retrieval):
       not grounded -> refine again (GROUNDING loop, up to MAX_GROUND_RETRIES
                       per refinement pass; reset by the router each pass)
       grounded     -> stance_check
  stance_check      -> classify the refined survivor:
       pro_israel           -> finalize -> END
       neutral_needs_refine -> router (refinement loop, MAX_REFINE_ITERS)
       off_topic_or_anti    -> generate_initial (late regen loop, MAX_LATE_REGEN_ITERS)
       gave_up              -> finalize -> END

Four caps (all per the generation/pass they scope, reset by the enclosing loop):
  - MAX_REFINE_ITERS  (stance_check -> router per generation, reset on regen)
  - MAX_EARLY_REGEN_ITERS (early_stance_check -> generate_initial, per generation,
                           reset by the late gate on each fresh generation)
  - MAX_LATE_REGEN_ITERS  (stance_check -> generate_initial, persists whole run)
  - MAX_GROUND_RETRIES (hallucination_check -> refine per refinement pass)

When the LATE regen budget is spent, the LATE stance_check sets gave_up=True
and routes to finalize, which reports the give-up honestly rather than shipping
a non-pro-Israel argument as the answer. The early gate has its OWN per-generation
regen budget (MAX_EARLY_REGEN_ITERS) and never decides give-up itself — once
its budget is spent for the current generation it just stops short-circuiting
and hands the draft to the refinement path. The late gate resets the early
budget on each fresh generation, so the two regen loops compose multiplicatively
(up to (1 + MAX_LATE_REGEN_ITERS) generations × MAX_EARLY_REGEN_ITERS early
regenerations each).

Two stance gates, complementary:
  - early_stance_check (before refinement) catches ONLY off-topic / anti-Israel
    survivors and regenerates immediately (while regen budget lasts), so a
    hopeless draft no longer burns a full refinement loop first. It has just two
    out-edges (generate_initial | router) and never routes to finalize: once the
    regen budget is spent it hands the draft to the router so the late gate is
    always reached (bounded termination).
  - stance_check (after refinement) is the SOLE authority on shipping and on
    give-up: every argument that reaches finalize via the late gate has passed
    through hallucination_check, so the grounding invariant is preserved.

Qwen is used only at eliminate_loser (no per-iteration ranking).
"""

from __future__ import annotations

import argparse

from langgraph.graph import END, StateGraph

from agents.graph.nodes import (
    early_stance_check,
    early_stance_router,
    eliminate_loser,
    finalize,
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
    g.add_node("early_stance_check", early_stance_check)
    g.add_node("router", router)
    g.add_node("retrieve_local", retrieve_local)
    g.add_node("retrieve_web", retrieve_web)
    g.add_node("grade_docs", grade_docs)
    g.add_node("web_search", web_search)
    g.add_node("reflect", reflect)
    g.add_node("refine", refine)
    g.add_node("hallucination_check", hallucination_check)
    g.add_node("stance_check", stance_check)
    g.add_node("finalize", finalize)

    g.set_entry_point("generate_initial")
    g.add_edge("generate_initial", "eliminate_loser")
    # Cheap pre-refinement stance gate on the raw survivor. It catches ONLY the
    # off-topic / anti-Israel case (which a refinement pass cannot rescue) and
    # regenerates immediately, instead of burning a full router -> retrieve ->
    # reflect -> refine -> ground cycle first. On-topic survivors fall through to
    # the router, so the late stance_check remains the authority on shipping and
    # every shipped argument still passes through the grounding pass.
    g.add_edge("eliminate_loser", "early_stance_check")
    g.add_conditional_edges(
        "early_stance_check",
        early_stance_router,
        {
            "generate_initial": "generate_initial",
            "router": "router",
        },
    )

    # Router -> retrieval arm (local, web) or straight to reflect ('none').
    g.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retrieve_local": "retrieve_local",
            "retrieve_web": "retrieve_web",
            # 'none' arm: skip retrieval/grade and go straight to reflect.
            # Refinement still runs, just without new retrieved evidence.
            "reflect": "reflect",
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

    # Stance check (used after eliminate_loser AND after hallucination_check):
    #   pro_israel           -> finalize (END)
    #   neutral_needs_refine -> router (refinement loop)
    #   off_topic_or_anti    -> generate_initial (regeneration loop)
    #   gave_up (caps spent) -> finalize (END), gave_up reported honestly
    g.add_conditional_edges(
        "stance_check",
        route_after_stance,
        {
            "finalize": "finalize",
            "router": "router",
            "generate_initial": "generate_initial",
        },
    )
    g.add_edge("finalize", END)

    return g.compile()


def run_refinement(topic: str, original_post: str) -> dict:
    """Run the full graph end-to-end and return the final state."""
    from agents.tracing import setup_tracing

    setup_tracing()  # no-op unless LangSmith is configured in the env
    graph = build_graph()
    return graph.invoke(
        {"topic": topic, "original_post": original_post},
        config={"recursion_limit": 150},
    )


def _main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Refine a pro-Israel argument against an anti-Israel CMV post.")
    parser.add_argument("--topic", required=True, help="Topic / CMV title.")
    parser.add_argument("--post", required=True, help="The anti-Israel original post body.")
    args = parser.parse_args()

    out = run_refinement(topic=args.topic, original_post=args.post)
    winner = out.get("winner", "?")
    winning_arg = out.get("generation") or out.get("argument", "")
    iters = out.get("iter", 0)
    print("\n========== FINAL ==========")
    print(f"Winner: {winner} (iters={iters})")
    print(f"Final scores: {out.get('final_scores')}")
    print(f"\n--- Winning argument ---\n{winning_arg}\n")
    print("--- History ---")
    for h in out.get("history", []):
        print(h)


if __name__ == "__main__":
    _main()
