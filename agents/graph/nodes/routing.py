"""Conditional-edge routing functions for the merged self-RAG graph.

Three independent loops + one forward branch:

  router -> retrieve_{local|web}
         -> grade_docs --(web_search?)--> web_search --> reflect   [forward branch]
                       --(else)----------------------> reflect
         -> reflect -> refine -> hallucination_check
              GROUNDING LOOP: not grounded -> refine again, up to
                MAX_GROUND_RETRIES per refinement pass (reset by router)
              grounded -> stance_check
                   pro_israel           -> finalize (END)
                   neutral_needs_refine -> router (REFINEMENT loop,
                     up to MAX_REFINE_ITERS passes per generation)
                   off_topic_or_anti    -> generate_initial (REGENERATION loop,
                     up to MAX_LATE_REGEN_ITERS restarts; each resets refine_iter)
                   both budgets exhausted -> finalize (gave_up=True)

Counters are incremented inside the nodes (hallucination_check, stance_check);
these functions only read them to choose an edge.
"""

from __future__ import annotations

from agents.graph.state import GraphState, MAX_GROUND_RETRIES


def route_after_router(state: GraphState) -> str:
    return {
        "local": "retrieve_local",
        "web": "retrieve_web",
        "none": "skip_retrieval",
    }[state["retrieval_mode"]]


def route_after_grade(state: GraphState) -> str:
    """If grade_docs flagged the docs irrelevant, run a web search first."""
    return "web_search" if state.get("web_search") else "reflect"


def route_after_hallucination(state: GraphState) -> str:
    """Grounding loop: re-refine while ungrounded and the per-pass budget lasts."""
    if state.get("grounded", True):
        return "stance_check"
    if state.get("ground_retries", 0) >= MAX_GROUND_RETRIES:
        # Out of grounding retries this pass: stop re-refining and let the stance
        # check + finalize report the best-available draft.
        return "stance_check"
    return "refine"


def route_after_stance(state: GraphState) -> str:
    """Three-way stance routing.

    - gave_up: end the graph honestly at finalize.
    - off_topic_or_anti (and not gave_up): regeneration loop; back to
      generate_initial.
    - neutral_needs_refine: refinement loop; back to router. The node already
      incremented refine_iter and will give_up via stance=off_topic_or_anti if
      the refinement budget was the trigger.
    - pro_israel: done; go to finalize. Every path to stance_check is
      via refine -> hallucination_check, so the grounding pass has already
      run by definition.
    """
    if state.get("gave_up"):
        return "finalize"
    stance = state.get("stance", "pro_israel")
    if stance == "off_topic_or_anti":
        return "generate_initial"
    if stance == "neutral_needs_refine":
        return "router"
    return "finalize"
