"""Conditional-edge routing functions for the merged self-RAG graph.

Two feedback loops (each cycles back to an earlier node, each with its own cap)
plus one forward branch:

  router -> retrieve_{local|web}
         -> grade_docs --(web_search?)--> web_search --> reflect   [forward branch,
                       --(else)----------------------> reflect      not a loop]
         -> reflect -> refine -> hallucination_check
              GROUNDING LOOP: not grounded -> refine again, up to
                MAX_GROUND_RETRIES per outer pass (reset each pass by router)
              grounded -> stance_check
                   pro-Israel -> final_compare (END)
                   OUTER/STANCE LOOP: not pro-Israel -> router again, up to
                     MAX_OUTER_ITERS passes (each carries a regen_reason so the
                     next refine knows it must fix the stance)
                   stance still failing at the cap -> force_regenerate -> END

There are three conditional routing functions below, but only two of them form
loops; route_after_grade only ever moves forward. Counters are incremented
inside the nodes (hallucination_check, stance_check); these functions only read
them to choose an edge.
"""

from __future__ import annotations

from agents.graph.state import GraphState, MAX_GROUND_RETRIES, MAX_OUTER_ITERS


def route_after_router(state: GraphState) -> str:
    return {"local": "retrieve_local", "web": "retrieve_web"}[state["retrieval_mode"]]


def route_after_grade(state: GraphState) -> str:
    """If grade_docs flagged the docs irrelevant, run a web search first."""
    return "web_search" if state.get("web_search") else "reflect"


def route_after_hallucination(state: GraphState) -> str:
    """Grounding loop: re-refine while ungrounded and the per-pass budget lasts."""
    if state.get("grounded", True):
        return "stance_check"
    if state.get("ground_retries", 0) >= MAX_GROUND_RETRIES:
        # Out of grounding retries this pass: stop re-refining and let the stance
        # check + final_compare report the best-available draft.
        return "stance_check"
    return "refine"


def route_after_stance(state: GraphState) -> str:
    """Outer/stance loop: reroute to the router while not pro-Israel and the
    outer budget lasts; force a targeted rewrite once the budget is spent."""
    if state.get("pro_israel_reply", True):
        return "final_compare"
    if state.get("outer_iter", 0) >= MAX_OUTER_ITERS:
        return "force_regenerate"
    return "router"
