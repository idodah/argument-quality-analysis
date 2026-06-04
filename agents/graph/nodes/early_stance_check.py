"""Node: cheap pre-refinement stance gate on the raw survivor.

Runs right after `eliminate_loser`, BEFORE any retrieval / reflect / refine.
It classifies the surviving raw draft and short-circuits the one case where a
full refinement pass would be pure waste: an off-topic or overtly anti-Israel
survivor. Those can only be fixed by a fresh generation, so there is no point
spending a router -> retrieve -> grade -> reflect -> refine -> ground cycle on
them first.

Routing (early_stance_router, via the stamped `early_action`):
  - off_topic_or_anti AND regen budget left -> generate_initial (regeneration)
  - off_topic_or_anti AND regen budget spent -> router (refine once; the late
        stance_check then records gave_up and finalizes)
  - everything else (pro_israel / neutral_needs_refine) -> router

The early gate NEVER routes to finalize and NEVER decides give-up. Give-up is
owned solely by the late `stance_check`, which sees the draft after a refinement
pass. The early gate only ever (a) regenerates while it has budget, or (b) hands
the draft to the refinement path. This preserves two invariants:
  - every shipped argument has passed through at least one refine -> grounding
    pass (nothing ships straight from the raw survivor), and
  - termination is bounded: once the regen budget is spent the early gate stops
    short-circuiting, so the late gate is guaranteed to be reached.

Counter semantics for the regeneration branch mirror `stance_check`: increment
early_regen_iter and reset the per-generation counters (refine_iter, ground_retries).
"""

from __future__ import annotations

from agents.graph.chains.stance_checker import check_stance
from agents.graph.state import MAX_EARLY_REGEN_ITERS, GraphState, current_argument


def early_stance_check(state: GraphState) -> GraphState:
    draft = current_argument(state)
    verdict = check_stance(state["original_post"], draft)
    stance = verdict["stance"]
    reason = verdict["reason"]
    early_regen_iter = state.get("early_regen_iter", 0)

    if stance != "off_topic_or_anti":
        # On-topic enough to refine. Don't decide shipping here — let the normal
        # pipeline (router ... late stance_check) handle it.
        print(f"[early_stance_check] stance={stance} -> proceed to refinement")
        history = state.get("history", []) + [{
            "stage": "early_stance_check", "stance": stance,
            "reason": reason, "action": "proceed",
        }]
        return {"stance": stance, "stance_reason": reason,
                "early_action": "router", "history": history}

    # Off-topic / anti-Israel raw survivor.
    #
    # While the EARLY regeneration budget lasts, skip refinement and regenerate
    # now (the early gate's whole point: don't burn a refinement loop on a
    # draft that's on the wrong track). The early gate has its OWN per-generation
    # budget; the late gate resets it whenever it starts a fresh generation, so
    # each generation gets a full set of early regenerations. Once it's spent
    # (within the current generation), the early gate stops short-circuiting and
    # lets the draft fall through to the refinement path
    # (-> router); the LATE stance_check is the single authority that records
    # gave_up and finalizes. This keeps termination bounded without an
    # early -> finalize edge.
    budget_left = early_regen_iter < MAX_EARLY_REGEN_ITERS
    if budget_left:
        new_early_regen_iter = early_regen_iter + 1
        print(f"[early_stance_check] stance=off_topic_or_anti ({reason}) "
              f"-> regenerate (early_regen {new_early_regen_iter}/{MAX_EARLY_REGEN_ITERS}) without burning a refinement pass")
        return {
            "stance": stance,
            "stance_reason": reason,
            "early_regen_iter": new_early_regen_iter,
            "refine_iter": 0,
            "ground_retries": 0,
            "consecutive_noop_refines": 0,
            "noop_streak_pass": -1,
            "regen_reason": reason or "the initial survivor was off-topic or anti-Israel",
            "early_action": "generate_initial",
            "history": state.get("history", []) + [{
                "stage": "early_stance_check", "stance": stance,
                "reason": reason, "early_regen_iter": new_early_regen_iter, "action": "regenerate",
            }],
        }

    # Budget spent: hand off to the refinement path; the late gate owns give-up.
    print(f"[early_stance_check] stance=off_topic_or_anti but early regen budget "
          f"spent ({early_regen_iter}/{MAX_EARLY_REGEN_ITERS}) -> refine once; late stance_check decides give-up")
    return {
        "stance": stance,
        "stance_reason": reason,
        "regen_reason": reason or "the initial survivor was off-topic or anti-Israel",
        "early_action": "router",
        "history": state.get("history", []) + [{
            "stage": "early_stance_check", "stance": stance,
            "reason": reason, "early_regen_iter": early_regen_iter, "action": "handoff_to_refine",
        }],
    }


def early_stance_router(state: GraphState) -> str:
    """Route the early gate using the action the node already decided.

    - "generate_initial": off-topic survivor with regen budget left (regenerate)
    - "router":           on-topic survivor, OR off-topic with the budget spent
                          (refine once; the late stance_check owns give-up)

    The early gate never routes to finalize: give-up is decided solely by the
    late stance_check after a refinement pass. Reading the node's stamped
    `early_action` avoids re-deriving the branch from early_regen_iter, which the node
    has already incremented.
    """
    return state.get("early_action", "router")
