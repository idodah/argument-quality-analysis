"""Node: cheap pre-refinement stance gate on the raw survivor.

Runs right after `eliminate_loser`, before any retrieval/refine. It catches the
one case a refinement pass can't rescue — an off-topic or anti-Israel draft —
and regenerates immediately instead of burning a full refine cycle on it first.

Routing (via the stamped `early_action`):
  - off_topic_or_anti, regen budget left  -> generate_initial (regenerate)
  - off_topic_or_anti, regen budget spent  -> router (refine once)
  - anything else                          -> router

The early gate never ships and never decides give-up — that stays with the late
`stance_check`, which always runs after a refinement pass. So two invariants
hold: nothing ships straight from a raw draft, and once the early budget is
spent the gate stops short-circuiting, guaranteeing the late gate is reached
(bounded termination). See `builder.py` for how the two gates compose.
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

    # Off-topic / anti-Israel raw survivor: regenerate while the early budget
    # lasts, else fall through to refinement (see the module docstring).
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

    - "generate_initial": off-topic survivor with regeneration budget left (regenerate)
    - "router":           on-topic survivor, OR off-topic with the budget spent
                          (refine once; the late stance_check owns give-up)

    The early gate never routes to finalize: give-up is decided solely by the
    late stance_check after a refinement pass. Reading the node's stamped
    `early_action` avoids re-deriving the branch from early_regen_iter, which the node
    has already incremented.
    """
    return state.get("early_action", "router")
