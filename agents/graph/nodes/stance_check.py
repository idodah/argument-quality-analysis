"""Node: classify the current argument's stance (pro / neutral / off-topic-or-anti).

Routes:
  - pro_israel           -> finalize (done)
  - neutral_needs_refine -> router (refinement loop, budget MAX_REFINE_ITERS)
  - off_topic_or_anti    -> generate_initial (regen loop, budget MAX_REGEN_ITERS)

When both budgets are exhausted, sets `gave_up=True` with a give_up_reason and
routes to finalize; finalize and run.py surface that honestly rather
than ship a non-pro-Israel argument as if it were one.

Counters owned here:
  - refine_iter: incremented on neutral_needs_refine, reset by regeneration.
  - regen_iter: incremented on off_topic_or_anti (never resets).
"""

from __future__ import annotations

from agents.graph.chains.stance_checker import check_stance
from agents.graph.state import (
    MAX_REFINE_ITERS,
    MAX_REGEN_ITERS,
    GraphState,
)


def stance_check(state: GraphState) -> GraphState:
    side = state["active_side"]
    draft = state["arg_a"] if side == "A" else state["arg_b"]
    verdict = check_stance(state["original_post"], draft)
    stance = verdict["stance"]
    reason = verdict["reason"]

    refine_iter = state.get("refine_iter", 0)
    regen_iter = state.get("regen_iter", 0)

    out: dict = {
        **state,
        "stance": stance,
        "pro_israel_reply": stance == "pro_israel",
        "stance_reason": reason,
    }

    if stance == "pro_israel":
        print(f"[stance_check] side={side} stance=pro_israel (refine={refine_iter}/{MAX_REFINE_ITERS}, regen={regen_iter}/{MAX_REGEN_ITERS}) — done")
    elif stance == "neutral_needs_refine":
        new_refine_iter = refine_iter + 1
        out["refine_iter"] = new_refine_iter
        out["regen_reason"] = reason or "the previous revision was on-topic but did not clearly argue the pro-Israel side"
        if new_refine_iter > MAX_REFINE_ITERS:
            # Refinement budget for this generation is spent. Promote to a
            # regeneration if we still have regen budget; otherwise give up.
            print(f"[stance_check] side={side} stance=neutral but refine budget exhausted ({new_refine_iter}/{MAX_REFINE_ITERS}) -> escalate")
            stance = "off_topic_or_anti"
            out["stance"] = stance

        else:
            print(f"[stance_check] side={side} stance=neutral_needs_refine ({reason}) -> router (refine {new_refine_iter}/{MAX_REFINE_ITERS})")

    if stance == "off_topic_or_anti":
        new_regen_iter = regen_iter + 1
        out["regen_iter"] = new_regen_iter
        # Reset the per-generation counters; the next generation starts fresh.
        out["refine_iter"] = 0
        out["ground_retries"] = 0
        out["regen_reason"] = reason or "the previous draft was off-topic or anti-Israel"
        if new_regen_iter > MAX_REGEN_ITERS:
            # Both budgets exhausted. Give up honestly rather than ship a
            # non-pro-Israel argument as the answer.
            out["gave_up"] = True
            out["give_up_reason"] = (
                f"Pipeline could not produce a pro-Israel argument after "
                f"{MAX_REFINE_ITERS} refinement and {MAX_REGEN_ITERS} regeneration "
                f"attempts. Last stance verdict: {state.get('stance', stance)} "
                f"({reason or 'no reason given'})."
            )
            print(f"[stance_check] side={side} budgets exhausted (refine cap + {new_regen_iter}/{MAX_REGEN_ITERS} regens) -> give up")
        else:
            print(f"[stance_check] side={side} stance=off_topic_or_anti ({reason}) -> generate_initial (regen {new_regen_iter}/{MAX_REGEN_ITERS})")

    out["history"] = state.get("history", []) + [{
        "side": side, "stage": "stance_check", "stance": stance, "reason": reason,
        "refine_iter": out.get("refine_iter", refine_iter),
        "regen_iter": out.get("regen_iter", regen_iter),
        "gave_up": out.get("gave_up", False),
    }]
    return out
