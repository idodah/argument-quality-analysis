"""Node: verify the current argument is a pro-Israel reply to the post.

Sets `pro_israel_reply`. On failure it increments `outer_iter` and records a
`regen_reason` so the next router pass + refine know the revision exists to fix
the stance. Routing (route_after_stance) sends a passing argument to
final_compare, a failing one back to the router, or to force_regenerate once the
outer-iteration cap is hit.
"""

from __future__ import annotations

from agents.graph.chains.stance_checker import check_stance
from agents.graph.state import GraphState


def stance_check(state: GraphState) -> GraphState:
    side = state["active_side"]
    draft = state["arg_a"] if side == "A" else state["arg_b"]
    verdict = check_stance(state["original_post"], draft)
    pro_israel = verdict["pro_israel_reply"]
    reason = verdict["reason"]

    out = {**state, "pro_israel_reply": pro_israel, "stance_reason": reason}
    if not pro_israel:
        # A failed stance opens a new outer pass; carry the reason into refine.
        out["outer_iter"] = state.get("outer_iter", 0) + 1
        out["regen_reason"] = reason or "the previous revision did not clearly argue the pro-Israel side"
    print(f"[stance_check] side={side} pro_israel_reply={pro_israel} "
          f"outer_iter={out.get('outer_iter', state.get('outer_iter', 0))} ({reason})")

    out["history"] = state.get("history", []) + [{
        "side": side, "outer_iter": out.get("outer_iter", state.get("outer_iter", 0)),
        "stage": "stance_check", "pro_israel_reply": pro_israel, "reason": reason,
    }]
    return out
