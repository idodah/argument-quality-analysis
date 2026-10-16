"""Node: classify the current argument's stance (pro / neutral / off-topic-or-anti).

Routes:
  - refutes_trope           -> finalize (done)
  - neutral_needs_refine -> router (refinement loop, budget MAX_REFINE_ITERS)
  - off_topic_or_anti    -> generate_initial (late regen loop, budget MAX_LATE_REGEN_ITERS)

When both budgets are exhausted, sets `gave_up=True` with a give_up_reason and
routes to finalize, which surfaces that honestly rather than shipping a
non-refutation as if it were one.

Counters owned here:
  - refine_iter: incremented on neutral_needs_refine, reset by regeneration.
  - late_regen_iter: incremented on off_topic_or_anti (never resets).
"""

from __future__ import annotations

from agents.graph.chains.stance_checker import check_stance
from agents.graph.state import (
    MAX_LATE_REGEN_ITERS,
    MAX_REFINE_ITERS,
    GraphState,
    current_argument,
)

# Consecutive critique-shaped refine outputs that promote
# `neutral_needs_refine` straight to a regeneration. Two no-ops in a row means
# the refiner is stuck and another refinement pass is unlikely to help.
_NOOP_STREAK_PROMOTION_THRESHOLD = 2


def stance_check(state: GraphState) -> GraphState:
    draft = current_argument(state)
    verdict = check_stance(state["original_post"], draft)
    stance = verdict["stance"]
    reason = verdict["reason"]

    refine_iter = state.get("refine_iter", 0)
    late_regen_iter = state.get("late_regen_iter", 0)
    noop_streak = state.get("consecutive_noop_refines", 0)

    out: dict = {
        "stance": stance,
        "refutes_trope": stance == "refutes_trope",
        "stance_reason": reason,
    }

    if stance == "refutes_trope":
        print(f"[stance_check] stance=refutes_trope "
              f"(refine={refine_iter}/{MAX_REFINE_ITERS}, late_regen={late_regen_iter}/{MAX_LATE_REGEN_ITERS}) — done")
    elif stance == "neutral_needs_refine":
        new_refine_iter = refine_iter + 1
        out["refine_iter"] = new_refine_iter
        out["regen_reason"] = reason or "the previous revision was on-topic but did not land a substantive refutation of the trope"
        if new_refine_iter >= MAX_REFINE_ITERS:
            # Refinement budget for this generation is spent. Promote to a
            # regeneration if we still have LATE regen budget; otherwise give up.
            print(f"[stance_check] stance=neutral but refine budget exhausted ({new_refine_iter}/{MAX_REFINE_ITERS}) -> escalate")
            stance = "off_topic_or_anti"
            out["stance"] = stance
        elif noop_streak >= _NOOP_STREAK_PROMOTION_THRESHOLD:
            # The refiner has been emitting critique-shaped no-ops for several
            # passes — the model is stuck and a fresh generation is more likely
            # to break out than another refinement pass.
            print(f"[stance_check] stance=neutral and no-op streak={noop_streak} >= {_NOOP_STREAK_PROMOTION_THRESHOLD} -> escalate to regeneration")
            stance = "off_topic_or_anti"
            out["stance"] = stance
        else:
            print(f"[stance_check] stance=neutral_needs_refine ({reason}) -> router (refine {new_refine_iter}/{MAX_REFINE_ITERS})")

    if stance == "off_topic_or_anti":
        new_late_regen_iter = late_regen_iter + 1
        out["late_regen_iter"] = new_late_regen_iter
        # Reset every per-generation counter — including the early-regen budget —
        # so the next generation starts fresh. Handing the early gate its budget
        # back is what makes the two regen loops compose multiplicatively (see
        # the cap block in agents.graph.state).
        out["refine_iter"] = 0
        out["ground_retries"] = 0
        out["consecutive_noop_refines"] = 0
        out["noop_streak_pass"] = -1
        out["early_regen_iter"] = 0
        out["regen_reason"] = reason or "the previous draft was off-topic, attacked the poster, or drifted into political advocacy"
        if new_late_regen_iter > MAX_LATE_REGEN_ITERS:
            # Late-regen budget exhausted. Give up honestly rather than ship a
            # non-refutation as the answer. (The early gate has its
            # own separate budget; nothing here looks at it.)
            out["gave_up"] = True
            # Use the CURRENT (post-escalation) stance in the give-up reason.
            # The check on `state["stance"]` would read a stale verdict from
            # before this node ran.
            passes = f"{MAX_REFINE_ITERS} refinement pass{'es' if MAX_REFINE_ITERS != 1 else ''}"
            attempts = f"{MAX_LATE_REGEN_ITERS} late regeneration attempt{'s' if MAX_LATE_REGEN_ITERS != 1 else ''}"
            out["give_up_reason"] = (
                f"Pipeline could not produce a substantive refutation after "
                f"{passes} per generation and {attempts}. "
                f"Final stance verdict: {stance} ({reason or 'no reason given'})."
            )
            print(f"[stance_check] late-regen budget exhausted "
                  f"({new_late_regen_iter}/{MAX_LATE_REGEN_ITERS}) -> give up")
        else:
            print(f"[stance_check] stance=off_topic_or_anti ({reason}) -> generate_initial (late_regen {new_late_regen_iter}/{MAX_LATE_REGEN_ITERS})")

    out["history"] = state.get("history", []) + [{
        "stage": "stance_check", "stance": stance, "reason": reason,
        "refine_iter": out.get("refine_iter", refine_iter),
        "late_regen_iter": out.get("late_regen_iter", late_regen_iter),
        "gave_up": out.get("gave_up", False),
    }]
    return out
