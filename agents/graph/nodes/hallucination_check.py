"""Node: Self-RAG hallucination check on the just-refined draft.

Placed between `refine` and `rank_against_prev`. On ungrounded verdict,
folds the grader's issues back into the active side's critique and loops
back to `refine` for a same-iteration retry. After MAX_REFINE_RETRIES
unsuccessful retries, reverts to the previous draft and marks the side
converged — treating persistent hallucination the same as losing on the
ranker.
"""

from __future__ import annotations

from agents.graph.chains.hallucination_grader import check_grounding
from agents.graph.state import GraphState, MAX_REFINE_RETRIES


def _retry_count(state: GraphState, side: str) -> int:
    key = f"refine_retries_{side.lower()}"
    return state.get(key, 0) or 0


def _reset_retry_state(state: dict, side: str) -> dict:
    state[f"refine_retries_{side.lower()}"] = 0
    return state


def hallucination_check(state: GraphState) -> GraphState:
    side = state["active_side"]
    if side == "A":
        draft = state["arg_a"]
        prev = state.get("arg_a_prev", "")
        critique_key = "critique_a"
        iter_key = "iter_a"
    else:
        draft = state["arg_b"]
        prev = state.get("arg_b_prev", "")
        critique_key = "critique_b"
        iter_key = "iter_b"

    verdict = check_grounding(draft, state.get("retrieved", []) or [])
    grounded = verdict["grounded"]
    issues = verdict.get("issues", [])
    print(f"[hallucination_check] side={side} grounded={grounded} issues={len(issues)}")

    history_entry = {
        "side": side,
        "iter": state.get(iter_key, 0),
        "stage": "hallucination_check",
        "grounded": grounded,
        "issues": issues,
    }
    new_history = state.get("history", []) + [history_entry]

    if grounded:
        updated = {**state, "grounded": True, "hallucination_issues": [], "history": new_history}
        return _reset_retry_state(updated, side)

    retries_used = _retry_count(state, side)
    if retries_used < MAX_REFINE_RETRIES:
        # Same-iter retry: fold issues into the critique, decrement iter so the
        # retry doesn't burn the side's MAX_ITERS budget, and route back to refine.
        existing_critique = state.get(critique_key, "")
        addendum = "**Hallucination grader flagged these ungrounded claims — REMOVE or correct them:**\n" + "\n".join(
            f"  - {it}" for it in issues
        )
        merged_critique = (existing_critique + "\n\n" + addendum).strip() if existing_critique else addendum
        cur_iter = state.get(iter_key, 0)
        updated: dict = {
            **state,
            "grounded": False,
            "hallucination_issues": issues,
            "critique": merged_critique,
            critique_key: merged_critique,
            f"refine_retries_{side.lower()}": retries_used + 1,
            iter_key: max(cur_iter - 1, 0),
            "history": new_history,
        }
        return updated

    # Retries exhausted: revert to prev draft and mark side converged.
    print(f"[hallucination_check] side={side} retries exhausted -> revert + converge")
    updated = {**state, "grounded": False, "hallucination_issues": issues, "history": new_history}
    if side == "A":
        updated["arg_a"] = prev
        updated["converged_a"] = True
    else:
        updated["arg_b"] = prev
        updated["converged_b"] = True
    return _reset_retry_state(updated, side)


def route_after_hallucination_check(state: GraphState) -> str:
    """Decide between same-iter refine retry and continuing to the ranker."""
    if state.get("grounded", True):
        return "rank_against_prev"
    side = state["active_side"]
    converged = state.get(f"converged_{side.lower()}", False)
    if converged:
        return "rank_against_prev"
    return "refine"
