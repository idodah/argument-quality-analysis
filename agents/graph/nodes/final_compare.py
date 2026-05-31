"""Node: declare the surviving side the winner and report the final state."""

from __future__ import annotations

from agents.graph.state import GraphState, split_for_display


def final_compare(state: GraphState) -> GraphState:
    """Declare the surviving side the winner.

    The A-vs-B comparison happens once, on the initial drafts, in
    `eliminate_loser`. By the time we reach this node only one side has been
    iterating; its final arg_{side} is the answer. Grounding and pro-Israel
    stance were already checked by their own nodes upstream — here we only
    surface warnings and publish the result.

    Output split: `generation` is the CLEAN argument (inline [n] markers and the
    '### Sources' footer removed) for display; `sources` is the list of URLs the
    model relied on, surfaced separately for human-in-the-loop verification. The
    raw cited draft is kept in `generation_raw` for debugging/traceability.
    """
    side = state.get("active_side", "A")
    winning_arg = state["arg_a"] if side == "A" else state["arg_b"]
    clean_arg, sources = split_for_display(winning_arg)
    iters = state.get(f"iter_{side.lower()}", 0)
    grounded = state.get("grounded", True)
    issues = state.get("hallucination_issues", [])
    pro_israel = state.get("pro_israel_reply", True)
    print(f"[final] winner={side} iters={iters} grounded={grounded} pro_israel={pro_israel} sources={len(sources)}")
    if not grounded:
        print(f"[final] WARNING: winning argument is NOT grounded — {len(issues)} unresolved issue(s):")
        for i, it in enumerate(issues, 1):
            print(f"  [issue {i}] {it}")
    if not pro_israel:
        print(f"[final] WARNING: winning argument is NOT a pro-Israel reply — {state.get('stance_reason', '')}")

    _print_score_trajectory(state.get("history", []))
    return {
        **state,
        "winner": side,
        "generation": clean_arg,
        "generation_raw": winning_arg,
        "sources": sources,
        "pro_israel_reply": pro_israel,
        "final_scores": {
            "winner": side,
            "iters": iters,
            "winning_argument_length": len(clean_arg),
            "grounded": grounded,
            "hallucination_issues": issues,
            "pro_israel_reply": pro_israel,
            "stance_reason": state.get("stance_reason", ""),
            "num_sources": len(sources),
        },
    }


def _print_score_trajectory(history: list[dict]) -> None:
    """Print the initial A-vs-B compare and each iteration's check verdicts."""
    print("--- trajectory ---")
    for h in history:
        stage = h.get("stage")
        if stage == "eliminate_loser":
            print(
                f"  [init A-vs-B] score_a={h['score_a']:.3f} score_b={h['score_b']:.3f} "
                f"-> survivor={h['survivor']}"
            )
        elif stage == "hallucination_check":
            print(f"  [grounding pass={h.get('outer_iter')} retries={h.get('ground_retries')}] "
                  f"grounded={h.get('grounded')}")
        elif stage == "stance_check":
            print(f"  [stance pass={h.get('outer_iter')}] pro_israel={h.get('pro_israel_reply')}")