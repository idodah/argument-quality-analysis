"""Node: declare the surviving side the winner and report the final state.

Terminal/publishing node. No model call, no comparison — the A-vs-B decision
happened in `eliminate_loser` and all gates (grounding, stance) ran upstream.
This node packages the result for the caller (split citations, roll up
final_scores, print warnings + trajectory) and routes to END.
"""

from __future__ import annotations

from agents.graph.state import GraphState, split_for_display


def finalize(state: GraphState) -> GraphState:
    """Publish the surviving side's argument and surface upstream warnings.

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
    stance = state.get("stance", "pro_israel")
    gave_up = state.get("gave_up", False)
    give_up_reason = state.get("give_up_reason", "")

    print(f"[final] winner={side} iters={iters} grounded={grounded} stance={stance} sources={len(sources)}")
    if gave_up:
        print(f"[final] GAVE UP: {give_up_reason}")
    if not grounded:
        print(f"[final] WARNING: winning argument is NOT grounded — {len(issues)} unresolved issue(s):")
        for i, it in enumerate(issues, 1):
            print(f"  [issue {i}] {it}")
    if not pro_israel and not gave_up:
        print(f"[final] WARNING: winning argument is NOT a pro-Israel reply — {state.get('stance_reason', '')}")

    _print_score_trajectory(state.get("history", []))
    return {
        **state,
        "winner": side,
        "generation": clean_arg,
        "generation_raw": winning_arg,
        "sources": sources,
        "pro_israel_reply": pro_israel,
        "stance": stance,
        "gave_up": gave_up,
        "give_up_reason": give_up_reason,
        "final_scores": {
            "winner": side,
            "iters": iters,
            "winning_argument_length": len(clean_arg),
            "grounded": grounded,
            "hallucination_issues": issues,
            "pro_israel_reply": pro_israel,
            "stance": stance,
            "stance_reason": state.get("stance_reason", ""),
            "gave_up": gave_up,
            "give_up_reason": give_up_reason,
            "regen_iter": state.get("regen_iter", 0),
            "refine_iter": state.get("refine_iter", 0),
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
            print(f"  [grounding refine_pass={h.get('refine_iter')} retries={h.get('ground_retries')}] "
                  f"grounded={h.get('grounded')}")
        elif stage == "stance_check":
            print(f"  [stance refine_pass={h.get('refine_iter')} regen={h.get('regen_iter')}] "
                  f"stance={h.get('stance')}")
        elif stage == "generate_initial":
            print(f"  [regen pass={h.get('regen_iter')}] reason={h.get('regen_reason')}")