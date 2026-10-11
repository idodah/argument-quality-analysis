"""Node: package the final result and route to END.

Terminal node — no model call. Every gate (A-vs-B, grounding, stance) has
already run upstream, so this just shapes `state["argument"]` for the caller:
splits it into a clean `generation` (inline [n] markers and the '### Sources'
footer stripped) plus the `sources` URL list surfaced for human verification,
rolls up `final_scores`, and prints any warnings and the run trajectory.
"""

from __future__ import annotations

from agents.graph.state import GraphState, split_for_display


def finalize(state: GraphState) -> GraphState:
    winner = state.get("winner", "A")
    final_argument = state.get("argument", "") or ""
    clean_arg, sources = split_for_display(final_argument)
    iters = state.get("iter", 0)
    grounded = state.get("grounded", True)
    # Whether the final pass's grounding was verified by the grader or merely
    # assumed (local/none retrieval had no factual evidence to check against).
    grounding_verified = state.get("grounding_verified", False)
    issues = state.get("hallucination_issues", [])
    pro_israel = state.get("pro_israel_reply", True)
    stance = state.get("stance", "pro_israel")
    gave_up = state.get("gave_up", False)
    give_up_reason = state.get("give_up_reason", "")

    print(f"[finalize] winner={winner} iters={iters} grounded={grounded} "
          f"(verified={grounding_verified}) stance={stance} sources={len(sources)}")
    if gave_up:
        print(f"[finalize] GAVE UP: {give_up_reason}")
    if not grounded:
        print(f"[finalize] WARNING: winning argument is NOT grounded — {len(issues)} unresolved issue(s):")
        for i, it in enumerate(issues, 1):
            print(f"  [issue {i}] {it}")
    if not pro_israel and not gave_up:
        print(f"[finalize] WARNING: winning argument is NOT a pro-Israel reply — {state.get('stance_reason', '')}")

    _print_score_trajectory(state.get("history", []))
    return {
        "winner": winner,
        "generation": clean_arg,
        "generation_raw": final_argument,
        "sources": sources,
        "pro_israel_reply": pro_israel,
        "stance": stance,
        "gave_up": gave_up,
        "give_up_reason": give_up_reason,
        "final_scores": {
            "winner": winner,
            "iters": iters,
            "winning_argument_length": len(clean_arg),
            "grounded": grounded,
            # "grounded" above can mean verified OR assumed; this disambiguates.
            "grounding_verified": grounding_verified,
            "hallucination_issues": issues,
            "pro_israel_reply": pro_israel,
            "stance": stance,
            "stance_reason": state.get("stance_reason", ""),
            "gave_up": gave_up,
            "give_up_reason": give_up_reason,
            "early_regen_iter": state.get("early_regen_iter", 0),
            "late_regen_iter": state.get("late_regen_iter", 0),
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
            print(f"  [stance refine_pass={h.get('refine_iter')} late_regen={h.get('late_regen_iter')}] "
                  f"stance={h.get('stance')}")
        elif stage == "early_stance_check":
            print(f"  [early_stance early_regen={h.get('early_regen_iter', 0)}] "
                  f"action={h.get('action')} stance={h.get('stance')}")
        elif stage == "generate_initial":
            print(f"  [regen early={h.get('early_regen_iter', 0)} late={h.get('late_regen_iter', 0)}] "
                  f"reason={h.get('regen_reason')}")