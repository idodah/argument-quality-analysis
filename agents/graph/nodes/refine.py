"""Node: rewrite the current draft using critique + evidence."""

from __future__ import annotations

from agents.graph.chains.refiner import looks_like_critique, refine_draft
from agents.graph.state import GraphState, current_argument, strip_empty_sources


def _build_fix_notes(state: GraphState) -> str:
    """Concrete problems with the PREVIOUS revision that this refine must fix:
    ungrounded claims from hallucination_check and/or a stance reroute reason."""
    notes: list[str] = []
    issues = state.get("hallucination_issues") or []
    if not state.get("grounded", True) and issues:
        notes.append(
            "Ungrounded claims to fix (remove, soften, or cite from the evidence):\n"
            + "\n".join(f"  - {i}" for i in issues)
        )
    regen = state.get("regen_reason") or ""
    if regen:
        notes.append(f"Stance problem to fix: {regen}")
    return "\n\n".join(notes)


def refine(state: GraphState) -> GraphState:
    current = current_argument(state)
    evidence = state.get("documents") or []
    fix_notes = _build_fix_notes(state)
    improved = refine_draft(
        state["topic"],
        state["original_post"],
        current,
        state.get("critique_history", []),
        evidence,
        fix_notes=fix_notes,
    )
    # Final guard: if the refiner wrote commentary ABOUT the argument instead of
    # the argument itself, drop it and keep the current draft. This must be a
    # deterministic check — the ranker has been seen to score critique-shaped
    # text HIGHER than a real argument, so we can't rely on it to reject it.
    noop_streak = state.get("consecutive_noop_refines", 0)
    noop_pass = state.get("noop_streak_pass", -1)
    pass_n = state.get("refine_iter", 0)
    if looks_like_critique(improved):
        # Count no-op PASSES (not calls) so stance_check can spot a stuck refiner
        # and regenerate. The grounding loop can call refine several times in one
        # pass; `noop_pass` guards against counting the same pass twice.
        if pass_n != noop_pass:
            noop_streak += 1
            noop_pass = pass_n
        print(f"[refine] output looked like critique -> keeping current draft "
              f"(no-op refine; streak={noop_streak})")
        improved = current
    else:
        # Drop an empty '### Sources' header the model emitted despite citing nothing.
        improved = strip_empty_sources(improved)
        noop_streak = 0  # productive output resets the streak
        noop_pass = -1

    # regen_reason is consumed once per refine; clear it so it doesn't leak into
    # later passes.
    return {
        "argument": improved,
        "generation": improved,
        "iter": state.get("iter", 0) + 1,
        "consecutive_noop_refines": noop_streak,
        "noop_streak_pass": noop_pass,
        "regen_reason": "",
    }