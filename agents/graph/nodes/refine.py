"""Node: rewrite the active side's draft using critique + evidence."""

from __future__ import annotations

from agents.graph.chains.refiner import looks_like_critique, refine_draft
from agents.graph.state import GraphState, active_view, strip_empty_sources


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
    side, current, _prev, _critique = active_view(state)
    evidence = state.get("documents") or state.get("retrieved") or []
    fix_notes = _build_fix_notes(state)
    improved = refine_draft(
        state["topic"],
        state["original_post"],
        current,
        state.get("critique_history", []),
        evidence,
        fix_notes=fix_notes,
    )
    # Final guard: if the refiner produced commentary about the argument
    # rather than the argument itself (despite the prompt), discard it and keep
    # the current draft. We cannot rely on the ranker to
    # reject critique-shaped text — it has been observed to score it HIGHER
    # than a real argument — so this must be a deterministic drop.
    if looks_like_critique(improved):
        print(f"[refine] side={side} output looked like critique -> keeping current draft (no-op refine)")
        improved = current
    else:
        # Drop an empty '### Sources' header the model emitted despite citing nothing.
        improved = strip_empty_sources(improved)
    # regen_reason is consumed once per refine; clear it so it doesn't leak into
    # later passes.
    out = {**state, "generation": improved, "regen_reason": ""}
    if side == "A":
        out.update({"arg_a_prev": current, "arg_a": improved, "iter_a": state.get("iter_a", 0) + 1})
    else:
        out.update({"arg_b_prev": current, "arg_b": improved, "iter_b": state.get("iter_b", 0) + 1})
    return out