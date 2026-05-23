"""Node: Self-RAG relevance grading over retrieved chunks."""

from __future__ import annotations

from agents.graph.chains.doc_grader import grade_chunks
from agents.graph.state import GraphState, active_view


def grade_docs(state: GraphState) -> GraphState:
    _side, current, _prev, _crit = active_view(state)
    chunks = state.get("retrieved", []) or []
    if not chunks:
        return {**state, "retrieved": []}
    kept = grade_chunks(current, chunks)
    print(f"[grade_docs] kept {len(kept)}/{len(chunks)} chunks")
    return {**state, "retrieved": kept}