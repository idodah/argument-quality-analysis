"""Node: Self-RAG binary relevance grading over retrieved chunks.

Grades the retrieved documents yes/no:
  - relevant   -> keep them in `documents` (the citation pool) and proceed.
  - irrelevant -> DROP them and set web_search=True, so only the web results
                  fetched next feed reflect/refine. Keeping irrelevant chunks
                  would pollute the citation pool (anti-Self-RAG).
"""

from __future__ import annotations

from agents.graph.chains.doc_grader import grade_docs_relevant
from agents.graph.state import GraphState, active_view


def grade_docs(state: GraphState) -> GraphState:
    _side, current, _prev, _crit = active_view(state)
    chunks = state.get("retrieved", []) or []
    if not chunks:
        print("[grade_docs] no documents -> web_search=True")
        return {**state, "retrieved": [], "documents": [], "web_search": True}

    relevant = grade_docs_relevant(current, chunks)
    if relevant:
        print(f"[grade_docs] relevant=True -> keep {len(chunks)} docs")
        return {**state, "retrieved": chunks, "documents": chunks, "web_search": False}

    print(f"[grade_docs] relevant=False -> drop {len(chunks)} docs, web_search=True")
    return {**state, "retrieved": [], "documents": [], "web_search": True}
