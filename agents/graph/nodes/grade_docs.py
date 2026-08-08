"""Node: CRAG (Corrective RAG) binary relevance grading over retrieved chunks.

Grades the retrieved documents yes/no:
  - relevant   -> keep them in `documents` (the citation pool) and proceed.
  - irrelevant -> DROP them and set web_search=True, so only the web results
                  fetched next feed reflect/refine. Keeping irrelevant chunks
                  would pollute the citation pool (anti-CRAG).

The drop-and-web-search path is the *corrective* action CRAG is named for:
rather than generating on evidence we've just judged irrelevant, the run
retrieves again from a different source before reflecting.
"""

from __future__ import annotations

from agents.graph.chains.doc_grader import grade_docs_relevant
from agents.graph.state import GraphState, current_argument


def grade_docs(state: GraphState) -> GraphState:
    current = current_argument(state)
    chunks = state.get("documents", []) or []
    if not chunks:
        print("[grade_docs] no documents -> web_search=True")
        return {"documents": [], "web_search": True}

    # Per-chunk relevance (canonical CRAG): keep the chunks that pass and
    # drop only the rest, instead of collapsing the whole pool to one verdict.
    kept = grade_docs_relevant(current, chunks)
    if kept:
        print(f"[grade_docs] kept {len(kept)}/{len(chunks)} relevant docs")
        return {"documents": kept, "web_search": False}

    # No chunk survived: drop everything and fall back to a web search.
    print(f"[grade_docs] 0/{len(chunks)} relevant -> drop all, web_search=True")
    return {"documents": [], "web_search": True}
