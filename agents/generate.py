"""Single-call entrypoint to the agentic graph.

``generate_refutation`` runs the full LangGraph refinement workflow on
one CMV post and returns a flat result dict. It is the shared generation path
used by the CLI (``run.py``), the web app (``webapp/app.py``), and the harvester
(CLI + MCP server), so all callers run identical generation logic.

``refutes_trope=True`` means the draft successfully refutes the trope the post
advances; see ``agents/prompts.py`` for the criteria the stance gate applies.
"""

from __future__ import annotations
from agents.graph.builder import run_refinement


def generate_refutation(title: str, body: str) -> dict:
    """Run the full agentic graph on one post and return the result."""
    
    out = run_refinement(topic=title, original_post=body)
    return {
        "generation": out.get("generation") or out.get("argument", ""),
        "sources": out.get("sources", []),
        "grounded": bool(out.get("grounded", True)),
        "refutes_trope": bool(out.get("refutes_trope", True)),
        "stance": out.get("stance", "refutes_trope"),
        "gave_up": bool(out.get("gave_up", False)),
        "give_up_reason": out.get("give_up_reason", ""),
        "final_scores": out.get("final_scores"),
    }
