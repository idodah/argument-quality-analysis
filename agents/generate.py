"""Single-call entrypoint to the agentic graph.

``generate_pro_israel_response`` runs the full LangGraph refinement workflow on
one CMV post and returns a flat result dict. It is the shared generation path
used by the CLI (``run.py``), the web app (``webapp/app.py``), and the harvester
(CLI + MCP server), so all callers run identical generation logic.
"""

from __future__ import annotations


def generate_pro_israel_response(title: str, body: str) -> dict:
    """Run the full agentic graph on one post and return the result.

    `agents.graph.builder` is imported lazily so importing this module does not
    pull in the graph / Qwen ranker until a generation is actually requested.
    """
    from agents.graph.builder import run_refinement

    out = run_refinement(topic=title, original_post=body)
    return {
        "generation": out.get("generation") or out.get("argument", ""),
        "sources": out.get("sources", []),
        "grounded": bool(out.get("grounded", True)),
        "pro_israel_reply": bool(out.get("pro_israel_reply", True)),
        "stance": out.get("stance", "pro_israel"),
        "gave_up": bool(out.get("gave_up", False)),
        "give_up_reason": out.get("give_up_reason", ""),
        "final_scores": out.get("final_scores"),
    }
