"""Graph nodes: GraphState in, partial GraphState out.

Nodes read the active side, call a chain or retriever, and return state deltas.
They own the bookkeeping (iter counters, history, convergence flags); chains
stay pure.
"""

from agents.graph.nodes.eliminate_loser import eliminate_loser
from agents.graph.nodes.final_compare import final_compare
from agents.graph.nodes.force_regenerate import force_regenerate
from agents.graph.nodes.generate_initial import generate_initial
from agents.graph.nodes.grade_docs import grade_docs
from agents.graph.nodes.hallucination_check import hallucination_check
from agents.graph.nodes.refine import refine
from agents.graph.nodes.reflect import reflect
from agents.graph.nodes.retrieve import retrieve_local, retrieve_web, web_search
from agents.graph.nodes.router import router
from agents.graph.nodes.routing import (
    route_after_grade,
    route_after_hallucination,
    route_after_router,
    route_after_stance,
)
from agents.graph.nodes.stance_check import stance_check

__all__ = [
    "eliminate_loser",
    "final_compare",
    "force_regenerate",
    "generate_initial",
    "grade_docs",
    "hallucination_check",
    "refine",
    "reflect",
    "retrieve_local",
    "retrieve_web",
    "web_search",
    "router",
    "stance_check",
    "route_after_grade",
    "route_after_hallucination",
    "route_after_router",
    "route_after_stance",
]
