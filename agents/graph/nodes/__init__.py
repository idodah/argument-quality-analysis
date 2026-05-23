"""Graph nodes: GraphState in, partial GraphState out.

Nodes read the active side, call a chain or retriever/ranker, and return
state deltas. They own the bookkeeping (iter counters, history, convergence
flags); chains stay pure.
"""

from agents.graph.nodes.final_compare import final_compare
from agents.graph.nodes.generate_initial import generate_initial
from agents.graph.nodes.grade_docs import grade_docs
from agents.graph.nodes.hallucination_check import hallucination_check, route_after_hallucination_check
from agents.graph.nodes.rank import rank_against_prev
from agents.graph.nodes.refine import refine
from agents.graph.nodes.reflect import reflect
from agents.graph.nodes.retrieve import retrieve_local, retrieve_web, skip_retrieval
from agents.graph.nodes.router import router
from agents.graph.nodes.routing import route_after_rank, route_after_router
from agents.graph.nodes.switch_side import switch_side

__all__ = [
    "final_compare",
    "generate_initial",
    "grade_docs",
    "hallucination_check",
    "rank_against_prev",
    "refine",
    "reflect",
    "retrieve_local",
    "retrieve_web",
    "skip_retrieval",
    "router",
    "route_after_hallucination_check",
    "route_after_rank",
    "route_after_router",
    "switch_side",
]