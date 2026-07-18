"""Refinement LangGraph: chains + nodes + builder.

``build_graph`` / ``run_refinement`` live in ``agents.graph.builder`` and
``GraphState`` in ``agents.graph.state``; import them from those modules
directly. Nothing is re-exported here so that importing the ``agents.graph``
package doesn't eagerly compile the builder.

Package layout
--------------

Modules:

  - ``builder.py`` — Wires the LangGraph topology and exposes
                     ``build_graph`` / ``run_refinement``.
  - ``state.py``   — ``GraphState`` (the shared TypedDict passed between nodes),
                     the iteration-cap constants for the four independent loops,
                     and small shared helpers.

Subpackages:

  - ``chains/``    — Pure, stateless LLM chains (prompt + model + parser). They
                     take primitive inputs and return primitive outputs.
  - ``nodes/``     — Graph nodes and routing edges: Nodes call a chain or 
                     retriever and own the bookkeeping (iteration counters, 
                     history, convergence flags) that the chains stay clear of.
"""
