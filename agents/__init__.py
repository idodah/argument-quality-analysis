"""Agentic argument-refinement workflow (LangGraph + LangChain).

The graph entrypoints live in ``agents.graph.builder`` (``build_graph`` /
``run_refinement``) and the state in ``agents.graph.state`` (``GraphState``);
import them from there. They are intentionally NOT re-exported here so that
importing a light submodule like ``agents.prompts`` doesn't eagerly pull in
LangGraph and the whole graph stack.

Package layout
--------------

First-level modules:

  - ``generate.py``   — Single-call entrypoint (``generate_pro_israel_response``)
                        that runs the full graph on one CMV post and returns a
                        flat result dict. Shared by the CLI, web app, and
                        harvester so every caller runs identical logic.
  - ``llm.py``        — Thin wrappers around the chat LLM (``creative_llm``,
                        ``deterministic_llm``, ``chat``). Hides the backend
                        (Bedrock Nova vs. OpenAI, via ``LLM_BACKEND``) from the
                        rest of the codebase.
  - ``ranker.py``     — Lazy singleton wrapper around the fine-tuned Qwen
                        pairwise ranker (``score_pair``). Backend is either a
                        SageMaker async endpoint or an in-process checkpoint.
  - ``retrieval.py``  — Retrieval backends for the Adaptive-RAG step: a local
                        Chroma corpus and Tavily web search, behind a common
                        ``.retrieve(query, k)`` interface.
  - ``prompts.py``    — The prompt strings / stance guide for each node of the
                        refinement graph.
  - ``tracing.py``    — Opt-in, idempotent LangSmith tracing helper (no-op when
                        the API key is absent).

Subpackages:

  - ``graph/``        — The refinement LangGraph itself: ``builder.py`` wires
                        the topology and exposes ``build_graph`` /
                        ``run_refinement``; ``state.py`` holds ``GraphState``,
                        the iteration-cap constants, and small helpers;
                        ``chains/`` holds the per-step LLM chains (initial
                        generator, router, doc/hallucination graders, reflector,
                        refiner, stance checker); ``nodes/`` holds the graph
                        nodes and routing edges that call those chains.
"""