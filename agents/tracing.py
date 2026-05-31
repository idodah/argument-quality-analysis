"""Opt-in LangSmith tracing for the agentic graph.

LangChain/LangGraph auto-trace every chain, LLM, and node run to LangSmith when
the standard env vars are set. This module is a tiny, idempotent helper that
reports whether tracing is active and lets you set a project name from one place.

To enable, put in your .env (no code change needed):
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=ls__...
    LANGSMITH_PROJECT=argument-quality   # optional

If the key is absent, this is a no-op and the graph runs untraced.
"""

from __future__ import annotations

import os

_DEFAULT_PROJECT = "argument-quality"


def _truthy(val: str | None) -> bool:
    return (val or "").strip().lower() in ("1", "true", "yes", "on")


def setup_tracing() -> bool:
    """Enable LangSmith tracing if configured. Returns True if active.

    Mirrors the legacy LANGCHAIN_* vars to the newer LANGSMITH_* names (and vice
    versa) so either spelling works, and defaults a project name. Safe to call
    more than once.
    """
    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    tracing = _truthy(os.environ.get("LANGSMITH_TRACING")) or _truthy(os.environ.get("LANGCHAIN_TRACING_V2"))

    if not api_key or not tracing:
        return False

    # Normalize both env-var spellings so LangChain picks them up regardless.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGSMITH_API_KEY"] = api_key
    project = os.environ.get("LANGSMITH_PROJECT") or os.environ.get("LANGCHAIN_PROJECT") or _DEFAULT_PROJECT
    os.environ["LANGCHAIN_PROJECT"] = project
    os.environ["LANGSMITH_PROJECT"] = project
    print(f"[tracing] LangSmith enabled (project={project!r})")
    return True
