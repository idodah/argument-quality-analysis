"""Structural smoke test for the package layout.

Guards the reorganization: the `rag/` pipeline package imports cleanly, and the
shared generation entrypoint lives in `agents.generate` (with `harvester.core`
re-exporting it, so the CLI/MCP still resolve it from the old name). Pure import
checks — no network, no API keys, no model loading.

Run: `uv run pytest tests/test_layout_imports.py`
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RAG_MODULES = [
    "rag",
    "rag.scrape_cmv_israel",
    "rag.classify_stance",
    "rag.ingest_rag",
    "rag.ingest_legal_sources",
]


@pytest.mark.parametrize("module", RAG_MODULES)
def test_rag_module_imports(module):
    importlib.import_module(module)  # raises -> test fails


def test_generation_entrypoint_exists():
    from agents.generate import generate_pro_israel_response
    assert callable(generate_pro_israel_response)


def test_harvester_reexports_same_entrypoint():
    """harvester is gitignored (Reddit-only); when present it must re-export the
    entrypoint from agents.generate rather than redefining it."""
    from agents.generate import generate_pro_israel_response as canonical
    try:
        import harvester.core as core
    except Exception:  # noqa: BLE001 - absent harvester is fine
        pytest.skip("harvester not present (gitignored)")
    assert getattr(core, "generate_pro_israel_response", None) is canonical


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
