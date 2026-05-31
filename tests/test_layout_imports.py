"""Structural smoke test for the package layout.

Guards the reorganization: the `rag/` pipeline package imports cleanly, and the
shared generation entrypoint lives in `agents.generate` (with `harvester.core`
re-exporting it, so the CLI/MCP still resolve it from the old name). Pure import
checks — no network, no API keys, no model loading.

Run: `uv run python tests/test_layout_imports.py`
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RAG_MODULES = [
    "rag",
    "rag.scrape_cmv_israel",
    "rag.classify_stance",
    "rag.ingest_rag",
    "rag.ingest_legal_sources",
]


def _check(name: str, cond: bool, detail: str = "") -> bool:
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    return cond


def main() -> int:
    ok = True

    print("rag/ pipeline package imports")
    for m in RAG_MODULES:
        try:
            importlib.import_module(m)
            ok &= _check(f"import {m}", True)
        except Exception as e:  # noqa: BLE001 - report any import failure
            ok &= _check(f"import {m}", False, f"{type(e).__name__}: {e}")

    print("\nshared generation entrypoint")
    from agents.generate import generate_pro_israel_response as gen_canonical

    ok &= _check("agents.generate.generate_pro_israel_response exists", callable(gen_canonical))

    # harvester is gitignored (Reddit-only), but when present it must re-export
    # the entrypoint from agents.generate rather than redefining it.
    try:
        import harvester.core as core
    except Exception:  # noqa: BLE001 - absent harvester is fine
        print("  [SKIP] harvester not present (gitignored) — re-export check skipped")
    else:
        ok &= _check(
            "harvester.core re-exports the SAME generate_pro_israel_response",
            getattr(core, "generate_pro_israel_response", None) is gen_canonical,
        )

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
