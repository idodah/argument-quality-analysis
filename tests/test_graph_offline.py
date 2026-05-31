"""Offline graph wiring/caps test.

Drives the REAL compiled graph (agents.graph.builder.build_graph) with every
LLM / retrieval / Qwen boundary stubbed, so no API keys or network are needed.
It exercises the worst case — stance always fails (forces the outer loop to its
cap and then force_regenerate) and grounding always fails (forces the grounding
loop to its cap each pass) — and asserts the run terminates and the caps hold.

Run: `uv run python tests/test_graph_offline.py`
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.graph.builder import build_graph
from agents.graph.state import MAX_GROUND_RETRIES, MAX_OUTER_ITERS


# Counts every node/chain invocation so we can assert on the caps afterwards.
calls: Counter[str] = Counter()


def _make_scenario(*, stance_ok: bool, grounded: bool):
    """Build a set of patches for one scenario.

    stance_ok=False, grounded=False is the worst case (max iterations).
    """

    def fake_generate_pair(topic, post):
        calls["generate"] += 1
        return ("ARG A initial draft", "ARG B initial draft")

    class FakeRanker:
        def score_pair(self, topic, post, arg_a, arg_b):
            calls["rank"] += 1
            # Deterministically pick A so the active side is stable.
            return {"winner": "A", "score_a": 0.9, "score_b": 0.1}

    fake_ranker = FakeRanker()

    def fake_get_ranker():
        return fake_ranker

    def fake_route_retrieval(post, draft, critique=""):
        calls["router"] += 1
        return "web", ["q1", "q2", "q3"]

    def fake_grade_relevant(draft, chunks):
        calls["grade"] += 1
        return True  # docs relevant -> no web_search detour

    def fake_reflect(post, draft, evidence):
        calls["reflect"] += 1
        return "**Missing**\n- add X\n**Superfluous**\n- cut Y"

    def fake_refine(topic, post, draft, critique_history, evidence, fix_notes=""):
        calls["refine"] += 1
        # Echo back a real-argument-shaped string (not critique-shaped, so the
        # looks_like_critique guard doesn't no-op it).
        return f"Refined argument addressing you directly (rev {calls['refine']})."

    def fake_check_grounding(draft, evidence):
        calls["grounding"] += 1
        if grounded:
            return {"grounded": True, "issues": []}
        return {"grounded": False, "issues": ["the 34,000 figure is unsupported"]}

    def fake_check_stance(post, draft):
        calls["stance"] += 1
        return {"pro_israel_reply": stance_ok, "reason": "" if stance_ok else "reads neutral"}

    def fake_force(post, draft):
        calls["force"] += 1
        return "Forced clearly pro-Israel rewrite addressed to you."

    def fake_force_stance(post, draft):
        # force_regenerate re-checks stance on its rewrite; the forced rewrite is
        # designed to pass, so report pro-Israel here.
        calls["force_stance"] += 1
        return {"pro_israel_reply": True, "reason": "forced pro-Israel rewrite"}

    # Web retriever: return some chunks so the pool isn't empty.
    def fake_web_retrieve(self, query, k=None):
        calls["web_retrieve"] += 1
        return [f"[url] http://example.com/{query}\nweb content for {query}"]

    def fake_local_retrieve(self, query, k=None, where=None):
        calls["local_retrieve"] += 1
        return [f"[argument] local chunk for {query}"]

    return [
        mock.patch("agents.graph.nodes.generate_initial.generate_initial_pair", fake_generate_pair),
        mock.patch("agents.graph.nodes.eliminate_loser.get_ranker", fake_get_ranker),
        mock.patch("agents.graph.nodes.router.route_retrieval", fake_route_retrieval),
        mock.patch("agents.graph.nodes.grade_docs.grade_docs_relevant", fake_grade_relevant),
        mock.patch("agents.graph.nodes.reflect.reflect_on_draft", fake_reflect),
        mock.patch("agents.graph.nodes.refine.refine_draft", fake_refine),
        mock.patch("agents.graph.nodes.hallucination_check.check_grounding", fake_check_grounding),
        mock.patch("agents.graph.nodes.stance_check.check_stance", fake_check_stance),
        mock.patch("agents.graph.nodes.force_regenerate.force_pro_israel", fake_force),
        mock.patch("agents.graph.nodes.force_regenerate.check_stance", fake_force_stance),
        mock.patch("agents.retrieval.WebRetriever.retrieve", fake_web_retrieve),
        mock.patch("agents.retrieval.LocalRetriever.retrieve", fake_local_retrieve),
        # Avoid constructing the real retrievers (which need API keys / chroma).
        mock.patch("agents.retrieval.WebRetriever.__init__", lambda self, *a, **k: None),
        mock.patch("agents.retrieval.LocalRetriever.__init__", lambda self, *a, **k: None),
        mock.patch("agents.graph.nodes.retrieve._LOCAL", None),
        mock.patch("agents.graph.nodes.retrieve._WEB", None),
    ]


def _run(*, stance_ok: bool, grounded: bool) -> dict:
    calls.clear()
    patches = _make_scenario(stance_ok=stance_ok, grounded=grounded)
    for p in patches:
        p.start()
    try:
        graph = build_graph()
        return graph.invoke(
            {"topic": "CMV: test", "original_post": "an anti-Israel post"},
            config={"recursion_limit": 100},
        )
    finally:
        for p in reversed(patches):
            p.stop()


def _check(name: str, cond: bool, detail: str = "") -> bool:
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    return cond


def main() -> int:
    ok = True

    print("Scenario 1: happy path (grounded + pro-Israel on first pass)")
    out = _run(stance_ok=True, grounded=True)
    ok &= _check("terminates with a winner", out.get("winner") in ("A", "B"))
    ok &= _check("reports pro_israel_reply=True", out.get("pro_israel_reply") is True)
    ok &= _check("exactly 1 refine", calls["refine"] == 1, f"refine={calls['refine']}")
    ok &= _check("exactly 1 stance check", calls["stance"] == 1, f"stance={calls['stance']}")
    ok &= _check("force_regenerate NOT used", calls["force"] == 0, f"force={calls['force']}")
    ok &= _check("eliminate_loser ranked once", calls["rank"] == 1, f"rank={calls['rank']}")
    print(f"  calls: {dict(calls)}")

    print("\nScenario 2: worst case (grounding always fails, stance always fails)")
    out = _run(stance_ok=False, grounded=False)
    ok &= _check("terminates with a winner", out.get("winner") in ("A", "B"))
    # Outer loop: MAX_OUTER_ITERS passes, last one -> force_regenerate.
    ok &= _check(
        "router called MAX_OUTER_ITERS times",
        calls["router"] == MAX_OUTER_ITERS,
        f"router={calls['router']} expected={MAX_OUTER_ITERS}",
    )
    ok &= _check("force_regenerate used exactly once", calls["force"] == 1, f"force={calls['force']}")
    # Each refine is followed by a grounding check; on failure it spends one
    # retry and refines again, so a pass that never grounds does exactly
    # MAX_GROUND_RETRIES refines. Worst case = MAX_OUTER_ITERS x MAX_GROUND_RETRIES.
    expected_refine = MAX_OUTER_ITERS * MAX_GROUND_RETRIES
    ok &= _check(
        "refine count = outer x ground_retries (3 x 2 = 6 worst case)",
        calls["refine"] == expected_refine,
        f"refine={calls['refine']} expected={expected_refine}",
    )
    # Grounding checks: one per refine.
    ok &= _check(
        "grounding checks = refine count",
        calls["grounding"] == expected_refine,
        f"grounding={calls['grounding']} refine={calls['refine']}",
    )
    # Stance checked once per outer pass.
    ok &= _check(
        "stance checks = MAX_OUTER_ITERS",
        calls["stance"] == MAX_OUTER_ITERS,
        f"stance={calls['stance']} expected={MAX_OUTER_ITERS}",
    )
    ok &= _check("final output flagged not pro-Israel? (force sets True)", out.get("pro_israel_reply") is True)
    print(f"  calls: {dict(calls)}")

    print("\nScenario 3: grounded but stance fails (no grounding retries)")
    out = _run(stance_ok=False, grounded=True)
    ok &= _check("terminates", out.get("winner") in ("A", "B"))
    ok &= _check("force_regenerate used once", calls["force"] == 1, f"force={calls['force']}")
    # 1 refine per outer pass (grounding passes immediately).
    ok &= _check(
        "refine count = MAX_OUTER_ITERS",
        calls["refine"] == MAX_OUTER_ITERS,
        f"refine={calls['refine']} expected={MAX_OUTER_ITERS}",
    )
    print(f"  calls: {dict(calls)}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
