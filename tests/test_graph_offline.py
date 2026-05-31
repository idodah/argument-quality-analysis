"""Offline graph wiring/caps test.

Drives the REAL compiled graph (agents.graph.builder.build_graph) with every
LLM / retrieval / Qwen boundary stubbed, so no API keys or network are needed.
It exercises the worst case — stance always fails (forces the outer loop to its
cap and then force_regenerate) and grounding always fails (forces the grounding
loop to its cap each pass) — and asserts the run terminates and the caps hold.

Run: `uv run pytest tests/test_graph_offline.py`
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from unittest import mock

import pytest

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


def test_happy_path_single_pass():
    """Grounded + pro-Israel on the first pass: one refine, one stance check, no force."""
    out = _run(stance_ok=True, grounded=True)
    assert out.get("winner") in ("A", "B")
    assert out.get("pro_israel_reply") is True
    assert calls["refine"] == 1
    assert calls["stance"] == 1
    assert calls["force"] == 0
    assert calls["rank"] == 1  # eliminate_loser ranks exactly once


def test_worst_case_caps_hold():
    """Grounding and stance both always fail: the loops hit their caps, then
    force_regenerate runs once and the graph still terminates."""
    out = _run(stance_ok=False, grounded=False)
    assert out.get("winner") in ("A", "B")
    # Outer loop: MAX_OUTER_ITERS passes, last one -> force_regenerate.
    assert calls["router"] == MAX_OUTER_ITERS
    assert calls["force"] == 1
    # A pass that never grounds does exactly MAX_GROUND_RETRIES refines, so the
    # worst case is MAX_OUTER_ITERS x MAX_GROUND_RETRIES refines (3 x 2 = 6).
    expected_refine = MAX_OUTER_ITERS * MAX_GROUND_RETRIES
    assert calls["refine"] == expected_refine
    assert calls["grounding"] == expected_refine  # one grounding check per refine
    assert calls["stance"] == MAX_OUTER_ITERS      # one stance check per outer pass
    assert out.get("pro_israel_reply") is True     # force_regenerate sets it True


def test_stance_fails_but_grounded():
    """Grounded immediately but stance keeps failing: one refine per outer pass
    (no grounding retries), force_regenerate once."""
    out = _run(stance_ok=False, grounded=True)
    assert out.get("winner") in ("A", "B")
    assert calls["force"] == 1
    assert calls["refine"] == MAX_OUTER_ITERS


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
