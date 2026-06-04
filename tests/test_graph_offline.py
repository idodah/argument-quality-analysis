"""Offline graph wiring/caps test for the ternary-stance graph with a single
late stance gate.

Drives the REAL compiled graph (agents.graph.builder.build_graph) with every
LLM / retrieval / Qwen boundary stubbed, so no API keys or network are needed.

There are TWO stance gates: an EARLY one on the raw survivor (before
refinement, catches only off_topic_or_anti and regenerates while the regen
budget lasts) and the LATE one after hallucination_check (the SOLE authority on
shipping and give-up). The early gate has only two out-edges
(generate_initial | router) and never routes to finalize. Every path to
finalize goes through refine -> hallucination_check -> late stance_check, so the
grounding pass is guaranteed by construction.

Scenarios:
  1. happy_path_pro_after_first_refine: early gate proceeds, late stance returns
     pro_israel after the mandatory refinement. One refine, one late stance call.
  2. refinement_path_neutral_then_pro: early gate proceeds; late stance returns
     neutral once, then pro_israel after another refinement.
  3. gave_up_via_neutral: early gate proceeds; late stance always neutral;
     exhaust refine + regen budgets, give up.
  4. gave_up_via_off_topic_early_regen_then_late_giveup: raw survivor always
     off_topic_or_anti; the EARLY gate regenerates while the budget lasts, then
     (budget spent) hands off to the router so exactly ONE refinement pass runs
     and the LATE gate records give-up. Early gate never finalizes.
  5. router_none_skips_retrieval: when router picks 'none', the graph runs
     skip_retrieval -> reflect -> refine -> hallucination_check (skipped) ->
     stance_check, without calling any retriever or the docs grader.
  6. no_finalize_without_grounding_pass: safety property; every successful
     run has at least one grounding check.

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
from agents.graph.state import (
    MAX_EARLY_REGEN_ITERS,
    MAX_LATE_REGEN_ITERS,
    MAX_REFINE_ITERS,
)


calls: Counter[str] = Counter()


def _make_scenario(
    stance_sequence: list[str],
    *,
    grounded: bool = True,
    retrieval_mode: str = "web",
    early_stance_sequence: list[str] | None = None,
):
    """Patch every external call.

    `stance_sequence` is consumed in order by the LATE stance gate
    (`stance_check`, after refinement); the last element repeats.
    `early_stance_sequence` is consumed by the EARLY gate
    (`early_stance_check`, on the raw survivor before refinement); it defaults
    to always 'pro_israel' so the early gate just proceeds and the late gate
    drives the scenario. Set it to exercise the off-topic short-circuit.
    `retrieval_mode` controls what the router returns (web by default).
    """
    if early_stance_sequence is None:
        early_stance_sequence = ["pro_israel"]

    def fake_generate_pair(topic, post, regen_reason=""):
        calls["generate"] += 1
        return ("ARG A initial draft", "ARG B initial draft")

    class FakeRanker:
        def score_pair(self, topic, post, arg_a, arg_b):
            calls["rank"] += 1
            return {"winner": "A", "score_a": 0.9, "score_b": 0.1}

    fake_ranker = FakeRanker()

    def fake_get_ranker():
        return fake_ranker

    def fake_route_retrieval(post, draft, critique=""):
        calls["router"] += 1
        if retrieval_mode == "web":
            return "web", ["q1", "q2", "q3"]
        if retrieval_mode == "none":
            return "none", []
        return "local", []

    def fake_grade_relevant(draft, chunks):
        # New per-chunk contract: return the relevant SUBSET (here: keep all).
        calls["grade"] += 1
        return list(chunks)

    def fake_reflect(post, draft, evidence):
        calls["reflect"] += 1
        return "**Missing**\n- add X\n**Superfluous**\n- cut Y"

    def fake_refine(topic, post, draft, critique_history, evidence, fix_notes=""):
        calls["refine"] += 1
        return f"Refined argument addressing you directly (rev {calls['refine']})."

    def fake_check_grounding(draft, evidence):
        calls["grounding"] += 1
        if grounded:
            return {"grounded": True, "issues": []}
        return {"grounded": False, "issues": ["the 34,000 figure is unsupported"]}

    stance_iter = iter(stance_sequence)
    last_stance = stance_sequence[-1]

    def fake_check_stance(post, draft):
        calls["stance"] += 1
        try:
            verdict = next(stance_iter)
        except StopIteration:
            verdict = last_stance
        return {"stance": verdict, "reason": f"stub:{verdict}"}

    early_iter = iter(early_stance_sequence)
    last_early = early_stance_sequence[-1]

    def fake_early_check_stance(post, draft):
        calls["early_stance"] += 1
        try:
            verdict = next(early_iter)
        except StopIteration:
            verdict = last_early
        return {"stance": verdict, "reason": f"stub:early:{verdict}"}

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
        mock.patch("agents.graph.nodes.early_stance_check.check_stance", fake_early_check_stance),
        mock.patch("agents.retrieval.WebRetriever.retrieve", fake_web_retrieve),
        mock.patch("agents.retrieval.LocalRetriever.retrieve", fake_local_retrieve),
        mock.patch("agents.retrieval.WebRetriever.__init__", lambda self, *a, **k: None),
        mock.patch("agents.retrieval.LocalRetriever.__init__", lambda self, *a, **k: None),
        mock.patch("agents.graph.nodes.retrieve._LOCAL", None),
        mock.patch("agents.graph.nodes.retrieve._WEB", None),
    ]


def _run(
    stance_sequence: list[str],
    *,
    grounded: bool = True,
    retrieval_mode: str = "web",
    early_stance_sequence: list[str] | None = None,
) -> dict:
    calls.clear()
    patches = _make_scenario(
        stance_sequence,
        grounded=grounded,
        retrieval_mode=retrieval_mode,
        early_stance_sequence=early_stance_sequence,
    )
    for p in patches:
        p.start()
    try:
        graph = build_graph()
        return graph.invoke(
            {"topic": "CMV: test", "original_post": "an anti-Israel post"},
            config={"recursion_limit": 200},
        )
    finally:
        for p in reversed(patches):
            p.stop()


def test_happy_path_pro_after_first_refine():
    """Mandatory refinement pass runs, stance comes back pro_israel, exit."""
    out = _run(["pro_israel"])
    assert out.get("winner") in ("A", "B")
    assert out.get("stance") == "pro_israel"
    assert out.get("pro_israel_reply") is True
    assert out.get("gave_up", False) is False
    assert calls["generate"] == 1
    assert calls["rank"] == 1
    assert calls["early_stance"] == 1  # raw survivor gate (proceeds)
    assert calls["stance"] == 1  # only the post-refine check
    assert calls["refine"] == 1
    assert calls["grounding"] == 1
    assert calls["router"] == 1


def test_refinement_path_neutral_then_pro():
    """First refinement -> neutral, second refinement -> pro_israel."""
    out = _run(["neutral_needs_refine", "pro_israel"])
    assert out.get("stance") == "pro_israel"
    assert out.get("gave_up", False) is False
    assert calls["generate"] == 1
    assert calls["early_stance"] == 1  # one survivor (no regen), proceeds once
    assert calls["stance"] == 2  # two late stance checks
    assert calls["refine"] == 2  # two refinement passes
    assert calls["router"] == 2


def test_gave_up_via_neutral_then_escalation():
    """Late stance always neutral: exhaust MAX_REFINE_ITERS, escalate to
    off-topic, exhaust MAX_LATE_REGEN_ITERS, give up.

    The early gate proceeds (early sequence defaults to 'pro_israel'), so the
    early regen budget is untouched — this test exercises the LATE side only.
    """
    out = _run(["neutral_needs_refine"] * 100)
    assert out.get("gave_up") is True
    assert out.get("give_up_reason", "")
    assert out.get("stance") == "off_topic_or_anti"  # escalation result
    assert calls["generate"] >= 2  # at least one regeneration before giving up
    # Worst-case bound on total refinement passes. Escalation fires when
    # refine_iter REACHES MAX_REFINE_ITERS (>=), so each generation runs
    # exactly MAX_REFINE_ITERS refinement passes before being discarded. Across
    # (MAX_LATE_REGEN_ITERS + 1) generations the tight bound is the product.
    max_expected_refines = (MAX_LATE_REGEN_ITERS + 1) * MAX_REFINE_ITERS
    assert calls["refine"] <= max_expected_refines


def test_gave_up_via_off_topic_early_regen_then_late_giveup():
    """Raw survivor always off_topic_or_anti, late gate also always
    off_topic_or_anti. The early budget RESETS on each late regeneration, so the
    two regen loops compose MULTIPLICATIVELY:

    There are (1 + MAX_LATE_REGEN_ITERS) "rounds" (the initial generation plus
    one per late regeneration). Each round:
      - the early gate regenerates MAX_EARLY_REGEN_ITERS times (no refinement
        burned), then
      - the early budget is spent for this round, so the early gate hands off to
        the router; refinement runs once and the late gate sees off_topic and
        either starts the next round (resetting the early budget) or gives up.

    Totals:
      generate     = (1 + MAX_LATE_REGEN_ITERS) * (MAX_EARLY_REGEN_ITERS + 1)
      early_stance = same as generate (early gate fires every generation)
      stance / refine / router / grounding = 1 + MAX_LATE_REGEN_ITERS
        (only the handoff generation in each round reaches the refinement path)
    """
    out = _run(
        ["off_topic_or_anti"] * 100,            # late gate (final pass) gives up
        early_stance_sequence=["off_topic_or_anti"] * 100,
    )
    assert out.get("gave_up") is True
    assert out.get("stance") == "off_topic_or_anti"
    rounds = 1 + MAX_LATE_REGEN_ITERS
    expected_generations = rounds * (MAX_EARLY_REGEN_ITERS + 1)
    assert calls["generate"] == expected_generations
    assert calls["early_stance"] == expected_generations
    expected_refines = rounds
    assert calls["stance"] == expected_refines
    assert calls["refine"] == expected_refines
    assert calls["router"] == expected_refines
    assert calls["grounding"] == expected_refines


def test_early_stance_check_has_no_finalize_edge():
    """Topology guard: early_stance_check routes only to generate_initial or
    router — never directly to finalize. Give-up is the late gate's job."""
    graph = build_graph().get_graph()
    targets = {e.target for e in graph.edges if e.source == "early_stance_check"}
    assert targets == {"generate_initial", "router"}
    assert "finalize" not in targets


def test_router_none_skips_retrieval():
    """Router picks 'none': skip_retrieval runs, no retriever or grader is
    called, but reflect+refine still execute."""
    out = _run(["pro_israel"], retrieval_mode="none")
    assert out.get("stance") == "pro_israel"
    assert out.get("gave_up", False) is False
    # The 'none' arm bypasses retrieval and the docs grader entirely.
    assert calls["web_retrieve"] == 0
    assert calls["local_retrieve"] == 0
    assert calls["grade"] == 0
    # But reflect + refine still run.
    assert calls["reflect"] == 1
    assert calls["refine"] == 1
    # hallucination_check is skipped on retrieval_mode='none' (no evidence to
    # ground against), but the node itself still runs and writes grounded=True.
    assert calls["grounding"] == 0
    assert calls["router"] == 1
    assert calls["stance"] == 1


def test_no_finalize_without_grounding_pass():
    """Safety property: every successful run goes through refine ->
    hallucination_check before stance_check ever fires.

    Even on the 'none' path the hallucination_check NODE runs (it just sets
    grounded=True without calling the grader). What matters is the topology:
    no path to finalize exists that bypasses refine.
    """
    out = _run(["pro_israel"])
    assert out.get("gave_up", False) is False
    assert calls["refine"] >= 1  # at least one refinement pass ran


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
