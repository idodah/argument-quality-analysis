"""Unit tests for the deterministic helper functions.

These are the pure string/parse functions that the offline graph test
(test_graph_offline.py) stubs out: citation/source extraction, the
critique-shape guard, the router's JSON parsing, chunk dedup, and the issue
classifier. They have real branching logic but no LLM/network/model
dependency, so they are cheap to test exhaustively and a good regression net.

Run: `uv run pytest tests/test_helpers.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.graph.chains.refiner import looks_like_critique
from agents.graph.chains.router import _parse_decision, _parse_queries
from agents.graph.nodes.hallucination_check import _is_citation_shaped
from agents.graph.nodes.retrieve import _dedupe
from agents.graph.state import (
    extract_sources,
    split_for_display,
    strip_citations,
    strip_empty_sources,
)


# --------------------------------------------------------------------------- #
# state.py: citation / sources handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text, want", [
    ("Israel withdrew in 2005 [1].", "Israel withdrew in 2005."),
    ("A claim [1, 2] stands.", "A claim stands."),
    ("Body text [1].\n\n### Sources\n[1] http://x.com", "Body text."),
    ("", ""),
    ("Plain prose.", "Plain prose."),
])
def test_strip_citations(text, want):
    assert strip_citations(text) == want


@pytest.mark.parametrize("text, want", [
    ("Argument body.\n\n### Sources\n", "Argument body."),
    # A Sources block that has [n] entries is left untouched.
    ("Argument body.\n\n### Sources\n[1] http://x.com",
     "Argument body.\n\n### Sources\n[1] http://x.com"),
    ("Just prose.", "Just prose."),
])
def test_strip_empty_sources(text, want):
    assert strip_empty_sources(text) == want


@pytest.mark.parametrize("text, want", [
    ("Body [1][2].\n\n### Sources\n[1] http://a.com\n[2] http://b.com",
     ["http://a.com", "http://b.com"]),
    # Repeated URLs are deduped, in first-seen order.
    ("### Sources\n[1] http://a.com\n[2] http://a.com", ["http://a.com"]),
    # Trailing punctuation is stripped.
    ("### Sources\n[1] http://a.com.", ["http://a.com"]),
    # A Sources block at the very start of the text is still detected
    # (regression: the regex once required a leading newline).
    ("### Sources\n[1] http://a.com", ["http://a.com"]),
    ("Body [1] with no footer.", []),
])
def test_extract_sources(text, want):
    assert extract_sources(text) == want


def test_split_for_display():
    clean, urls = split_for_display("Claim [1].\n\n### Sources\n[1] http://a.com")
    assert clean == "Claim."
    assert urls == ["http://a.com"]


# --------------------------------------------------------------------------- #
# refiner.py: critique-shape guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text, expected", [
    ("Your draft should cite a source here.", True),
    ("**Missing**\n- add the 1947 partition context", True),   # leaked reflect header
    ("You claim Israel controls Gaza's water, but in fact...", False),  # genuine rebuttal
    ("", False),
])
def test_looks_like_critique(text, expected):
    assert looks_like_critique(text) is expected


# --------------------------------------------------------------------------- #
# hallucination_check.py: citation-vs-fact issue classifier
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("issue, expected", [
    ("marker [2] points to the wrong chunk", True),
    ("wrong-pointer on the second claim", True),
    ("citation does not match the source", True),
    ("the 34,000 casualty figure is unsupported", False),  # fabricated fact
])
def test_is_citation_shaped(issue, expected):
    assert _is_citation_shaped(issue) is expected


# --------------------------------------------------------------------------- #
# router.py: LLM-output JSON parsing (with fallbacks)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text, n, want", [
    ('{"mode": "web", "queries": ["a", "b", "c", "d"]}', 3, ("web", ["a", "b", "c"])),
    ('{"mode": "local", "queries": ["a", "b"]}', 3, ("local", [])),  # local drops queries
    ("the model rambled with no json", 3, ("local", [])),           # unparseable -> safe default
    ('{"mode": "web", queries: [}', 3, ("local", [])),              # malformed json -> safe default
    ('{"mode": "web", "queries": "oops"}', 3, ("web", [])),         # non-list queries -> empty
])
def test_parse_decision(text, n, want):
    assert _parse_decision(text, n) == want


@pytest.mark.parametrize("text, n, want", [
    ('{"queries": ["x", "y", "z", "w"]}', 2, ["x", "y"]),
    ('{"queries": ["a", "  ", "b"]}', 5, ["a", "b"]),  # blanks dropped
    ("nope", 3, []),
])
def test_parse_queries(text, n, want):
    assert _parse_queries(text, n) == want


# --------------------------------------------------------------------------- #
# retrieve.py: chunk dedup
# --------------------------------------------------------------------------- #
def test_dedupe_by_url_header():
    a = "[url] http://x.com\nfirst copy"
    b = "[url] http://x.com\nsecond copy (same url)"
    c = "[url] http://y.com\nother"
    assert _dedupe([a, b, c]) == [a, c]  # dedup by [url], keep first


def test_dedupe_keeps_reference_chunks_sharing_one_url():
    # Reference articles are split into many chunks that share a url by design.
    # Keying on the url alone would collapse a whole article to its first
    # passage, starving hallucination_check of the evidence it verifies against.
    a = "[url] http://x.com\n\n[title] T\n\n[reference article — authoritative source: ushmm]\nfirst passage"
    b = "[url] http://x.com\n\n[title] T\n\n[reference article — authoritative source: ushmm]\nsecond passage"
    assert _dedupe([a, b]) == [a, b]


def test_dedupe_still_collapses_identical_reference_chunks():
    a = "[url] http://x.com\n\n[reference article — authoritative source: ushmm]\nsame passage"
    assert _dedupe([a, a]) == [a]


def test_dedupe_headerless_by_content():
    assert _dedupe(["same", "same", "diff"]) == ["same", "diff"]


def test_dedupe_drops_empty():
    assert _dedupe(["", "keep", ""]) == ["keep"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# --------------------------------------------------------------------------- #
# hallucination_check: when the grader is skipped vs. actually run
# --------------------------------------------------------------------------- #
# The local corpus holds two kinds of evidence: CMV comments (example arguments
# — nothing to verify a fact against) and reference articles (authoritative,
# url-bearing, citable). Skipping the grader on every mode=='local' pass would
# bypass verification on exactly the evidence the reference corpus supplies, so
# the skip is keyed on whether citable evidence is actually present.
#
# These drive the real node with check_grounding stubbed, so they fail if the
# skip condition regresses.

REFERENCE_DOC = "[url] https://encyclopedia.ushmm.org/x\n\n[reference article]\nbody"
CMV_DOC = "[topic] CMV: x\n\n[argument (earned a delta on CMV)]\nbody"


def _run_hallucination_check(mode, documents):
    """Invoke the real node, stubbing the LLM grader to a grounded verdict."""
    import importlib
    from unittest import mock

    # NB: the module and the node function share a name, so a plain
    # `import ... as hc` binds the FUNCTION (re-exported on the package), not
    # the module. Fetch the module explicitly so patching finds check_grounding.
    hc = importlib.import_module("agents.graph.nodes.hallucination_check")

    state = {
        "argument": "a draft",
        "documents": documents,
        "retrieval_mode": mode,
        "history": [],
        "refine_iter": 0,
    }
    with mock.patch.object(hc, "check_grounding",
                           return_value={"grounded": True, "issues": []}) as grader:
        out = hc.hallucination_check(state)
    return out, grader


def test_grounding_runs_on_local_reference_evidence():
    out, grader = _run_hallucination_check("local", [REFERENCE_DOC])
    assert grader.called, "reference evidence is citable and must be verified"
    assert out["grounding_verified"] is True


def test_grounding_skipped_on_local_cmv_only_evidence():
    out, grader = _run_hallucination_check("local", [CMV_DOC])
    assert not grader.called
    assert out["grounding_verified"] is False


def test_grounding_always_skipped_on_none_mode():
    out, grader = _run_hallucination_check("none", [REFERENCE_DOC])
    assert not grader.called
    assert out["grounding_verified"] is False


def test_grounding_skipped_when_local_returned_nothing():
    out, grader = _run_hallucination_check("local", [])
    assert not grader.called
    assert out["grounding_verified"] is False


# --------------------------------------------------------------------------- #
# prompts: the length cap must be stated consistently
# --------------------------------------------------------------------------- #
# The output length is set purely by prompt text, in three places that have to
# agree: the initial generator writes the drafts, the refiner rewrites them, and
# reflect decides how much new material to demand. If one drifts, the nodes
# fight each other — reflect asking for more than the cap holds is how a
# refinement loop burns passes without converging.

def test_initial_generator_asks_for_one_or_two_paragraphs():
    from agents import prompts
    assert "one or two paragraphs" in prompts.INITIAL_GEN_USER.lower()


def test_refiner_states_the_cap_as_hard_and_outranking():
    from agents import prompts
    system = prompts.REFINE_SYSTEM.lower()
    assert "one or two paragraphs" in system
    # The cap has to beat the critique, or "ADD the Missing bullets" wins.
    assert "outranks" in system


def test_reflect_is_aware_of_the_same_cap():
    from agents import prompts
    assert "one or two paragraphs" in prompts.REFLECT_SYSTEM.lower()


def test_no_prompt_still_asks_for_the_old_longer_range():
    from agents import prompts
    for name in ("INITIAL_GEN_USER", "REFINE_SYSTEM", "REFLECT_SYSTEM"):
        assert "3-6 paragraph" not in getattr(prompts, name).lower(), name


# --------------------------------------------------------------------------- #
# orchestrate: query -> individual search terms
# --------------------------------------------------------------------------- #
# Lemmy and PieFed pass `q` straight to their APIs, which match it as one
# phrase. A multi-word query therefore has to be issued as separate searches —
# the single-call form found nothing on either platform across a week of runs.

def test_search_terms_splits_on_whitespace():
    from harvester.orchestrate import search_terms
    assert search_terms("rothschild holohoax zog") == ["rothschild", "holohoax", "zog"]


def test_search_terms_keeps_quoted_phrases_whole():
    # "blood libel" is meaningless split into two searches.
    from harvester.orchestrate import search_terms
    assert search_terms('rothschild "blood libel" zog') == [
        "rothschild", "blood libel", "zog"]


def test_search_terms_handles_empty_and_blank():
    from harvester.orchestrate import search_terms
    assert search_terms("") == []
    assert search_terms("   ") == []


def test_default_query_is_a_term_list_not_a_phrase():
    from harvester.orchestrate import DEFAULT_QUERY, search_terms
    terms = search_terms(DEFAULT_QUERY)
    assert len(terms) > 1, "the default must issue several searches, not one phrase"
    assert "blood libel" in terms, "multi-word terms must survive as one term"
