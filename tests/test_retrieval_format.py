"""Offline tests for LocalRetriever._format.

The corpus holds two kinds of document in one collection, and _format is what
tells the generator which is which. Getting this wrong has a specific failure
mode: a Reddit comment presented as an authoritative source invites the model
to cite it as evidence, and the citation contract in REFINE_SYSTEM keys off the
'[url]' line, so provenance labelling is load-bearing rather than cosmetic.

Pure string formatting — no network, no embeddings, no Chroma.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.retrieval import LocalRetriever


class _Doc:
    """Minimal stand-in for a langchain Document."""

    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


REFERENCE_META = {
    "source_type": "reference",
    "source": "ushmm",
    "title": "Combating Holocaust Denial: Origins of Holocaust Denial",
    "url": "https://encyclopedia.ushmm.org/content/en/article/x",
    "trope": "holocaust_denial",
}

CMV_META = {
    "source_type": "cmv_delta",
    "source": "cmv_israel",
    "topic": "CMV: something",
    "original_post": "the original post body",
}


def test_reference_doc_emits_citable_url_line():
    # REFINE_SYSTEM tells the model each chunk starts with '[url] <URL>' and that
    # those URLs are its ONLY legitimate sources, so the line must come first.
    out = LocalRetriever._format(_Doc("Nazi policy did a great deal...", REFERENCE_META))
    assert out.startswith("[url] https://encyclopedia.ushmm.org/content/en/article/x")
    assert "[title] Combating Holocaust Denial: Origins of Holocaust Denial" in out
    assert "Nazi policy did a great deal..." in out


def test_reference_doc_names_its_authoritative_source():
    out = LocalRetriever._format(_Doc("body", REFERENCE_META))
    assert "authoritative source: ushmm" in out


def test_cmv_doc_is_not_labelled_authoritative_and_has_no_url():
    # The key anti-requirement: a delta-awarded Reddit comment must never look
    # like a citable factual source.
    out = LocalRetriever._format(_Doc("some persuasive comment", CMV_META))
    assert "[url]" not in out
    assert "authoritative" not in out
    assert "[argument (earned a delta on CMV)]" in out
    assert "[topic] CMV: something" in out
    assert "[original_post]" in out


def test_reference_doc_without_url_still_renders_body():
    meta = {k: v for k, v in REFERENCE_META.items() if k != "url"}
    out = LocalRetriever._format(_Doc("body text", meta))
    assert "[url]" not in out
    assert "body text" in out


def test_bare_doc_falls_back_to_page_content():
    out = LocalRetriever._format(_Doc("just the text", {}))
    assert out == "just the text"


def test_headers_use_brackets_not_markdown():
    # '###' headers inside a chunk collide with the prompt-level section headers
    # used by reflect / refine / hallucination_check.
    for meta in (REFERENCE_META, CMV_META):
        assert "###" not in LocalRetriever._format(_Doc("body", meta))
