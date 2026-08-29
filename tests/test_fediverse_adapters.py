"""Offline tests for the Fediverse adapters' parsing + the Thread shaping.

No network: we feed canned API payloads through the parsing paths and check the
uniform PostRef/Thread output (especially the canonical_id used for cross-platform
dedup). Network calls are stubbed at the adapter's `_get`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harvester.fediverse.base import PostRef, Thread


def test_thread_rebuttal_inputs_merges_body_and_comments():
    post = PostRef("ap1", "lemmy", "1", "CMV: title", "post body", "u", 0.0)
    thread = Thread(post=post, comments=["first comment", "second comment"])
    topic, body = thread.rebuttal_inputs()
    assert topic == "CMV: title"
    assert "post body" in body
    assert "first comment" in body and "second comment" in body


def test_thread_rebuttal_inputs_handles_empty_body_link_post():
    # Link posts often have no body — the comments must carry the argument.
    post = PostRef("ap2", "piefed", "2", "Israel headline", "", "u", 0.0)
    thread = Thread(post=post, comments=["the real argument is here"])
    _topic, body = thread.rebuttal_inputs()
    assert "the real argument is here" in body


def test_lemmy_search_parses_canonical_id(monkeypatch):
    from harvester.fediverse.lemmy import LemmyAdapter

    payload = {"posts": [{"post": {
        "id": 99, "ap_id": "https://other.instance/post/99",
        "name": "Israel post", "body": "body text",
        "published": "2026-06-09T15:08:01Z",
    }}]}
    a = LemmyAdapter()
    monkeypatch.setattr(a, "_get", lambda path, params: payload)
    refs = a.search("israel")
    assert len(refs) == 1
    r = refs[0]
    assert r.canonical_id == "https://other.instance/post/99"  # ap_id, not local id
    assert r.platform == "lemmy" and r.local_id == "99"
    assert r.title == "Israel post" and r.created_utc > 0


def test_piefed_uses_title_field(monkeypatch):
    from harvester.fediverse.piefed import PieFedAdapter

    payload = {"posts": [{"post": {
        "id": 5, "ap_id": "https://x/post/5", "title": "PieFed title", "body": "b",
        "published": "2026-06-09T00:00:00Z",
    }}]}
    a = PieFedAdapter()
    monkeypatch.setattr(a, "_get", lambda path, params: payload)
    refs = a.search("israel")
    assert refs[0].title == "PieFed title"
    assert refs[0].canonical_id == "https://x/post/5"


def test_reddit_fetches_the_feed_once_across_many_searches(monkeypatch):
    """The orchestrator searches once PER TERM. Reddit has no search API, so
    every call would re-fetch the same /new RSS feed — observed tripping its
    rate limiter (429) on 5 of 6 terms, leaving the Reddit arm empty. The feed
    is identical for every term, so it must be fetched only once."""
    from harvester.fediverse import reddit as reddit_mod

    calls = {"n": 0}

    class _Post:
        def __init__(self, i):
            self.id = str(i)
            self.title = f"CMV: rothschild post {i}"
            self.body = "blood libel and khazar and zog"
            self.url = f"https://reddit.com/{i}"
            self.created_utc = 1_700_000_000.0
            self.permalink = self.url

    def fake_fetch(limit=25):
        calls["n"] += 1
        return [_Post(i) for i in range(3)]

    monkeypatch.setattr(reddit_mod, "fetch_from_rss", fake_fetch)
    a = reddit_mod.RedditAdapter()
    for term in ("rothschild", "blood libel", "khazar", "zog", "holohoax"):
        a.search(term, limit=10)
    assert calls["n"] == 1, f"feed fetched {calls['n']} times; must be 1"
