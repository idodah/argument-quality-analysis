"""Reddit adapter: wraps the existing RSS fetch behind the Platform interface.

Reddit's public read surface is the r/changemyview `/new` Atom feed (the data API
is approval-gated). There's no topic-search endpoint we can use unauthenticated,
so `search(query)` fetches the newest posts and keeps those whose title/body
mention the query terms — a local filter that keeps the interface uniform with
the Fediverse adapters. The RSS body is already the full post, so `thread()` has
no extra comments to fetch.
"""

from __future__ import annotations

from harvester.fediverse.base import PostRef, Thread
from harvester.fetch import fetch_from_rss


def _to_ref(post) -> PostRef:
    # The RSS Post.id is already "t3_<base36>" and the url is the permalink —
    # both stable across views, so the permalink is our canonical id.
    return PostRef(
        canonical_id=post.url or post.id,
        platform="reddit",
        local_id=post.id,
        title=post.title,
        body=post.body,
        url=post.url,
        created_utc=post.created_utc,
    )


class RedditAdapter:
    name = "reddit"

    def __init__(self):
        # Cache the last feed fetch so search() and a following thread() don't
        # double-hit the network within one cycle.
        self._cache: dict[str, PostRef] = {}
        # The fetched feed itself, reused across search() calls. The
        # orchestrator issues one search PER TERM, and Reddit has no
        # unauthenticated search API — every call would otherwise re-fetch the
        # same /new RSS feed and trip its rate limiter (observed: 429 on 5 of 6
        # terms, so the Reddit arm returned nothing at all). The feed is
        # identical for every term, so fetch once per adapter instance and
        # filter locally; the orchestrator already keeps one instance per run.
        self._feed: list[PostRef] | None = None

    def _feed_once(self, limit: int) -> list[PostRef]:
        """The /new feed, fetched at most once per adapter instance."""
        if self._feed is None:
            self._feed = [_to_ref(p) for p in fetch_from_rss(limit=max(limit, 25))]
            for ref in self._feed:
                self._cache[ref.local_id] = ref
        return self._feed

    def search(self, query: str, limit: int = 25) -> list[PostRef]:
        """Keep the newest CMV posts matching any query term (Reddit has no
        unauthenticated search, so we filter the cached feed locally)."""
        terms = [t.lower() for t in query.split() if t.strip()]
        out: list[PostRef] = []
        for ref in self._feed_once(limit):
            hay = f"{ref.title}\n{ref.body}".lower()
            if not terms or any(t in hay for t in terms):
                out.append(ref)
                if len(out) >= limit:
                    break
        return out

    def thread(self, local_id: str) -> Thread:
        """Return the cached post as a Thread (RSS carried the full body and no
        comments, so search() already stashed everything). A miss means the id
        isn't in this run's feed — which the /new feed can't surface — so raise
        rather than waste a round-trip."""
        ref = self._cache.get(local_id)
        if ref is None:
            raise RuntimeError(
                f"reddit post {local_id!r} not in this run's feed cache; "
                "call search() before thread()."
            )
        return Thread(post=ref, comments=[])
