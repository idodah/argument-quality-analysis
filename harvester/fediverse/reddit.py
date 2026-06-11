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

    def search(self, query: str, limit: int = 25) -> list[PostRef]:
        terms = [t.lower() for t in query.split() if t.strip()]
        out: list[PostRef] = []
        for post in fetch_from_rss(limit=max(limit, 25)):
            ref = _to_ref(post)
            self._cache[ref.local_id] = ref
            hay = f"{ref.title}\n{ref.body}".lower()
            if not terms or any(t in hay for t in terms):
                out.append(ref)
                if len(out) >= limit:
                    break
        return out

    def thread(self, local_id: str) -> Thread:
        # RSS already carried the full post body; comments aren't in the feed, so
        # there is nothing to fetch here — `search()` populated `_cache` with the
        # full PostRef. The orchestrator always calls search() before thread()
        # within a cycle, so a miss means the id was never in this run's feed;
        # re-fetching wouldn't surface it (the /new feed only carries the newest
        # ~25), so raise rather than burn a second round-trip on a stale guess.
        ref = self._cache.get(local_id)
        if ref is None:
            raise RuntimeError(
                f"reddit post {local_id!r} not in this run's feed cache; "
                "call search() before thread()."
            )
        return Thread(post=ref, comments=[])
