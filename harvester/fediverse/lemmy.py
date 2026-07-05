"""Lemmy adapter: read-only search + thread via the public /api/v3 endpoints.

No auth needed for reading. `LEMMY_INSTANCE` (default lemmy.world) picks the home
instance; its search federates across the network.
"""

from __future__ import annotations

import os
from datetime import datetime

from harvester.fediverse.base import PostRef, Thread, http_get_json

_UA = "python:argument-quality-harvester:1.0 (fediverse read)"


def _instance() -> str:
    return (os.environ.get("LEMMY_INSTANCE") or "https://lemmy.world").rstrip("/")


def _ts(published: str | None) -> float:
    """Parse a Lemmy ISO timestamp to epoch seconds; 0.0 if missing/unparseable."""
    if not published:
        return 0.0
    try:
        # Lemmy returns e.g. "2026-06-09T15:08:01.123456Z" (sometimes no tz).
        s = published.replace("Z", "+00:00")
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return 0.0


class LemmyAdapter:
    name = "lemmy"

    def __init__(self, instance: str | None = None):
        self.base = (instance or _instance()).rstrip("/")
        self.headers = {"User-Agent": _UA}

    def _get(self, path: str, params: dict) -> dict:
        return http_get_json(f"{self.base}/api/v3/{path}", params, self.headers)

    def search(self, query: str, limit: int = 25) -> list[PostRef]:
        """Newest posts matching `query`, federated across the network."""
        data = self._get("search", {
            "q": query, "type_": "Posts", "listing_type": "All",
            "sort": "New", "limit": limit,
        })
        out: list[PostRef] = []
        for pv in data.get("posts", []):
            p = pv.get("post", {})
            out.append(PostRef(
                canonical_id=p.get("ap_id") or f"lemmy:{p.get('id')}",
                platform=self.name,
                local_id=str(p.get("id")),
                title=p.get("name") or "",
                body=p.get("body") or "",
                url=p.get("ap_id") or "",
                created_utc=_ts(p.get("published")),
            ))
        return out

    def thread(self, local_id: str) -> Thread:
        """Fetch the post plus its top comments by Lemmy post id."""
        pdata = self._get("post", {"id": local_id})
        p = pdata.get("post_view", {}).get("post", {})
        post = PostRef(
            canonical_id=p.get("ap_id") or f"lemmy:{p.get('id')}",
            platform=self.name,
            local_id=str(p.get("id") or local_id),
            title=p.get("name") or "",
            body=p.get("body") or "",
            url=p.get("ap_id") or "",
            created_utc=_ts(p.get("published")),
        )
        cdata = self._get("comment/list", {
            "post_id": local_id, "sort": "Top", "limit": 10,
        })
        comments = [
            cv["comment"]["content"]
            for cv in cdata.get("comments", [])
            if cv.get("comment", {}).get("content")
        ]
        return Thread(post=post, comments=comments)
