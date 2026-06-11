"""PieFed adapter: read-only search + thread via the public /api/alpha endpoints.

PieFed's API is Lemmy-shaped (same post_view / ap_id / comment-list structure),
but lives under /api/alpha and uses `title` (not `name`) for the post title. No
auth needed for reading. `PIEFED_INSTANCE` (default piefed.social) picks the host.
"""

from __future__ import annotations

import os
from datetime import datetime

from harvester.fediverse.base import PostRef, Thread, http_get_json

_UA = "python:argument-quality-harvester:1.0 (fediverse read)"


def _instance() -> str:
    return (os.environ.get("PIEFED_INSTANCE") or "https://piefed.social").rstrip("/")


def _ts(published: str | None) -> float:
    if not published:
        return 0.0
    try:
        return datetime.fromisoformat(published.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _to_ref(p: dict, name: str) -> PostRef:
    return PostRef(
        canonical_id=p.get("ap_id") or f"piefed:{p.get('id')}",
        platform=name,
        local_id=str(p.get("id")),
        title=p.get("title") or p.get("name") or "",
        body=p.get("body") or "",
        url=p.get("ap_id") or p.get("url") or "",
        created_utc=_ts(p.get("published")),
    )


class PieFedAdapter:
    name = "piefed"

    def __init__(self, instance: str | None = None):
        self.base = (instance or _instance()).rstrip("/")
        self.headers = {"User-Agent": _UA}

    def _get(self, path: str, params: dict) -> dict:
        return http_get_json(f"{self.base}/api/alpha/{path}", params, self.headers)

    def search(self, query: str, limit: int = 25) -> list[PostRef]:
        data = self._get("search", {
            "q": query, "type_": "Posts", "sort": "New", "limit": limit,
        })
        return [_to_ref(pv.get("post", {}), self.name) for pv in data.get("posts", [])]

    def thread(self, local_id: str) -> Thread:
        pdata = self._get("post", {"id": local_id})
        p = pdata.get("post_view", {}).get("post", {})
        post = _to_ref(p or {"id": local_id}, self.name)
        cdata = self._get("comment/list", {"post_id": local_id, "sort": "Top", "limit": 10})
        comments = [
            cv["comment"]["content"]
            for cv in cdata.get("comments", [])
            if cv.get("comment", {}).get("content")
        ]
        return Thread(post=post, comments=comments)
