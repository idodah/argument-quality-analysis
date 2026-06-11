"""Platform-adapter interface for multi-platform harvesting.

One uniform contract over heterogeneous backends (Lemmy, PieFed, Reddit). Each
adapter turns its platform's API into the same `PostRef` / `Thread` shapes, so the
MCP tools and the orchestrator agent never touch platform-specific details.

See harvester/FEDIVERSE_DESIGN.md for the rationale.
"""

from __future__ import annotations

import ipaddress
import socket
import time
from dataclasses import asdict, dataclass, field
from typing import Protocol
from urllib.parse import urlparse


def assert_safe_url(url: str) -> None:
    """SSRF guard: reject non-http(s) schemes and URLs whose host resolves to a
    private / loopback / link-local / reserved address.

    The link-local block (169.254.0.0/16) is what stops a malicious instance or
    post-derived URL from reaching the cloud metadata service (169.254.169.254)
    and stealing the job's IAM credentials once this runs on AWS.

    Defense-in-depth: today the instance hosts come from env vars (trusted), but
    any future path that fetches a URL taken from post content (an `ap_id`, a
    `url` field) is covered by this central check. Raises RuntimeError if unsafe.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise RuntimeError(f"refusing non-http(s) URL: {url!r}")
    host = parsed.hostname
    if not host:
        raise RuntimeError(f"URL has no host: {url!r}")
    try:
        # Check EVERY address the host maps to (defeats DNS rebinding where a name
        # resolves to a public IP on one lookup and a private one on the next).
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise RuntimeError(f"cannot resolve host {host!r}: {e}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise RuntimeError(
                f"refusing URL {url!r}: host {host!r} resolves to non-public {ip}"
            )


def http_get_json(url: str, params: dict, headers: dict, *, timeout: int = 20,
                  retries: int = 2) -> dict:
    """GET JSON with a small retry on transient errors. Fediverse instances are
    small and throw intermittent 5xx / timeouts; one blip shouldn't lose a whole
    platform's batch. Raises RuntimeError after the last attempt.

    The URL is SSRF-checked (assert_safe_url) before the request."""
    import requests

    assert_safe_url(url)
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last = e
            # Retry only on transient-looking failures (5xx / timeout / conn).
            status = getattr(getattr(e, "response", None), "status_code", None)
            transient = status is None or status >= 500
            if attempt < retries and transient:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
    raise RuntimeError(f"GET {url} failed: {last}")


@dataclass
class PostRef:
    """A post discovered via search. `canonical_id` is the cross-platform dedup
    key (ActivityPub ap_id / permalink) — the SAME federated post seen on two
    instances must share it, so we never answer it twice."""

    canonical_id: str
    platform: str          # "lemmy" | "piefed" | "reddit"
    local_id: str          # the id this platform's get_thread needs
    title: str
    body: str              # may be empty for link posts -> rely on comments
    url: str
    created_utc: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Thread:
    """A post plus its top comments — the material to rebut (the post body is
    often empty for link shares, so comments carry the argument)."""

    post: PostRef
    comments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"post": self.post.to_dict(), "comments": self.comments}

    def rebuttal_inputs(self) -> tuple[str, str]:
        """(topic, body) for the generation graph: the title is the topic; the
        body is the post body plus salient comments so there's something to
        argue against even when the post is a bare link."""
        topic = self.post.title
        parts = []
        if self.post.body.strip():
            parts.append(self.post.body.strip())
        if self.comments:
            parts.append("\n\n--- Comments ---\n" + "\n\n".join(self.comments))
        return topic, "\n\n".join(parts).strip()


class Platform(Protocol):
    """A read-only source of posts. Adapters implement this; the MCP server and
    the orchestrator depend only on it."""

    name: str

    def search(self, query: str, limit: int = 25) -> list[PostRef]:
        """Return recent posts matching `query` (newest-ish first)."""
        ...

    def thread(self, local_id: str) -> Thread:
        """Fetch one post + its top comments by this platform's local id."""
        ...
