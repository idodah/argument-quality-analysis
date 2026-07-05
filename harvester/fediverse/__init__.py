"""Multi-platform read adapters (Lemmy, PieFed, Reddit) behind one interface.

Each adapter implements `Platform` (base.py), turning its backend into uniform
`PostRef` / `Thread` shapes. `get_platform(name)` is the registry the MCP server
and orchestrator use.
"""

from __future__ import annotations

from harvester.fediverse.base import Platform, PostRef, Thread


def get_platform(name: str) -> Platform:
    """Return the adapter for a platform name."""
    n = name.lower()
    if n == "lemmy":
        from harvester.fediverse.lemmy import LemmyAdapter
        return LemmyAdapter()
    if n == "piefed":
        from harvester.fediverse.piefed import PieFedAdapter
        return PieFedAdapter()
    if n == "reddit":
        from harvester.fediverse.reddit import RedditAdapter
        return RedditAdapter()
    raise RuntimeError(f"unknown platform {name!r} (expected lemmy|piefed|reddit).")


PLATFORMS = ("lemmy", "piefed", "reddit")

__all__ = ["Platform", "PostRef", "Thread", "get_platform", "PLATFORMS"]
