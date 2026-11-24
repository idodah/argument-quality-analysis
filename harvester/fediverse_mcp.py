"""MCP server exposing multi-platform post reading as agent tools (read-only).

An orchestrator agent decides — across Lemmy, PieFed, and Reddit — which trope-advancing 
threads are worth answering and where, then drafts replies. That runtime decision 
over heterogeneous-but-uniform backends is a real agent/tool boundary, which is
what MCP is for. The tools are READ-ONLY: detect + draft + notify; never post.

Tools:
  search_posts(platform, query, limit) -> list of post refs
  get_thread(platform, local_id)       -> post + top comments

Run (stdio transport):
    uv run python -m harvester.fediverse_mcp

The scheduled pipeline (harvester.orchestrate) is this server's main client: it
spawns it over stdio via harvester.mcp_client and does all platform reads
through these tools. Interactive MCP hosts (Claude Code / Desktop) can attach to
the same server for manual exploration.
"""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from harvester.fediverse import PLATFORMS, Platform, get_platform

load_dotenv()

# WARNING, not the default INFO: FastMCP logs every request to stderr, which the
# orchestrator inherits — a line per search term per platform in CloudWatch.
mcp = FastMCP("fediverse-reader", log_level="WARNING")

# Adapter instances are kept per platform for the life of the server, so state
# the adapters carry between calls survives across tool calls: the Reddit
# adapter fetches the RSS feed ONCE and serves every search term and every
# get_thread from it (a fresh adapter per call would re-hit the feed per term
# -> 429s, and get_thread would miss the post cache). A long-lived interactive
# session would otherwise see a stale feed forever, so instances expire after
# ADAPTER_TTL_SECONDS (default 10 min — longer than one orchestrator run).
_ADAPTER_TTL = float(os.environ.get("ADAPTER_TTL_SECONDS", "600"))
_adapters: dict[str, tuple[float, Platform]] = {}


def _adapter(platform: str) -> Platform:
    """The cached adapter for `platform`, recreated once it is older than the TTL."""
    key = platform.lower()
    now = time.monotonic()
    hit = _adapters.get(key)
    if hit is None or now - hit[0] > _ADAPTER_TTL:
        hit = (now, get_platform(key))  # raises RuntimeError on unknown platform
        _adapters[key] = hit
    return hit[1]


@mcp.tool()
def list_platforms() -> list[str]:
    """The platforms available to search/read (lemmy, piefed, reddit)."""
    return list(PLATFORMS)


@mcp.tool()
def search_posts(platform: str, query: str, limit: int = 25) -> list[dict]:
    """Search one platform ('lemmy'|'piefed'|'reddit') for recent posts matching
    `query`, up to `limit`. Returns post refs (canonical_id, platform, local_id
    for get_thread, title, body, url, created_utc), or an error dict on failure.
    """
    try:
        return [p.to_dict() for p in _adapter(platform).search(query, limit=limit)]
    except RuntimeError as e:
        return [{"error": str(e), "platform": platform}]


@mcp.tool()
def get_thread(platform: str, local_id: str) -> dict:
    """Fetch one post plus its top comments (the material to rebut), by the
    `local_id` from search_posts on platform ('lemmy'|'piefed'|'reddit').
    Returns {post: {...}, comments: [...]}, or {error: ...} on failure.
    """
    try:
        return _adapter(platform).thread(local_id).to_dict()
    except RuntimeError as e:
        return {"error": str(e), "platform": platform, "local_id": local_id}


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
