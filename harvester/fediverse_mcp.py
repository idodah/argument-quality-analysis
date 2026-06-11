"""MCP server exposing multi-platform post reading as agent tools (read-only).

This is the justified MCP placement (see harvester/FEDIVERSE_DESIGN.md): an
orchestrator agent decides — across Lemmy, PieFed, and Reddit — which anti-Israel
threads are worth answering and where, then drafts replies. That runtime decision
over heterogeneous-but-uniform backends is a real agent/tool boundary, which is
what MCP is for. The tools are READ-ONLY: detect + draft + notify; never post.

Tools:
  search_posts(platform, query, limit) -> list of post refs
  get_thread(platform, local_id)       -> post + top comments

Run (stdio transport):
    uv run python -m harvester.fediverse_mcp
"""

from __future__ import annotations

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from harvester.fediverse import PLATFORMS, get_platform

load_dotenv()

mcp = FastMCP("fediverse-reader")


@mcp.tool()
def list_platforms() -> list[str]:
    """The platforms available to search/read (lemmy, piefed, reddit)."""
    return list(PLATFORMS)


@mcp.tool()
def search_posts(platform: str, query: str, limit: int = 25) -> list[dict]:
    """Search one platform for recent posts matching `query`.

    Args:
        platform: 'lemmy', 'piefed', or 'reddit'.
        query: search terms (e.g. 'israel palestine gaza').
        limit: max posts to return.

    Returns a list of post refs, each with: canonical_id (the cross-platform
    dedup key), platform, local_id (pass to get_thread), title, body, url,
    created_utc. Returns an error dict on failure instead of raising.
    """
    try:
        return [p.to_dict() for p in get_platform(platform).search(query, limit=limit)]
    except RuntimeError as e:
        return [{"error": str(e), "platform": platform}]


@mcp.tool()
def get_thread(platform: str, local_id: str) -> dict:
    """Fetch one post plus its top comments (the material to rebut).

    Args:
        platform: 'lemmy', 'piefed', or 'reddit'.
        local_id: the post's `local_id` from search_posts.

    Returns {post: {...}, comments: [...]} — or {error: ...} on failure.
    """
    try:
        return get_platform(platform).thread(local_id).to_dict()
    except RuntimeError as e:
        return {"error": str(e), "platform": platform, "local_id": local_id}


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
