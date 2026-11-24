"""MCP client for the fediverse-reader server (harvester.fediverse_mcp).

The orchestrator reads every platform THROUGH the MCP server rather than
importing the adapters: it spawns the server as a subprocess over the stdio
transport, performs the MCP handshake, checks the tools it needs are offered,
and then calls `search_posts` / `get_thread` as MCP tool calls. The server is
the only component that talks to Lemmy / PieFed / Reddit.

The MCP SDK is async, while the orchestrator is a plain synchronous loop. The
SDK's stdio transport runs inside anyio task groups that must be entered and
exited by the same task, so the session can't be opened in one `asyncio.run()`
and used from another. Instead `MCPHarvesterClient` runs one event loop on a
background thread that owns the session for its whole life, and the sync
methods submit tool calls to that loop and wait for the result.

    with MCPHarvesterClient() as client:
        reddit = client.platform("reddit")
        refs = reddit.search("rothschild")
        thread = reddit.thread(refs[0].local_id)

`client.platform(name)` returns an object with the same `search()` / `thread()`
shape as the in-process adapters (harvester.fediverse.base.Platform), so the
orchestrator's logic doesn't change with the transport.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from harvester.fediverse.base import PostRef, Thread

SERVER_MODULE = "harvester.fediverse_mcp"
REQUIRED_TOOLS = {"search_posts", "get_thread"}
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _tool_payload(result) -> Any:
    """Decode a CallToolResult into plain Python data.

    FastMCP returns list-typed tools as structured content wrapped in
    {"result": [...]}, and bare-dict tools as a single JSON text block; handle
    both. A protocol-level tool failure (isError) becomes RuntimeError.
    """
    texts = [getattr(c, "text", "") for c in result.content]
    if result.isError:
        raise RuntimeError("MCP tool error: " + " ".join(texts).strip())
    sc = result.structuredContent
    if sc is not None:
        if isinstance(sc, dict) and set(sc) == {"result"}:
            return sc["result"]
        return sc
    if len(texts) == 1:
        return json.loads(texts[0])
    return [json.loads(t) for t in texts]


def _raise_if_error(payload: Any) -> None:
    """The server reports adapter failures as {"error": ...} data (so an LLM
    host can react to them); for the pipeline, turn them back into the
    RuntimeError the in-process adapters raise."""
    if isinstance(payload, dict) and "error" in payload:
        raise RuntimeError(payload["error"])
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and "error" in item:
                raise RuntimeError(item["error"])


def _ref(d: dict) -> PostRef:
    return PostRef(**{k: d[k] for k in PostRef.__dataclass_fields__})


class MCPHarvesterClient:
    """Sync handle on one stdio session with the fediverse-reader MCP server."""

    def __init__(self, *, timeout: float = 120.0):
        self.timeout = timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._stop: asyncio.Event | None = None
        self._ready: Future = Future()
        self._closed: Future = Future()
        self.tools: list[str] = []

    # --- lifecycle -------------------------------------------------------

    def _server_params(self) -> StdioServerParameters:
        # stdio_client passes the child only a minimal default environment
        # (PATH/HOME/...); pass ours so the server sees LEMMY_INSTANCE,
        # REDDIT_USER_AGENT, etc. Same interpreter, so the same venv.
        return StdioServerParameters(
            command=sys.executable,
            args=["-m", SERVER_MODULE],
            env=dict(os.environ),
            cwd=str(_PROJECT_ROOT),
        )

    async def _serve(self) -> None:
        """Own the session for its whole life: open, handshake, wait for close."""
        try:
            self._stop = asyncio.Event()
            async with stdio_client(self._server_params()) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    self.tools = sorted(t.name for t in listed.tools)
                    missing = REQUIRED_TOOLS - set(self.tools)
                    if missing:
                        raise RuntimeError(
                            f"MCP server is missing tools {sorted(missing)}; offers {self.tools}")
                    self._session = session
                    self._ready.set_result(None)
                    await self._stop.wait()
        except BaseException as e:  # surface startup failures to start()
            if not self._ready.done():
                self._ready.set_exception(e)
            raise
        finally:
            self._session = None

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            loop.run_until_complete(self._serve())
            self._closed.set_result(None)
        except BaseException as e:
            self._closed.set_exception(e)
        finally:
            loop.close()

    def start(self) -> "MCPHarvesterClient":
        self._thread = threading.Thread(target=self._run_loop, name="mcp-client", daemon=True)
        self._thread.start()
        self._ready.result(timeout=self.timeout)  # re-raises a failed handshake
        return self

    def close(self) -> None:
        if self._loop is not None and self._stop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=self.timeout)
            self._thread = None

    def __enter__(self) -> "MCPHarvesterClient":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # --- tool calls ------------------------------------------------------

    def call(self, tool: str, **arguments) -> Any:
        """Call one MCP tool synchronously; returns its decoded payload."""
        if self._session is None or self._loop is None:
            raise RuntimeError("MCP session is not open; use `with MCPHarvesterClient()`.")
        fut = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(tool, arguments), self._loop)
        try:
            result = fut.result(timeout=self.timeout)
        except TimeoutError as e:
            fut.cancel()
            raise RuntimeError(f"MCP tool {tool!r} timed out after {self.timeout}s") from e
        return _tool_payload(result)

    def platform(self, name: str) -> "MCPPlatform":
        return MCPPlatform(self, name)


class MCPPlatform:
    """One platform, read through the MCP server. Same search()/thread() shape
    as the in-process adapters, so callers can't tell the difference."""

    def __init__(self, client: MCPHarvesterClient, name: str):
        self.client = client
        self.name = name

    def search(self, query: str, limit: int = 25) -> list[PostRef]:
        payload = self.client.call("search_posts", platform=self.name, query=query, limit=limit)
        _raise_if_error(payload)
        return [_ref(d) for d in payload]

    def thread(self, local_id: str) -> Thread:
        payload = self.client.call("get_thread", platform=self.name, local_id=local_id)
        _raise_if_error(payload)
        return Thread(post=_ref(payload["post"]), comments=list(payload.get("comments", [])))
