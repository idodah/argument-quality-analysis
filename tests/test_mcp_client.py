"""The fediverse-reader MCP server + the orchestrator's MCP client, offline.

The server's tools are exercised over a real MCP session (in-memory transport,
same protocol as stdio) with a fake adapter standing in for the network, and the
results are decoded with the client's own helpers — so this covers the whole
tool-call round trip: server serialization -> protocol -> client decoding.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from harvester import fediverse_mcp
from harvester.fediverse.base import PostRef, Thread
from harvester.mcp_client import _raise_if_error, _ref, _tool_payload
from mcp.shared.memory import create_connected_server_and_client_session

_POST = PostRef(canonical_id="https://lemmy.example/post/1", platform="lemmy",
                local_id="1", title="t", body="b", url="https://lemmy.example/post/1",
                created_utc=1700000000.0)


class _FakeAdapter:
    instances = 0

    def __init__(self):
        type(self).instances += 1

    def search(self, query, limit=25):
        if query == "boom":
            raise RuntimeError("upstream down")
        return [_POST]

    def thread(self, local_id):
        return Thread(post=_POST, comments=["c1", "c2"])


@pytest.fixture
def fake_server(monkeypatch):
    _FakeAdapter.instances = 0
    monkeypatch.setattr(fediverse_mcp, "get_platform", lambda name: _FakeAdapter())
    monkeypatch.setattr(fediverse_mcp, "_adapters", {})
    return fediverse_mcp.mcp


def _call(server, tool, **args):
    async def go():
        async with create_connected_server_and_client_session(server) as s:
            return _tool_payload(await s.call_tool(tool, args))
    return asyncio.run(go())


def test_search_round_trip_decodes_to_postref(fake_server):
    payload = _call(fake_server, "search_posts", platform="lemmy", query="zog")
    _raise_if_error(payload)
    assert [_ref(d) for d in payload] == [_POST]


def test_get_thread_round_trip(fake_server):
    payload = _call(fake_server, "get_thread", platform="lemmy", local_id="1")
    _raise_if_error(payload)
    assert _ref(payload["post"]) == _POST
    assert payload["comments"] == ["c1", "c2"]


def test_adapter_error_becomes_runtimeerror_on_client(fake_server):
    payload = _call(fake_server, "search_posts", platform="lemmy", query="boom")
    with pytest.raises(RuntimeError, match="upstream down"):
        _raise_if_error(payload)


def test_server_reuses_adapter_across_calls(fake_server):
    # Reddit's feed cache lives on the adapter instance; get_thread only works
    # if it hits the same instance search_posts populated.
    async def go():
        async with create_connected_server_and_client_session(fake_server) as s:
            await s.call_tool("search_posts", {"platform": "lemmy", "query": "a"})
            await s.call_tool("search_posts", {"platform": "lemmy", "query": "b"})
            await s.call_tool("get_thread", {"platform": "lemmy", "local_id": "1"})
    asyncio.run(go())
    assert _FakeAdapter.instances == 1


def test_tool_payload_protocol_error_raises():
    res = SimpleNamespace(isError=True, structuredContent=None,
                          content=[SimpleNamespace(text="bad args")])
    with pytest.raises(RuntimeError, match="bad args"):
        _tool_payload(res)
