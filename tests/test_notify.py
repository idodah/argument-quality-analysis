"""Offline tests for the Discord notifier backend.

The bug these guard against is the one that drove two backend changes: a long
argument being silently cut. An earlier ntfy backend capped a notification at
4096 bytes and split past it; Discord instead sends anything over its 2000-char
`content` limit as a single `.md` attachment, so the text arrives whole.

No network: `requests.post` is stubbed and the calls are inspected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harvester import notify, notify_discord

WEBHOOK = "https://discord.com/api/webhooks/123/abc"


@pytest.fixture
def dc_env(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", WEBHOOK)


def _ok_response(status=204):
    resp = mock.Mock()
    resp.status_code = status
    resp.json.return_value = {}
    resp.text = ""
    return resp


# --------------------------------------------------------------------------- #
# short vs. long
# --------------------------------------------------------------------------- #
def test_short_message_goes_inline(dc_env):
    with mock.patch.object(notify_discord.requests, "post",
                           return_value=_ok_response()) as post:
        notify_discord.send("a short argument")
    (url,), kwargs = post.call_args
    assert url == WEBHOOK
    assert kwargs["json"]["content"] == "a short argument"
    assert kwargs.get("files") is None


def test_long_message_is_sent_whole_as_an_attachment(dc_env):
    # The regression that matters: nothing may be truncated or split.
    body = "x" * (notify_discord.MAX_CONTENT + 5000)
    with mock.patch.object(notify_discord.requests, "post",
                           return_value=_ok_response()) as post:
        notify_discord.send(body)
    kwargs = post.call_args.kwargs
    _, buf, _ = kwargs["files"]["files[0]"]
    assert buf.getvalue().decode() == body, "attachment must carry the FULL text"


def test_long_message_sends_exactly_one_request(dc_env):
    body = "y" * (notify_discord.MAX_CONTENT * 3)
    with mock.patch.object(notify_discord.requests, "post",
                           return_value=_ok_response()) as post:
        notify_discord.send(body)
    assert post.call_count == 1, "must not split into multiple notifications"


def test_inline_content_respects_the_api_limit(dc_env):
    body = "z" * (notify_discord.MAX_CONTENT + 100)
    with mock.patch.object(notify_discord.requests, "post",
                           return_value=_ok_response()) as post:
        notify_discord.send(body, click_url="https://reddit.com/r/x/1")
    payload = json.loads(post.call_args.kwargs["data"]["payload_json"])
    assert len(payload["content"]) <= notify_discord.MAX_CONTENT


def _sent_content(post):
    """The `content` field, whichever path was taken (inline json or multipart)."""
    kwargs = post.call_args.kwargs
    if kwargs.get("json") is not None:
        return kwargs["json"]["content"]
    return json.loads(kwargs["data"]["payload_json"])["content"]


def test_message_near_the_cap_never_overflows_once_the_link_is_added(dc_env):
    # A body just under 2000 plus an appended url would exceed the cap, so it
    # must fall to the attachment path rather than being sent over-length.
    body = "q" * (notify_discord.MAX_CONTENT - 10)
    with mock.patch.object(notify_discord.requests, "post",
                           return_value=_ok_response()) as post:
        notify_discord.send(body, click_url="https://reddit.com/r/x/1")
    assert len(_sent_content(post)) <= notify_discord.MAX_CONTENT
    # and the full body still goes out, in the attachment
    _, buf, _ = post.call_args.kwargs["files"]["files[0]"]
    assert buf.getvalue().decode() == body


def test_click_url_is_appended_as_a_link(dc_env):
    with mock.patch.object(notify_discord.requests, "post",
                           return_value=_ok_response()) as post:
        notify_discord.send("hi", click_url="https://reddit.com/r/x/1")
    assert "https://reddit.com/r/x/1" in post.call_args.kwargs["json"]["content"]


def test_non_http_click_url_is_dropped(dc_env):
    with mock.patch.object(notify_discord.requests, "post",
                           return_value=_ok_response()) as post:
        notify_discord.send("hi", click_url="javascript:alert(1)")
    assert "javascript" not in post.call_args.kwargs["json"]["content"]


def test_http_200_is_also_accepted(dc_env):
    with mock.patch.object(notify_discord.requests, "post",
                           return_value=_ok_response(200)):
        notify_discord.send("hi")  # must not raise


def test_error_response_raises_with_discord_message(dc_env):
    bad = mock.Mock()
    bad.status_code = 404
    bad.json.return_value = {"message": "Unknown Webhook", "code": 10015}
    bad.text = ""
    with mock.patch.object(notify_discord.requests, "post", return_value=bad):
        with pytest.raises(RuntimeError, match="Unknown Webhook"):
            notify_discord.send("hi")


def test_missing_webhook_raises_with_setup_help(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    with pytest.raises(RuntimeError, match="Webhooks"):
        notify_discord.send("hi")


# --------------------------------------------------------------------------- #
# notify.py re-exports the discord transport
# --------------------------------------------------------------------------- #
def test_notify_send_is_the_discord_send():
    # notify.py owns the message shape and re-exports the transport; there is
    # no dispatch layer, so these must be the same function.
    assert notify.send is notify_discord.send
    assert notify.configured is notify_discord.configured


def test_configured_reflects_the_env(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    assert notify.configured() is False
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", WEBHOOK)
    assert notify.configured() is True


def test_no_removed_backend_env_is_read_anywhere():
    """Regression: no NTFY_* or TELEGRAM_* env var may be read.

    Checks the uppercase env-var prefixes specifically; lowercase prose in the
    module docstring explains why those backends were dropped and is deliberate.
    """
    import inspect
    for mod in (notify, notify_discord):
        src = inspect.getsource(mod)
        assert "NTFY" not in src, mod.__name__
        assert "TELEGRAM_" not in src, mod.__name__
