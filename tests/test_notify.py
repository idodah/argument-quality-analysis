"""Offline tests for the ntfy notifier backend.

The bug these guard against, across four backend changes, is a long argument
being silently cut. The ORIGINAL ntfy code chunked at 4000 bytes into several
independent pushes that arrived unordered and mid-sentence. This version sends
one request and lets ntfy store an oversized body as a `.txt` attachment, so
the regression to protect is: one request, full text, never split.

No network: `requests.post` is stubbed and the calls are inspected.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harvester import notify, notify_ntfy


@pytest.fixture
def ntfy_env(monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "cmv-test-topic")
    monkeypatch.delenv("NTFY_SERVER", raising=False)
    monkeypatch.delenv("NTFY_TOKEN", raising=False)


def _ok():
    resp = mock.Mock()
    resp.status_code = 200
    resp.text = ""
    return resp


def _sent_body(post):
    return post.call_args.kwargs["data"].decode("utf-8")


# --------------------------------------------------------------------------- #
# the regression that caused the original complaint
# --------------------------------------------------------------------------- #
def test_long_body_is_sent_in_one_request(ntfy_env):
    body = "x" * (notify_ntfy.INLINE_LIMIT * 3)
    with mock.patch.object(notify_ntfy.requests, "post", return_value=_ok()) as post:
        notify_ntfy.send(body)
    assert post.call_count == 1, "must not split into multiple pushes"


def test_long_body_is_sent_completely_intact(ntfy_env):
    body = "y" * (notify_ntfy.INLINE_LIMIT * 3)
    with mock.patch.object(notify_ntfy.requests, "post", return_value=_ok()) as post:
        notify_ntfy.send(body)
    assert _sent_body(post) == body


def test_short_body_is_unchanged(ntfy_env):
    with mock.patch.object(notify_ntfy.requests, "post", return_value=_ok()) as post:
        notify_ntfy.send("a short refutation")
    assert _sent_body(post) == "a short refutation"


# --------------------------------------------------------------------------- #
# url, headers, auth
# --------------------------------------------------------------------------- #
def test_posts_to_the_configured_topic(ntfy_env):
    with mock.patch.object(notify_ntfy.requests, "post", return_value=_ok()) as post:
        notify_ntfy.send("hi")
    assert post.call_args.args[0] == "https://ntfy.sh/cmv-test-topic"


def test_custom_server_is_used_and_trailing_slash_stripped(ntfy_env, monkeypatch):
    monkeypatch.setenv("NTFY_SERVER", "https://ntfy.example.com/")
    with mock.patch.object(notify_ntfy.requests, "post", return_value=_ok()) as post:
        notify_ntfy.send("hi")
    assert post.call_args.args[0] == "https://ntfy.example.com/cmv-test-topic"


def test_click_url_becomes_click_and_action_headers(ntfy_env):
    with mock.patch.object(notify_ntfy.requests, "post", return_value=_ok()) as post:
        notify_ntfy.send("hi", click_url="https://reddit.com/r/x/1")
    headers = post.call_args.kwargs["headers"]
    assert headers["Click"] == "https://reddit.com/r/x/1"
    assert "Open post" in headers["Actions"]


def test_non_http_click_url_is_dropped(ntfy_env):
    with mock.patch.object(notify_ntfy.requests, "post", return_value=_ok()) as post:
        notify_ntfy.send("hi", click_url="javascript:alert(1)")
    headers = post.call_args.kwargs["headers"]
    assert "Click" not in headers and "Actions" not in headers


def test_token_becomes_bearer_auth(ntfy_env, monkeypatch):
    monkeypatch.setenv("NTFY_TOKEN", "tk_secret")
    with mock.patch.object(notify_ntfy.requests, "post", return_value=_ok()) as post:
        notify_ntfy.send("hi")
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer tk_secret"


def test_no_auth_header_without_a_token(ntfy_env):
    with mock.patch.object(notify_ntfy.requests, "post", return_value=_ok()) as post:
        notify_ntfy.send("hi")
    assert "Authorization" not in post.call_args.kwargs["headers"]


def test_unicode_body_is_utf8_encoded(ntfy_env):
    with mock.patch.object(notify_ntfy.requests, "post", return_value=_ok()) as post:
        notify_ntfy.send("refutation — with an em dash and “quotes”")
    assert _sent_body(post) == "refutation — with an em dash and “quotes”"


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
def test_missing_topic_raises_with_setup_help(monkeypatch):
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    with pytest.raises(RuntimeError, match="NTFY_TOPIC"):
        notify_ntfy.send("hi")


def test_non_200_raises(ntfy_env):
    bad = mock.Mock()
    bad.status_code = 403
    bad.text = "forbidden"
    with mock.patch.object(notify_ntfy.requests, "post", return_value=bad):
        with pytest.raises(RuntimeError, match="403"):
            notify_ntfy.send("hi")


def test_connection_error_is_wrapped(ntfy_env):
    with mock.patch.object(notify_ntfy.requests, "post",
                           side_effect=requests.ConnectionError("boom")):
        with pytest.raises(RuntimeError, match="ntfy send failed"):
            notify_ntfy.send("hi")


# --------------------------------------------------------------------------- #
# notify.py re-exports the ntfy transport
# --------------------------------------------------------------------------- #
def test_notify_send_is_the_ntfy_send():
    assert notify.send is notify_ntfy.send
    assert notify.configured is notify_ntfy.configured


def test_configured_reflects_the_env(monkeypatch):
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    assert notify.configured() is False
    monkeypatch.setenv("NTFY_TOPIC", "t")
    assert notify.configured() is True


def test_no_removed_backend_env_is_read_anywhere():
    """Regression: no TELEGRAM_*, DISCORD_* or SMTP_* env var may be read."""
    import inspect
    for mod in (notify, notify_ntfy):
        src = inspect.getsource(mod)
        for prefix in ("TELEGRAM_", "DISCORD_", "SMTP_", "EMAIL_"):
            assert prefix not in src, f"{mod.__name__} references {prefix}"
