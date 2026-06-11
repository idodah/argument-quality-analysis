"""Offline tests for agents.ranker.SageMakerRanker's async polling.

No AWS: fake `_sm` / `_s3` clients are injected into an instance built via
__new__ (bypassing the boto3-creating __init__). Covers the success path, the
failure-path short-circuit (a regression guard for the bug where only
OutputLocation was polled, so a failed inference hung until timeout), and the
timeout itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import ranker as ranker_mod
from agents.ranker import SageMakerRanker


class _NoSuchKey(Exception):
    pass


class _FakeS3:
    """Minimal S3 stub: `objects` maps 'bucket/key' -> python obj (JSON body).
    Missing keys raise NoSuchKey, like boto3's get_object."""

    def __init__(self, objects: dict):
        self.objects = objects
        self.puts = []

        class _Exc:
            NoSuchKey = _NoSuchKey

        self.exceptions = _Exc()

    def get_object(self, Bucket, Key):
        full = f"{Bucket}/{Key}"
        if full not in self.objects:
            raise _NoSuchKey()
        body = json.dumps(self.objects[full]).encode()
        return {"Body": _Body(body)}

    def put_object(self, Bucket, Key, Body, ContentType):
        self.puts.append((Bucket, Key))


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data


class _FakeSM:
    def __init__(self, response: dict):
        self._response = response
        self.calls = []

    def invoke_endpoint_async(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def _make_ranker(sm, s3) -> SageMakerRanker:
    r = SageMakerRanker.__new__(SageMakerRanker)  # skip boto3-building __init__
    r.endpoint = "ep"
    r.input_bucket = "in-bucket"
    r.input_prefix = "ranker-requests"
    r._sm = sm
    r._s3 = s3
    return r


def test_score_pair_success(monkeypatch):
    out = {"score_a": 1.5, "score_b": 0.5, "winner": "A", "prob_a_better": 0.7}
    s3 = _FakeS3({"out-bucket/result.json": out})
    sm = _FakeSM({"OutputLocation": "s3://out-bucket/result.json"})
    r = _make_ranker(sm, s3)

    result = r.score_pair("t", "p", "a", "b")

    assert result == {"score_a": 1.5, "score_b": 0.5, "winner": "A", "prob_a_better": 0.7}
    # The request payload was uploaded before invoking.
    assert s3.puts and s3.puts[0][0] == "in-bucket"
    assert sm.calls[0]["EndpointName"] == "ep"


def test_score_pair_raises_on_failure_location(monkeypatch):
    # OutputLocation never appears; FailureLocation does -> fail fast, no hang.
    monkeypatch.setattr(ranker_mod, "_ASYNC_POLL_S", 0)
    s3 = _FakeS3({"out-bucket/fail.json": {"ErrorCode": "ModelError", "Message": "boom"}})
    sm = _FakeSM({
        "OutputLocation": "s3://out-bucket/result.json",   # missing in S3
        "FailureLocation": "s3://out-bucket/fail.json",     # present
    })
    r = _make_ranker(sm, s3)

    with pytest.raises(RuntimeError, match="inference failed"):
        r.score_pair("t", "p", "a", "b")


def test_score_pair_times_out(monkeypatch):
    # Neither output nor failure ever appears -> TimeoutError (not an infinite loop).
    monkeypatch.setattr(ranker_mod, "_ASYNC_POLL_S", 0)
    monkeypatch.setattr(ranker_mod, "_ASYNC_TIMEOUT_S", 0)  # deadline already passed
    s3 = _FakeS3({})
    sm = _FakeSM({"OutputLocation": "s3://out-bucket/missing.json"})
    r = _make_ranker(sm, s3)

    with pytest.raises(TimeoutError, match="not ready"):
        r.score_pair("t", "p", "a", "b")


def test_get_ranker_selects_sagemaker(monkeypatch):
    monkeypatch.setattr(ranker_mod, "_RANKER", None)
    monkeypatch.setenv("SAGEMAKER_RANKER_ENDPOINT", "ep")
    created = {}

    def fake_init(self):
        created["yes"] = True

    monkeypatch.setattr(SageMakerRanker, "__init__", lambda self: fake_init(self))
    r = ranker_mod.get_ranker()
    assert isinstance(r, SageMakerRanker)
    assert created.get("yes")
    monkeypatch.setattr(ranker_mod, "_RANKER", None)  # reset singleton


def test_get_ranker_defaults_to_local(monkeypatch):
    monkeypatch.setattr(ranker_mod, "_RANKER", None)
    monkeypatch.delenv("SAGEMAKER_RANKER_ENDPOINT", raising=False)
    created = {}
    monkeypatch.setattr(
        ranker_mod, "LocalRanker", lambda *a, **k: created.setdefault("local", object())
    )
    ranker_mod.get_ranker()
    assert "local" in created
    monkeypatch.setattr(ranker_mod, "_RANKER", None)
