"""Offline tests for the SageMaker ranker inference handler.

Exercises the four SageMaker entrypoints (model_fn/input_fn/predict_fn/output_fn)
with `models.qwen.load_model` / `score_pair` stubbed, so no model, GPU, or AWS is
needed. Verifies request parsing, field validation, the score_pair call wiring,
and JSON serialization — the glue that could drift from the contract.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The handler ships under infra/ (not importable as a package); load it by path.
_HANDLER_PATH = (
    Path(__file__).resolve().parent.parent / "infra" / "sagemaker" / "code" / "inference.py"
)


@pytest.fixture
def handler(monkeypatch):
    # Stub the heavy scoring path before importing the handler module so its
    # top-level `from models.qwen import ...` binds to our fakes.
    import models.qwen as qwen

    monkeypatch.setattr(qwen, "load_model", lambda model_dir: ("MODEL", "TOK"))

    def fake_score_pair(model, tok, topic, post, arg_a, arg_b):
        assert (model, tok) == ("MODEL", "TOK")
        # Deterministic, order-sensitive so the test can assert on it.
        return {
            "score_a": 1.5, "score_b": 0.5,
            "winner": "A", "prob_a_better": 0.73,
        }

    monkeypatch.setattr(qwen, "score_pair", fake_score_pair)

    spec = importlib.util.spec_from_file_location("sm_inference", _HANDLER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_model_fn_loads_bundle(handler):
    bundle = handler.model_fn("/opt/ml/model")
    assert bundle == {"model": "MODEL", "tokenizer": "TOK"}


def test_input_fn_parses_valid_json(handler):
    body = json.dumps({"topic": "t", "post": "p", "arg_a": "a", "arg_b": "b"})
    assert handler.input_fn(body) == {"topic": "t", "post": "p", "arg_a": "a", "arg_b": "b"}


def test_input_fn_accepts_bytes(handler):
    body = json.dumps({"topic": "t", "post": "p", "arg_a": "a", "arg_b": "b"}).encode()
    assert handler.input_fn(body)["topic"] == "t"


def test_input_fn_rejects_missing_fields(handler):
    with pytest.raises(ValueError, match="missing required fields"):
        handler.input_fn(json.dumps({"topic": "t", "post": "p"}))


def test_input_fn_rejects_wrong_content_type(handler):
    with pytest.raises(ValueError, match="Unsupported content type"):
        handler.input_fn("{}", content_type="text/plain")


def test_predict_fn_calls_score_pair(handler):
    bundle = handler.model_fn("/opt/ml/model")
    out = handler.predict_fn(
        {"topic": "t", "post": "p", "arg_a": "a", "arg_b": "b"}, bundle
    )
    assert out == {"score_a": 1.5, "score_b": 0.5, "winner": "A", "prob_a_better": 0.73}


def test_output_fn_serializes_json(handler):
    pred = {"score_a": 1.5, "score_b": 0.5, "winner": "A", "prob_a_better": 0.73}
    body, content_type = handler.output_fn(pred)
    assert content_type == "application/json"
    assert json.loads(body) == pred


def test_roundtrip_input_predict_output(handler):
    bundle = handler.model_fn("/opt/ml/model")
    parsed = handler.input_fn(
        json.dumps({"topic": "t", "post": "p", "arg_a": "a", "arg_b": "b"})
    )
    body, _ = handler.output_fn(handler.predict_fn(parsed, bundle))
    assert json.loads(body)["winner"] == "A"
