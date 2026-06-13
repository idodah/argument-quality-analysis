"""SageMaker inference handler for the Qwen pairwise ranker.

Deployed behind a SageMaker **async** endpoint. It reuses the *exact* scoring
path from `models.qwen` (`load_model` + `score_pair`), so the endpoint produces
byte-identical scores to the in-process `LocalRanker` — the only difference is
where the GPU forward pass runs. Keeping one scoring implementation (in
`models/qwen.py`) avoids drift between the local and deployed rankers.

SageMaker calls these four functions:
  - `model_fn(model_dir)`        -> load once at container start
  - `input_fn(body, type)`       -> parse the request JSON
  - `predict_fn(input, model)`   -> score the pair
  - `output_fn(pred, accept)`    -> serialize the result JSON

Request JSON  : {"topic", "post", "arg_a", "arg_b"}
Response JSON : {"score_a", "score_b", "winner", "prob_a_better"}  (== score_pair)
"""

from __future__ import annotations

import json

# The model artifact is unpacked into `model_dir`; the repo's `models/` package
# is shipped under `code/` (see package_model.sh) and is importable here.
from models.qwen import load_model, score_pair

_JSON = "application/json"


def model_fn(model_dir: str):
    """Load the fine-tuned ranker once. `model_dir` holds the unpacked artifact
    (adapter/, tokenizer/, score_head.pt) — the same layout `load_model` reads."""
    model, tokenizer = load_model(model_dir)
    return {"model": model, "tokenizer": tokenizer}


def input_fn(request_body: str | bytes, content_type: str = _JSON) -> dict:
    if content_type != _JSON:
        raise ValueError(f"Unsupported content type: {content_type!r} (expected {_JSON})")
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    payload = json.loads(request_body)
    missing = [k for k in ("topic", "post", "arg_a", "arg_b") if k not in payload]
    if missing:
        raise ValueError(f"Request missing required fields: {missing}")
    return payload


def predict_fn(payload: dict, bundle: dict) -> dict:
    return score_pair(
        bundle["model"], bundle["tokenizer"],
        payload["topic"], payload["post"], payload["arg_a"], payload["arg_b"],
    )


def output_fn(prediction: dict, accept: str = _JSON) -> tuple[str, str]:
    # score_pair already returns JSON-native floats/strings.
    return json.dumps(prediction), _JSON
