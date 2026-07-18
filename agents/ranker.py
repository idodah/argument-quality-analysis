"""Lazy wrapper around the fine-tuned Qwen pairwise ranker.

Two backends, selected by environment:

  - **SageMaker** (deployed): set `SAGEMAKER_RANKER_ENDPOINT` to the async
    endpoint name. `score_pair` POSTs the request to S3 and calls
    `invoke_endpoint_async`, then polls S3 for the result.
  - **In-process** (local / offline): set `RANKER_PATH` to a checkpoint dir or
    HF repo id. The QLoRA-fine-tuned Qwen is loaded once via transformers/peft
    and scored in-process. This is the path the offline tests and CLI use.

Both backends expose the same `score_pair(topic, post, arg_a, arg_b) -> dict`,
returning `{score_a, score_b, winner, prob_a_better}`, so the graph never knows
which one is active.
"""

from __future__ import annotations

import json
import os
import time
import uuid

_RANKER: "Ranker | None" = None

# How long to wait for an async SageMaker result before giving up, and how often
# to poll S3 for it. A scale-to-zero endpoint may cold-start, so the timeout is
# generous.
_ASYNC_TIMEOUT_S = float(os.environ.get("SAGEMAKER_RANKER_TIMEOUT_S", "600"))
_ASYNC_POLL_S = float(os.environ.get("SAGEMAKER_RANKER_POLL_S", "2"))


class Ranker:
    """Common interface; subclasses implement `score_pair`."""

    def score_pair(self, topic: str, post: str, arg_a: str, arg_b: str) -> dict:
        raise NotImplementedError


class DisabledRanker(Ranker):
    """No-op ranker for deployments without the Qwen endpoint (RANKER_DISABLED=1).

    Picks side A unconditionally (a fixed, cheap choice). The A-vs-B elimination
    is the only place the ranker is consulted, and the downstream stance gate
    overrides a bad pick by regenerating — so the graph stays correct, it just
    loses the persuasiveness tie-break. Lets the CPU-only image run the full
    Bedrock pipeline with no GPU/SageMaker cost.
    """

    def score_pair(self, topic: str, post: str, arg_a: str, arg_b: str) -> dict:
        print("[ranker] DISABLED (RANKER_DISABLED) — defaulting to side A, no scoring.")
        return {"score_a": 0.0, "score_b": 0.0, "winner": "A", "prob_a_better": 0.5}


class LocalRanker(Ranker):
    """Loads the QLoRA-fine-tuned Qwen ranker in-process and scores locally."""

    def __init__(self, model_path: str | None = None):
        from models import qwen

        path = model_path or os.environ.get("RANKER_PATH")
        if not path:
            raise EnvironmentError(
                "Set RANKER_PATH to a local checkpoint dir or HF repo id "
                "for the fine-tuned ranker (e.g. ./checkpoints/qwen_qlora_rank/final), "
                "or SAGEMAKER_RANKER_ENDPOINT to use the deployed endpoint."
            )
        print(f"[ranker] loading in-process from {path} ...")
        self.model, self.tokenizer = qwen.load_model(path)
        self._score_pair = qwen.score_pair

    def score_pair(self, topic: str, post: str, arg_a: str, arg_b: str) -> dict:
        return self._score_pair(self.model, self.tokenizer, topic, post, arg_a, arg_b)


class SageMakerRanker(Ranker):
    """Calls the Qwen ranker on a SageMaker **async** inference endpoint.

    Async (vs. real-time) lets the endpoint scale to zero between the harvester's
    hourly runs, so there is no idle-GPU bill; the request/response are exchanged
    via S3 and we poll for the result.
    """

    def __init__(self, endpoint: str | None = None, input_bucket: str | None = None):
        import boto3

        self.endpoint = endpoint or os.environ["SAGEMAKER_RANKER_ENDPOINT"]
        # S3 location async invocations upload their request payloads to.
        self.input_bucket = input_bucket or os.environ["SAGEMAKER_RANKER_INPUT_BUCKET"]
        self.input_prefix = os.environ.get("SAGEMAKER_RANKER_INPUT_PREFIX", "ranker-requests")
        region = os.environ.get("SAGEMAKER_REGION") or os.environ.get("AWS_REGION")
        self._sm = boto3.client("sagemaker-runtime", region_name=region)
        self._s3 = boto3.client("s3", region_name=region)
        print(f"[ranker] using SageMaker async endpoint {self.endpoint!r}")

    def _put_request(self, payload: dict) -> str:
        key = f"{self.input_prefix}/{uuid.uuid4().hex}.json"
        self._s3.put_object(
            Bucket=self.input_bucket, Key=key,
            Body=json.dumps(payload).encode("utf-8"),
            ContentType="application/json",
        )
        return f"s3://{self.input_bucket}/{key}"

    def _read_s3_json(self, s3_uri: str):
        """Return the parsed JSON at `s3_uri`, or None if it doesn't exist yet."""
        bucket, _, key = s3_uri.replace("s3://", "").partition("/")
        try:
            obj = self._s3.get_object(Bucket=bucket, Key=key)
            return json.loads(obj["Body"].read())
        except self._s3.exceptions.NoSuchKey:
            return None

    def _wait_for_output(self, output_s3_uri: str, failure_s3_uri: str | None) -> dict:
        """Poll for the async result. On success SageMaker writes `output_s3_uri`;
        on inference failure it writes `failure_s3_uri` and `output_s3_uri` never
        appears — so we must watch BOTH, or a failed call would hang until the
        timeout and then raise a misleading 'not ready' error."""
        deadline = time.time() + _ASYNC_TIMEOUT_S
        while True:
            result = self._read_s3_json(output_s3_uri)
            if result is not None:
                return result
            if failure_s3_uri:
                failure = self._read_s3_json(failure_s3_uri)
                if failure is not None:
                    raise RuntimeError(f"SageMaker async inference failed: {failure}")
            if time.time() >= deadline:
                raise TimeoutError(
                    f"SageMaker async result not ready after {_ASYNC_TIMEOUT_S}s "
                    f"({output_s3_uri})"
                )
            time.sleep(_ASYNC_POLL_S)

    def score_pair(self, topic: str, post: str, arg_a: str, arg_b: str) -> dict:
        payload = {"topic": topic, "post": post, "arg_a": arg_a, "arg_b": arg_b}
        input_uri = self._put_request(payload)
        resp = self._sm.invoke_endpoint_async(
            EndpointName=self.endpoint,
            InputLocation=input_uri,
            ContentType="application/json",
        )
        result = self._wait_for_output(resp["OutputLocation"], resp.get("FailureLocation"))
        # The handler returns the same shape as models.qwen.score_pair.
        return {
            "score_a": float(result["score_a"]),
            "score_b": float(result["score_b"]),
            "winner": result["winner"],
            "prob_a_better": float(result["prob_a_better"]),
        }


def get_ranker() -> Ranker:
    """Return the singleton ranker, choosing the backend from the environment:
    disabled if `RANKER_DISABLED` is truthy (no GPU/endpoint — defaults to A),
    SageMaker if `SAGEMAKER_RANKER_ENDPOINT` is set, else the in-process load."""
    global _RANKER
    if _RANKER is None:
        if os.environ.get("RANKER_DISABLED", "").lower() in ("1", "true", "yes"):
            _RANKER = DisabledRanker()
        elif os.environ.get("SAGEMAKER_RANKER_ENDPOINT"):
            _RANKER = SageMakerRanker()
        else:
            _RANKER = LocalRanker()
    return _RANKER
