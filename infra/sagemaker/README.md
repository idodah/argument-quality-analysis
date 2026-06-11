# SageMaker — Qwen pairwise ranker (async endpoint)

Serves the QLoRA-fine-tuned Qwen ranker behind a SageMaker **async** inference
endpoint (scale-to-zero, so no idle-GPU bill between the harvester's hourly runs).
The endpoint is called by `agents.ranker.SageMakerRanker` when
`SAGEMAKER_RANKER_ENDPOINT` is set; otherwise the graph loads the ranker
in-process. Both paths score through the same `models.qwen` code, so results are
identical.

## Contents

| Path | Role |
|------|------|
| `code/inference.py` | The SageMaker handler. Reuses `models.qwen.load_model` + `score_pair` for byte-identical scoring. |
| `code/requirements.txt` | Extra deps layered on the PyTorch DLC (transformers/peft/bitsandbytes/…), pinned to `uv.lock`. |
| `package_model.sh` | Builds the `model.tar.gz` (artifact + `code/`) and optionally uploads it to S3. |

## Request / response contract

```
POST  {"topic": ..., "post": ..., "arg_a": ..., "arg_b": ...}
->    {"score_a": float, "score_b": float, "winner": "A"|"B", "prob_a_better": float}
```

Identical to `models.qwen.score_pair`, which both the endpoint and the local
ranker call.

## Deploy flow

1. **Train / locate the checkpoint** — a `RANKER_PATH` dir containing `adapter/`,
   `tokenizer/`, and `score_head.pt` (produced by `uv run python -m models.qwen`).

2. **Package + upload the artifact:**
   ```bash
   infra/sagemaker/package_model.sh ./checkpoints/qwen_qlora_rank/final \
     s3://<artifact-bucket>/ranker/model.tar.gz
   ```

3. **Create the model + async endpoint** — done by Terraform (next step of the
   deployment plan): a SageMaker Model pointing at the S3 `model.tar.gz` on a GPU
   DLC image, an endpoint config with an **async** inference block (S3 output
   path + scale-to-zero autoscaling to 0 instances), and the endpoint.

4. **Point the app at it** — set on the Fargate task:
   ```
   SAGEMAKER_RANKER_ENDPOINT=<endpoint-name>
   SAGEMAKER_RANKER_INPUT_BUCKET=<bucket for async request payloads>
   ```

## Base image note

The handler needs a torch that matches `uv.lock` (2.11.x) and a CUDA the GPU
instance supports. Pick a SageMaker PyTorch GPU DLC accordingly and do **not**
pin torch in `requirements.txt` (it ships in the base image; a mismatched pin
risks a CUDA break).

## Why reuse `models/qwen.py` instead of reimplementing

The scoring path (prompt build, length-trim, mean-pool, score head) is subtle and
must match training exactly. Shipping the two source modules (`qwen.py`,
`data.py`) into the artifact and calling them keeps **one** implementation, so the
endpoint can never silently drift from the local ranker. `datasets`/`pandas` get
imported transitively but are unused at request time — a possible future slim-down
if image size matters.
