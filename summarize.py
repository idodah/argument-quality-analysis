"""
Summarize original_post only, using openai/gpt-oss-20b via Together AI batch inference.

original_post is deduplicated by thread_id — each unique original
is sent only once, then the summary is joined back onto all rows.
Short originals (under 100 words) are returned as-is by the model.

Usage:
    uv run python summarize.py

Outputs:
    arguments_data/pairs_summarized.xlsx  (local backup)
    Pushes updated dataset to idodah/argument-quality-cmv on Hugging Face Hub
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from together import Together

load_dotenv()

HF_REPO_ID = "idodah/argument-quality-cmv"
MODEL = "openai/gpt-oss-20b"
POLL_INTERVAL = 60   # seconds between status checks
BATCH_SIZE = 200     # max texts per batch job

SYSTEM_PROMPT = (
    "Rewrite the following argumentative text as a concise first-person summary, "
    "as if you are the original author restating your own argument. "
    "Preserve the key claims, reasoning, persuasive intent, and the author's tone and style. "
    "Use 'I' and 'my' throughout. Do not refer to 'the author'. "
    "Output only the summary, nothing else."
)

_REFUSAL_PREFIXES = (
    "i'm sorry",
    "i am sorry",
    "i cannot",
    "i can't",
    "as an ai",
)

TERMINAL_STATUSES = {"COMPLETED", "FAILED", "EXPIRED", "CANCELLED"}

def _is_valid_summary(summary: str) -> bool:
    if not summary:
        return False
    lower = summary.lower()
    if any(lower.startswith(p) for p in _REFUSAL_PREFIXES):
        return False
    return True


def _build_jsonl(thread_ids: list[str], texts: list[str]) -> str:
    lines = []
    for tid, text in zip(thread_ids, texts):
        record = {
            "custom_id": tid,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.0
            },
        }
        lines.append(json.dumps(record))
    return "\n".join(lines).encode("utf-8")


def _submit_batch(client: Together, jsonl_bytes: bytes, label: str) -> str:
    """Upload JSONL and submit a batch job. Returns batch job id."""
    print(f"  Uploading {label} input file ({len(jsonl_bytes) // 1024} KB)...")
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="wb") as tmp:
        tmp.write(jsonl_bytes)
        tmp_path = Path(tmp.name)
    try:
        uploaded = client.files.upload(file=tmp_path, purpose="batch-api", check=False)
    finally:
        tmp_path.unlink(missing_ok=True)

    print(f"  Submitting batch job for {label}...")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  Batch job id: {batch.job.id}")
    return batch.job.id


def _poll_until_done(client: Together, batch_id: str, label: str) -> str:
    """Poll until terminal status. Returns output_file_id."""
    while True:
        job = client.batches.retrieve(batch_id)
        status = job.status or "UNKNOWN"
        progress = job.progress or 0.0
        print(f"  [{label}] status={status}  progress={progress:.1f}%", flush=True)

        if status in TERMINAL_STATUSES:
            if status != "COMPLETED":
                raise RuntimeError(
                    f"Batch {batch_id} ended with status {status}: {job.error}"
                )
            return job.output_file_id

        time.sleep(POLL_INTERVAL)



def _fetch_results(client: Together, output_file_id: str, ids: list[str], originals: list[str]) -> list[str]:
    """Download output JSONL and return summaries aligned to input order.

    Falls back to the original text for empty or refused responses.
    """
    raw = client.files.content(output_file_id).read().decode()
    result_map: dict[str, str] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        custom_id = obj["custom_id"]
        summary = obj["response"]["body"]["choices"][0]["message"]["content"].strip()
        result_map[custom_id] = summary

    original_map = dict(zip(ids, originals))
    results = []
    n_fallbacks = 0
    for i in ids:
        summary = result_map.get(i, "")
        if _is_valid_summary(summary):
            results.append(summary)
        else:
            results.append(original_map[i])
            n_fallbacks += 1
    if n_fallbacks:
        print(f"  Fallback to original for {n_fallbacks}/{len(ids)} texts (empty/refused)")
    return results


def batch_summarize(texts: list[str], client: Together, label: str, test: bool = False) -> list[str]:
    """End-to-end: chunk into BATCH_SIZE jobs → upload → submit → poll → fetch."""
    all_summaries: list[str] = []
    chunks = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    if test:
        chunks = chunks[:1]
    n_chunks = len(chunks)

    for chunk_idx, chunk in enumerate(chunks):
        chunk_label = f"{label}_chunk{chunk_idx}"
        print(f"  Chunk {chunk_idx + 1}/{n_chunks} ({len(chunk)} texts)...")
        ids = [f"{chunk_label}_{i}" for i in range(len(chunk))]
        jsonl_bytes = _build_jsonl(ids, chunk)
        batch_id = _submit_batch(client, jsonl_bytes, chunk_label)
        output_file_id = _poll_until_done(client, batch_id, chunk_label)
        all_summaries.extend(_fetch_results(client, output_file_id, ids, chunk))
    return all_summaries

def main() -> None:
    token_hf = os.environ.get("HF_TOKEN")
    token_together = os.environ.get("TOGETHER_API_KEY")
    if not token_hf:
        raise EnvironmentError("HF_TOKEN not set in .env")
    if not token_together:
        raise EnvironmentError("TOGETHER_API_KEY not set in .env")

    client = Together(api_key=token_together)

    # ------------------------------------------------------------------
    # 1. Load data from Hugging Face
    # ------------------------------------------------------------------
    print(f"Loading dataset from {HF_REPO_ID}...")
    hf_ds = load_dataset(HF_REPO_ID, split="full", token=token_hf)
    df = hf_ds.to_pandas()
    print(f"Loaded {len(df)} rows, {df['thread_id'].nunique()} unique submissions.")
    
    # ------------------------------------------------------------------
    # 2. Summarize original_post (deduplicated by thread_id)
    # ------------------------------------------------------------------
    print("\nSummarizing original_post (deduplicated)...")
    originals = (
        df[["thread_id", "original_post"]]
        .drop_duplicates(subset="thread_id")
        .reset_index(drop=True)
    )
    originals["original_post_summary"] = batch_summarize(
        originals["original_post"].tolist(), client, label="original"
    )

    df = df.merge(
        originals[["thread_id", "original_post_summary"]],
        on="thread_id",
        how="left",
    )

    # ------------------------------------------------------------------
    # 3. Save locally
    # ------------------------------------------------------------------
    out_path = Path("./arguments_data/pairs_summarized.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\nSaved locally to {out_path}")

    # ------------------------------------------------------------------
    # 6. Push to Hugging Face Hub
    # ------------------------------------------------------------------
    print(f"\nUploading to {HF_REPO_ID} (split='summarized')...")
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds.push_to_hub(HF_REPO_ID, split="summarized", token=token_hf, private=False)
    print(f"Done. https://huggingface.co/datasets/{HF_REPO_ID}")



if __name__ == "__main__":
    main()