"""
Summarize original_post only, using openai/gpt-oss-20b via Together AI direct inference.

original_post is deduplicated by thread_id — each unique original
is sent only once, then the summary is joined back onto all rows.

Usage:
    uv run python summarize_inference.py

Outputs:
    data/pairs_summarized_inference.xlsx     (local backup)
    data/pairs_summarized_inference.parquet  (safety net — always written)
    Pushes updated dataset to idodah/argument-quality-cmv on Hugging Face Hub
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from together import Together

load_dotenv()

HF_REPO_ID = "idodah/argument-quality-cmv"
MODEL = "openai/gpt-oss-20b"
MAX_WORKERS = 8

SYSTEM_PROMPT = (
    "Rewrite the following argumentative text as a concise first-person summary, "
    "as if you are the original author restating your own argument. "
    "Preserve the key claims, reasoning, persuasive intent, and the author's tone and style. "
    "Use 'I' and 'my' throughout. Do not refer to 'the author'. "
    "The topic of the argument is provided as context only — do not summarize it. "
    "The summary must be shorter than the original argument. "
    "Respond with a JSON object of the form {\"summary\": \"...\"} and nothing else."
)

USER_TEMPLATE = "Topic (context only, do not summarize): {topic}\n\nArgument:\n{text}"

RESPONSE_FORMAT = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    },
}

_REFUSAL_PREFIXES = (
    "i'm sorry",
    "i am sorry",
    "i cannot",
    "i can't",
    "as an ai",
    "i’m sorry"
)

def _is_valid_summary(summary: str) -> bool:
    if not summary:
        return False
    lower = summary.lower()
    if any(lower.startswith(p) for p in _REFUSAL_PREFIXES):
        return False
    return True


MAX_ATTEMPTS = 3


def _summarize_one(client: Together, text: str, topic: str) -> str | None:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(topic=topic, text=text)},
            ],
            temperature=0.0 if attempt == 1 else 0.7,
            reasoning_effort="low",
            response_format=RESPONSE_FORMAT,
        )
        content = (response.choices[0].message.content or "").strip()
        try:
            summary = json.loads(content).get("summary", "").strip()
        except json.JSONDecodeError:
            summary = ""
        if _is_valid_summary(summary):
            return summary
    return None


def summarize_texts(
    texts: list[str], topics: list[str], client: Together, max_workers: int = MAX_WORKERS
) -> tuple[list[str | None], list[int]]:
    n = len(texts)
    summaries: list[str | None] = [None] * n
    skipped_indices: list[int] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_summarize_one, client, text, topics[i]): i
            for i, text in enumerate(texts)
        }
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                summaries[i] = future.result()
            except Exception as e:
                print(f"  [skip] row {i}: {type(e).__name__}: {e}", flush=True)
                skipped_indices.append(i)
            completed += 1
            print(f"  {completed}/{n}...", flush=True)
            if completed%10==0:
                print(summaries[i])

    skipped_indices.sort()
    return summaries, skipped_indices


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
        df[["thread_id", "topic", "original_post"]]
        .drop_duplicates(subset="thread_id")
        .reset_index(drop=True)
    )
    summaries, skipped_indices = summarize_texts(
        originals["original_post"].tolist(),
        originals["topic"].tolist(),
        client,
    )
    originals["summary"] = summaries

    if skipped_indices:
        skipped = originals.iloc[skipped_indices].reset_index(drop=True)
        print(
            f"\n{len(skipped_indices)} rows skipped",
            flush=True,
        )
        for tid in skipped["thread_id"].tolist():
            print(f"  thread_id={tid}", flush=True)

    if "summary" in df.columns:
        df = df.drop(columns=["summary"])
    df = df.merge(
        originals[["thread_id", "summary"]],
        on="thread_id",
        how="left",
    )

    # Drop rows whose original_post could not be summarized after MAX_ATTEMPTS.
    before = len(df)
    df = df[df["summary"].notna()].reset_index(drop=True)
    print(f"\nDropped {before - len(df)} rows with no summary; {len(df)} remain.")

    # ------------------------------------------------------------------
    # 3. Save locally
    # ------------------------------------------------------------------
    # Always write parquet first as a safety net — the API spend is the
    # expensive part, and xlsx can fail on row count / cell length limits.
    parquet_path = Path("./data/pairs_summarized_inference.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"Saved locally to {parquet_path}")

    df.to_excel("./data/pairs_summarized_inference.xlsx",index=False)

    # ------------------------------------------------------------------
    # 4. Push to Hugging Face Hub
    # ------------------------------------------------------------------
    print(f"\nUploading to {HF_REPO_ID} (split='summarization')...")
    ds = Dataset.from_pandas(df, preserve_index=False)
    from huggingface_hub import HfApi
    api = HfApi(token=token_hf)
    files = api.list_repo_files(HF_REPO_ID, repo_type="dataset")
    stale = [f for f in files if f.startswith("data/summarization-") or f.startswith("summarization/")]
    for f in stale:
        api.delete_file(path_in_repo=f, repo_id=HF_REPO_ID, repo_type="dataset")
        print(f"  deleted {f}")
    ds.push_to_hub(HF_REPO_ID, split="summarization", token=token_hf, private=False)
    print(f"Done. https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    main()