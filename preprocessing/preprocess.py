"""
Preprocess argument quality pairs from both Webis-CMV-20 and winning-args-corpus.

Pipeline:
  1. Parse raw pairs from both sources
  2. Clean text, normalize topics, drop empty/duplicate pairs
  3. Keep the longest original_post per thread
  4. Save a local parquet copy and upload to the Hugging Face Hub (split='full')
"""

import os
from datetime import date as _date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from schemas import ArgumentPair

from .data_creation import build_webis_raw, build_winning_raw
from .text_utils import clean_text, normalize_topic, strip_edit_paragraphs

from datasets import Dataset, Features, Value

load_dotenv()

HF_REPO_ID = "idodah/argument-quality-cmv"
OUT_PARQUET = Path("./data/argument_quality_preprocessed.parquet")


def _enrich(pairs: list[ArgumentPair]) -> pd.DataFrame:
    """Normalize topics, clean/strip text, drop empty or identical pairs, dedup on
    (delta, nodelta), and coerce dates into a DataFrame."""
    rows = []
    for p in pairs:
        topic = normalize_topic(p.topic)
        original_post = strip_edit_paragraphs(clean_text(p.original_post))
        delta_argument = strip_edit_paragraphs(clean_text(p.delta_argument))
        nodelta_argument = strip_edit_paragraphs(clean_text(p.nodelta_argument))

        if not original_post or not delta_argument or not nodelta_argument:
            continue
        if delta_argument == nodelta_argument:
            continue

        rows.append(ArgumentPair(
            thread_id=p.thread_id,
            topic=topic,
            original_post=original_post,
            delta_argument=delta_argument,
            nodelta_argument=nodelta_argument,
            date=p.date,
        ))

    seen = set()
    deduped = []
    for r in rows:
        key = (r.delta_argument, r.nodelta_argument)
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    records = []
    for r in deduped:
        row = r.model_dump()
        d = row.get("date")
        if d is not None and not isinstance(d, _date):
            try:
                row["date"] = _date.fromisoformat(str(d))
            except (ValueError, TypeError):
                row["date"] = None
        records.append(row)
    return pd.DataFrame(records)


def upload_to_hub(df: pd.DataFrame, repo_id: str = HF_REPO_ID) -> None:
    """Push `df` to the Hugging Face Hub as split='full' with an explicit string/date schema."""

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN not set. Add it to your .env file.")

    features = Features({
        "thread_id": Value("large_string"),
        "topic": Value("large_string"),
        "original_post": Value("large_string"),
        "delta_argument": Value("large_string"),
        "nodelta_argument": Value("large_string"),
        "date": Value("date32"),
    })
    ds = Dataset.from_pandas(df, preserve_index=False, features=features)
    ds.push_to_hub(repo_id, token=token, split="full", private=False)
    print(f"Uploaded to https://huggingface.co/datasets/{repo_id}")


def _use_longest_post_per_thread(df: pd.DataFrame) -> pd.DataFrame:
    """Give every row of a thread the longest ``original_post`` seen for that thread
    (different pairs from one thread may carry differently-truncated copies)."""
    longest = (
        df.assign(_len=df["original_post"].str.len())
        .sort_values("_len", ascending=False)
        .drop_duplicates("thread_id")
        .set_index("thread_id")["original_post"]
    )
    df["original_post"] = df["thread_id"].map(longest)
    return df


def main() -> None:
    """Parse both sources, clean/dedup into the 'full' split, and upload it to the Hub."""
    if not os.environ.get("HF_TOKEN"):
        raise EnvironmentError("HF_TOKEN not set. Add it to your .env file.")

    print("Parsing Webis-CMV-20...")
    webis_rows = build_webis_raw()
    print(f"  {len(webis_rows)} raw pairs")

    print("Parsing winning-args-corpus...")
    winning_rows = build_winning_raw()
    print(f"  {len(winning_rows)} raw pairs")

    all_rows = webis_rows + winning_rows
    print(f"\nTotal raw pairs: {len(all_rows)}")

    print("\nCleaning")
    df = _enrich(all_rows)

    if df.empty:
        raise RuntimeError("No rows survived filtering — aborting before upload.")

    df = _use_longest_post_per_thread(df)

    print(f"\nFinal dataset: {len(df)} rows, {df['thread_id'].nunique()} unique threads")
    print(f"\nNull counts:\n{df.isnull().sum().to_string()}")

    print(f"\nSaving parquet to {OUT_PARQUET}...")
    df.to_parquet(OUT_PARQUET, index=False)
    print("Done.")

    upload_to_hub(df)


if __name__ == "__main__":
    main()