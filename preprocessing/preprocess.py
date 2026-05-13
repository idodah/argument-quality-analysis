"""
Preprocess argument quality pairs from both Webis-CMV-20 and winning-args-corpus.

Pipeline:
  1. Parse raw pairs from both sources
  2. Clean text and count tokens
  3. Compute OpenAI embeddings and cosine similarities
  4. Upload to Hugging Face Hub and save a local Excel copy
"""

import os
import re
from datetime import date as _date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from schemas import ArgumentPair

from .data_creation import build_webis_raw, build_winning_raw
from .text_utils import clean_text #, count_tokens

load_dotenv()

HF_REPO_ID = "idodah/argument-quality-cmv"
OUT_PARQUET = Path("./data/argument_quality_preprocessed.parquet")


_EDIT_PARAGRAPH_RE = re.compile(r"^\s*edit\b.*?(?=\n\s*\n|\Z)", re.IGNORECASE | re.DOTALL | re.MULTILINE)


def _strip_edit_paragraphs(text: str) -> str:
    text = _EDIT_PARAGRAPH_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _enrich(pairs: list[ArgumentPair]) -> pd.DataFrame:
    rows = []
    for p in pairs:
        topic = re.sub(r"^CMV:\s*", "", p.topic, flags=re.IGNORECASE).strip()
        topic = re.sub(r"[.\s]*cmv\s*\.?\s*$", "", topic, flags=re.IGNORECASE).strip()
        original_post = _strip_edit_paragraphs(clean_text(p.original_post))
        delta_argument = _strip_edit_paragraphs(clean_text(p.delta_argument))
        nodelta_argument = _strip_edit_paragraphs(clean_text(p.nodelta_argument))

        if not original_post or not delta_argument or not nodelta_argument:
            continue
        if delta_argument == nodelta_argument:
            continue

        # delta_tokens = count_tokens(delta_argument)
        # nodelta_tokens = count_tokens(nodelta_argument)

        # if delta_tokens < 20 or nodelta_tokens < 20:
        #     continue

        rows.append(ArgumentPair(
            thread_id= p.thread_id,
            topic= topic,
            original_post= original_post,
            delta_argument= delta_argument,
            nodelta_argument= nodelta_argument,
            date= p.date
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
    from datasets import Dataset, Features, Value

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


if __name__ == "__main__":
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

    longest_per_thread = (
        df.assign(_len=df["original_post"].str.len())
        .sort_values("_len", ascending=False)
        .drop_duplicates("thread_id")
        .set_index("thread_id")["original_post"]
    )
    df["original_post"] = df["thread_id"].map(longest_per_thread)

    print(f"\nFinal dataset: {len(df)} rows, {df['thread_id'].nunique()} unique threads")
    print(f"\nNull counts:\n{df.isnull().sum().to_string()}")

    print(f"\nSaving parquet to {OUT_PARQUET}...")
    df.to_parquet(OUT_PARQUET, index=False)
    print("Done.")

    upload_to_hub(df)