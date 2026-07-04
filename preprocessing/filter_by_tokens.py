"""
Load the 'filtered' split from the HF hub, drop rows where delta_argument or
nodelta_argument has fewer than MIN_TOKENS or more than MAX_TOKENS tokens
(per the ranker model's tokenizer), and push the result to a new split.
"""

import os

import pandas as pd
from datasets import Dataset, Features, Value, load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer

from models.data import HF_REPO_ID
from models.qwen import MODEL_ID

INPUT_SPLIT = "filtered"
OUTPUT_SPLIT = "filtered_v2"
OUTPUT_EXCEL = "data/filtered_v2.xlsx"
MIN_TOKENS = 20
MAX_TOKENS = 2000
MAX_ORIGINAL_POST_TOKENS = 2000
MAX_LENGTH_RATIO = 3.0

FEATURES = Features({
    "thread_id": Value("large_string"),
    "topic": Value("large_string"),
    "original_post": Value("large_string"),
    "delta_argument": Value("large_string"),
    "nodelta_argument": Value("large_string"),
    "date": Value("date32"),
})


def token_length(tokenizer, text) -> int:
    """Number of tokens in ``text`` under the ranker's tokenizer (0 if empty/None)."""
    if text is None or not str(text).strip():
        return 0
    return len(tokenizer.encode(str(text), add_special_tokens=False))


def add_token_counts(df: pd.DataFrame, tokenizer) -> list[str]:
    """Add a ``{field}_n_tokens`` column for each text field present; return their names."""
    token_cols = []
    for field in ["topic", "original_post", "delta_argument", "nodelta_argument"]:
        if field not in df.columns:
            continue
        col = f"{field}_n_tokens"
        df[col] = [token_length(tokenizer, t) for t in df[field].values]
        token_cols.append(col)
    return token_cols


def apply_token_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Drop pairs whose arguments fall outside [MIN_TOKENS, MAX_TOKENS], whose post
    exceeds MAX_ORIGINAL_POST_TOKENS, or whose two arguments differ in length by
    more than MAX_LENGTH_RATIO — printing how many rows each step keeps."""
    before = len(df)
    mask = (
        df["delta_argument_n_tokens"].between(MIN_TOKENS, MAX_TOKENS)
        & df["nodelta_argument_n_tokens"].between(MIN_TOKENS, MAX_TOKENS)
    )
    df = df[mask].reset_index(drop=True)
    after_len = len(df)
    print(f"Kept {after_len}/{before} rows after filtering arguments to [{MIN_TOKENS}, {MAX_TOKENS}] tokens.")

    df = df[df["original_post_n_tokens"] <= MAX_ORIGINAL_POST_TOKENS].reset_index(drop=True)
    after_post = len(df)
    print(f"Kept {after_post}/{after_len} rows after filtering original_post to <= {MAX_ORIGINAL_POST_TOKENS} tokens.")

    longer = df[["delta_argument_n_tokens", "nodelta_argument_n_tokens"]].max(axis=1)
    shorter = df[["delta_argument_n_tokens", "nodelta_argument_n_tokens"]].min(axis=1)
    df = df[(longer / shorter) <= MAX_LENGTH_RATIO].reset_index(drop=True)
    print(f"Kept {len(df)}/{after_post} rows after length-ratio filter (<= {MAX_LENGTH_RATIO}x).")
    return df


def main() -> None:
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN not set. Add it to your .env file.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)

    df = load_dataset(HF_REPO_ID, split=INPUT_SPLIT).to_pandas()
    print(f"Loaded {len(df)} rows from split '{INPUT_SPLIT}'.")

    token_cols = add_token_counts(df, tokenizer)
    print("\nToken length stats:")
    print(df[token_cols].describe().to_string())

    df = apply_token_filters(df)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"Saved filtered rows to {OUTPUT_EXCEL}.")

    df_hub = df.drop(columns=token_cols)
    out_ds = Dataset.from_pandas(df_hub, preserve_index=False, features=FEATURES)
    out_ds.push_to_hub(HF_REPO_ID, token=hf_token, split=OUTPUT_SPLIT, private=False)
    print(f"Pushed {len(df)} rows to https://huggingface.co/datasets/{HF_REPO_ID} (split='{OUTPUT_SPLIT}').")


if __name__ == "__main__":
    main()
