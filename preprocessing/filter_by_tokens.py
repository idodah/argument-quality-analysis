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
    if text is None or not str(text).strip():
        return 0
    return len(tokenizer.encode(str(text), add_special_tokens=False))


def main() -> None:
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN not set. Add it to your .env file.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)

    ds = load_dataset(HF_REPO_ID, split=INPUT_SPLIT)
    df = ds.to_pandas()
    before = len(df)
    print(f"Loaded {before} rows from split '{INPUT_SPLIT}'.")

    token_cols = []
    for field in ["topic", "original_post", "delta_argument", "nodelta_argument"]:
        if field not in df.columns:
            continue
        col = f"{field}_n_tokens"
        df[col] = [token_length(tokenizer, t) for t in df[field].values]
        token_cols.append(col)

    print("\nToken length stats:")
    print(df[token_cols].describe().to_string())

    mask = (
        df["delta_argument_n_tokens"].between(MIN_TOKENS, MAX_TOKENS)
        & df["nodelta_argument_n_tokens"].between(MIN_TOKENS, MAX_TOKENS)
    )
    df = df[mask].reset_index(drop=True)
    after_len = len(df)
    print(f"Kept {after_len}/{before} rows after filtering arguments to [{MIN_TOKENS}, {MAX_TOKENS}] tokens.")

    post_mask = df["original_post_n_tokens"] <= MAX_ORIGINAL_POST_TOKENS
    df = df[post_mask].reset_index(drop=True)
    after_post = len(df)
    print(f"Kept {after_post}/{after_len} rows after filtering original_post to <= {MAX_ORIGINAL_POST_TOKENS} tokens.")

    longer = df[["delta_argument_n_tokens", "nodelta_argument_n_tokens"]].max(axis=1)
    shorter = df[["delta_argument_n_tokens", "nodelta_argument_n_tokens"]].min(axis=1)
    ratio_mask = (longer / shorter) <= MAX_LENGTH_RATIO
    df = df[ratio_mask].reset_index(drop=True)
    after = len(df)
    print(f"Kept {after}/{after_post} rows after length-ratio filter (<= {MAX_LENGTH_RATIO}x).")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"Saved filtered rows to {OUTPUT_EXCEL}.")

    df_hub = df.drop(columns=token_cols)
    out_ds = Dataset.from_pandas(df_hub, preserve_index=False, features=FEATURES)
    out_ds.push_to_hub(HF_REPO_ID, token=hf_token, split=OUTPUT_SPLIT, private=False)
    print(f"Pushed {after} rows to https://huggingface.co/datasets/{HF_REPO_ID} (split='{OUTPUT_SPLIT}').")


if __name__ == "__main__":
    main()
