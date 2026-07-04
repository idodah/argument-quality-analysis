"""
Classify each argument in the CMV-Israel RAG corpus as pro-Israel / anti-Israel
/ neutral / mixed using GPT-4o-mini zero-shot. Filter to high-confidence
pro-Israel arguments for the RAG corpus.

Inputs:
    data/cmv_israel_rag.parquet  (from rag.scrape_cmv_israel)

Outputs:
    data/cmv_israel_rag_classified.parquet   all rows + stance fields
    data/cmv_israel_rag_pro.parquet          filtered to stance=pro_israel, confidence>=0.8
    Hub split 'israel_2023_plus_rag_pro' on idodah/argument-quality-cmv

Each row gets:
    stance       ∈ {pro_israel, anti_israel, neutral, mixed}
    confidence   ∈ [0, 1]
    rationale    one-sentence explanation from the classifier

Usage:
    uv run python -m rag.classify_stance
    uv run python -m rag.classify_stance --no-hub
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from datasets import Dataset, Features, Value
from dotenv import load_dotenv
from openai import OpenAI

from models.data import HF_REPO_ID

INPUT_PARQUET = Path("data/cmv_israel_rag.parquet")
OUT_CLASSIFIED = Path("data/cmv_israel_rag_classified.parquet")
OUT_PRO = Path("data/cmv_israel_rag_pro.parquet")
HUB_SPLIT = "israel_2023_plus_rag_pro"

MODEL_ID = "gpt-4o-mini"
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
ARG_TRUNC_CHARS = 6000  # keep prompt within model context cheaply

PRO_CONFIDENCE_THRESHOLD = 0.8

SYSTEM_PROMPT = (
    "You are an expert at analyzing political argumentation about the Israel-Palestine conflict. "
    "Given an argument from a Reddit ChangeMyView thread, classify its stance toward Israel.\n\n"
    "Definitions:\n"
    "- pro_israel: defends Israel's actions, security needs, or right to exist; criticizes Hamas or anti-Israel framing.\n"
    "- anti_israel: criticizes Israel's actions or legitimacy; uses framing like apartheid, genocide, occupation, colonialism.\n"
    "- neutral: presents facts or analysis without taking a side, or argues a tangential point.\n"
    "- mixed: takes positions that meaningfully favor both sides, or rejects the framing of both sides.\n\n"
    "Be careful with: quoted rebuttals (the speaker doesn't endorse the quoted view), sarcasm, hedged language. "
    "Judge the speaker's own position, not surface keywords.\n\n"
    "Respond with strict JSON: "
    '{"stance": "pro_israel|anti_israel|neutral|mixed", "confidence": <float 0..1>, "rationale": "<one sentence>"}'
)

OUT_FEATURES = Features({
    "thread_id": Value("large_string"),
    "comment_id": Value("large_string"),
    "topic": Value("large_string"),
    "original_post": Value("large_string"),
    "argument": Value("large_string"),
    "score": Value("int32"),
    "date": Value("date32"),
    "stance": Value("string"),
    "confidence": Value("float32"),
    "rationale": Value("large_string"),
})


def _classify_one(client: OpenAI, topic: str, argument: str) -> dict:
    """Zero-shot classify one argument's stance; retries, returns neutral on failure."""
    user = (
        f"### Thread topic\n{topic}\n\n"
        f"### Argument\n{argument[:ARG_TRUNC_CHARS]}\n\n"
        "Classify the argument's stance toward Israel. Respond with JSON only."
    )
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
            stance = data.get("stance", "neutral")
            if stance not in {"pro_israel", "anti_israel", "neutral", "mixed"}:
                stance = "neutral"
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            rationale = str(data.get("rationale", ""))[:500]
            return {"stance": stance, "confidence": confidence, "rationale": rationale}
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"  [error after {MAX_RETRIES} retries] {e}")
                return {"stance": "neutral", "confidence": 0.0, "rationale": f"classifier error: {e}"}
            time.sleep(RETRY_BACKOFF * (2 ** attempt))
    return {"stance": "neutral", "confidence": 0.0, "rationale": "unreachable"}


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Classify every row, adding stance / confidence / rationale columns."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    results = []
    for i, row in enumerate(df.itertuples(index=False), 1):
        results.append(_classify_one(client, row.topic, row.argument))
        if i % 25 == 0 or i == len(df):
            print(f"  classified {i}/{len(df)}")
    df = df.copy()
    df["stance"] = [r["stance"] for r in results]
    df["confidence"] = [r["confidence"] for r in results]
    df["rationale"] = [r["rationale"] for r in results]
    return df


def push_to_hub(df: pd.DataFrame) -> None:
    """Upload the classified rows to the Hub split (no-op if HF_TOKEN is unset)."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN not set; skipping Hub upload.")
        return
    hub_df = df.copy()
    hub_df["date"] = pd.to_datetime(hub_df["date"]).dt.date
    ds = Dataset.from_pandas(hub_df, preserve_index=False, features=OUT_FEATURES)
    ds.push_to_hub(HF_REPO_ID, token=hf_token, split=HUB_SPLIT, private=False)
    print(f"Pushed {len(ds)} rows to https://huggingface.co/datasets/{HF_REPO_ID} (split='{HUB_SPLIT}').")


def main(push: bool = True) -> None:
    """Load the scraped parquet, classify, save all + pro-Israel subsets, push."""
    load_dotenv()
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"{INPUT_PARQUET} not found. Run rag.scrape_cmv_israel first.")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df)} delta arguments from {INPUT_PARQUET}.")

    df = classify(df)
    df.to_parquet(OUT_CLASSIFIED, index=False)
    print(f"\nClassified all rows -> {OUT_CLASSIFIED}.")
    print("\nStance distribution:")
    print(df["stance"].value_counts().to_string())
    print("\nConfidence quantiles:")
    print(df["confidence"].describe().to_string())

    pro_df = df[(df["stance"] == "pro_israel") & (df["confidence"] >= PRO_CONFIDENCE_THRESHOLD)].reset_index(drop=True)
    pro_df.to_parquet(OUT_PRO, index=False)
    print(f"\nFiltered to {len(pro_df)} pro-Israel arguments (confidence >= {PRO_CONFIDENCE_THRESHOLD}) -> {OUT_PRO}.")

    if push:
        push_to_hub(pro_df)


def cli() -> None:
    """Parse CLI args and run the classification pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-hub", action="store_true", help="Skip pushing to Hugging Face Hub")
    args = parser.parse_args()
    main(push=not args.no_hub)


if __name__ == "__main__":
    cli()