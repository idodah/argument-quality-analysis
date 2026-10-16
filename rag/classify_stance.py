"""
Classify each argument in the CMV RAG corpus by how it handles antisemitic
tropes, using GPT-4o-mini zero-shot. Filter to high-confidence *refutations* —
arguments that debunk a trope on the factual record — for the RAG corpus.

The retrieval corpus is what the agent cites, so it must contain refutations
(the myth's origin, who spread it, the contradicting evidence) and NOT
political advocacy about Israel. `political_argument` exists as its own class
precisely so that material is identified and excluded rather than silently
mixed into the evidence store.

Inputs:
    data/cmv_israel_rag.parquet  (from rag.scrape_cmv_israel)

Outputs:
    data/cmv_israel_rag_classified.parquet   all rows + stance fields
    data/cmv_israel_rag_pro.parquet          filtered to stance=refutation, confidence>=0.8
    Hub split 'israel_2023_plus_rag_pro' on idodah/argument-quality-cmv

Each row gets:
    stance       ∈ {refutation, trope, political_argument, neutral, mixed}
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

VALID_STANCES = {"refutation", "trope", "political_argument", "neutral", "mixed"}
# Only refutations enter the evidence store the agent cites.
CORPUS_STANCE = "refutation"
PRO_CONFIDENCE_THRESHOLD = 0.8

SYSTEM_PROMPT = (
    "You are an expert at analyzing argumentation about antisemitism. Given an "
    "argument from a Reddit ChangeMyView thread, classify how it handles "
    "antisemitic tropes (blood libel, Jewish control of banking/media/government, "
    "Holocaust denial, dual loyalty, the Khazar myth, Great Replacement, the "
    "Protocols, collective blame of Jews as a group).\n\n"
    "Definitions:\n"
    "- refutation: debunks a trope on the factual or historical record — its origin, "
    "who propagated it, the evidence that contradicts it. This is the class we want.\n"
    "- trope: asserts a trope as true.\n"
    "- political_argument: argues about the Israeli government's policies, conduct, or "
    "legitimacy. This is political speech, NOT a trope and NOT a refutation of one — "
    "classify it here regardless of which side it takes.\n"
    "- neutral: presents facts or analysis without doing any of the above, or argues a "
    "tangential point.\n"
    "- mixed: meaningfully does more than one of the above.\n\n"
    "The key distinction: claims about JEWS AS A GROUP (their nature, loyalty, or "
    "secret coordinated power) are tropes; claims about a STATE'S CONDUCT are "
    "political_argument.\n\n"
    "Be careful with: quoted tropes (a speaker refuting a myth must quote it — that is "
    "'refutation', not 'trope'), sarcasm, hedged language. "
    "Judge the speaker's own position, not surface keywords.\n\n"
    "Respond with strict JSON: "
    '{"stance": "refutation|trope|political_argument|neutral|mixed", '
    '"confidence": <float 0..1>, "rationale": "<one sentence>"}'
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
        "Classify how the argument handles antisemitic tropes. Respond with JSON only."
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
            if stance not in VALID_STANCES:
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
    """Load the scraped parquet, classify, save all + refutation subsets, push."""
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

    pro_df = df[(df["stance"] == CORPUS_STANCE) & (df["confidence"] >= PRO_CONFIDENCE_THRESHOLD)].reset_index(drop=True)
    pro_df.to_parquet(OUT_PRO, index=False)
    print(f"\nFiltered to {len(pro_df)} trope refutations (confidence >= {PRO_CONFIDENCE_THRESHOLD}) -> {OUT_PRO}.")

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