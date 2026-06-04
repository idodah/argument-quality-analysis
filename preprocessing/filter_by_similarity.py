"""
Filter the raw dataset by semantic similarity (no summary step required).

Steps:
  1. Load the 'full' split from HF (output of preprocess.py).
  2. Embed original_post / delta_argument / nodelta_argument once each
     (deduplicated to save API calls).
  3. Compute three cosine-similarity columns:
       - sim_orig_delta      (post  <-> delta_argument)
       - sim_orig_nodelta    (post  <-> nodelta_argument)
       - sim_delta_nodelta   (delta <-> nodelta_argument)
  4. Save the full df WITH similarity columns to Excel (local).
  5. Apply thresholds: keep rows where both arguments are on-topic w.r.t. the
     post, and drop near-duplicate pairs.
  6. Push filtered df WITHOUT similarity columns to HF as split='filtered'.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

HF_REPO_ID = "idodah/argument-quality-cmv"
INPUT_SPLIT = "full"
OUTPUT_SPLIT = "filtered"

EXCEL_OUT = Path("./data/pairs_with_similarity.xlsx")
EXCEL_FILTERED_OUT = Path("./data/pairs_filtered.xlsx")

SIM_THRESHOLDS = {
    "sim_orig_delta": 0.3,
    "sim_orig_nodelta": 0.3,
}
MAX_DELTA_NODELTA_SIM = 0.95

_OPENAI_EMBED_MODEL = "text-embedding-3-small"
_OPENAI_BATCH_SIZE = 512


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts in batches using OpenAI text-embedding-3-small.

    Returns an (N, D) float32 array in the same order as input texts.
    """
    client = OpenAI()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), _OPENAI_BATCH_SIZE):
        batch = texts[i : i + _OPENAI_BATCH_SIZE]
        response = client.embeddings.create(model=_OPENAI_EMBED_MODEL, input=batch)
        batch_embs = [e.embedding for e in sorted(response.data, key=lambda e: e.index)]
        all_embeddings.extend(batch_embs)
    return np.array(all_embeddings, dtype=np.float32)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between each row of a and corresponding row of b. Returns 1-D array."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return (a_norm * b_norm).sum(axis=1)


def _embed_unique(texts: pd.Series) -> dict[str, np.ndarray]:
    unique = [t for t in texts.dropna().unique().tolist() if t]
    print(f"  embedding {len(unique)} unique texts...")
    embs = embed_texts(unique)
    return dict(zip(unique, embs))


def _series_to_matrix(series: pd.Series, cache: dict[str, np.ndarray]) -> np.ndarray:
    dim = next(iter(cache.values())).shape[0]
    out = np.zeros((len(series), dim), dtype=np.float32)
    for i, t in enumerate(series.tolist()):
        if t in cache:
            out[i] = cache[t]
    return out


def main() -> None:
    token_hf = os.environ.get("HF_TOKEN")
    if not token_hf:
        raise EnvironmentError("HF_TOKEN not set in .env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set in .env")

    print(f"Loading '{INPUT_SPLIT}' split from {HF_REPO_ID}...")
    df = load_dataset(HF_REPO_ID, split=INPUT_SPLIT, token=token_hf).to_pandas()
    print(f"  {len(df)} rows")

    before = len(df)
    df = df.dropna(subset=["original_post", "delta_argument", "nodelta_argument"]).reset_index(drop=True)
    if len(df) < before:
        print(f"  dropped {before - len(df)} rows with nulls in required text columns")

    print("\nEmbedding original_post...")
    cache_orig = _embed_unique(df["original_post"])
    print("Embedding delta_argument...")
    cache_delta = _embed_unique(df["delta_argument"])
    print("Embedding nodelta_argument...")
    cache_nodelta = _embed_unique(df["nodelta_argument"])

    emb_orig = _series_to_matrix(df["original_post"], cache_orig)
    emb_delta = _series_to_matrix(df["delta_argument"], cache_delta)
    emb_nodelta = _series_to_matrix(df["nodelta_argument"], cache_nodelta)

    df["sim_orig_delta"] = cosine_sim_matrix(emb_orig, emb_delta)
    df["sim_orig_nodelta"] = cosine_sim_matrix(emb_orig, emb_nodelta)
    df["sim_delta_nodelta"] = cosine_sim_matrix(emb_delta, emb_nodelta)

    print("\nSimilarity distributions:")
    print(df[list(SIM_THRESHOLDS.keys()) + ["sim_delta_nodelta"]].describe().to_string())

    print(f"\nSaving full df with similarities to {EXCEL_OUT}...")
    df.to_excel(EXCEL_OUT, index=False)

    mask = pd.Series(True, index=df.index)
    for col, thr in SIM_THRESHOLDS.items():
        col_mask = df[col] >= thr
        print(f"  {col} >= {thr}: keeps {col_mask.sum()}/{len(df)}")
        mask &= col_mask

    dup_mask = df["sim_delta_nodelta"] <= MAX_DELTA_NODELTA_SIM
    print(f"  sim_delta_nodelta <= {MAX_DELTA_NODELTA_SIM}: keeps {dup_mask.sum()}/{len(df)}")
    mask &= dup_mask

    sim_cols = list(SIM_THRESHOLDS.keys()) + ["sim_delta_nodelta"]
    filtered = df[mask].drop(columns=sim_cols).reset_index(drop=True)
    print(f"\nFiltered: {len(filtered)}/{len(df)} rows survived all thresholds")

    print(f"Saving filtered df to {EXCEL_FILTERED_OUT}...")
    filtered.to_excel(EXCEL_FILTERED_OUT, index=False)

    print(f"\nUploading filtered df to {HF_REPO_ID} (split='{OUTPUT_SPLIT}')...")
    ds = Dataset.from_pandas(filtered, preserve_index=False)
    ds.push_to_hub(HF_REPO_ID, split=OUTPUT_SPLIT, token=token_hf, private=False)
    print(f"Done. https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
