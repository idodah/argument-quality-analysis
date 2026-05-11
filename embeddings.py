"""
OpenAI embedding utilities for the Webis-CMV-20 dataset.
"""

import numpy as np
from openai import OpenAI

from text_utils import clean_text

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


def build_embedding_cache(raw: list[dict]) -> dict[str, np.ndarray]:
    """Collect all unique cleaned texts across all records, embed once, return lookup dict."""
    unique_texts: list[str] = []
    seen: set[str] = set()

    for record in raw:
        for field in ("original_post", "delta_argument", "nodelta_argument"):
            text = clean_text(record.get(field, ""))
            if text and text not in seen:
                unique_texts.append(text)
                seen.add(text)

    print(f"Embedding {len(unique_texts)} unique texts via OpenAI ({_OPENAI_EMBED_MODEL})...")
    embeddings = embed_texts(unique_texts)
    return dict(zip(unique_texts, embeddings))


def comment_similarities(
    comment_bodies: list[str],
    original_emb: np.ndarray,
    emb_cache: dict[str, np.ndarray],
) -> list[float]:
    """Return cosine similarity of each comment to the original argument embedding."""
    comment_embs = np.array(
        [emb_cache[b] for b in comment_bodies if b in emb_cache], dtype=np.float32
    )
    if len(comment_embs) == 0:
        return []
    orig = original_emb.reshape(1, -1).repeat(len(comment_embs), axis=0)
    sims = cosine_sim_matrix(orig, comment_embs)
    return [round(float(s), 4) for s in sims]