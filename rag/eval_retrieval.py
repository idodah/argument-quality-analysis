"""Small retrieval-quality check for the local Chroma corpus.

Measures whether the semantic retriever surfaces the *right* documents, so you
can tell — with numbers rather than vibes — whether the current dense-only
setup is good enough, and notice if quality regresses as the corpus grows (the
point at which adding chunking or hybrid search would start to pay off; see the
discussion in the module docstrings of `rag/ingest_rag.py`).

Two probe sets, no hand-labeling required:

1. Leave-one-out self-retrieval over the CMV arguments. For each argument we
   query with the same `topic` + `original_post` string the live graph uses
   (see agents/graph/nodes/retrieve.py::retrieve_local) and check whether that
   argument's own document comes back in the top-k. Reports Recall@k and MRR.
   This is the metric that matters: the graph retrieves arguments that
   countered a *similar* opposing post, and a document is the most relevant
   answer to its own post.

2. Legal-keyword probes. A handful of queries containing the rare exact tokens
   ("San Remo paragraph 67", "Palmer Report blockade") that dense embeddings
   handle worst — this is the narrow case where hybrid/BM25 would help, so it's
   the canary to watch.

Needs the real corpus and OPENAI_API_KEY (it embeds queries), so it lives in
`rag/` as a script, not in the offline `tests/` suite.

Usage:
    uv run python -m rag.eval_retrieval
    uv run python -m rag.eval_retrieval --k 4
"""

from __future__ import annotations

import argparse

import pandas as pd
from dotenv import load_dotenv

from agents.retrieval import CHROMA_DIR, COLLECTION_NAME, LocalRetriever, doc_count
from rag.ingest_rag import INPUT_PARQUET

# Legal-keyword probes: (query, substring that must appear in a retrieved chunk).
# These target the exact-token weakness of dense-only retrieval.
LEGAL_PROBES = [
    ("Is a naval blockade lawful under international law?", "blockade"),
    ("San Remo Manual paragraph 67 capture of vessels breaching a blockade", "San Remo"),
    ("Palmer Report conclusion on the Gaza flotilla blockade", "Palmer"),
]


def _self_retrieval(df: pd.DataFrame, retriever: LocalRetriever, k: int) -> None:
    """Leave-one-out: each argument should retrieve itself from its own post."""
    hits = 0
    reciprocal_ranks = 0.0
    misses: list[str] = []
    for row in df.itertuples(index=False):
        post = (row.original_post or "").strip()
        query = f"{row.topic}\n\n{post[:600]}"  # mirror retrieve_local's query shape
        results = retriever.store.similarity_search(
            query, k=k, filter={"source": "cmv_israel"}
        )
        # Match on the argument text we embedded (page_content), truncation-safe.
        target = (row.argument or "").strip()[:200]
        rank = next(
            (i for i, d in enumerate(results) if d.page_content[:200] == target),
            None,
        )
        if rank is not None:
            hits += 1
            reciprocal_ranks += 1.0 / (rank + 1)
        else:
            misses.append(str(row.topic)[:70])

    n = len(df)
    print(f"\nSelf-retrieval over {n} CMV arguments (k={k}):")
    print(f"  Recall@{k}: {hits}/{n} = {hits / n:.2%}")
    print(f"  MRR:       {reciprocal_ranks / n:.3f}")
    if misses:
        print(f"  Missed ({len(misses)}) — not in top-{k} for their own post:")
        for m in misses[:10]:
            print(f"    - {m}")
        if len(misses) > 10:
            print(f"    ... and {len(misses) - 10} more")


def _legal_probes(retriever: LocalRetriever, k: int) -> None:
    print(f"\nLegal-keyword probes (k={k}):")
    for query, needle in LEGAL_PROBES:
        chunks = retriever.retrieve(query, k=k)
        hit = any(needle.lower() in c.lower() for c in chunks)
        mark = "ok " if hit else "MISS"
        print(f"  [{mark}] {needle!r:14} <- {query[:60]!r}")


def main(k: int = 4) -> None:
    load_dotenv()
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(
            f"{INPUT_PARQUET} not found. Run rag.classify_stance then rag.ingest_rag first."
        )
    retriever = LocalRetriever(k=k)
    n_docs = doc_count(retriever.store)
    if n_docs == 0:
        raise SystemExit(
            f"Collection '{COLLECTION_NAME}' at {CHROMA_DIR} is empty. "
            "Run `uv run python -m rag.ingest_rag` (and rag.ingest_legal_sources) first."
        )
    print(f"Corpus '{COLLECTION_NAME}': {n_docs} documents.")

    df = pd.read_parquet(INPUT_PARQUET)
    _self_retrieval(df, retriever, k)
    _legal_probes(retriever, k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=4, help="top-k to retrieve (default 4)")
    args = parser.parse_args()
    main(k=args.k)
