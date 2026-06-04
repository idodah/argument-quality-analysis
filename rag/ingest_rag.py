"""
Ingest the classified pro-Israel CMV arguments into the Chroma vector store
used by agents.retrieval.LocalRetriever.

Reads data/cmv_israel_rag_pro.parquet and adds one document per argument to the
`pro_israel_corpus` collection at .chroma/ — the same collection retrieve_local
queries, so no graph changes are needed.

Each document's page_content is the argument text alone — so the embedding
matches on the argument, not on the (long) post. Topic and the original post
are kept in metadata; retrieve_local reassembles the full context block for
the generator.

Usage:
    python -m rag.ingest_rag
    python -m rag.ingest_rag --reset   # drop existing collection first
"""

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from agents.retrieval import CHROMA_DIR, COLLECTION_NAME, _doc_count

INPUT_PARQUET = Path("data/cmv_israel_rag_pro.parquet")
POST_EXCERPT_CHARS = 1200
ARGUMENT_MAX_CHARS = 6000


def main(reset: bool = False) -> None:
    load_dotenv()
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"{INPUT_PARQUET} not found. Run rag.classify_stance first.")

    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df)} pro-Israel arguments from {INPUT_PARQUET}.")
    if df.empty:
        print("Nothing to ingest.")
        return

    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    if reset:
        existing = _doc_count(store)
        if existing:
            store.delete_collection()
            print(f"Reset: dropped {existing} existing documents from '{COLLECTION_NAME}'.")
            store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=str(CHROMA_DIR),
            )

    docs, ids = [], []
    for row in df.itertuples(index=False):
        post = (row.original_post or "").strip()
        if len(post) > POST_EXCERPT_CHARS:
            post = post[:POST_EXCERPT_CHARS].rstrip() + " […]"
        # page_content = argument alone -> the embedding matches on the argument.
        page_content = (row.argument or "").strip()[:ARGUMENT_MAX_CHARS]
        metadata = {
            "source": "cmv_israel",
            "thread_id": str(row.thread_id),
            "comment_id": str(row.comment_id),
            "topic": str(row.topic),
            "original_post": post,
            "score": int(row.score),
            "date": str(row.date),
            "stance": str(row.stance),
            "confidence": float(row.confidence),
        }
        docs.append(Document(page_content=page_content, metadata=metadata))
        ids.append(f"cmv_israel::{row.comment_id}")

    # idempotent: same comment_id -> same id, so re-running upserts rather than duplicates
    store.add_documents(docs, ids=ids)
    print(f"Ingested {len(docs)} documents into '{COLLECTION_NAME}' at {CHROMA_DIR}.")
    print(f"Collection now holds {_doc_count(store)} documents.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Drop the collection before ingesting")
    args = parser.parse_args()
    main(reset=args.reset)