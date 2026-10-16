"""
Ingest authoritative trope-refutation documents into the Chroma vector store.

Companion to `rag.ingest_rag`, which ingests delta-awarded CMV comments. Both
write to the same `trope_refutation_corpus` collection, so `retrieve_local`
needs no graph changes; documents are told apart by their `source_type`
metadata:

    source_type="cmv_delta"  -> persuasive, human-validated, NOT a factual source
    source_type="reference"  -> authoritative and citable (carries a real url)

`agents.retrieval.LocalRetriever._format` reads that field and labels each
chunk accordingly, so the generator knows which chunks it may cite as evidence.

Reference articles are long (up to ~95k chars), so unlike CMV comments they are
split into overlapping chunks before embedding. Chunk ids are deterministic
(`<doc_id>::<n>`), making re-runs upserts rather than duplicates.

Usage:
    uv run python -m rag.ingest_reference
    uv run python -m rag.ingest_reference --reset   # drop collection first
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agents.retrieval import CHROMA_DIR, COLLECTION_NAME, doc_count

INPUT_PARQUET = Path("data/trope_reference.parquet")
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200
MIN_CHUNK_CHARS = 200   # drop trailing fragments too small to be useful evidence


def build_documents(df: pd.DataFrame) -> tuple[list[Document], list[str]]:
    """Split each reference article into chunks and build Documents + ids."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs: list[Document] = []
    ids: list[str] = []
    for row in df.itertuples(index=False):
        chunks = [c for c in splitter.split_text(row.text or "")
                  if len(c.strip()) >= MIN_CHUNK_CHARS]
        for n, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk.strip(),
                metadata={
                    "source_type": "reference",
                    "source": str(row.source),
                    "trope": str(row.trope),
                    "title": str(row.title),
                    "url": str(row.url),
                    "doc_id": str(row.doc_id),
                    "chunk": n,
                    "retrieved": str(row.retrieved),
                },
            ))
            ids.append(f"{row.doc_id}::{n}")
    return docs, ids


def main(reset: bool = False) -> None:
    """Chunk, embed, and upsert the reference corpus into Chroma."""
    load_dotenv()
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(
            f"{INPUT_PARQUET} not found. Run rag.scrape_reference first."
        )

    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df)} reference documents from {INPUT_PARQUET}.")
    if df.empty:
        print("Nothing to ingest.")
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    if reset:
        existing = doc_count(store)
        if existing:
            store.delete_collection()
            print(f"Reset: dropped {existing} existing documents from '{COLLECTION_NAME}'.")
            store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=str(CHROMA_DIR),
            )

    docs, ids = build_documents(df)
    if not docs:
        print("No chunks produced; nothing ingested.")
        return

    store.add_documents(docs, ids=ids)
    print(f"Ingested {len(docs)} chunks from {len(df)} articles "
          f"into '{COLLECTION_NAME}' at {CHROMA_DIR}.")
    print(f"Collection now holds {doc_count(store)} documents.")

    counts = pd.Series([d.metadata["trope"] for d in docs]).value_counts()
    print("\nChunks per trope:")
    print(counts.to_string())


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Drop the collection before ingesting")
    args = parser.parse_args()
    main(reset=args.reset)


if __name__ == "__main__":
    cli()
