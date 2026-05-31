"""RAG corpus pipeline for the pro-Israel argument retriever.

End-to-end the pipeline scrapes Israel-related r/ChangeMyView delta arguments,
classifies their stance, and ingests the high-confidence pro-Israel ones — plus
a handful of legal primary sources — into the Chroma vector store that
``agents.retrieval.LocalRetriever`` queries.

Run order (each is a module, run from the repo root):
    uv run python -m rag.scrape_cmv_israel      # -> data/cmv_israel_rag.parquet
    uv run python -m rag.classify_stance        # -> data/cmv_israel_rag_pro.parquet
    uv run python -m rag.ingest_rag             # -> .chroma/ pro_israel_corpus
    uv run python -m rag.ingest_legal_sources   # -> .chroma/ legal primary chunks
"""
