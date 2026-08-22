"""RAG corpus pipeline for the trope-refutation retriever.

End-to-end the pipeline scrapes r/ChangeMyView delta arguments, classifies how
each handles antisemitic tropes, and ingests the high-confidence *refutations*
into the Chroma vector store that ``agents.retrieval.LocalRetriever`` queries.
Arguments classified ``political_argument`` (criticism of the Israeli
government — political speech, not a trope) are excluded from the corpus.

The ``*_pro*`` data filenames are retained from an earlier iteration of this
project to avoid a data migration; the rows they hold are refutations.

Run order (each is a module, run from the repo root):
    uv run python -m rag.scrape_cmv_israel      # -> data/cmv_israel_rag.parquet
    uv run python -m rag.classify_stance        # -> data/cmv_israel_rag_pro.parquet
    uv run python -m rag.ingest_rag             # -> .chroma/ trope_refutation_corpus
"""
