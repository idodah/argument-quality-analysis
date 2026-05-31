---
description: Run the RAG corpus pipeline (scrape -> classify -> ingest) in order, checking artifacts between steps
---

Build (or rebuild) the local pro-Israel RAG corpus that the agentic graph's
`retrieve_local` arm queries. This is a **4-step ordered pipeline**; each step
reads the previous step's artifact, so the order is mandatory.

Before starting, confirm `OPENAI_API_KEY` is in `.env` — steps 2-4 all need it
(classification + embeddings). `HF_TOKEN` is optional (only step 2's Hub push).
If `$ARGUMENTS` names a single step (e.g. "ingest" or "step 3"), run just that
step and the ones it depends on; otherwise run all four.

## Steps

1. **Scrape** — `uv run python -m rag.scrape_cmv_israel`
   - Writes `data/cmv_israel_rag.parquet` (+ `.jsonl`). Pages arctic-shift from
     2023-10-01 to now, so it is slow and network-bound; expect several minutes.
   - To sanity-check connectivity first without a full run, suggest
     `--smoke` (5 threads, dumps raw JSON to `data/cmv_israel_smoke.json`).
   - After it finishes, confirm the parquet exists and is non-empty before
     moving on.

2. **Classify stance** — `uv run python -m rag.classify_stance`
   - Reads `data/cmv_israel_rag.parquet`; writes
     `data/cmv_israel_rag_classified.parquet` (all rows) and
     `data/cmv_israel_rag_pro.parquet` (stance=pro_israel, confidence>=0.8).
   - Calls GPT-4o-mini once per argument — costs money and scales with the
     scrape size. Pass `--no-hub` to skip the Hugging Face upload (default
     pushes if `HF_TOKEN` is set).
   - Report the printed stance distribution and how many rows survived the
     pro-Israel filter. **If `cmv_israel_rag_pro.parquet` is empty, stop** —
     ingesting it would leave the corpus empty and the local retrieval arm
     would silently return nothing.

3. **Ingest CMV arguments** — `uv run python -m rag.ingest_rag`
   - Reads `data/cmv_israel_rag_pro.parquet`; upserts one document per argument
     into the `pro_israel_corpus` Chroma collection at `.chroma/`.
   - Idempotent (keyed on `comment_id`), so re-running upserts rather than
     duplicates. Use `--reset` to drop the collection first for a clean rebuild.
   - Report the final "Collection now holds N documents" count.

4. **Ingest legal sources** — `uv run python -m rag.ingest_legal_sources`
   - Adds the fixed Palmer Report / San Remo legal chunks to the same
     collection. Idempotent (skips ids already present). Don't skip this — the
     refiner relies on these for citable primary-source claims.

## After

Report the final Chroma document count and confirm the collection is non-empty.
A populated `pro_israel_corpus` is what keeps `agents.retrieval.LocalRetriever`
from printing its "corpus is empty" warning and returning no local results.

Notes:
- All artifacts (`data/`, `.chroma/`) are gitignored — do not commit them.
- This pipeline hits the network and paid APIs; it is **not** part of the
  offline test suite. Don't run it just to verify code changes — use `/test`.

$ARGUMENTS
