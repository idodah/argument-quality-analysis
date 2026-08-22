"""
Retrieval backends for the Adaptive-RAG step.

Two arms:
  - local: Chroma vector store (`trope_refutation_corpus`) — authoritative
    trope-refutation articles plus delta-awarded CMV comments, ingested by the
    `rag/` pipeline (`rag.ingest_reference` / `rag.ingest_rag`).
  - web:   Tavily search. Requires TAVILY_API_KEY in the environment.

Both retrievers expose the same interface: `.retrieve(query, k) -> list[str]`.
"""

from __future__ import annotations

import os
from pathlib import Path

# Chroma opens its SQLite store read-WRITE even for queries (lock/WAL files), so
# the corpus dir must be writable.
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR") or (Path(__file__).resolve().parent.parent / ".chroma"))
COLLECTION_NAME = "trope_refutation_corpus"


def doc_count(store) -> int:
    """Number of documents in a langchain-chroma store, via the public API.

    `store.get(include=[])` returns only the ids (no embeddings/documents
    payload), so this is a cheap count that doesn't poke at the private
    `_collection` attribute.
    """
    return len(store.get(include=[]).get("ids", []))


class LocalRetriever:
    """Chroma-backed local corpus. Returns empty results until the `rag/`
    ingestion scripts have populated the `trope_refutation_corpus` collection."""

    def __init__(self, k: int = 4):
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings

        self.k = k
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        if doc_count(self.store) == 0:
            print(
                "[LocalRetriever] trope_refutation_corpus is empty; local arm will return "
                "no results. Populate it with `uv run python -m rag.ingest_rag`."
            )

    def retrieve(self, query: str, k: int | None = None, where: dict | None = None) -> list[str]:
        if doc_count(self.store) == 0:
            return []
        results = self.store.similarity_search(query, k=k or self.k, filter=where)
        return [self._format(d) for d in results]

    @staticmethod
    def _format(doc) -> str:
        """Reassemble a retrieved argument with its post context for the generator.

        The collection holds two kinds of document, distinguished by the
        `source_type` metadata field:

          - "cmv_delta"  — a delta-awarded CMV comment. Embeds the argument
            alone but carries topic/original_post in metadata, which are
            reassembled here as context.
          - "reference"  — an authoritative article (USHMM, Wikipedia). Carries
            a real `url`, so it is emitted with a '[url]' line and is therefore
            citable; refine/hallucination_check treat a chunk's '[url]' as the
            only legitimate citation target.

        Provenance is stated explicitly in the label because the two carry very
        different authority: a CMV comment persuaded one human and is NOT a
        factual source, while a reference article is. Mislabeling the former as
        the latter would invite the generator to cite it as evidence.

        A doc with neither shape falls back to the bare page_content.

        Labels use bracket prefixes ('[topic]', '[original_post]', '[argument]')
        rather than markdown '###' headers. The ### form collides with the
        prompt structure used by reflect/refine/hallucination_check (all of
        which use '###' for their own section headers), causing downstream
        LLMs to parse chunk-internal headers as prompt-level structure and
        treat the rest of the evidence as missing. Bracket labels are also
        consistent with the web arm's '[url]' / '[title]' header format.
        """
        meta = doc.metadata or {}

        if meta.get("source_type") == "reference":
            parts = []
            if meta.get("url"):
                parts.append(f"[url] {meta['url']}")
            if meta.get("title"):
                parts.append(f"[title] {meta['title']}")
            origin = meta.get("source", "reference")
            parts.append(
                f"[reference article — authoritative source: {origin}]\n{doc.page_content}"
            )
            return "\n\n".join(parts)

        topic = meta.get("topic")
        post = meta.get("original_post")
        if not topic and not post:
            return doc.page_content
        parts = []
        if topic:
            parts.append(f"[topic] {topic}")
        if post:
            parts.append(f"[original_post]\n{post}")
        parts.append(f"[argument (earned a delta on CMV)]\n{doc.page_content}")
        return "\n\n".join(parts)


class WebRetriever:
    """Tavily-backed web retriever."""

    def __init__(self, k: int = 4, allowed_domains: list[str] | None = None):
        self.k = k
        # When non-empty, Tavily restricts results to these domains. When empty
        # or None, behavior matches the unfiltered baseline.
        self.allowed_domains = list(allowed_domains) if allowed_domains else []
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            self.client = None
            print("[WebRetriever] TAVILY_API_KEY not set; web arm will return empty results.")
            return
        from tavily import TavilyClient

        self.client = TavilyClient(api_key=api_key)

    def retrieve(self, query: str, k: int | None = None) -> list[str]:
        if self.client is None:
            return []
        search_kwargs = {
            "query": query,
            "max_results": k or self.k,
            "search_depth": "advanced",
        }
        if self.allowed_domains:
            search_kwargs["include_domains"] = self.allowed_domains
        try:
            resp = self.client.search(**search_kwargs)
        except Exception as e:
            print(f"[WebRetriever] Search failed: {e}")
            return []
        chunks: list[str] = []
        for r in resp.get("results", []):
            content = r.get("content", "")
            if not content:
                continue
            url = r.get("url", "")
            title = r.get("title", "")
            header_lines = []
            if url:
                header_lines.append(f"[url] {url}")
            if title:
                header_lines.append(f"[title] {title}")
            header = "\n".join(header_lines)
            chunks.append(f"{header}\n{content}" if header else content)
        return chunks