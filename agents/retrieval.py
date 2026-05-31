"""
Retrieval backends for the Adaptive-RAG step.

Two arms:
  - local: Chroma vector store backed by ./docs (placeholder; populate by
    dropping .md/.txt files into the docs/ directory).
  - web:   Tavily search. Requires TAVILY_API_KEY in the environment.

Both retrievers expose the same interface: `.retrieve(query, k) -> list[str]`.
"""

from __future__ import annotations

import os
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
CHROMA_DIR = Path(__file__).resolve().parent.parent / ".chroma"
COLLECTION_NAME = "pro_israel_corpus"


class LocalRetriever:
    """Chroma-backed local corpus. Empty until docs/ is populated."""

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
        if self.store._collection.count() == 0:
            self._ingest_docs()

    def _ingest_docs(self) -> None:
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        if not DOCS_DIR.exists():
            return
        text_files = list(DOCS_DIR.glob("**/*.txt")) + list(DOCS_DIR.glob("**/*.md"))
        if not text_files:
            print(f"[LocalRetriever] No documents found in {DOCS_DIR}. Local arm will return empty results until populated.")
            return

        loader = DirectoryLoader(
            str(DOCS_DIR),
            glob="**/*.{md,txt}",
            loader_cls=TextLoader,
            show_progress=False,
        )
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        chunks = splitter.split_documents(docs)
        if chunks:
            self.store.add_documents(chunks)
            print(f"[LocalRetriever] Indexed {len(chunks)} chunks from {len(text_files)} files.")

    def retrieve(self, query: str, k: int | None = None, where: dict | None = None) -> list[str]:
        if self.store._collection.count() == 0:
            return []
        results = self.store.similarity_search(query, k=k or self.k, filter=where)
        return [self._format(d) for d in results]

    @staticmethod
    def _format(doc) -> str:
        """Reassemble a retrieved argument with its post context for the generator.

        CMV-Israel docs embed the argument alone but carry topic/original_post
        in metadata; loose docs/ chunks have neither, so we fall back to the
        bare page_content.

        Labels use bracket prefixes ('[topic]', '[original_post]', '[argument]')
        rather than markdown '###' headers. The ### form collides with the
        prompt structure used by reflect/refine/hallucination_check (all of
        which use '###' for their own section headers), causing downstream
        LLMs to parse chunk-internal headers as prompt-level structure and
        treat the rest of the evidence as missing. Bracket labels are also
        consistent with the web arm's '[url]' / '[title]' header format.
        """
        meta = doc.metadata or {}
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