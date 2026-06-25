"""Ingest legal primary-source chunks into the local Chroma corpus.

Instead of hardcoding verbatim text, each source is declared as a URL (the
source of truth); the text is FETCHED at ingest time, stripped to paragraphs,
filtered to the passages that are actually on-topic (the blockade keywords),
and chunked. This keeps the file to (id, url, title, keywords) and lets the
corpus be refreshed by re-running ingestion rather than editing pasted text.

The local retriever (agents/graph/nodes/retrieve.py) filters on metadata
`source == "cmv_israel"`, so these chunks carry that source tag to be
retrievable, plus `doc_type == "legal_primary"` to distinguish them from the
CMV delta-comment corpus. Each chunk begins with a '[url] <URL>' line so the
refiner can cite it the same way it cites web results.

Fetching is SSRF-guarded (reuses harvester.fediverse.base.assert_safe_url) and
fails LOUDLY if a source yields too little usable text — a broken fetch must
never silently poison a corpus that grounds legal claims. Run once:
    uv run python -m rag.ingest_legal_sources
Idempotent: skips chunks whose id already exists in the collection.
"""

from __future__ import annotations

import html
import re

import requests

from agents.retrieval import CHROMA_DIR, COLLECTION_NAME
from harvester.fediverse.base import assert_safe_url

# (id_prefix, url, title, on_topic_keywords)
# The URL is the source of truth; text is fetched and filtered to paragraphs
# matching any keyword (case-insensitive), so we embed only the relevant passages
# instead of an entire article. Chunk ids are f"{id_prefix}_{n}".
LEGAL_SOURCES = [
    (
        "palmer",
        "https://www.un.org/unispal/document/auto-insert-202356/",
        "UN Palmer Report (2011) — Panel of Inquiry on the 31 May 2010 Flotilla Incident",
        ["blockade", "naval", "international law", "self-defence", "flotilla",
         "humanitarian", "security"],
    ),
    (
        "sanremo",
        "https://en.wikipedia.org/wiki/San_Remo_Manual_on_International_Law_Applicable_to_Armed_Conflicts_at_Sea",
        "San Remo Manual on International Law Applicable to Armed Conflicts at Sea (1994)",
        ["blockade", "merchant", "contraband", "neutral", "capture",
         "starving", "civilian population"],
    ),
]

# A source that yields fewer than this many on-topic paragraphs is treated as a
# failed/garbage fetch and aborts ingestion, rather than poisoning the corpus.
_MIN_PARAGRAPHS_PER_SOURCE = 2
# Keep paragraphs within a sane size for embedding; skip tiny fragments.
_MIN_PARA_CHARS = 120
_MAX_PARA_CHARS = 1500

_REQUEST_TIMEOUT = 30
_USER_AGENT = "argument-quality-rag/1.0 (legal-source ingestion)"

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"[ \t]+")


def _fetch_text(url: str) -> str:
    """Fetch a URL (SSRF-checked) and return its decoded HTML/text body."""
    assert_safe_url(url)
    resp = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def _html_to_paragraphs(raw: str) -> list[str]:
    """Strip HTML to a list of plain-text paragraphs.

    Crude but dependency-free: drop script/style, split on block boundaries,
    strip remaining tags, unescape entities, and collapse whitespace. Good
    enough to recover article body paragraphs for keyword filtering; we are not
    trying to faithfully render the page.
    """
    raw = _SCRIPT_STYLE_RE.sub(" ", raw)
    # Treat closing block tags and <br> as paragraph separators.
    raw = re.sub(r"(?i)</(p|div|li|h[1-6]|tr|section|article)>", "\n\n", raw)
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    text = _TAG_RE.sub("", raw)
    text = html.unescape(text)
    paras = []
    for block in text.split("\n\n"):
        para = _WS_RE.sub(" ", block).strip()
        if para:
            paras.append(para)
    return paras


def _on_topic(para: str, keywords: list[str]) -> bool:
    low = para.lower()
    return any(kw.lower() in low for kw in keywords)


def _extract_chunks(url: str, keywords: list[str]) -> list[str]:
    """Fetch `url` and return the on-topic, sensibly-sized paragraphs from it."""
    paras = _html_to_paragraphs(_fetch_text(url))
    chunks = []
    seen: set[str] = set()
    for para in paras:
        if not (_MIN_PARA_CHARS <= len(para) <= _MAX_PARA_CHARS):
            continue
        if not _on_topic(para, keywords):
            continue
        if para in seen:
            continue
        seen.add(para)
        chunks.append(para)
    return chunks


def collect_chunks() -> list[tuple[str, str, str, str]]:
    """Fetch every source and return (id, url, title, text) chunks.

    Raises RuntimeError if any source yields too few on-topic paragraphs, so a
    broken fetch fails the whole run instead of silently ingesting nothing (or
    junk) for that source.
    """
    out: list[tuple[str, str, str, str]] = []
    for prefix, url, title, keywords in LEGAL_SOURCES:
        try:
            paras = _extract_chunks(url, keywords)
        except (requests.RequestException, RuntimeError) as e:
            raise RuntimeError(f"failed to fetch legal source {prefix!r} from {url}: {e}") from e
        if len(paras) < _MIN_PARAGRAPHS_PER_SOURCE:
            raise RuntimeError(
                f"legal source {prefix!r} ({url}) yielded only {len(paras)} on-topic "
                f"paragraph(s); expected >= {_MIN_PARAGRAPHS_PER_SOURCE}. The page "
                f"layout or URL may have changed — refusing to ingest a degraded corpus."
            )
        for n, para in enumerate(paras):
            out.append((f"{prefix}_{n}", url, title, para))
        print(f"  {prefix}: {len(paras)} on-topic paragraphs")
    return out


def main() -> None:
    import chromadb
    from dotenv import load_dotenv
    from langchain_openai import OpenAIEmbeddings

    load_dotenv()

    print("Fetching legal sources...")
    legal_chunks = collect_chunks()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection(COLLECTION_NAME)
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    existing = set(col.get(ids=[c[0] for c in legal_chunks]).get("ids", []))
    to_add = [c for c in legal_chunks if c[0] not in existing]
    if not to_add:
        print(f"All {len(legal_chunks)} legal chunks already present; nothing to do.")
        return

    ids, docs, metas = [], [], []
    for cid, url, title, body in to_add:
        ids.append(cid)
        # Lead with the [url]/[title] header so the refiner cites it like a web chunk.
        docs.append(f"[url] {url}\n[title] {title}\n{body}")
        metas.append({"source": "cmv_israel", "doc_type": "legal_primary",
                      "stance": "pro_israel", "url": url, "title": title})

    embeddings = embedder.embed_documents(docs)
    col.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    print(f"Ingested {len(to_add)} legal chunks. Collection now has {col.count()} docs.")


if __name__ == "__main__":
    main()
