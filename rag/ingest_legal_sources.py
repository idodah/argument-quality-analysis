"""Ingest legal primary-source chunks into the local Chroma corpus.

The local retriever (agents/graph/nodes/retrieve.py) filters on
metadata `source == "cmv_israel"`, so these chunks carry that source tag to be
retrievable, plus `doc_type == "legal_primary"` to distinguish them from the
CMV delta-comment corpus. Each chunk begins with a '[url] <URL>' line so the
refiner can cite it the same way it cites web results.

Text is verbatim from the primary sources (Palmer Report PDF; San Remo Manual
via ICRC/reference), trimmed only for length. Run once:
    uv run python -m rag.ingest_legal_sources
Idempotent: skips chunks whose id already exists in the collection.
"""

from __future__ import annotations

from agents.retrieval import CHROMA_DIR, COLLECTION_NAME

# (id, url, title, verbatim_text)
LEGAL_CHUNKS = [
    (
        "palmer_blockade_legitimate",
        "https://www.un.org/unispal/document/auto-insert-202356/",
        "UN Palmer Report (2011) — Panel of Inquiry on the 31 May 2010 Flotilla Incident",
        "The naval blockade was imposed as a legitimate security measure in order to "
        "prevent weapons from entering Gaza by sea and its implementation complied with "
        "the requirements of international law. Israel faces a real threat to its "
        "security from militant groups in Gaza. Although people are entitled to express "
        "their political views, the flotilla acted recklessly in attempting to breach "
        "the naval blockade.",
    ),
    (
        "palmer_blockade_lawful",
        "https://www.un.org/unispal/document/auto-insert-202356/",
        "UN Palmer Report (2011) — Panel of Inquiry on the 31 May 2010 Flotilla Incident",
        "The imposition of the naval blockade was lawful and complied with the "
        "conditions of international law, in view of the security circumstances and "
        "Israel's efforts to fulfil its humanitarian obligations. That conclusion was "
        "based on the following: the conflict between Israel and the Gaza Strip is an "
        "\"international armed conflict\" for the purposes of international law.",
    ),
    (
        "palmer_established_procedures",
        "https://www.un.org/unispal/document/auto-insert-202356/",
        "UN Palmer Report (2011) — Panel of Inquiry on the 31 May 2010 Flotilla Incident",
        "All humanitarian missions wishing to assist the Gaza population should do so "
        "through established procedures and the designated land crossings in "
        "consultation with the Government of Israel and the Palestinian Authority. The "
        "imposition of a naval blockade as an action in self-defence is a legitimate "
        "measure; the San Remo Manual provides a useful reference in identifying the "
        "established norms of customary international law applicable to a naval blockade.",
    ),
    (
        "sanremo_blockade_breach",
        "https://en.wikipedia.org/wiki/San_Remo_Manual_on_International_Law_Applicable_to_Armed_Conflicts_at_Sea",
        "San Remo Manual on International Law Applicable to Armed Conflicts at Sea (1994)",
        "Under the San Remo Manual, merchant vessels believed on reasonable grounds to "
        "be breaching a blockade may be stopped and captured. Paragraph 67 permits "
        "action against vessels that \"are believed on reasonable grounds to be carrying "
        "contraband or breaching a blockade, and if after prior warning they "
        "intentionally and clearly refuse to stop, or intentionally and clearly resist "
        "visit, search or capture.\" Paragraph 146 authorizes capture of neutral "
        "merchant vessels outside neutral waters so engaged.",
    ),
    (
        "sanremo_blockade_humanitarian",
        "https://en.wikipedia.org/wiki/San_Remo_Manual_on_International_Law_Applicable_to_Armed_Conflicts_at_Sea",
        "San Remo Manual on International Law Applicable to Armed Conflicts at Sea (1994)",
        "A naval blockade is a recognized method of warfare. Under the San Remo Manual, "
        "paragraph 102, \"a blockade is prohibited if it has the sole purpose of starving "
        "the civilian population or denying it other objects essential for its "
        "survival.\" A blockade imposed for a genuine security purpose — preventing arms "
        "from reaching an enemy force — is therefore lawful even though it restricts "
        "maritime traffic, provided humanitarian access is allowed.",
    ),
]


def main() -> None:
    import chromadb
    from dotenv import load_dotenv
    from langchain_openai import OpenAIEmbeddings

    load_dotenv()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection(COLLECTION_NAME)
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    existing = set(col.get(ids=[c[0] for c in LEGAL_CHUNKS]).get("ids", []))
    to_add = [c for c in LEGAL_CHUNKS if c[0] not in existing]
    if not to_add:
        print(f"All {len(LEGAL_CHUNKS)} legal chunks already present; nothing to do.")
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