"""
Part 1 / Task 3 (continued) -- embed both chunk sets with the embedding
backend and index each chunking strategy in its own ChromaDB collection.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.embedding_backend import encode, backend_name  # noqa: E402
from app.rag.chunking import (  # noqa: E402
    load_kb_documents, fixed_size_overlap_chunks, sentence_based_chunks, Chunk,
)

CHROMA_DIR = str(Path(__file__).resolve().parent.parent.parent / "chroma_store")
FIXED_COLLECTION = "kb_fixed_overlap"
SENTENCE_COLLECTION = "kb_sentence"

_client = None


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _upsert_chunks(collection_name: str, chunks: list[Chunk]) -> None:
    client = get_client()
    # Fresh build every run so re-running this module is idempotent.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]
    metadatas = [{"doc_id": c.doc_id, "strategy": c.strategy} for c in chunks]
    embeddings = encode(texts).tolist()
    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return collection


def build_indexes() -> tuple[int, int]:
    docs = load_kb_documents()
    fixed = fixed_size_overlap_chunks(docs)
    sent = sentence_based_chunks(docs)
    _upsert_chunks(FIXED_COLLECTION, fixed)
    _upsert_chunks(SENTENCE_COLLECTION, sent)
    return len(fixed), len(sent)


def add_document(doc_id: str, title: str, text: str) -> tuple[int, int]:
    """Writes a new KB markdown document to disk, then rebuilds both chunking
    strategies' ChromaDB collections so the new document is immediately
    retrievable (Part 3 / Task 11's `POST /add-document` endpoint).
    """
    from app.rag.chunking import KB_DIR

    path = KB_DIR / f"{doc_id}.md"
    path.write_text(f"# {title}\n\n{text}\n", encoding="utf-8")
    return build_indexes()


def retrieve(query: str, collection_name: str, top_k: int = 3) -> list[dict]:
    """Returns list of {chunk_id, doc_id, text, similarity} sorted best-first."""
    client = get_client()
    collection = client.get_collection(collection_name)
    query_emb = encode([query]).tolist()
    res = collection.query(query_embeddings=query_emb, n_results=top_k)
    out = []
    for i in range(len(res["ids"][0])):
        distance = res["distances"][0][i]
        # Chroma's cosine "distance" = 1 - cosine_similarity for normalized vectors.
        similarity = 1.0 - distance
        out.append({
            "chunk_id": res["ids"][0][i],
            "doc_id": res["metadatas"][0][i]["doc_id"],
            "text": res["documents"][0][i],
            "similarity": similarity,
        })
    return out


if __name__ == "__main__":
    n_fixed, n_sent = build_indexes()
    print(f"Embedding backend: {backend_name()}")
    print(f"Indexed {n_fixed} fixed-overlap chunks into '{FIXED_COLLECTION}'")
    print(f"Indexed {n_sent} sentence chunks into '{SENTENCE_COLLECTION}'")

    sample_query = "How long do I have to return a pair of shoes?"
    print(f"\nSample query: {sample_query!r}")
    for coll in (FIXED_COLLECTION, SENTENCE_COLLECTION):
        print(f"\n  Top matches from {coll}:")
        for r in retrieve(sample_query, coll, top_k=3):
            print(f"    sim={r['similarity']:.3f} doc={r['doc_id']} :: {r['text'][:80]}...")
