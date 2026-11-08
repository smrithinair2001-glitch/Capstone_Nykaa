"""
Part 1 / Task 3 -- two independent chunking strategies over the knowledge
base, used to populate two separate ChromaDB collections so they can be
compared head to head in Task 5.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent / "kb"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    strategy: str


def load_kb_documents() -> dict[str, str]:
    """doc_id -> raw text, for every .md file in app/kb/."""
    docs = {}
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        # Drop the leading markdown H1 title line; keep the body prose.
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("# ")]
        docs[path.stem] = " ".join(ln.strip() for ln in lines if ln.strip())
    return docs


def split_sentences(text: str) -> list[str]:
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def fixed_size_overlap_chunks(docs: dict[str, str], chunk_size_words: int = 40,
                               overlap_words: int = 10) -> list[Chunk]:
    """Fixed-size-with-overlap chunking: slide a word-count window over each
    document's whitespace-tokenized text.
    """
    assert overlap_words < chunk_size_words
    chunks: list[Chunk] = []
    step = chunk_size_words - overlap_words
    for doc_id, text in docs.items():
        words = text.split()
        idx = 0
        n = 0
        start = 0
        while start < len(words):
            window = words[start:start + chunk_size_words]
            if not window:
                break
            chunk_text = " ".join(window)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::fixed::{n}",
                doc_id=doc_id,
                text=chunk_text,
                strategy="fixed_overlap",
            ))
            n += 1
            if start + chunk_size_words >= len(words):
                break
            start += step
    return chunks


def sentence_based_chunks(docs: dict[str, str], sentences_per_chunk: int = 2) -> list[Chunk]:
    """Sentence-based chunking: split each document into sentences, then
    group consecutive sentences (sentences_per_chunk at a time) into a chunk,
    never crossing a document boundary.
    """
    chunks: list[Chunk] = []
    for doc_id, text in docs.items():
        sentences = split_sentences(text)
        n = 0
        for i in range(0, len(sentences), sentences_per_chunk):
            group = sentences[i:i + sentences_per_chunk]
            chunk_text = " ".join(group)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::sent::{n}",
                doc_id=doc_id,
                text=chunk_text,
                strategy="sentence",
            ))
            n += 1
    return chunks


if __name__ == "__main__":
    docs = load_kb_documents()
    print(f"Loaded {len(docs)} KB documents: {list(docs.keys())}")

    fixed = fixed_size_overlap_chunks(docs)
    sent = sentence_based_chunks(docs)
    print(f"\nFixed-size-with-overlap: {len(fixed)} chunks")
    for c in fixed[:3]:
        print(f"  [{c.chunk_id}] {c.text[:90]}...")

    print(f"\nSentence-based: {len(sent)} chunks")
    for c in sent[:3]:
        print(f"  [{c.chunk_id}] {c.text[:90]}...")
