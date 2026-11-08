"""
Part 1 / Task 4 -- grounded generation under MOCK_LLM, with an empirically
calibrated "I don't know" threshold (retrieval similarity is the only
groundedness signal available since there is no real model to judge it).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.rag.index import retrieve, FIXED_COLLECTION, SENTENCE_COLLECTION  # noqa: E402

FALLBACK_ANSWER = (
    "I don't have grounded information in the Nykaa policy knowledge base to "
    "answer that confidently, so I can't give you a reliable answer here. "
    "I'll escalate this to a human support associate."
)

# Set empirically by calibrate_threshold() against the kb_sentence collection
# (see transcripts/03_grounded_generation.txt): min in-scope top-1 similarity
# observed was 0.901, max out-of-scope top-1 similarity observed was 0.783,
# so the midpoint 0.842 is used as the production default. Kept as a module
# constant so generation.py has no import-order dependency on calibration
# having been re-run.
SIMILARITY_THRESHOLD = 0.842


def _compose_answer(query: str, chunks: list[dict]) -> str:
    """MOCK_LLM grounded "generation": under MOCK_LLM there is no real model
    call, so the answer is composed deterministically and *only* from the
    retrieved chunk text -- no token of this answer comes from anywhere
    else, which is what "grounded generation" means in this project.
    """
    bullet_lines = []
    seen_docs = set()
    for c in chunks:
        if c["doc_id"] in seen_docs:
            continue
        seen_docs.add(c["doc_id"])
        bullet_lines.append(f"- {c['text']}")
    body = "\n".join(bullet_lines)
    return (
        f"Based on Nykaa policy documents, here is what applies:\n{body}"
    )


def grounded_generate(query: str, collection: str = SENTENCE_COLLECTION,
                       top_k: int = 3,
                       threshold: float = SIMILARITY_THRESHOLD) -> dict:
    retrieved = retrieve(query, collection, top_k=top_k)
    top1_similarity = retrieved[0]["similarity"] if retrieved else 0.0

    if not retrieved or top1_similarity < threshold:
        return {
            "query": query,
            "answer": FALLBACK_ANSWER,
            "is_fallback": True,
            "top1_similarity": top1_similarity,
            "retrieved": retrieved,
            "collection": collection,
        }

    answer = _compose_answer(query, retrieved)
    return {
        "query": query,
        "answer": answer,
        "is_fallback": False,
        "top1_similarity": top1_similarity,
        "retrieved": retrieved,
        "collection": collection,
    }


IN_SCOPE_CALIBRATION_QUERIES = [
    "How many days do I have to return a kurta I bought?",
    "What is the refund timeline for a cash on delivery order?",
    "Is my electronics order covered under warranty?",
]

OUT_OF_SCOPE_CALIBRATION_QUERIES = [
    "Who directed the movie Inception?",
    "Tell me a joke about programmers.",
]

DEMO_IN_SCOPE_QUERIES = IN_SCOPE_CALIBRATION_QUERIES + [
    "Can I cancel my order after it has already shipped?",
    "Can I exchange my shoes for a different size?",
]

DEMO_OUT_OF_SCOPE_QUERY = "Can you write a poem about the ocean?"


def calibrate_threshold(collection: str = SENTENCE_COLLECTION, top_k: int = 3) -> dict:
    print(f"=== Threshold calibration against collection '{collection}' ===")
    in_scope_sims = []
    for q in IN_SCOPE_CALIBRATION_QUERIES:
        r = retrieve(q, collection, top_k=top_k)
        sim = r[0]["similarity"] if r else 0.0
        in_scope_sims.append(sim)
        print(f"  [in-scope]     top1_sim={sim:.3f}  query={q!r}")

    oos_sims = []
    for q in OUT_OF_SCOPE_CALIBRATION_QUERIES:
        r = retrieve(q, collection, top_k=top_k)
        sim = r[0]["similarity"] if r else 0.0
        oos_sims.append(sim)
        print(f"  [out-of-scope] top1_sim={sim:.3f}  query={q!r}")

    min_in_scope = min(in_scope_sims)
    max_oos = max(oos_sims)
    print(f"\n  min(in-scope top1 sim)     = {min_in_scope:.3f}")
    print(f"  max(out-of-scope top1 sim) = {max_oos:.3f}")

    if max_oos < min_in_scope:
        chosen = round((min_in_scope + max_oos) / 2, 3)
        print(f"  Clusters are separated -> choosing midpoint threshold = {chosen}")
    else:
        # Clusters overlap: fall back to a threshold just below the lowest
        # in-scope score, biasing toward not falsely refusing a real query.
        chosen = round(min_in_scope - 0.01, 3)
        print(f"  Clusters overlap -> choosing threshold just under min in-scope = {chosen}")

    return {
        "in_scope_sims": in_scope_sims,
        "oos_sims": oos_sims,
        "min_in_scope": min_in_scope,
        "max_oos": max_oos,
        "chosen_threshold": chosen,
    }


if __name__ == "__main__":
    calib = calibrate_threshold()
    threshold = calib["chosen_threshold"]
    print(f"\n>>> SIMILARITY_THRESHOLD set to {threshold} for the rest of this demo <<<\n")

    print("=== Grounded-generation demo: 5 in-scope queries ===")
    for q in DEMO_IN_SCOPE_QUERIES:
        result = grounded_generate(q, threshold=threshold)
        print(f"\nQ: {q}")
        print(f"   top1_similarity={result['top1_similarity']:.3f}  "
              f"fallback={result['is_fallback']}")
        print(f"   A: {result['answer'][:300]}")

    print("\n=== Grounded-generation demo: 1 deliberately out-of-scope query ===")
    result = grounded_generate(DEMO_OUT_OF_SCOPE_QUERY, threshold=threshold)
    print(f"\nQ: {DEMO_OUT_OF_SCOPE_QUERY}")
    print(f"   top1_similarity={result['top1_similarity']:.3f}  fallback={result['is_fallback']}")
    print(f"   A: {result['answer']}")
    assert result["is_fallback"], "Out-of-scope query should have triggered the fallback!"
    print("\n[PASS] Out-of-scope query correctly triggered the 'I don't know' fallback.")
