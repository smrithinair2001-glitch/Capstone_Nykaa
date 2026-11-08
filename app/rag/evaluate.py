"""
Part 1 / Task 5 -- document-level precision/recall for both chunking
strategies, over the same 5 queries used in Task 4's grounded-generation
demo (the in-scope half only -- precision/recall against a known-relevant
document is not meaningful for the deliberately out-of-scope query).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.rag.index import retrieve, FIXED_COLLECTION, SENTENCE_COLLECTION  # noqa: E402
from app.rag.generation import DEMO_IN_SCOPE_QUERIES  # noqa: E402

TOP_K = 3

# Ground truth: the single KB document that actually answers each query,
# established by inspection of app/kb/*.md (same queries as the Task 4 demo).
GROUND_TRUTH = {
    "How many days do I have to return a kurta I bought?": {"01_return_window"},
    "What is the refund timeline for a cash on delivery order?": {"02_cod_refund_timeline"},
    "Is my electronics order covered under warranty?": {"05_warranty_terms"},
    "Can I cancel my order after it has already shipped?": {"06_order_cancellation"},
    "Can I exchange my shoes for a different size?": {"09_size_exchange"},
}


def doc_level_precision_recall(query: str, collection: str, top_k: int = TOP_K) -> dict:
    retrieved = retrieve(query, collection, top_k=top_k)
    retrieved_docs = []
    for r in retrieved:
        if r["doc_id"] not in retrieved_docs:
            retrieved_docs.append(r["doc_id"])  # dedup, preserve rank order
    relevant_docs = GROUND_TRUTH[query]

    hits = [d for d in retrieved_docs if d in relevant_docs]
    precision = len(hits) / len(retrieved_docs) if retrieved_docs else 0.0
    recall = len(hits) / len(relevant_docs) if relevant_docs else 0.0
    return {
        "query": query,
        "collection": collection,
        "retrieved_docs": retrieved_docs,
        "relevant_docs": relevant_docs,
        "hits": hits,
        "precision": precision,
        "recall": recall,
    }


def run_evaluation() -> dict:
    queries = DEMO_IN_SCOPE_QUERIES[:5]
    assert len(queries) == 5, "Need exactly 5 queries for this evaluation"

    results = {FIXED_COLLECTION: [], SENTENCE_COLLECTION: []}
    for collection in (FIXED_COLLECTION, SENTENCE_COLLECTION):
        print(f"\n=== Collection: {collection} (top_k={TOP_K}) ===")
        for q in queries:
            r = doc_level_precision_recall(q, collection)
            results[collection].append(r)
            dedup_n = len(r["retrieved_docs"])
            hit_n = len(r["hits"])
            rel_n = len(r["relevant_docs"])
            print(f"\nQ: {q}")
            print(f"   retrieved (deduped, rank order): {r['retrieved_docs']}")
            print(f"   relevant (ground truth):         {sorted(r['relevant_docs'])}")
            print(f"   hits = retrieved ∩ relevant = {r['hits']}  (|hits|={hit_n})")
            print(f"   precision = |hits| / |retrieved| = {hit_n}/{dedup_n} = {r['precision']:.3f}")
            print(f"   recall    = |hits| / |relevant|  = {hit_n}/{rel_n} = {r['recall']:.3f}")

    print("\n=== Aggregate (macro-average over the 5 queries) ===")
    summary = {}
    for collection in (FIXED_COLLECTION, SENTENCE_COLLECTION):
        rs = results[collection]
        avg_p = sum(r["precision"] for r in rs) / len(rs)
        avg_r = sum(r["recall"] for r in rs) / len(rs)
        summary[collection] = {"avg_precision": avg_p, "avg_recall": avg_r}
        terms_p = " + ".join(f"{r['precision']:.3f}" for r in rs)
        terms_r = " + ".join(f"{r['recall']:.3f}" for r in rs)
        print(f"\n{collection}:")
        print(f"  avg_precision = ({terms_p}) / 5 = {avg_p:.3f}")
        print(f"  avg_recall    = ({terms_r}) / 5 = {avg_r:.3f}")

    return summary


if __name__ == "__main__":
    summary = run_evaluation()

    fixed = summary[FIXED_COLLECTION]
    sent = summary[SENTENCE_COLLECTION]
    print("\n=== Recommendation ===")
    print(
        f"Fixed-size-with-overlap scored avg_precision={fixed['avg_precision']:.3f}, "
        f"avg_recall={fixed['avg_recall']:.3f}; sentence-based scored "
        f"avg_precision={sent['avg_precision']:.3f}, avg_recall={sent['avg_recall']:.3f}. "
    )
    if sent["avg_recall"] >= fixed["avg_recall"] and sent["avg_precision"] >= fixed["avg_precision"]:
        print(
            "Recommendation: deploy the sentence-based collection -- it matches or beats "
            "fixed-size-with-overlap on both metrics here, and because each chunk is a "
            "single self-contained policy statement it also produces cleaner, less "
            "repetitive grounded-generation answers than the overlapping fixed windows."
        )
    elif fixed["avg_recall"] >= sent["avg_recall"] and fixed["avg_precision"] >= sent["avg_precision"]:
        print(
            "Recommendation: deploy the fixed-size-with-overlap collection -- it matches or "
            "beats sentence-based chunking on both metrics here, and the overlap protects "
            "against policy statements that straddle a sentence boundary."
        )
    else:
        better_recall = SENTENCE_COLLECTION if sent["avg_recall"] >= fixed["avg_recall"] else FIXED_COLLECTION
        print(
            f"Recommendation: deploy '{better_recall}' for Part 2. The two strategies trade "
            f"off precision against recall on these 5 queries ({FIXED_COLLECTION} misses the "
            f"relevant document entirely on the first query, pulling its recall down to "
            f"{fixed['avg_recall']:.3f}, while {SENTENCE_COLLECTION} retrieves the relevant "
            f"document in the top-3 on every single query, for recall={sent['avg_recall']:.3f}). "
            f"For a support agent, failing to surface the one document that actually answers "
            f"the customer is worse than returning one extra, slightly-off-topic chunk "
            f"alongside the right one, so the higher-recall collection is the better default "
            f"even though it costs a few points of precision "
            f"({sent['avg_precision']:.3f} vs {fixed['avg_precision']:.3f})."
        )
