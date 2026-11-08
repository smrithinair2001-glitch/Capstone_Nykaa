"""
Part 2 / Task 7 (tools) -- the two CrewAI tools used by the crew.

``rag_lookup`` is deliberately named to contain the substring "lookup"
even though it is the RAG/policy-retrieval tool, not the order-lookup
tool -- see app/crew/mock_llm.py's module docstring for why.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from crewai.tools import tool  # noqa: E402

from app.rag.generation import grounded_generate  # noqa: E402
from app.rag.index import SENTENCE_COLLECTION  # noqa: E402
from app.tools.order_lookup import check_order_status as _check_order_status  # noqa: E402

# Simple call log so the demo/tests can verify "this tool actually ran"
# without depending on CrewAI's own (summary-only) usage counters.
TOOL_CALL_LOG: list[dict] = []


@tool("rag_lookup")
def rag_lookup(query: str) -> str:
    """Answer a Nykaa customer-support POLICY question (returns, refunds,
    warranty, cancellation, loyalty points, shipping, escalation, etc.) by
    retrieving grounded passages from the Nykaa policy knowledge base.
    Input is the customer's question as plain text. Returns either a
    grounded answer composed only from retrieved policy text, or an
    explicit "I don't know" fallback if nothing in the knowledge base is
    similar enough to the question.
    """
    TOOL_CALL_LOG.append({"tool": "rag_lookup", "args": {"query": query}})
    result = grounded_generate(query, collection=SENTENCE_COLLECTION)
    return result["answer"]


@tool("check_order_status")
def check_order_status(record_id: str) -> str:
    """Look up a specific Nykaa ORDER by its record_id (format NYK-XXXXX)
    and return its status, value, and escalation signal. Input is the
    order's record_id as plain text, e.g. "NYK-00002".
    """
    TOOL_CALL_LOG.append({"tool": "check_order_status", "args": {"record_id": record_id}})
    try:
        result = _check_order_status(record_id)
    except KeyError as e:
        return f"Order lookup failed: {e}"
    return (
        f"Order {result['record_id']}: status={result['status']}, "
        f"order_value_inr={result['order_value_inr']}, "
        f"escalation_score={result['escalation_score']}, "
        f"should_escalate={result['should_escalate']}"
    )


if __name__ == "__main__":
    print(rag_lookup.name, "::", rag_lookup.args_schema.model_fields.keys())
    print(check_order_status.name, "::", check_order_status.args_schema.model_fields.keys())
    print()
    print(rag_lookup.run(query="How long do I have to return a kurta?"))
    print()
    print(check_order_status.run(record_id="NYK-00002"))
