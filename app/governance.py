"""
Part 4 / Tasks 15-16 -- the four-layer AI-governance model (Application
layer: least autonomy + risk classification; Runtime layer: per-request
cost-budget cap) and response caching.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# -- Application layer: least autonomy ---------------------------------------


def verify_least_autonomy() -> dict:
    """Verify that only the Lookup Agent's tool list includes
    check_order_status. We never wire the tool to the Retrieval Agent or
    the Composer Agent in the first place (see app/crew/crew_build.py) --
    this function demonstrates and locks in that guarantee rather than
    just hoping no one adds it elsewhere later.
    """
    from app.crew.crew_build import build_agents
    from app.crew.tools import check_order_status

    agents = build_agents()
    report = {}
    for name, agent in agents.items():
        tool_names = [t.name for t in (agent.tools or [])]
        report[name] = tool_names

    lookup_has_it = check_order_status.name in report["lookup"]
    others_dont = all(
        check_order_status.name not in tools
        for key, tools in report.items()
        if key != "lookup"
    )
    return {
        "tools_by_agent": report,
        "only_lookup_agent_has_check_order_status": lookup_has_it and others_dont,
    }


RISK_CLASSIFICATION = "Medium"
RISK_JUSTIFICATION = (
    "This system falls into the Medium risk tier of the Low/Medium/High scheme: it is a "
    "customer-support ticket system (the scheme's own example of a Medium-risk workload), "
    "not a Low-risk pure summarization/transcription tool, and not a High-risk system making "
    "medical, hiring, or financial decisions. It does read order value (order_value_inr) and "
    "escalation signals, but only to route a support ticket, not to approve a loan, set a "
    "price, or make any irreversible financial decision on its own -- every should_escalate "
    "outcome still lands on a human associate (see the escalation matrix in the knowledge "
    "base), so the system's own autonomy stops at routing/drafting, consistent with Medium "
    "rather than High risk."
)


# -- Runtime layer: per-request cost-budget cap ------------------------------

# Deliberately small for this demo so a normal ticket passes and an
# artificially long one is rejected -- in a real deployment this would be
# sized to the model's actual context window / a real per-request $ budget.
MAX_REQUEST_TOKENS = 500
CHARS_PER_TOKEN_ESTIMATE = 4


class BudgetExceededError(Exception):
    pass


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def enforce_request_budget(text: str, max_tokens: int = MAX_REQUEST_TOKENS) -> int:
    """Raises BudgetExceededError if the request is estimated to exceed the
    per-request token/cost budget. Returns the estimated token count
    otherwise.
    """
    estimated = estimate_tokens(text)
    if estimated > max_tokens:
        raise BudgetExceededError(
            f"Request is estimated at {estimated} tokens, which exceeds the "
            f"per-request budget cap of {max_tokens} tokens."
        )
    return estimated


# -- Response caching ---------------------------------------------------------

_RESPONSE_CACHE: dict[str, dict] = {}
_CACHE_STATS = {"hits": 0, "misses": 0}


def _normalize(query: str) -> str:
    return " ".join(query.strip().lower().split())


def cached_grounded_generate(query: str, **kwargs) -> dict:
    """Thin cache wrapper around app.rag.generation.grounded_generate, keyed
    by normalized query text.
    """
    from app.rag.generation import grounded_generate

    key = _normalize(query)
    if key in _RESPONSE_CACHE:
        _CACHE_STATS["hits"] += 1
        return {**_RESPONSE_CACHE[key], "cache_hit": True}

    _CACHE_STATS["misses"] += 1
    result = grounded_generate(query, **kwargs)
    _RESPONSE_CACHE[key] = result
    return {**result, "cache_hit": False}


def cache_stats() -> dict:
    return dict(_CACHE_STATS)


if __name__ == "__main__":
    print("=== Task 15a: Application-layer least-autonomy ===")
    report = verify_least_autonomy()
    for name, tools in report["tools_by_agent"].items():
        print(f"  {name} agent tools: {tools}")
    print(f"  only_lookup_agent_has_check_order_status = "
          f"{report['only_lookup_agent_has_check_order_status']}")
    assert report["only_lookup_agent_has_check_order_status"]
    print("  [PASS] check_order_status is wired to exactly one agent (Lookup Agent); "
          "it is never attached to Retrieval or Composer, so they cannot call it even "
          "if their prompts asked them to.")

    print("\n=== Task 15b: Risk classification ===")
    print(f"  Risk level: {RISK_CLASSIFICATION}")
    print(f"  Justification: {RISK_JUSTIFICATION}")

    print("\n=== Task 15c: Runtime per-request cost-budget cap ===")
    normal_ticket = "How many days do I have to return a kurta I bought?"
    est = enforce_request_budget(normal_ticket)
    print(f"  normal ticket ({len(normal_ticket)} chars) -> estimated {est} tokens -> accepted "
          f"(cap={MAX_REQUEST_TOKENS})")

    oversized_ticket = (
        "I need help with my order. " * 120
    )  # deliberately oversized
    print(f"  oversized ticket length: {len(oversized_ticket)} chars, "
          f"estimated {estimate_tokens(oversized_ticket)} tokens")
    try:
        enforce_request_budget(oversized_ticket)
        raise SystemExit("Expected BudgetExceededError was not raised!")
    except BudgetExceededError as e:
        print(f"  [PASS] guardrail fired: {e}")

    print("\n=== Task 16: Response caching ===")
    from app.rag.index import SENTENCE_COLLECTION

    q = "How many days do I have to return a kurta I bought?"

    t0 = time.perf_counter()
    r1 = cached_grounded_generate(q, collection=SENTENCE_COLLECTION)
    t1 = time.perf_counter()
    print(f"  1st call: cache_hit={r1['cache_hit']}  elapsed={t1 - t0:.4f}s")

    t0 = time.perf_counter()
    r2 = cached_grounded_generate(q, collection=SENTENCE_COLLECTION)
    t1 = time.perf_counter()
    print(f"  2nd call (identical query): cache_hit={r2['cache_hit']}  elapsed={t1 - t0:.4f}s")

    stats = cache_stats()
    print(f"  cache stats: {stats}")
    assert r1["cache_hit"] is False and r2["cache_hit"] is True
    assert stats["hits"] == 1 and stats["misses"] == 1
    print("  [PASS] repeated identical query produced a cache hit and skipped the "
          "redundant retrieval call (evidenced by the hit/miss counters above, and the "
          "2nd call's near-zero elapsed time vs. the 1st).")
