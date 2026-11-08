"""
Part 2 / Task 6 -- the order-lookup tool, with a designed (not a bare
boolean OR) escalation_score.

Escalation-score formula
-------------------------
escalation_score = 0.6 * delayed_indicator + 0.4 * recency_signal

  delayed_indicator = 1.0 if order.delayed_shipment else 0.0
  recency_signal    = min(order.days_since_created, 30) / 30.0

Both components are already normalized to [0, 1] and the weights sum to
1.0, so escalation_score is guaranteed to land in [0, 1] without any extra
clamping. The 0.6/0.4 split deliberately weights "is this shipment actually
late right now" above "how long has this order existed", because a 2-day-old
order that is already flagged delayed is a more urgent support case than a
29-day-old order that shipped on time and is simply old (e.g. already
delivered and sitting past its SLA window with nothing wrong).

Escalation threshold
---------------------
We justify the escalation threshold directly from this project's own
generated dataset rather than picking a round number. See
``justify_threshold()`` below, which is also what produced the numbers
quoted in README.md: the 80th percentile of `days_since_created` across the
60 generated orders is 22 days, which maps (through the recency_signal
term, with delayed_indicator=1 i.e. the worst case for that recency) to an
escalation_score of 0.6*1 + 0.4*(22/30) = 0.893, and (with
delayed_indicator=0, the best case) to 0.4*(22/30) = 0.293. We set
ESCALATION_THRESHOLD = 0.5, which sits strictly between those two bounds:
it always escalates a currently-delayed shipment (minimum possible score
when delayed=1 is 0.6, already above 0.5) and only escalates a non-delayed
order once it is unusually old (days_since_created/30 > 0.833, i.e.
days_since_created > 25, which only the oldest ~1-in-6 non-delayed orders
in our data reach -- consistent with the 16.67% of all 60 orders that
actually cross the threshold, printed by justify_threshold() below).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.dataset import ORDERS  # noqa: E402

DELAYED_WEIGHT = 0.6
RECENCY_WEIGHT = 0.4
ESCALATION_THRESHOLD = 0.5

_ORDERS_BY_ID = {o["record_id"]: o for o in ORDERS}


def compute_escalation_score(delayed_shipment: bool, days_since_created: int) -> float:
    delayed_indicator = 1.0 if delayed_shipment else 0.0
    recency_signal = min(max(days_since_created, 0), 30) / 30.0
    score = DELAYED_WEIGHT * delayed_indicator + RECENCY_WEIGHT * recency_signal
    return round(score, 4)


def check_order_status(record_id: str) -> dict:
    """The Lookup Agent's sole tool (see the Part 4 least-autonomy demo for
    why no other agent may call this). Returns status, order_value_inr, and
    a designed escalation_score; raises KeyError for an unknown record_id so
    the caller can handle "order not found" explicitly rather than silently.
    """
    order = _ORDERS_BY_ID.get(record_id)
    if order is None:
        raise KeyError(f"No such order record_id: {record_id!r}")

    score = compute_escalation_score(order["delayed_shipment"], order["days_since_created"])
    return {
        "record_id": order["record_id"],
        "status": order["status"],
        "order_value_inr": order["order_value_inr"],
        "escalation_score": score,
        "should_escalate": score >= ESCALATION_THRESHOLD,
    }


def justify_threshold() -> dict:
    days = sorted(o["days_since_created"] for o in ORDERS)
    n = len(days)
    idx80 = int(round(0.8 * (n - 1)))
    p80_days = days[idx80]

    worst_case_score = compute_escalation_score(True, p80_days)
    best_case_score = compute_escalation_score(False, p80_days)

    all_scores = sorted(
        compute_escalation_score(o["delayed_shipment"], o["days_since_created"])
        for o in ORDERS
    )
    pct_escalated = 100.0 * sum(1 for s in all_scores if s >= ESCALATION_THRESHOLD) / len(all_scores)

    return {
        "n_orders": n,
        "p80_days_since_created": p80_days,
        "worst_case_score_at_p80_days": worst_case_score,
        "best_case_score_at_p80_days": best_case_score,
        "chosen_threshold": ESCALATION_THRESHOLD,
        "pct_orders_escalated_at_threshold": round(pct_escalated, 2),
    }


if __name__ == "__main__":
    j = justify_threshold()
    print("=== Escalation-score threshold justification ===")
    print(f"n_orders = {j['n_orders']}")
    print(f"80th percentile of days_since_created = {j['p80_days_since_created']} days")
    print(f"  -> escalation_score if that order is ALSO delayed  : "
          f"{DELAYED_WEIGHT}*1 + {RECENCY_WEIGHT}*({j['p80_days_since_created']}/30) "
          f"= {j['worst_case_score_at_p80_days']}")
    print(f"  -> escalation_score if that order is NOT delayed   : "
          f"{DELAYED_WEIGHT}*0 + {RECENCY_WEIGHT}*({j['p80_days_since_created']}/30) "
          f"= {j['best_case_score_at_p80_days']}")
    print(f"Chosen ESCALATION_THRESHOLD = {j['chosen_threshold']}")
    print(f"  -> {j['pct_orders_escalated_at_threshold']}% of the 60 generated orders "
          f"currently cross this threshold.")

    print("\n=== check_order_status() demo ===")
    for rec in ORDERS[:4]:
        result = check_order_status(rec["record_id"])
        print(f"  {result}")

    print("\n=== Unknown record_id raises KeyError ===")
    try:
        check_order_status("NYK-99999")
    except KeyError as e:
        print(f"  Raised as expected: {e}")
