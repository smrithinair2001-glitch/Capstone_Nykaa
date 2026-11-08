"""Part 2 / Task 9 -- the structured-output schema every crew response must
conform to.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SupportResponse(BaseModel):
    """Structured response the Composer agent must produce for every ticket."""

    answer: str = Field(..., description="The final, customer-facing answer text.")
    used_policy_kb: bool = Field(
        ..., description="True if the policy knowledge base (rag_lookup) was consulted."
    )
    used_order_lookup: bool = Field(
        ..., description="True if a specific order was looked up (check_order_status)."
    )
    should_escalate: bool = Field(
        ..., description="True if this ticket should be escalated to a human associate."
    )
