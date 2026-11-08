"""Pydantic verdict model produced by the Autogen Final-Editor agent (Part 4, Task 14)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ReviewVerdict(BaseModel):
    """Structured verdict the Final-Editor agent must emit via `output_content_type`."""

    approved: bool = Field(
        ..., description="True if the draft answer was approved unchanged; False if it was revised."
    )
    final_answer: str = Field(
        ..., description="The answer to actually send to the customer (unchanged draft, or the revised text)."
    )
    reason: str = Field(
        ..., description="One or two sentences explaining the approve/revise decision."
    )
