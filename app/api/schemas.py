"""Pydantic request/response models for the FastAPI deployment (Part 3 / Task 11)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The customer's support question.")
    session_id: str | None = Field(
        default=None, description="Optional session id; unused by /ask (stateless), used by the WebSocket chat."
    )


class AskResponse(BaseModel):
    trace_id: str
    answer: str
    is_fallback: bool
    top1_similarity: float
    cache_hit: bool
    blocked_prompt_injection: bool


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(..., min_length=1, description="Slug for the new KB document, e.g. '13_gift_cards'.")
    title: str = Field(..., min_length=1, description="Document title (written as the markdown H1).")
    text: str = Field(..., min_length=1, description="Policy document body text.")


class AddDocumentResponse(BaseModel):
    trace_id: str
    doc_id: str
    fixed_overlap_chunks_total: int
    sentence_chunks_total: int


class ChatMessage(BaseModel):
    role: str
    content: str
