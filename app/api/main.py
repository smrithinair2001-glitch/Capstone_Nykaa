"""Part 3 / Task 11 -- FastAPI deployment.

Two HTTP endpoints (``POST /ask``, ``POST /add-document``), both typed with
Pydantic request/response models, plus one WebSocket endpoint
(``/ws/chat/{session_id}``) for real-time multi-turn chat that gracefully
handles a client disconnect (catches ``WebSocketDisconnect`` so the server
keeps running for every other connected client).

Every request is logged as one structured JSON line (Part 3 / Task 12) with
a trace id and timing, and the logged text is PII-masked with the same
guardrail applied to what the model sees -- no raw unmasked phone number or
payment-card number ever reaches the log file.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import app  # noqa: E402  (sets MOCK_LLM / telemetry env vars on import)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402

from app.api.logging_utils import log_event, new_trace_id  # noqa: E402
from app.api.schemas import (  # noqa: E402
    AddDocumentRequest,
    AddDocumentResponse,
    AskRequest,
    AskResponse,
)
from app.crew.memory_demo import ask as memory_ask  # noqa: E402
from app.governance import cached_grounded_generate  # noqa: E402
from app.guardrails import detect_prompt_injection, mask_pii  # noqa: E402
from app.rag.index import add_document as _add_document  # noqa: E402

api = FastAPI(title="Nykaa Domain Support Agent API")


@api.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    trace_id = new_trace_id()
    start = time.monotonic()

    if detect_prompt_injection(req.query):
        log_event(
            trace_id, "/ask", start, raw_text=req.query,
            extra={"blocked_prompt_injection": True},
        )
        return AskResponse(
            trace_id=trace_id,
            answer=(
                "This request looks like an attempt to manipulate the assistant's "
                "instructions or access another customer's data, so I can't act on it. "
                "It has been escalated to Trust & Safety."
            ),
            is_fallback=True,
            top1_similarity=0.0,
            cache_hit=False,
            blocked_prompt_injection=True,
        )

    result = cached_grounded_generate(req.query)
    log_event(
        trace_id, "/ask", start, raw_text=req.query,
        extra={
            "is_fallback": result["is_fallback"],
            "top1_similarity": result["top1_similarity"],
            "cache_hit": result["cache_hit"],
            "blocked_prompt_injection": False,
        },
    )
    return AskResponse(
        trace_id=trace_id,
        answer=result["answer"],
        is_fallback=result["is_fallback"],
        top1_similarity=result["top1_similarity"],
        cache_hit=result["cache_hit"],
        blocked_prompt_injection=False,
    )


@api.post("/add-document", response_model=AddDocumentResponse)
def add_document(req: AddDocumentRequest) -> AddDocumentResponse:
    trace_id = new_trace_id()
    start = time.monotonic()

    n_fixed, n_sent = _add_document(req.doc_id, req.title, req.text)

    log_event(
        trace_id, "/add-document", start, raw_text=f"{req.title}: {req.text}",
        extra={"doc_id": req.doc_id, "fixed_overlap_chunks_total": n_fixed,
               "sentence_chunks_total": n_sent},
    )
    return AddDocumentResponse(
        trace_id=trace_id,
        doc_id=req.doc_id,
        fixed_overlap_chunks_total=n_fixed,
        sentence_chunks_total=n_sent,
    )


@api.websocket("/ws/chat/{session_id}")
async def ws_chat(websocket: WebSocket, session_id: str) -> None:
    """Real-time multi-turn chat. Session memory (Part 2 / Task 8) is keyed
    by ``session_id``, so multiple clients can chat concurrently without
    sharing history, and a disconnect from one client never takes the
    server down for anyone else.
    """
    await websocket.accept()
    try:
        while True:
            raw_text = await websocket.receive_text()
            trace_id = new_trace_id()
            start = time.monotonic()

            masked_text, _ = mask_pii(raw_text)
            if detect_prompt_injection(raw_text):
                answer = (
                    "This message looks like an attempt to manipulate the assistant's "
                    "instructions, so I can't act on it."
                )
                blocked = True
            else:
                # CrewAI's synchronous `.kickoff()` refuses to run from inside
                # a running asyncio event loop (which this WebSocket handler
                # is), so the (synchronous) memory-backed crew call is
                # offloaded to a worker thread instead of awaited directly.
                answer = await asyncio.to_thread(memory_ask, session_id, raw_text)
                blocked = False

            log_event(
                trace_id, "/ws/chat", start, raw_text=raw_text,
                extra={"session_id": session_id, "blocked_prompt_injection": blocked},
            )
            await websocket.send_json({
                "trace_id": trace_id,
                "session_id": session_id,
                "answer": answer,
            })
    except WebSocketDisconnect:
        # Expected on normal client close/navigate-away. Swallow it here so
        # this one connection's teardown never crashes the server process or
        # affects any other client's open WebSocket.
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api, host="0.0.0.0", port=8000)
