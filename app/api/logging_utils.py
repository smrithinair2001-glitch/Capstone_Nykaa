"""Structured JSON-Lines (ELK-style) per-request logging for Part 3 / Task 12.

Each request gets one JSON object per line, written to ``logs/requests.jsonl``,
with a trace id and timing. The logged query text is PII-masked with the
exact same ``app.guardrails.mask_pii`` function applied on the model-input
side, so raw unmasked phone numbers / payment-card numbers are never
persisted to disk.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from app.guardrails import mask_pii

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_PATH = LOG_DIR / "requests.jsonl"


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def log_event(
    trace_id: str,
    endpoint: str,
    start_time: float,
    *,
    raw_text: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Appends one masked, structured JSON line to logs/requests.jsonl."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    duration_ms = round((time.monotonic() - start_time) * 1000, 2)

    record: dict[str, Any] = {
        "trace_id": trace_id,
        "endpoint": endpoint,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_ms": duration_ms,
    }
    if raw_text is not None:
        masked_text, pii_found = mask_pii(raw_text)
        record["request_text_masked"] = masked_text
        record["pii_fields_masked"] = pii_found
    if extra:
        record.update(extra)

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
