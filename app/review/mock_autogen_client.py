"""Deterministic, keyless mock `ChatCompletionClient` for the Autogen review stage.

Part 4 / Task 14 asks for a 2-agent Autogen ``RoundRobinGroupChat`` team
(a Policy-Compliance-Reviewer and a Final-Editor) that reviews the CrewAI
Composer's draft answer against the originally retrieved RAG context, and
either approves it unchanged or revises it. Everything must run under
``MOCK_LLM`` with zero API keys and zero network access, exactly like the
CrewAI side (see ``app/crew/mock_llm.py``).

Autogen's model-client abstraction is ``autogen_core.models.ChatCompletionClient``,
an async, ABC-based interface (very different from CrewAI's synchronous
``BaseLLM``). Rather than hand-scripting a fixed reply sequence with the
built-in ``autogen_ext.models.replay.ReplayChatCompletionClient`` (which
cannot branch on the actual content of a draft answer), this module
implements a small custom client with real, content-aware decision logic,
mirroring the spirit of the CrewAI ``MockLLM``:

- One client instance is built per *role* (``"reviewer"`` or ``"editor"``),
  since in this design each role is a dedicated agent with its own client,
  so no cross-call state tracking is required (unlike the CrewAI MockLLM,
  which had to share one client across multiple agents and task turns).
- The **Reviewer** scans the draft answer for a small set of deliberately
  "ungrounded claim" marker phrases -- sentences a draft could contain that
  are NOT supported by the retrieved policy context. If any marker is
  present, it emits a "REVISE: ..." verdict naming the issue; otherwise it
  emits "APPROVE: ...".
- The **Final-Editor** reads the Reviewer's verdict text plus the original
  draft, and emits the structured ``ReviewVerdict`` JSON: unchanged draft
  when approved, or the draft with the offending sentence(s) stripped out
  when revised.
"""

from __future__ import annotations

import json
import re
from typing import Any, AsyncGenerator, Literal, Mapping, Optional, Sequence, Union

from autogen_core import CancellationToken
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelFamily,
    ModelInfo,
    RequestUsage,
)
from autogen_core.tools import Tool, ToolSchema
from pydantic import BaseModel

from app.review.verdict import ReviewVerdict

# Deliberately "ungrounded" claim markers. A draft answer containing any of
# these phrases is making a promise that is NOT backed by any of the 12 KB
# policy documents -- used to force a demonstrable "REVISE" branch in the
# demo without requiring a real LLM to hallucinate on command.
UNGROUNDED_CLAIM_MARKERS: list[str] = [
    "lifetime warranty on all items",
    "completely free of all shipping and handling charges",
    "guaranteed overnight delivery to any location",
]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _text(msg: Any) -> str:
    """Best-effort plain-text extraction from an Autogen LLMMessage."""
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return " ".join(parts)
    return str(content)


def _extract_section(text: str, start_marker: str, end_marker: Optional[str] = None) -> str:
    lower = text.lower()
    idx = lower.find(start_marker.lower())
    if idx == -1:
        return ""
    body = text[idx + len(start_marker):]
    if end_marker:
        end_idx = body.lower().find(end_marker.lower())
        if end_idx != -1:
            body = body[:end_idx]
    return body.strip()


def _find_marker(draft_text: str) -> Optional[str]:
    lower = draft_text.lower()
    for marker in UNGROUNDED_CLAIM_MARKERS:
        if marker.lower() in lower:
            return marker
    return None


def _strip_sentences_containing(text: str, marker: str) -> str:
    sentences = _SENTENCE_SPLIT_RE.split(text)
    kept = [s for s in sentences if marker.lower() not in s.lower()]
    return " ".join(s.strip() for s in kept if s.strip())


class MockAutogenChatCompletionClient(ChatCompletionClient):
    """Deterministic, keyless stand-in for a real LLM model client.

    ``role`` selects the agent behavior: ``"reviewer"`` scans the draft for
    ungrounded claims and emits a plain-text APPROVE/REVISE verdict;
    ``"editor"`` reads that verdict plus the original task and emits the
    final structured :class:`ReviewVerdict` JSON.
    """

    def __init__(self, role: Literal["reviewer", "editor"]) -> None:
        if role not in ("reviewer", "editor"):
            raise ValueError(f"Unknown role: {role!r}")
        self.role = role
        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    # -- core API -----------------------------------------------------
    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Any = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        joined = "\n".join(_text(m) for m in messages)
        prompt_tokens = max(1, len(joined) // 4)

        if self.role == "reviewer":
            content = self._reviewer_reply(messages)
        else:
            content = self._editor_reply(messages)

        completion_tokens = max(1, len(content) // 4)
        self._actual_usage = RequestUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        self._total_usage = RequestUsage(
            prompt_tokens=self._total_usage.prompt_tokens + prompt_tokens,
            completion_tokens=self._total_usage.completion_tokens + completion_tokens,
        )
        return CreateResult(finish_reason="stop", content=content, usage=self._actual_usage, cached=False)

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Any = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        result = await self.create(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )

        async def _gen() -> AsyncGenerator[Union[str, CreateResult], None]:
            yield result

        return _gen()

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._actual_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return max(1, sum(len(_text(m)) for m in messages) // 4)

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return 100_000 - self.count_tokens(messages, tools=tools)

    @property
    def capabilities(self) -> dict:
        return {"vision": False, "function_calling": False, "json_output": True}

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            vision=False,
            function_calling=False,
            json_output=True,
            family=ModelFamily.UNKNOWN,
            structured_output=True,
        )

    # -- role-specific logic -------------------------------------------
    def _reviewer_reply(self, messages: Sequence[LLMMessage]) -> str:
        joined = "\n".join(_text(m) for m in messages)
        draft_text = _extract_section(joined, "Draft answer:", "Retrieved policy context:")
        marker = _find_marker(draft_text or joined)
        if marker is not None:
            return (
                f"REVISE: The draft contains a claim not supported by the retrieved policy "
                f"context: \"{marker}\". This must be removed before the answer is sent to "
                f"the customer."
            )
        return "APPROVE: The draft is fully consistent with the retrieved policy context."

    def _editor_reply(self, messages: Sequence[LLMMessage]) -> str:
        joined = "\n".join(_text(m) for m in messages)
        draft_text = _extract_section(joined, "Draft answer:", "Retrieved policy context:") or joined
        reviewer_text = _text(messages[-1]) if messages else ""

        if reviewer_text.strip().upper().startswith("REVISE"):
            marker = _find_marker(draft_text)
            final_answer = _strip_sentences_containing(draft_text, marker) if marker else draft_text
            reason = reviewer_text.split(":", 1)[-1].strip() if ":" in reviewer_text else reviewer_text
            verdict = ReviewVerdict(approved=False, final_answer=final_answer, reason=reason)
        else:
            reason = reviewer_text.split(":", 1)[-1].strip() if ":" in reviewer_text else reviewer_text
            verdict = ReviewVerdict(approved=True, final_answer=draft_text, reason=reason or "Approved as drafted.")

        return verdict.model_dump_json()
