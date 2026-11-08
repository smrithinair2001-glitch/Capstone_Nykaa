"""
Part 2 / Task 7 -- MOCK_LLM, a deterministic drop-in LLM for CrewAI that
needs zero API keys and zero network access.

Implemented by extending ``crewai.llms.base_llm.BaseLLM`` (CrewAI's own
documented extension point for a non-litellm LLM), per the brief.

Two documented pitfalls this file deliberately avoids
-------------------------------------------------------
1. CrewAI's built-in ReAct system prompt (see
   ``crewai/translations/en.json``, key "tools") contains the literal
   example line ``Observation: the result of the action`` as part of the
   few-shot instructions shown to the agent -- *before* any tool has ever
   run. A parser that scans the *whole* conversation for the substring
   "Observation:" to decide "has a tool result already come back?" will
   match that template text on the very first LLM call and silently treat
   the placeholder text as a real answer. We never scan the whole
   conversation: we track per-task state ourselves (keyed by
   ``id(from_task)``/``id(from_agent)``, which CrewAI passes into every
   ``call()``) and, when we do need to read a tool's result, we read it
   from exactly one place -- the content of ``messages[-1]`` -- which is
   only ever the real ``Observation: <result>`` message CrewAI appends
   right before re-invoking the LLM (see
   ``crewai/agents/step_executor.py::_build_observation_message``).

2. We never decide which tool applies by testing whether a keyword is a
   substring of the tool's *name* (e.g. checking ``"lookup" in tool.name``
   would misclassify a tool literally named ``rag_lookup`` as the
   order-lookup tool, when it is actually the RAG retrieval tool). Instead
   we dispatch off each tool's own declared argument schema
   (``tool.args_schema.model_fields``, read straight from the ``BaseTool``
   objects CrewAI hands us via ``from_agent.tools``): a tool whose schema
   requires ``record_id`` is the order-lookup tool; a tool whose schema
   requires ``query`` is the RAG tool. The RAG tool in this project is in
   fact named ``rag_lookup`` specifically to exercise this.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from crewai.llms.base_llm import BaseLLM  # noqa: E402
from pydantic import PrivateAttr  # noqa: E402

ORDER_ID_RE = re.compile(r"\bNYK-\d{4,6}\b", re.IGNORECASE)


class MockLLM(BaseLLM):
    """Deterministic MOCK_LLM. model name is cosmetic; no network call is
    ever made by this class.
    """

    model: str = "mock/deterministic-v1"

    # Per-task state: id(from_task or from_agent) -> "awaiting_observation" | "done"
    _task_state: dict[int, str] = PrivateAttr(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("model", "mock/deterministic-v1")
        super().__init__(**kwargs)

    def supports_function_calling(self) -> bool:
        # Deliberately False: this project exercises CrewAI's *text*
        # (ReAct) tool-calling path, not native OpenAI-style function
        # calling, which is what surfaces the two pitfalls documented above.
        return False

    # -- dispatch helpers ---------------------------------------------

    @staticmethod
    def _extract_order_id(text: str) -> str | None:
        m = ORDER_ID_RE.search(text)
        return m.group(0).upper() if m else None

    @staticmethod
    def _last_message_content(messages: list[dict]) -> str:
        if not messages:
            return ""
        content = messages[-1].get("content", "")
        if isinstance(content, list):
            # Multimodal content blocks; concatenate any text parts.
            parts = [c.get("text", "") for c in content if isinstance(c, dict)]
            return " ".join(parts)
        return str(content)

    def _select_tool_for_agent(self, from_agent: Any):
        """Return the single BaseTool configured on this agent, or None."""
        tools = getattr(from_agent, "tools", None) or []
        return tools[0] if tools else None

    @staticmethod
    def _tool_schema_fields(bt) -> set[str]:
        schema = getattr(bt, "args_schema", None)
        if schema is None:
            return set()
        return set(schema.model_fields.keys())

    def _build_action_text(self, bt, query_text: str) -> str | None:
        """Build the 'Thought/Action/Action Input' text for a tool call,
        choosing the tool's arguments by its declared schema fields
        (never by matching a keyword against the tool's name). Returns
        None if this tool needs a record_id and none is present in the
        query -- the caller should then skip the tool entirely rather than
        invoke it with a made-up id.
        """
        fields = self._tool_schema_fields(bt)
        if "record_id" in fields:
            order_id = self._extract_order_id(query_text)
            if order_id is None:
                return None
            args = {"record_id": order_id}
        elif "query" in fields:
            args = {"query": query_text.strip()}
        else:
            # Unknown schema shape: pass the raw text under the first field.
            first_field = next(iter(fields), "input")
            args = {first_field: query_text.strip()}

        return (
            f"Thought: I need to use the {bt.name} tool to answer this.\n"
            f"Action: {bt.name}\n"
            f"Action Input: {json.dumps(args)}"
        )

    # -- BaseLLM interface ----------------------------------------------

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        task_key = id(from_task) if from_task is not None else id(from_agent)
        state = self._task_state.get(task_key, "start")

        bt = self._select_tool_for_agent(from_agent)

        if bt is not None and state == "start":
            # First turn for a tool-equipped agent: decide + emit an Action
            # (or, if this tool needs a record_id that simply isn't present
            # in the ticket, skip the tool and finish immediately).
            query_text = self._last_message_content(messages)
            action_text = self._build_action_text(bt, query_text)
            if action_text is None:
                self._task_state[task_key] = "done"
                return (
                    "Thought: This ticket does not reference a specific order id, "
                    "so there is nothing for the order-lookup tool to check.\n"
                    "Final Answer: No order id referenced in this ticket; "
                    "order lookup skipped."
                )
            self._task_state[task_key] = "awaiting_observation"
            return action_text

        if bt is not None and state == "awaiting_observation":
            # Second turn. CrewAI's executor appends the real tool result as
            # a trailing "Observation: ..." suffix on the message that
            # echoes *our own* previous Action text (messages[-2]), then
            # appends its own generic "Analyze the tool result..." reasoning
            # prompt as the actual last message (messages[-1]). So: we read
            # from our own echoed message, not the system-prompt template
            # (pitfall 1), and specifically take the text after the LAST
            # "Observation:" marker in that one message -- never scanning
            # the full conversation/system prompt for the substring.
            own_echoed_message = messages[-2] if len(messages) >= 2 else messages[-1]
            echoed_content = own_echoed_message.get("content", "")
            if isinstance(echoed_content, list):
                echoed_content = " ".join(
                    c.get("text", "") for c in echoed_content if isinstance(c, dict)
                )
            observation = str(echoed_content).rsplit("Observation:", 1)[-1].strip()
            self._task_state[task_key] = "done"
            return (
                "Thought: I now have the information I need.\n"
                f"Final Answer: {observation}"
            )

        # No tools on this agent (the Composer) -- synthesize directly from
        # whatever context CrewAI has placed in the latest message (which
        # includes the prior tasks' outputs via Task(context=[...])), and
        # return it as PURE JSON matching SupportResponse: CrewAI's own
        # fast-path (crewai/utilities/converter.py: convert_to_model) tries
        # `json.loads()` on our Final-Answer text before ever considering an
        # extra "reformat it" LLM round trip, so returning clean JSON here
        # means that extra round trip never happens.
        context_text = self._last_message_content(messages)
        answer_json = self._compose_final_answer(context_text)
        return (
            "Thought: I will combine the retrieved policy information and "
            "the order lookup result into one customer-facing answer.\n"
            f"Final Answer: {answer_json}"
        )

    def _compose_final_answer(self, context_text: str) -> str:
        """Deterministic composition for the Response Composer agent. Pulls
        out whatever the upstream agents (Retrieval, Lookup) actually
        produced from the task context and merges them into one
        SupportResponse JSON object -- produced with zero model calls,
        using only the grounded text already present in context_text.
        """
        from app.rag.generation import FALLBACK_ANSWER

        # CrewAI's task prompt wraps the actual upstream context between a
        # "This is the context you're working with:" marker and a trailing
        # "Provide your complete response:" marker (with an output_pydantic
        # schema preamble before it, which we deliberately strip here so it
        # is never mistaken for real retrieval/lookup content). Isolate just
        # the real context before splitting on CrewAI's own
        # "----------" separator between the two upstream tasks' outputs
        # (observed empirically; see transcripts).
        working_text = context_text
        marker = "context you're working with:"
        idx = working_text.lower().find(marker)
        if idx != -1:
            working_text = working_text[idx + len(marker):]
        end_marker = "Provide your complete response:"
        end_idx = working_text.find(end_marker)
        if end_idx != -1:
            working_text = working_text[:end_idx]

        parts = [p.strip() for p in working_text.split("----------") if p.strip()]
        retrieval_part = parts[0] if len(parts) >= 1 else ""
        lookup_part = parts[1] if len(parts) >= 2 else ""

        used_policy_kb = bool(retrieval_part) and FALLBACK_ANSWER.split(",")[0] not in retrieval_part
        used_order_lookup = "Order NYK-" in lookup_part or "Order NYK-" in working_text
        should_escalate = (
            "should_escalate=True" in lookup_part
            or "should_escalate=True" in working_text
            or not used_policy_kb
        )

        answer_bits = []
        if retrieval_part:
            answer_bits.append(retrieval_part)
        if lookup_part and "No order id referenced" not in lookup_part:
            answer_bits.append(lookup_part)
        if not answer_bits:
            answer_bits.append(FALLBACK_ANSWER)
        answer_text = "\n\n".join(answer_bits)

        obj = {
            "answer": answer_text,
            "used_policy_kb": used_policy_kb,
            "used_order_lookup": used_order_lookup,
            "should_escalate": should_escalate,
        }
        return json.dumps(obj)


_SHARED_MOCK_LLM: MockLLM | None = None


def get_mock_llm() -> MockLLM:
    global _SHARED_MOCK_LLM
    if _SHARED_MOCK_LLM is None:
        _SHARED_MOCK_LLM = MockLLM()
    return _SHARED_MOCK_LLM
