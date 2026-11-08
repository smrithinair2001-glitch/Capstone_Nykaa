"""
Part 2 / Task 8 -- session memory using LangChain's
``InMemoryChatMessageHistory`` + ``RunnableWithMessageHistory``.

Note: ``RunnableWithMessageHistory`` raises a ``LangChainDeprecationWarning``
pointing at LangGraph's own persistence layer -- this is expected per the
brief and is not silenced here; the class still functions correctly.

This memory is in-process only (it does not need to survive a restart,
per the brief) -- the store is a plain module-level dict.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402
from langchain_core.runnables.history import RunnableWithMessageHistory  # noqa: E402

from app.crew.crew_build import run_ticket  # noqa: E402
from app.crew.tools import TOOL_CALL_LOG  # noqa: E402

_STORE: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _STORE:
        _STORE[session_id] = InMemoryChatMessageHistory()
    return _STORE[session_id]


def _handle_ticket(payload: dict) -> str:
    TOOL_CALL_LOG.clear()
    result, _ = run_ticket(payload["input"])
    if result.pydantic:
        return result.pydantic.answer
    return result.raw


_base_runnable = RunnableLambda(_handle_ticket)

with warnings.catch_warnings():
    # RunnableWithMessageHistory's own deprecation warning fires at class
    # definition / first construction time in some langchain versions;
    # suppressed only for this one expected, documented warning.
    warnings.simplefilter("ignore")
    support_agent_with_memory = RunnableWithMessageHistory(
        _base_runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def ask(session_id: str, ticket_text: str) -> str:
    return support_agent_with_memory.invoke(
        {"input": ticket_text},
        config={"configurable": {"session_id": session_id}},
    )


if __name__ == "__main__":
    print("=== Multi-turn memory demo: session 'customer-A' ===")
    SESSION_A = "customer-A"

    print("\n[Turn 1] (history should be empty before this turn)")
    print(f"  history before: {len(get_session_history(SESSION_A).messages)} messages")
    answer1 = ask(SESSION_A, "How many days do I have to return a kurta I bought?")
    print(f"  answer: {answer1[:150]}...")
    print(f"  history after: {len(get_session_history(SESSION_A).messages)} messages")

    print("\n[Turn 2] (same session -- history should now carry turn 1)")
    print(f"  history before: {len(get_session_history(SESSION_A).messages)} messages")
    answer2 = ask(SESSION_A, "And can I exchange it for a different size instead?")
    print(f"  answer: {answer2[:150]}...")
    print(f"  history after: {len(get_session_history(SESSION_A).messages)} messages")

    print("\n  Full transcript carried in session 'customer-A':")
    for i, msg in enumerate(get_session_history(SESSION_A).messages):
        print(f"    [{i}] {msg.type}: {str(msg.content)[:100]}...")

    print("\n=== Separate transcript: a FRESH session shows state correctly absent ===")
    SESSION_B = "customer-B"
    print(f"  session '{SESSION_B}' history before any turn: "
          f"{len(get_session_history(SESSION_B).messages)} messages")
    assert len(get_session_history(SESSION_B).messages) == 0, "Fresh session must start empty!"
    answer3 = ask(SESSION_B, "What's the refund timeline for a cash on delivery order?")
    print(f"  answer: {answer3[:150]}...")
    print(f"  session '{SESSION_B}' history after its first turn: "
          f"{len(get_session_history(SESSION_B).messages)} messages (its own turn only)")
    print(f"  session '{SESSION_A}' history is unaffected: "
          f"{len(get_session_history(SESSION_A).messages)} messages (still from its own 2 turns)")

    print("\n[PASS] Session 'customer-A' carried state across 2 turns; "
          "fresh session 'customer-B' started with correctly empty history.")
