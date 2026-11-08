"""Part 4 / Task 14: Autogen review stage on top of the CrewAI draft answer.

Builds a 2-agent Autogen ``RoundRobinGroupChat`` team -- a Policy-Compliance-
Reviewer and a Final-Editor -- that reviews the CrewAI Composer's draft
answer against the originally retrieved RAG context, and either approves it
unchanged or revises it.

Two brief-mandated gotchas, both handled below:

1. ``RoundRobinGroupChat`` is bounded with ``max_turns=2`` (a real
   constructor parameter) so exactly one Reviewer turn and one Editor turn
   happen, not ``MaxMessageTermination`` with an off-by-one message count.
2. The Final-Editor is given structured output via
   ``output_content_type=ReviewVerdict``. Doing this WITHOUT also passing
   ``custom_message_types=[StructuredMessage[ReviewVerdict]]`` to the team
   constructor crashes with
   ``ValueError: Message type ... is not registered``, so both are wired
   together here.
"""

from __future__ import annotations

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat

from app.review.mock_autogen_client import MockAutogenChatCompletionClient
from app.review.verdict import ReviewVerdict


def build_review_team() -> RoundRobinGroupChat:
    reviewer = AssistantAgent(
        name="Policy_Compliance_Reviewer",
        model_client=MockAutogenChatCompletionClient(role="reviewer"),
        system_message=(
            "You are a policy-compliance reviewer for Nykaa customer support. Given a "
            "draft answer and the policy context it was supposed to be grounded in, decide "
            "whether the draft makes any claim that the context does not support. Reply "
            "with either 'APPROVE: <reason>' or 'REVISE: <reason>'."
        ),
    )
    editor = AssistantAgent(
        name="Final_Editor",
        model_client=MockAutogenChatCompletionClient(role="editor"),
        system_message=(
            "You are the final editor. Read the reviewer's verdict and the original draft, "
            "then produce the final structured verdict: approve the draft unchanged, or "
            "revise it to remove any unsupported claim the reviewer flagged."
        ),
        output_content_type=ReviewVerdict,
    )

    team = RoundRobinGroupChat(
        participants=[reviewer, editor],
        max_turns=2,
        custom_message_types=[StructuredMessage[ReviewVerdict]],
    )
    return team


def build_review_task(draft_answer: str, retrieved_context: str) -> str:
    return (
        f"Draft answer:\n{draft_answer}\n\n"
        f"Retrieved policy context:\n{retrieved_context}\n\n"
        "Review the draft against the context and produce the final verdict."
    )


async def run_review(draft_answer: str, retrieved_context: str) -> ReviewVerdict:
    """Runs the 2-turn review team on one draft and returns the parsed verdict."""
    team = build_review_team()
    task = build_review_task(draft_answer, retrieved_context)
    result = await team.run(task=task)

    verdict_message = None
    for message in reversed(result.messages):
        if isinstance(message, StructuredMessage):
            verdict_message = message
            break
    if verdict_message is None:
        raise RuntimeError("Final-Editor did not produce a structured ReviewVerdict message.")

    content = verdict_message.content
    if isinstance(content, ReviewVerdict):
        return content
    return ReviewVerdict.model_validate(content)
