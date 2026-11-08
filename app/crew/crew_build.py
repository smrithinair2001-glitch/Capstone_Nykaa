"""
Part 2 / Task 8 -- the CrewAI crew: 3 agents (Retrieval, Lookup, Composer),
each backed by the shared MOCK_LLM, run via ``.kickoff()``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from crewai import Agent, Crew, Process, Task  # noqa: E402

from app.crew.mock_llm import get_mock_llm  # noqa: E402
from app.crew.schemas import SupportResponse  # noqa: E402
from app.crew.tools import TOOL_CALL_LOG, check_order_status, rag_lookup  # noqa: E402

MOCK_LLM = get_mock_llm()


def build_agents() -> dict[str, Agent]:
    retrieval_agent = Agent(
        role="Retrieval Agent",
        goal="Answer Nykaa policy questions using only grounded knowledge-base passages.",
        backstory=(
            "You are a policy-retrieval specialist for Nykaa customer support. "
            "You never answer from memory -- you always consult the policy "
            "knowledge base through your tool."
        ),
        tools=[rag_lookup],
        llm=MOCK_LLM,
        verbose=True,
        allow_delegation=False,
    )

    lookup_agent = Agent(
        role="Lookup Agent",
        goal="Look up the status of a specific order when the ticket references one.",
        backstory=(
            "You are an order-status specialist. You only ever check order "
            "records through your tool; you never guess an order's status."
        ),
        tools=[check_order_status],
        llm=MOCK_LLM,
        verbose=True,
        allow_delegation=False,
    )

    composer_agent = Agent(
        role="Response Composer",
        goal="Combine policy and order-lookup findings into one clear customer-facing answer.",
        backstory=(
            "You are the final-draft writer for the support desk. You merge what "
            "the Retrieval Agent and the Lookup Agent found into a single, "
            "friendly, accurate reply -- you have no tools of your own and never "
            "invent information that isn't already in front of you."
        ),
        tools=[],
        llm=MOCK_LLM,
        verbose=True,
        allow_delegation=False,
    )

    return {"retrieval": retrieval_agent, "lookup": lookup_agent, "composer": composer_agent}


def build_tasks(agents: dict[str, Agent], query: str) -> dict[str, Task]:
    retrieval_task = Task(
        description=(
            f"A Nykaa customer wrote this support ticket: \"{query}\"\n"
            "Use the rag_lookup tool to find and return the grounded policy "
            "answer for this ticket."
        ),
        expected_output=(
            "The grounded policy answer text returned by the rag_lookup tool "
            "(verbatim), or its fallback message if nothing was found."
        ),
        agent=agents["retrieval"],
    )

    lookup_task = Task(
        description=(
            f"The same Nykaa support ticket: \"{query}\"\n"
            "If this ticket references a specific order id (format NYK-XXXXX), "
            "use the check_order_status tool to look it up and report the result. "
            "If no order id is referenced, say so without guessing one."
        ),
        expected_output=(
            "The order status/escalation info returned by check_order_status, "
            "or a statement that no order id was referenced."
        ),
        agent=agents["lookup"],
    )

    composer_task = Task(
        description=(
            f"Original customer ticket: \"{query}\"\n"
            "You have been given the Retrieval Agent's policy findings and the "
            "Lookup Agent's order-status findings as context. Write one final, "
            "customer-facing answer that merges both, in the structured format "
            "requested."
        ),
        expected_output="A SupportResponse with answer, used_policy_kb, used_order_lookup, should_escalate.",
        agent=agents["composer"],
        context=[retrieval_task, lookup_task],
        output_pydantic=SupportResponse,
    )

    return {"retrieval": retrieval_task, "lookup": lookup_task, "composer": composer_task}


def build_crew(query: str) -> tuple[Crew, dict[str, Task]]:
    agents = build_agents()
    tasks = build_tasks(agents, query)
    crew = Crew(
        agents=[agents["retrieval"], agents["lookup"], agents["composer"]],
        tasks=[tasks["retrieval"], tasks["lookup"], tasks["composer"]],
        process=Process.sequential,
        verbose=True,
    )
    return crew, tasks


def run_ticket(query: str):
    crew, tasks = build_crew(query)
    result = crew.kickoff()
    return result, tasks


if __name__ == "__main__":
    import json as _json

    SAMPLE_QUERIES = [
        "How many days do I have to return a kurta I bought?",
        "Can you check the status of order NYK-00002 and tell me if I can still cancel it?",
    ]

    for q in SAMPLE_QUERIES:
        print("\n" + "=" * 100)
        print(f"TICKET: {q}")
        print("=" * 100)
        TOOL_CALL_LOG.clear()
        result, tasks = run_ticket(q)

        print("\n--- Tool calls actually made during this kickoff() ---")
        for entry in TOOL_CALL_LOG:
            print(f"  {entry['tool']}({entry['args']})")
        if not TOOL_CALL_LOG:
            print("  (none)")

        print("\n--- Final structured output (schema-validated in code) ---")
        # Task 9: validate every response against the Pydantic schema
        # ourselves, in code -- not just trust that CrewAI's own
        # output_pydantic machinery did it.
        validated = SupportResponse.model_validate_json(result.raw)
        assert validated == result.pydantic
        print(_json.dumps(validated.model_dump(), indent=2))
        print("[PASS] SupportResponse.model_validate_json succeeded.")
