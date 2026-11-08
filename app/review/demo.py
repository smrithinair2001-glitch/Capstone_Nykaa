"""Part 4 / Task 14 demo: Autogen review stage on top of real CrewAI output.

Runs an actual CrewAI ticket end to end (Part 2's crew, under MOCK_LLM),
takes the Composer's genuine draft answer plus the Retrieval Agent's
genuinely retrieved policy context, and feeds both into the Autogen
review team:

- Case 1 (approve-unchanged): the real, grounded draft the crew produced.
- Case 2 (actually-revised): the SAME grounded context, but a copy of the
  draft with one deliberately injected ungrounded claim spliced in, to
  demonstrate the Reviewer catching it and the Final-Editor stripping it
  back out.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import app  # noqa: E402  (sets telemetry/mock env vars on import)
from app.crew.crew_build import run_ticket  # noqa: E402
from app.review.mock_autogen_client import UNGROUNDED_CLAIM_MARKERS  # noqa: E402
from app.review.team import run_review  # noqa: E402

TICKET_QUERY = "How many days do I have to return a kurta I bought?"
INJECTED_CLAIM = UNGROUNDED_CLAIM_MARKERS[0]  # "lifetime warranty on all items"


def main() -> None:
    print(f"Running CrewAI ticket: {TICKET_QUERY!r}\n")
    result, tasks = run_ticket(TICKET_QUERY)
    support_response = result.pydantic
    retrieved_context = tasks["retrieval"].output.raw

    print("Composer's draft answer (genuine, from the live crew run):")
    print(support_response.answer)
    print()
    print("Retrieval Agent's retrieved policy context (genuine):")
    print(retrieved_context)
    print()

    corrupted_draft = (
        support_response.answer
        + f" Also, Nykaa offers a {INJECTED_CLAIM} regardless of category."
    )

    print("=" * 70)
    print("CASE 1 -- genuine grounded draft -> expect APPROVE (unchanged)")
    print("=" * 70)
    verdict1 = asyncio.run(run_review(support_response.answer, retrieved_context))
    print(json.dumps(verdict1.model_dump(), indent=2))
    assert verdict1.approved is True, "Expected the genuine draft to be approved unchanged."
    assert verdict1.final_answer == support_response.answer
    print("[PASS] Case 1: genuine draft approved unchanged.\n")

    print("=" * 70)
    print("CASE 2 -- draft with a deliberately injected ungrounded claim")
    print("=" * 70)
    print("Injected sentence fragment: "
          f"'...Also, Nykaa offers a {INJECTED_CLAIM} regardless of category.'\n")
    verdict2 = asyncio.run(run_review(corrupted_draft, retrieved_context))
    print(json.dumps(verdict2.model_dump(), indent=2))
    assert verdict2.approved is False, "Expected the injected claim to trigger a revision."
    assert INJECTED_CLAIM not in verdict2.final_answer.lower()
    print("[PASS] Case 2: injected ungrounded claim detected and stripped out by the review team.")


if __name__ == "__main__":
    main()
