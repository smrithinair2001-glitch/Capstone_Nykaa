"""
Nykaa Domain Support Agent -- package init.

Disables CrewAI's (and other libraries') default outbound telemetry calls
as early as possible, before crewai or chromadb get imported anywhere in
the project. See Submission Guidelines in the brief: "CrewAI's own
telemetry attempts an outbound network call by default when
crew.kickoff() runs... Set CREWAI_DISABLE_TELEMETRY=true."
"""

import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MOCK_LLM", os.environ.get("MOCK_LLM", "1"))
