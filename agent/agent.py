"""
Drug Safety Signal Detection Agent — local edition

Fully local pipeline. No API keys, no cloud services, no licenses.
Requires: Docker Desktop with Model Runner (Gemma4 E4B) + OpenSearch

Pipeline:
  prr_agent       → Python PRR via opensearch-py (no query syntax bugs)
  label_agent     → openFDA label API
  literature_agent → PubMed search
  report_agent    → final briefing
"""

import os
import socket
from dotenv import load_dotenv

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm

from agent.tools.prr import calculate_prr, get_drug_names, get_signal_timeline
from agent.tools.openfda import get_drug_label
from agent.tools.pubmed import search_literature

load_dotenv()


# ---------------------------------------------------------------------------
# Tracing — silently disabled if Phoenix isn't running
# ---------------------------------------------------------------------------

def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def setup_tracing():
    phoenix_url = os.getenv("PHOENIX_URL", "http://localhost:6006")
    grpc_port = 4317
    grpc_host = phoenix_url.split("://")[-1].split(":")[0]

    if not _port_open(grpc_host, grpc_port):
        print("[phoenix] Not reachable — tracing disabled")
        return None

    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource

        exporter = OTLPSpanExporter(
            endpoint=f"{grpc_host}:{grpc_port}", insecure=True
        )
        provider = TracerProvider(resource=Resource({"service.name": "drug-safety-agent"}))
        provider.add_span_processor(BatchSpanProcessor(exporter))
        otel_trace.set_tracer_provider(provider)

        try:
            from openinference.instrumentation.litellm import LiteLLMInstrumentor
            LiteLLMInstrumentor().instrument(tracer_provider=provider)
        except Exception:
            pass

        print(f"[phoenix] Tracing → http://{grpc_host}:6006")
        return provider
    except Exception as e:
        print(f"[phoenix] Setup failed: {e}")
        return None


_tracing_active = setup_tracing()


# ---------------------------------------------------------------------------
# Model — Gemma4 E4B via Docker Model Runner (zero API cost)
# ---------------------------------------------------------------------------

def _get_model():
    return LiteLlm(
        model="openai/docker.io/ai/gemma4:E4B",
        base_url=os.getenv("LOCAL_MODEL_URL", "http://localhost:12434/v1"),
        api_key="docker",
    )


# ---------------------------------------------------------------------------
# Tool wrappers — thin ADK-compatible async functions
# ---------------------------------------------------------------------------

async def get_prr_signals(drug_name: str) -> dict:
    """
    Calculate PRR safety signals for a drug from FAERS data.
    Automatically resolves brand names and computes against full population baseline.

    Args:
        drug_name: Generic or brand name (e.g. "semaglutide", "rofecoxib")
    """
    names_result = await get_drug_names(drug_name)
    drug_names = names_result["found_names"]
    return await calculate_prr(drug_names)


async def get_labeled_reactions(drug_name: str) -> dict:
    """Fetch current FDA label and extract all documented adverse reactions."""
    return await get_drug_label(drug_name)


async def find_literature_evidence(drug_name: str, reaction_term: str) -> dict:
    """Search PubMed for evidence on a drug-reaction pair."""
    return await search_literature(drug_name, reaction_term)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

prr_agent = LlmAgent(
    name="prr_analyst",
    model=_get_model(),
    instruction="""Call get_prr_signals with the drug name from the user request.
OUTPUT ONLY this table (no explanations):
drug_total=N  faers_total=N
| Reaction | PRR | Reports |
Only rows where PRR>=2.0 AND reports>=3. Max 20 rows.""",
    tools=[get_prr_signals],
)

label_agent = LlmAgent(
    name="label_analyst",
    model=_get_model(),
    instruction="""Call get_labeled_reactions with the drug name.
Try generic name first, then brand name (semaglutide→ozempic, rofecoxib→vioxx).
OUTPUT ONLY a comma-separated list of reaction terms in ALL CAPS. No explanations.""",
    tools=[get_labeled_reactions],
)

literature_agent = LlmAgent(
    name="literature_analyst",
    model=_get_model(),
    instruction="""For unlabeled PRR signals from prr_analyst, call find_literature_evidence.
Search max 3 signals (highest PRR first).
OUTPUT ONLY:
| Signal | Papers | Supports? |
No explanations.""",
    tools=[find_literature_evidence],
)

report_agent = LlmAgent(
    name="report_writer",
    model=_get_model(),
    instruction="""Write the final drug safety briefing from the prior agents' outputs.

## Drug Safety Briefing: <DRUG>
**FAERS reports analysed**: N  |  **Method**: PRR (EMA standard)

### Signals Detected (PRR ≥ 2.0, n ≥ 3)
| Reaction | PRR | Reports | In FDA Label? | Literature |
|----------|-----|---------|---------------|------------|

### Key Findings
1-3 bullet points on the most important signals.

**Risk**: LOW / MEDIUM / HIGH
**Action**: MONITOR / INVESTIGATE / ESCALATE

> FAERS data only. For research purposes. Requires clinical validation.""",
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

root_agent = SequentialAgent(
    name="drug_safety_pipeline",
    description="Local drug safety signal detection: PRR → Label → Literature → Report",
    sub_agents=[prr_agent, label_agent, literature_agent, report_agent],
)
