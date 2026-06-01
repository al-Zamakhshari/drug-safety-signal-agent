"""
Optional observability setup — Arize Phoenix via OpenTelemetry.

Tracing is entirely optional:
  - Only activates if the `observability` extras are installed
    (uv sync --extra observability)
  - Only activates if Phoenix is reachable on port 4317
  - Silently no-ops in all other cases — pipeline runs fine without it

To enable:
    uv sync --extra observability
    docker compose --profile observability up -d phoenix
    # Traces appear at http://localhost:6006
"""

import os
import socket
from dotenv import load_dotenv

load_dotenv()


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def setup_tracing():
    """
    Set up Phoenix OTLP tracing.

    Instruments LangChain/LangGraph ChatOpenAI calls — the actual LLM path
    used by this pipeline (not LiteLLM, which is not used here).

    Returns the TracerProvider if tracing is active, else None.
    """
    phoenix_url = os.getenv("PHOENIX_URL", "http://localhost:6006")
    grpc_host   = phoenix_url.split("://")[-1].split(":")[0]
    grpc_port   = 4317

    if not _port_open(grpc_host, grpc_port):
        return None   # Phoenix not running — silent no-op

    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource

        exporter  = OTLPSpanExporter(endpoint=f"{grpc_host}:{grpc_port}", insecure=True)
        provider  = TracerProvider(resource=Resource({"service.name": "drug-safety-agent"}))
        provider.add_span_processor(BatchSpanProcessor(exporter))
        otel_trace.set_tracer_provider(provider)

        # Instrument LangChain — this is what the pipeline actually uses
        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor
            LangChainInstrumentor().instrument(tracer_provider=provider)
        except ImportError:
            pass   # observability extras not installed

        print(f"[phoenix] Tracing → http://{grpc_host}:6006")
        return provider

    except ImportError:
        # opentelemetry packages not installed (observability extra not requested)
        return None
    except Exception as e:
        print(f"[phoenix] Setup failed: {e}")
        return None


_tracing_active = setup_tracing()
