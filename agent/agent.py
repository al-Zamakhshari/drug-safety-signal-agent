"""
Observability setup for the drug safety agent.

Configures Arize Phoenix tracing (optional — silently skipped if Phoenix
is not running). Call setup_tracing() at startup.
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
    """Set up Phoenix OTLP tracing. Silently disabled if Phoenix is not running."""
    phoenix_url = os.getenv("PHOENIX_URL", "http://localhost:6006")
    grpc_host = phoenix_url.split("://")[-1].split(":")[0]
    grpc_port = 4317

    if not _port_open(grpc_host, grpc_port):
        print(f"[phoenix] Port {grpc_port} not reachable — tracing disabled")
        return None

    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource

        exporter = OTLPSpanExporter(endpoint=f"{grpc_host}:{grpc_port}", insecure=True)
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
