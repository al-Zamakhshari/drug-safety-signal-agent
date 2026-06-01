"""
Drug Safety Signal Agent — Web UI

FastAPI server with Server-Sent Events streaming.
Each LangGraph node completion is streamed to the client in real-time.

Usage:
    uv run python -m app.server
    # or via docker compose up
"""

import asyncio
import json
import sys
import io
from contextlib import redirect_stdout
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Drug Safety Signal Agent", version="1.0.0")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


async def _run_pipeline_streaming(drug_name: str) -> AsyncIterator[dict]:
    """
    Run the LangGraph pipeline and yield SSE events for each node completion.
    Uses astream_events to get fine-grained progress without modifying the pipeline.
    """
    from agent.pipeline import pipeline, DrugSafetyState

    initial_state: DrugSafetyState = {
        "drug_name":       drug_name,
        "drug_names":      [],
        "prr_signals":     [],
        "drug_total":      0,
        "faers_total":     0,
        "anomaly_signals": [],
        "label_text":      "",
        "literature":      [],
        "past_findings":   [],
        "investigation":   [],
        "briefing":        "",
        "error":           None,
    }

    # Node display names for the UI
    NODE_LABELS = {
        "resolve_names":     "🔍 Resolving drug names (RxNorm)…",
        "load_memory":       "🧠 Loading ML Memory (prior findings)…",
        "calculate_prr":     "📊 Calculating PRR signals (OpenSearch)…",
        "anomaly_detection": "📈 Querying anomaly detection…",
        "fetch_label":       "📋 Fetching FDA label (openFDA)…",
        "search_lit":        "📚 Searching PubMed literature…",
        "investigate":       "🔬 Investigating signals (Qwen3.5 + tools)…",
        "write_report":      "✍️  Writing briefing (Qwen3.5)…",
        "save_memory":       "💾 Saving findings to ML Memory…",
    }

    yield {"event": "start", "data": json.dumps({"drug": drug_name})}

    # Capture print() output to surface pipeline logs in the SSE stream
    final_briefing = ""
    last_node = ""

    async for event in pipeline.astream_events(initial_state, version="v2"):
        kind = event.get("event", "")
        name = event.get("name", "")

        # Node started
        if kind == "on_chain_start" and name in NODE_LABELS:
            last_node = name
            yield {
                "event": "node_start",
                "data": json.dumps({
                    "node": name,
                    "label": NODE_LABELS[name],
                })
            }
            await asyncio.sleep(0)

        # Node completed — extract useful data from output
        elif kind == "on_chain_end" and name in NODE_LABELS:
            output = event.get("data", {}).get("output", {})
            detail = ""

            if name == "calculate_prr":
                n_signals = len(output.get("prr_signals", []))
                total = output.get("drug_total", 0)
                detail = f"{total:,} drug reports · {n_signals} signals (PRR≥2)"

            elif name == "resolve_names":
                names = output.get("drug_names", [])
                detail = " · ".join(names[:4])

            elif name == "fetch_label":
                chars = len(output.get("label_text", ""))
                detail = f"{chars:,} chars of label text"

            elif name == "search_lit":
                papers = sum(l.get("papers", 0) for l in output.get("literature", []))
                detail = f"{papers} papers found"

            elif name == "anomaly_detection":
                n = len(output.get("anomaly_signals", []))
                detail = f"{n} class-ratio signals"

            elif name == "load_memory":
                n = len(output.get("past_findings", []))
                detail = f"{n} prior run(s) in ML Memory" if n else "first run"

            elif name == "investigate":
                inv = output.get("investigation", [])
                if inv:
                    n_tools = inv[0].get("tool_calls_made", 0)
                    detail = f"{n_tools} tool calls"

            elif name == "write_report":
                briefing = output.get("briefing", "")
                if briefing:
                    final_briefing = briefing
                    detail = f"{len(briefing)} chars"

            elif name == "save_memory":
                detail = "findings persisted"

            yield {
                "event": "node_done",
                "data": json.dumps({
                    "node": name,
                    "label": NODE_LABELS[name].replace("…", " ✓"),
                    "detail": detail,
                })
            }
            await asyncio.sleep(0)

    # Send the final briefing
    if final_briefing:
        yield {
            "event": "briefing",
            "data": json.dumps({"markdown": final_briefing})
        }

    yield {"event": "done", "data": json.dumps({"drug": drug_name})}


@app.get("/analyze")
async def analyze(drug: str):
    """SSE endpoint — streams pipeline progress then the final briefing."""
    if not drug or len(drug.strip()) < 2:
        return {"error": "Drug name required"}

    async def event_generator():
        try:
            async for evt in _run_pipeline_streaming(drug.strip()):
                yield evt
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)})
            }

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="0.0.0.0", port=8080, reload=False)
