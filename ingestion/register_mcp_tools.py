"""
Register tools in OpenSearch built-in MCP server (OS 3.6.0+).

Run once after docker compose up. Tools persist in .plugins-ml-mcp-tools index.
The investigator agent discovers these at runtime via list_opensearch_tools().

Usage:
    uv run python -m ingestion.register_mcp_tools
"""

import asyncio, os
import httpx
from dotenv import load_dotenv

load_dotenv()

BASE = os.getenv("OPENSEARCH_URL", "https://localhost:9200")
AUTH = (os.getenv("OPENSEARCH_USER", "admin"),
        os.getenv("OPENSEARCH_PASSWORD", "Pharma@2024!"))
HDR  = {"Content-Type": "application/json"}

TOOLS = [
    # ── Already registered (idempotent — skipped if exists) ──────────────
    {
        "type": "DataDistributionTool",
        "name": "analyze_reaction_distribution",
        "description": (
            "Analyzes how FAERS reaction distribution changed between two time periods. "
            "Returns divergence score and top changed reactions. "
            "Use to identify EMERGING (absent in baseline) vs GROWING (increased) signals. "
            "Required: selectionTimeRangeStart/End, baselineTimeRangeStart/End "
            "(format: yyyy-MM-dd HH:mm:ss)"
        ),
        "attributes": {"index": "faers_reports", "timeField": "receivedate"},
    },
    {
        "type": "SearchIndexTool",
        "name": "search_faers",
        "description": (
            "Search FAERS adverse event reports using OpenSearch DSL. "
            "Investigate drug-reaction by demographics, seriousness, country, reporter type. "
            "Required: query (OpenSearch DSL JSON string)"
        ),
        "attributes": {"index": "faers_reports", "size": 20},
    },

    # ── New: Anomaly Detection tools ──────────────────────────────────────
    {
        "type": "SearchAnomalyDetectorsTool",
        "name": "list_anomaly_detectors",
        "description": (
            "List all anomaly detectors in OpenSearch. Use to find the "
            "drug_class_ratio_detector ID for querying class-ratio anomaly results. "
            "No required parameters."
        ),
        "attributes": {},
    },
    {
        "type": "SearchAnomalyResultsTool",
        "name": "get_anomaly_results",
        "description": (
            "Query anomaly detection results from a specific detector. "
            "Use after list_anomaly_detectors to get the detector ID. "
            "Returns anomaly grades (0-1) and confidence scores per time period. "
            "Optional: detectorId, anomalyGradeThreshold, dataStartTime, dataEndTime, size."
        ),
        "attributes": {},
    },

    # Note: LogPatternAnalysisTool (analyze_narrative_patterns) removed.
    # The `narrative` field is empty in FAERS ingestion — the openFDA API
    # and ZIP ingestion paths do not include free-text narratives. Registering
    # the tool would burn a Phase-2 tool-call slot on a guaranteed-empty result.
    # Re-register if narrative ingestion is added in the future.
]


async def main():
    async with httpx.AsyncClient(verify=False, timeout=30) as client:
        # Enable MCP server
        r = await client.put(
            f"{BASE}/_cluster/settings", auth=AUTH, headers=HDR,
            json={"persistent": {"plugins.ml_commons.mcp_server_enabled": True}},
        )
        print(f"MCP server enabled: {r.json().get('acknowledged')}")

        # Check existing tools
        r = await client.post(
            f"{BASE}/_plugins/_ml/mcp", auth=AUTH, headers=HDR,
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 1},
        )
        existing = {t["name"] for t in r.json().get("result", {}).get("tools", [])}
        print(f"Existing MCP tools: {existing or '(none)'}")

        # Register new tools
        new_tools = [t for t in TOOLS if t["name"] not in existing]
        if not new_tools:
            print("All tools already registered.")
            return

        r = await client.post(
            f"{BASE}/_plugins/_ml/mcp/tools/_register",
            auth=AUTH, headers=HDR,
            json={"tools": new_tools},
        )
        result = r.json()
        if r.status_code in (200, 201):
            print(f"✅ Registered {len(new_tools)} tools:")
            for t in new_tools:
                print(f"   - {t['name']} ({t['type']})")
        else:
            print(f"❌ Registration failed: {result}")

        # Verify
        r = await client.post(
            f"{BASE}/_plugins/_ml/mcp", auth=AUTH, headers=HDR,
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 2},
        )
        tools = r.json().get("result", {}).get("tools", [])
        print(f"\nTotal MCP tools now available: {len(tools)}")
        for t in tools:
            print(f"   {t['name']}: {t['description'][:70]}")


if __name__ == "__main__":
    asyncio.run(main())
