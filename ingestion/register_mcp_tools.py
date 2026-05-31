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
    {
        "type": "DataDistributionTool",
        "name": "analyze_reaction_distribution",
        "description": (
            "Analyzes how reaction distribution in FAERS changed between two time periods. "
            "Returns divergence score and top changed reactions. "
            "Use to identify EMERGING (absent in baseline) vs GROWING (increased) signals. "
            "Required: selectionTimeRangeStart/End, baselineTimeRangeStart/End "
            "(ISO format: YYYY-MM-DDT00:00:00.000Z)"
        ),
        "attributes": {
            "index": "faers_reports",
            "timeField": "receivedate",
        },
    },
    {
        "type": "SearchIndexTool",
        "name": "search_faers",
        "description": (
            "Search FAERS adverse event reports using OpenSearch DSL. "
            "Investigate drug-reaction signals by demographics, seriousness, "
            "country, reporter type, or any combination. "
            "Required: query (OpenSearch DSL JSON string)"
        ),
        "attributes": {
            "index": "faers_reports",
            "size":  20,
        },
    },
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
