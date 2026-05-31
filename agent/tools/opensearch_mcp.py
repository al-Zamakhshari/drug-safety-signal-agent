"""
Universal OpenSearch MCP client tool.

Connects to the built-in MCP server in OpenSearch 3.6.0 ML Commons.
The investigator agent can discover available tools at runtime and call
any of them freely — no hardcoded Python wrappers needed.

Built-in OS MCP endpoint: POST /_plugins/_ml/mcp
Protocol: JSON-RPC 2.0 (stateless)

Registered tools (via ingestion/register_mcp_tools.py):
  - analyze_reaction_distribution (DataDistributionTool)
  - search_faers (SearchIndexTool)
  + any future tools registered without code changes
"""

import json
import os
from typing import Annotated

import httpx
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

_BASE  = os.getenv("OPENSEARCH_URL", "https://localhost:9200")
_AUTH  = (os.getenv("OPENSEARCH_USER", "admin"),
          os.getenv("OPENSEARCH_PASSWORD", "Pharma@2024!"))
_MCP   = f"{_BASE}/_plugins/_ml/mcp"
_HDR   = {"Content-Type": "application/json"}
_ID    = 0


def _next_id() -> int:
    global _ID
    _ID += 1
    return _ID


async def _mcp(method: str, params: dict) -> dict:
    async with httpx.AsyncClient(verify=False, timeout=30) as client:
        r = await client.post(
            _MCP, auth=_AUTH, headers=_HDR,
            json={"jsonrpc": "2.0", "method": method,
                  "params": params, "id": _next_id()},
        )
        return r.json()


@tool
async def list_opensearch_tools() -> str:
    """
    List all tools available on the OpenSearch built-in MCP server.
    Call this first to discover what tools you can use for investigation.
    Returns tool names, descriptions, and required parameters.
    """
    resp = await _mcp("tools/list", {})
    tools = resp.get("result", {}).get("tools", [])
    if not tools:
        return json.dumps({"tools": [], "note": "No tools registered. Run: uv run python -m ingestion.register_mcp_tools"})

    summary = []
    for t in tools:
        schema = t.get("inputSchema", {})
        props  = schema.get("properties", {})
        required = schema.get("required", [])
        summary.append({
            "name":        t["name"],
            "description": t.get("description", ""),
            "required_params": required,
            "all_params":  list(props.keys()),
        })
    return json.dumps({"tools": summary, "count": len(tools)}, indent=2)


@tool
async def call_opensearch_tool(
    tool_name: Annotated[str, "Name of the tool to call (from list_opensearch_tools)"],
    arguments: Annotated[str, "JSON string of arguments for the tool"],
) -> str:
    """
    Call any tool registered on the OpenSearch built-in MCP server.

    Use list_opensearch_tools first to discover available tools and their parameters.

    Key tools available:
      analyze_reaction_distribution — compares reaction distribution between
        two time periods. Required: selectionTimeRangeStart, selectionTimeRangeEnd,
        baselineTimeRangeStart, baselineTimeRangeEnd (ISO format: YYYY-MM-DDT00:00:00.000Z)

      search_faers — flexible DSL search of FAERS reports.
        Required: query (OpenSearch DSL JSON)

    Returns the tool's result as a JSON string.
    """
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        return json.dumps({"error": f"Invalid JSON in arguments: {arguments[:200]}"})

    resp = await _mcp("tools/call", {"name": tool_name, "arguments": args})

    if "error" in resp:
        return json.dumps({"error": resp["error"]})

    # Extract text content from MCP response
    content = resp.get("result", {}).get("content", [])
    if content and isinstance(content, list):
        text = content[0].get("text", "")
        try:
            return json.dumps(json.loads(text), indent=2)
        except Exception:
            return text

    return json.dumps(resp.get("result", {}))
