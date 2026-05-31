"""
Signal registry backed by OpenSearch ML Memory (OS 3.6.0+).

Uses the native /_plugins/_ml/memory API — no extra services needed.
One memory container per drug stores investigation findings across runs,
completing the roadmap item:  [ ] Signal registry (NEW/VALIDATED/DISMISSED)

Architecture:
  One memory container per drug (created on first run, reused after).
  Each pipeline run appends its findings as messages.
  Next run queries memory to inform investigation priorities.

Memory container IDs are stored in the `agent-memory-index` OS index
so they persist across process restarts.
"""

import os
import json
from typing import Any
import httpx
from dotenv import load_dotenv

load_dotenv()

BASE  = os.getenv("OPENSEARCH_URL", "https://localhost:9200")
USER  = os.getenv("OPENSEARCH_USER", "admin")
PASS  = os.getenv("OPENSEARCH_PASSWORD", "Pharma@2024!")
AUTH  = (USER, PASS)
HDR   = {"Content-Type": "application/json"}

# OS index that maps drug → memory_id (persists across restarts)
_REGISTRY_INDEX = "agent-signal-memory-index"

# In-process cache: drug → memory_id
_memory_cache: dict[str, str] = {}


async def _os_post(path: str, body: dict) -> dict:
    async with httpx.AsyncClient(verify=False, timeout=15) as client:
        r = await client.post(f"{BASE}{path}", auth=AUTH, headers=HDR, json=body)
        return r.json()


async def _os_get(path: str) -> dict:
    async with httpx.AsyncClient(verify=False, timeout=15) as client:
        r = await client.get(f"{BASE}{path}", auth=AUTH, headers=HDR)
        return r.json()


async def _ensure_registry_index():
    """Create the OS index that stores drug→memory_id mapping."""
    async with httpx.AsyncClient(verify=False, timeout=10) as client:
        r = await client.put(
            f"{BASE}/{_REGISTRY_INDEX}",
            auth=AUTH, headers=HDR,
            json={"mappings": {"properties": {
                "drug":      {"type": "keyword"},
                "memory_id": {"type": "keyword"},
            }}, "settings": {"number_of_shards": 1, "number_of_replicas": 0}},
        )
    # 400 = already exists, fine
    return r.status_code in (200, 201, 400)


async def get_or_create_memory(drug: str) -> str:
    """
    Return the ML Memory container ID for a drug, creating it if needed.
    IDs are persisted in OpenSearch so they survive process restarts.
    """
    drug_upper = drug.upper()

    # 1. In-process cache
    if drug_upper in _memory_cache:
        return _memory_cache[drug_upper]

    # 2. Check persisted index
    await _ensure_registry_index()
    async with httpx.AsyncClient(verify=False, timeout=10) as client:
        r = await client.post(
            f"{BASE}/{_REGISTRY_INDEX}/_search",
            auth=AUTH, headers=HDR,
            json={"query": {"term": {"drug": drug_upper}}},
        )
        hits = r.json().get("hits", {}).get("hits", [])
        if hits:
            mid = hits[0]["_source"]["memory_id"]
            _memory_cache[drug_upper] = mid
            return mid

    # 3. Create new memory container
    resp = await _os_post(
        "/_plugins/_ml/memory",
        {"name": f"drug_safety_{drug_upper}"}
    )
    mid = resp.get("memory_id")
    if not mid:
        raise RuntimeError(f"Failed to create ML Memory container: {resp}")

    # Persist the mapping
    async with httpx.AsyncClient(verify=False, timeout=10) as client:
        await client.post(
            f"{BASE}/{_REGISTRY_INDEX}/_doc",
            auth=AUTH, headers=HDR,
            json={"drug": drug_upper, "memory_id": mid},
        )
    _memory_cache[drug_upper] = mid
    return mid


async def save_finding(
    drug: str,
    signals: list[dict],
    investigation: str,
    risk: str = "MEDIUM",
) -> dict[str, Any]:
    """
    Persist this run's findings to ML Memory.

    Each run is stored as one message in the memory container.
    The input field records detected signals; the response field
    records the LLM investigation classification.
    """
    mid = await get_or_create_memory(drug)

    # Compact signal summary for the input field
    top_signals = [
        f"{s['reaction']} PRR={s['prr']} n={s['drug_count']} sig={'✓' if s.get('significant') else '~'}"
        for s in signals[:10]
    ]
    input_text = f"DRUG: {drug.upper()} | RISK: {risk}\nSIGNALS: {'; '.join(top_signals)}"

    resp = await _os_post(
        f"/_plugins/_ml/memory/{mid}/messages",
        {
            "input":    input_text,
            "response": investigation or "No investigation performed",
        }
    )
    return {"memory_id": mid, "message_id": resp.get("message_id"), "drug": drug}


async def search_findings(drug: str, top_n: int = 5) -> list[dict[str, Any]]:
    """
    Retrieve past investigation findings for a drug from ML Memory.
    Returns the most recent `top_n` messages.
    """
    try:
        mid = await get_or_create_memory(drug)
        # Correct endpoint: GET /messages (not POST /_search)
        resp = await _os_get(f"/_plugins/_ml/memory/{mid}/messages")
        raw_messages = resp.get("messages", [])
        # Sort by create_time descending, take top_n
        raw_messages.sort(key=lambda m: m.get("create_time", ""), reverse=True)
        messages = []
        for msg in raw_messages[:top_n]:
            messages.append({
                "input":      msg.get("input", ""),
                "response":   msg.get("response", ""),
                "created_at": msg.get("create_time", ""),
            })
        return messages
    except Exception as e:
        return [{"error": str(e)}]
