"""
Signal registry backed by OpenSearch ML Memory (OS 3.6.0+).

Two complementary stores:

1. ML Memory (`/_plugins/_ml/memory`) — human-readable text trail.
   One container per drug; each run appends one message (free-text).
   Used for the investigator's contextual "PRIOR RUN FINDINGS" note.

2. Structured run index (`agent-signal-runs`) — machine-readable per-run state.
   One document per (drug, run). Enables:
     - Cross-run PRR delta ("PANCREATITIS was PRR=8.2, now PRR=9.1 → GROWING")
     - NEW / VALIDATED / DISMISSED signal lifecycle
     - Enriched memory_context for the Phase-1 investigator prompt

Architecture:
  One memory container per drug (created on first run, reused after).
  Container IDs persisted in `agent-signal-memory-index` (survives restarts).
  Structured run docs in `agent-signal-runs` (separate index).
"""

import os
import json
from datetime import datetime, timezone
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
# OS index for structured per-run signal state
_RUNS_INDEX = "agent-signal-runs"

# In-process cache: drug → memory_id
_memory_cache: dict[str, str] = {}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

async def _os_post(path: str, body: dict) -> dict:
    async with httpx.AsyncClient(verify=False, timeout=15) as client:
        r = await client.post(f"{BASE}{path}", auth=AUTH, headers=HDR, json=body)
        return r.json()


async def _os_get(path: str) -> dict:
    async with httpx.AsyncClient(verify=False, timeout=15) as client:
        r = await client.get(f"{BASE}{path}", auth=AUTH, headers=HDR)
        return r.json()


async def _os_put_index(index: str, mapping: dict) -> None:
    """Create an index if it doesn't exist (ignores 400 = already exists)."""
    async with httpx.AsyncClient(verify=False, timeout=10) as client:
        await client.put(f"{BASE}/{index}", auth=AUTH, headers=HDR, json=mapping)


# ---------------------------------------------------------------------------
# ML Memory (text trail — store 1)
# ---------------------------------------------------------------------------

async def _ensure_registry_index():
    await _os_put_index(_REGISTRY_INDEX, {
        "mappings": {"properties": {
            "drug":      {"type": "keyword"},
            "memory_id": {"type": "keyword"},
        }},
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    })


async def get_or_create_memory(drug: str) -> str:
    """
    Return the ML Memory container ID for a drug, creating if needed.
    IDs persisted in OpenSearch so they survive process restarts.

    Race-condition fix: uses PUT /{index}/_create/{drug_upper} (op_type=create)
    instead of POST /{index}/_doc (auto-id). The _create endpoint returns 409 if
    a document for this drug already exists — the 409 loser re-reads the winner's
    memory_id, ensuring all workers converge on the same container with zero
    orphaned ML Memory objects under concurrent uvicorn worker startup.
    """
    drug_upper = drug.upper()
    if drug_upper in _memory_cache:
        return _memory_cache[drug_upper]

    await _ensure_registry_index()
    # Always read from OpenSearch first — authoritative across all workers
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

    # Not found — create the ML Memory container
    resp = await _os_post("/_plugins/_ml/memory", {"name": f"drug_safety_{drug_upper}"})
    mid = resp.get("memory_id")
    if not mid:
        raise RuntimeError(f"Failed to create ML Memory container: {resp}")

    # Write with op_type=create and a deterministic doc ID.
    # If another worker already wrote, we get a 409 — read the winner's ID.
    async with httpx.AsyncClient(verify=False, timeout=10) as client:
        r = await client.put(
            f"{BASE}/{_REGISTRY_INDEX}/_create/{drug_upper}",
            auth=AUTH, headers=HDR,
            json={"drug": drug_upper, "memory_id": mid},
        )
        if r.status_code == 409:
            # Another worker won the race — read their memory_id
            r2 = await client.post(
                f"{BASE}/{_REGISTRY_INDEX}/_search",
                auth=AUTH, headers=HDR,
                json={"query": {"term": {"drug": drug_upper}}},
            )
            existing = r2.json().get("hits", {}).get("hits", [])
            if existing:
                mid = existing[0]["_source"]["memory_id"]

    _memory_cache[drug_upper] = mid
    return mid


async def save_finding(
    drug: str,
    signals: list[dict],
    investigation: str,
    risk: str = "MEDIUM",
) -> dict[str, Any]:
    """
    Persist this run's text findings to ML Memory (human-readable trail).
    Separate from the structured run store — both are maintained.
    """
    mid = await get_or_create_memory(drug)
    top_signals = [
        f"{s['reaction']} PRR={s['prr']} n={s['drug_count']} sig={'✓' if s.get('significant') else '~'}"
        for s in signals[:10]
    ]
    input_text = f"DRUG: {drug.upper()} | RISK: {risk}\nSIGNALS: {'; '.join(top_signals)}"
    resp = await _os_post(
        f"/_plugins/_ml/memory/{mid}/messages",
        {"input": input_text, "response": investigation or "No investigation performed"},
    )
    return {"memory_id": mid, "message_id": resp.get("message_id"), "drug": drug}


async def search_findings(drug: str, top_n: int = 5) -> list[dict[str, Any]]:
    """Retrieve past text findings from ML Memory (most recent first)."""
    try:
        mid = await get_or_create_memory(drug)
        resp = await _os_get(f"/_plugins/_ml/memory/{mid}/messages")
        raw_messages = resp.get("messages", [])
        raw_messages.sort(key=lambda m: m.get("create_time", ""), reverse=True)
        return [
            {"input": msg.get("input", ""), "response": msg.get("response", ""),
             "created_at": msg.get("create_time", "")}
            for msg in raw_messages[:top_n]
        ]
    except Exception as e:
        return [{"error": str(e)}]


# ---------------------------------------------------------------------------
# Structured run store (machine-readable — store 2)
# ---------------------------------------------------------------------------

_RUNS_MAPPING = {
    "mappings": {"properties": {
        "drug":     {"type": "keyword"},
        "run_ts":   {"type": "date"},
        "signals":  {"type": "nested", "properties": {
            "reaction":   {"type": "keyword"},
            "prr":        {"type": "float"},
            "prr_lower":  {"type": "float"},
            "prr_upper":  {"type": "float"},   # needed for CI-overlap lifecycle test
            "drug_count": {"type": "integer"},
            "effect":     {"type": "keyword"},   # CLASS_EFFECT | DRUG_SPECIFIC | None
            "trend":      {"type": "keyword"},   # GROWING | STABLE | EMERGING | None
            "labeled":    {"type": "boolean"},
            "status":     {"type": "keyword"},   # NEW | VALIDATED | DISMISSED
        }},
    }},
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
}


async def save_run_signals(drug: str, signals: list[dict]) -> None:
    """
    Persist one structured run document per drug.
    Each signal carries: reaction, prr, prr_lower, prr_upper, drug_count,
    effect (from Phase-1 classifier), trend, labeled, status (NEW/VALIDATED/DISMISSED).
    Logs a warning if persistence fails — never raises (supplemental store).
    """
    await _os_put_index(_RUNS_INDEX, _RUNS_MAPPING)
    doc = {
        "drug":    drug.upper(),
        "run_ts":  datetime.now(timezone.utc).isoformat(),
        "signals": signals,
    }
    try:
        async with httpx.AsyncClient(verify=False, timeout=15) as client:
            r = await client.post(
                f"{BASE}/{_RUNS_INDEX}/_doc",
                auth=AUTH, headers=HDR, json=doc,
            )
            if r.status_code >= 300:
                print(f"  [status] WARNING: run persistence failed HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"  [status] WARNING: run persistence failed: {e}")


async def load_last_run(drug: str) -> dict | None:
    """
    Fetch the most recent structured run document for a drug.
    Returns None if no prior run exists.
    """
    try:
        await _os_put_index(_RUNS_INDEX, _RUNS_MAPPING)
        async with httpx.AsyncClient(verify=False, timeout=10) as client:
            r = await client.post(
                f"{BASE}/{_RUNS_INDEX}/_search",
                auth=AUTH, headers=HDR,
                json={
                    "query": {"term": {"drug": drug.upper()}},
                    "sort":  [{"run_ts": {"order": "desc"}}],
                    "size":  1,
                },
            )
            hits = r.json().get("hits", {}).get("hits", [])
            return hits[0]["_source"] if hits else None
    except Exception:
        return None


def build_memory_context(current_signals: list[dict], prior_run: dict | None) -> str:
    """
    Build a structured per-reaction PRR delta string for the Phase-1 investigator
    prompt — replaces the old truncated prose note.

    Example output:
      PANCREATITIS: DRUG_SPECIFIC PRR=8.2→9.1 (+11%) | PERSISTENT+GROWING
      BLOOD_GLUCOSE_INCR: DRUG_SPECIFIC PRR=6.0 last run | now PRR=2.1 → RESOLVED
      NAUSEA: NEW (first time seen)
    """
    if not prior_run:
        return ""

    prior_map = {s["reaction"]: s for s in prior_run.get("signals", [])}
    current_map = {s["reaction"]: s for s in current_signals}
    lines = []

    for rxn, cur in current_map.items():
        prior = prior_map.get(rxn)
        if prior is None:
            lines.append(f"  {rxn}: NEW (not seen in prior run)")
        else:
            p_prr = prior.get("prr") or 0
            c_prr = cur.get("prr") or 0
            delta_pct = int((c_prr - p_prr) / p_prr * 100) if p_prr else 0
            direction = f"+{delta_pct}%" if delta_pct > 0 else f"{delta_pct}%"
            tags = []
            if prior.get("effect"):
                tags.append(prior["effect"])
            if prior.get("status") == "VALIDATED":
                tags.append("PERSISTENT")
            tag_str = " ".join(tags)
            lines.append(
                f"  {rxn}: {tag_str} PRR={p_prr:.1f}→{c_prr:.1f} ({direction})"
            )

    for rxn, prior in prior_map.items():
        if rxn not in current_map and prior.get("status") == "VALIDATED":
            lines.append(f"  {rxn}: PRR={prior.get('prr'):.1f} last run → RESOLVED (gone)")

    if not lines:
        return ""
    return "PRIOR RUN SIGNAL TRAJECTORY:\n" + "\n".join(lines[:10]) + "\n"
