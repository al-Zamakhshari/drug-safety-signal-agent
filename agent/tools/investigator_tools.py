"""
OpenSearch-backed tools for the investigator agent.

These are Python functions decorated with @tool — LangChain extracts the
JSON schema from the type hints and docstring automatically.

Each tool does one focused query against OpenSearch and returns a JSON string.
The investigator agent (Qwen3.5-9B, thinking=ON) decides which tools to call
and in what order.
"""

import json
import os
from typing import Annotated

from langchain_core.tools import tool
from agent.os_client import client as _client
from dotenv import load_dotenv

load_dotenv()

INDEX = os.getenv("OPENSEARCH_INDEX", "faers_reports")


async def _get_prr(drug_names: list[str], reaction: str) -> dict:
    """
    Compute PRR for a specific drug+reaction pair in one round-trip.

    Uses a single query with two filter aggs:
      - drug_rxn:  reports for this drug that have the reaction
      - all_rxn:   all FAERS reports that have the reaction (baseline)
    Plus track_total_hits for drug_total and a separate count for faers_total.
    """
    client = _client()
    try:
        rxn_upper = reaction.upper()
        resp = await client.search(index=INDEX, body={
            "size": 0,
            "track_total_hits": True,
            "query": {"terms": {"drug_names": drug_names}},
            "aggs": {
                "drug_rxn": {"filter": {"term": {"reactions": rxn_upper}}},
                "global_rxn": {
                    "global": {},
                    "aggs": {"rxn": {"filter": {"term": {"reactions": rxn_upper}}}},
                },
            },
        })

        drug_total  = resp["hits"]["total"]["value"]
        drug_count  = resp["aggregations"]["drug_rxn"]["doc_count"]
        faers_total = resp["aggregations"]["global_rxn"]["doc_count"]
        baseline    = resp["aggregations"]["global_rxn"]["rxn"]["doc_count"]

        if drug_total == 0 or baseline == 0 or faers_total == 0:
            return {"prr": 0, "drug_count": drug_count, "drug_total": drug_total}

        non_drug_total = faers_total - drug_total
        non_drug_count = baseline - drug_count
        if non_drug_total <= 0 or non_drug_count <= 0:
            return {"prr": 0, "drug_count": drug_count, "drug_total": drug_total,
                    "baseline": baseline, "faers_total": faers_total}

        prr = (drug_count / drug_total) / (non_drug_count / non_drug_total)
        return {
            "prr":         round(prr, 2),
            "drug_count":  drug_count,
            "drug_total":  drug_total,
            "baseline":    baseline,
            "faers_total": faers_total,
        }
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Tool 1: PRR for any drug+reaction
# ---------------------------------------------------------------------------

@tool
async def get_prr(
    drug: Annotated[str, "Drug name in ALL CAPS as stored in FAERS drug_names field"],
    reaction: Annotated[str, "MedDRA Preferred Term in ALL CAPS"],
) -> str:
    """
    Calculate Proportional Reporting Ratio (PRR) for a specific drug-reaction
    pair in the FAERS database. PRR >= 2.0 with n >= 3 is an EMA signal threshold.
    Returns prr, drug_count, drug_total, baseline, faers_total.
    """
    result = await _get_prr([drug], reaction)
    return json.dumps({**result, "drug": drug, "reaction": reaction})


# ---------------------------------------------------------------------------
# Tool 2: Class effect check — compare PRR across a drug class
# ---------------------------------------------------------------------------

@tool
async def check_class_effect(
    reaction: Annotated[str, "MedDRA reaction term in ALL CAPS"],
    comparator_drugs: Annotated[list[str], "List of comparator drug names in ALL CAPS"],
) -> str:
    """
    Check whether a reaction is a class-wide effect by comparing PRR
    across multiple drugs in the same therapeutic class.
    If all comparators show elevated PRR, the signal is likely a class effect,
    not specific to the index drug.
    Returns PRR for each comparator and a class_effect boolean.
    """
    results = {}
    for drug in comparator_drugs:
        r = await _get_prr([drug], reaction)
        results[drug] = {"prr": r["prr"], "count": r["drug_count"]}

    elevated = [d for d, v in results.items() if v["prr"] >= 2.0]
    class_effect = len(elevated) >= len(comparator_drugs) * 0.6

    return json.dumps({
        "reaction":     reaction,
        "comparators":  results,
        "elevated":     elevated,
        "class_effect": class_effect,
        "interpretation": (
            f"CLASS EFFECT — {len(elevated)}/{len(comparator_drugs)} drugs show PRR≥2"
            if class_effect else
            f"DRUG-SPECIFIC — only {len(elevated)}/{len(comparator_drugs)} comparators elevated"
        ),
    })


# ---------------------------------------------------------------------------
# Tool 3: Temporal trend — when did the signal emerge?
# ---------------------------------------------------------------------------

@tool
async def get_signal_trend(
    drug: Annotated[str, "Drug name in ALL CAPS"],
    reaction: Annotated[str, "Reaction term in ALL CAPS"],
) -> str:
    """
    Get the yearly report count for a drug+reaction pair to identify
    when a signal first emerged and whether it is growing or stable.
    A rapidly growing signal is higher priority than a stable one.
    """
    client = _client()
    try:
        resp = await client.search(index=INDEX, body={
            "size": 0,
            "query": {"bool": {"must": [
                {"term": {"drug_names": drug}},
                {"term": {"reactions": reaction}},
            ]}},
            "aggs": {"by_year": {"date_histogram": {
                "field": "receivedate",
                "calendar_interval": "year",
                "format": "yyyy",
            }}},
        })

        buckets  = resp["aggregations"]["by_year"]["buckets"]
        timeline = [{"year": b["key_as_string"], "count": b["doc_count"]}
                    for b in buckets if b["doc_count"] > 0]

        if len(timeline) >= 2:
            first_year  = timeline[0]["year"]
            last_count  = timeline[-1]["count"]
            first_count = timeline[0]["count"]
            trend = "GROWING" if last_count > first_count * 1.5 else "STABLE"
        else:
            first_year = timeline[0]["year"] if timeline else "unknown"
            trend = "INSUFFICIENT_DATA"

        return json.dumps({
            "drug": drug, "reaction": reaction,
            "first_seen": first_year,
            "trend": trend,
            "timeline": timeline[-6:],
        })
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Tool 4: Two-window temporal comparison — EMERGING / GROWING / STABLE
#
# Replaces the old DataDistributionTool wrapper. DataDistributionTool only
# analyses low-cardinality scalar fields (sex, reporter_type, age, country)
# and silently excludes the multi-valued `reactions[]` array, so it always
# returned "reaction not in top distribution changes" — unconditionally dead.
#
# This implementation does a direct two-window count query on the actual
# reactions field, computing per-reaction rates in each period.
# ---------------------------------------------------------------------------


def _classify_periods(rc: int, rt: int, bc: int, bt: int) -> str:
    """
    Pure classification logic for two-window temporal comparison.

    Extracted as a standalone function so tests can call the real implementation
    rather than duplicating the logic. Used by compare_time_periods below.

    Args:
        rc: reaction count in recent period
        rt: total drug reports in recent period
        bc: reaction count in baseline period
        bt: total drug reports in baseline period

    Returns:
        One of: EMERGING / GROWING / DECLINING / NOT REPORTED / STABLE
    """
    r_rate = rc / rt if rt else 0.0
    b_rate = bc / bt if bt else 0.0

    if bc == 0 and rc > 0:
        return "EMERGING — absent in baseline, present recently"
    elif bt > 0 and rt > 0 and r_rate > b_rate * 1.5:
        return "GROWING — reporting rate increased vs baseline"
    elif bt > 0 and rt > 0 and r_rate < b_rate * 0.67:
        return "DECLINING — reporting rate fell vs baseline"
    elif rc == 0 and bc == 0:
        return "NOT REPORTED — no reports in either period"
    else:
        return "STABLE — similar rate in both periods"

@tool
async def compare_time_periods(
    drug: Annotated[str, "Drug name in ALL CAPS"],
    reaction: Annotated[str, "MedDRA reaction term in ALL CAPS"],
    recent_start: Annotated[str, "ISO date for recent period start — use the drug's most recent reporting years (e.g. last 2 years of available data). Format: yyyy-MM-dd"],
    recent_end:   Annotated[str, "ISO date for recent period end — typically the latest date in the dataset. Format: yyyy-MM-dd"],
    baseline_start: Annotated[str, "ISO date for baseline period start — use the drug's earliest reporting years. Format: yyyy-MM-dd"],
    baseline_end:   Annotated[str, "ISO date for baseline period end — typically 2+ years before the recent period start. Format: yyyy-MM-dd"],
) -> str:
    """
    Compare how a specific drug-reaction pair changed between a baseline and
    a recent time period, using direct two-window count queries.

    Returns reaction counts and rates for both periods, and classifies the
    signal as EMERGING (absent in baseline, present recently), GROWING
    (rate increased >50% vs baseline), DECLINING, or STABLE.

    Use this to confirm whether a signal is a new phenomenon or pre-existing,
    and whether reporting is accelerating.
    """
    client = _client()
    rxn = reaction.upper()
    try:
        resp = await client.search(index=INDEX, body={
            "size": 0,
            "query": {"terms": {"drug_names": [drug]}},
            "aggs": {
                "recent": {"filter": {"bool": {"must": [
                    {"term":  {"reactions":    rxn}},
                    {"range": {"receivedate": {"gte": recent_start, "lte": recent_end}}},
                ]}}},
                "recent_total": {"filter": {"range": {
                    "receivedate": {"gte": recent_start, "lte": recent_end}
                }}},
                "baseline": {"filter": {"bool": {"must": [
                    {"term":  {"reactions":    rxn}},
                    {"range": {"receivedate": {"gte": baseline_start, "lte": baseline_end}}},
                ]}}},
                "baseline_total": {"filter": {"range": {
                    "receivedate": {"gte": baseline_start, "lte": baseline_end}
                }}},
            },
        })

        a  = resp["aggregations"]
        rc = a["recent"]["doc_count"]
        rt = a["recent_total"]["doc_count"]
        bc = a["baseline"]["doc_count"]
        bt = a["baseline_total"]["doc_count"]

        r_rate = rc / rt if rt else 0.0
        b_rate = bc / bt if bt else 0.0
        interpretation = _classify_periods(rc, rt, bc, bt)

        return json.dumps({
            "drug":     drug,
            "reaction": rxn,
            "periods": {
                "recent":   f"{recent_start[:10]} → {recent_end[:10]}",
                "baseline": f"{baseline_start[:10]} → {baseline_end[:10]}",
            },
            "recent":   {"count": rc, "total": rt, "rate": round(r_rate, 5)},
            "baseline": {"count": bc, "total": bt, "rate": round(b_rate, 5)},
            "interpretation": interpretation,
        })
    except Exception as e:
        return json.dumps({"error": str(e), "drug": drug, "reaction": rxn})
    finally:
        await client.close()
