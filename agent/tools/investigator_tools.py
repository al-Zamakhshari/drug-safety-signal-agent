"""
Real OpenSearch-backed tools for the investigator agent.

These are Python functions decorated with @tool — LangChain extracts the
JSON schema from the type hints and docstring automatically.

Each tool does one focused query against OpenSearch and returns a JSON string.
The investigator agent (Gemma4 E4B) decides which tools to call and in what order.
"""

import json
import os
from typing import Annotated

from langchain_core.tools import tool
from opensearchpy import AsyncOpenSearch
from dotenv import load_dotenv

load_dotenv()

INDEX = os.getenv("OPENSEARCH_INDEX", "faers_reports")


def _client() -> AsyncOpenSearch:
    return AsyncOpenSearch(
        hosts=[os.getenv("OPENSEARCH_URL", "https://localhost:9200")],
        http_auth=(
            os.getenv("OPENSEARCH_USER", "admin"),
            os.getenv("OPENSEARCH_PASSWORD", "Pharma@2024!"),
        ),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
    )


async def _get_prr(drug_names: list[str], reaction: str) -> dict:
    """Helper: compute PRR for a specific drug+reaction pair."""
    client = _client()
    try:
        drug_resp = await client.search(index=INDEX, body={
            "size": 0, "track_total_hits": True,
            "query": {"terms": {"drug_names": drug_names}},
            "aggs": {"rxn": {"filter": {"term": {"reactions": reaction.upper()}}}},
        })
        drug_total = drug_resp["hits"]["total"]["value"]
        drug_count = drug_resp["aggregations"]["rxn"]["doc_count"]

        base_resp = await client.search(index=INDEX, body={
            "size": 0, "track_total_hits": True,
            "aggs": {"rxn": {"filter": {"term": {"reactions": reaction.upper()}}}},
        })
        faers_total = base_resp["hits"]["total"]["value"]
        baseline   = base_resp["aggregations"]["rxn"]["doc_count"]

        if drug_total == 0 or baseline == 0 or faers_total == 0:
            return {"prr": 0, "drug_count": drug_count, "drug_total": drug_total}

        prr = (drug_count / drug_total) / (baseline / faers_total)
        return {
            "prr":        round(prr, 2),
            "drug_count": drug_count,
            "drug_total": drug_total,
            "baseline":   baseline,
            "faers_total": faers_total,
        }
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Tool 1: PRR for any drug+reaction
# ---------------------------------------------------------------------------

@tool
async def get_prr(
    drug: Annotated[str, "Drug name in ALL CAPS (e.g. SEMAGLUTIDE, OZEMPIC)"],
    reaction: Annotated[str, "MedDRA reaction term in ALL CAPS (e.g. ASTHMA, PANCREATITIS)"],
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
    class_effect = len(elevated) >= len(comparator_drugs) * 0.6  # 60% threshold

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
# Tool 3: Drug-drug interaction check
# ---------------------------------------------------------------------------

@tool
async def check_ddi(
    primary_drug: Annotated[str, "Primary drug in ALL CAPS"],
    suspect_drug: Annotated[str, "Suspected interacting drug in ALL CAPS"],
    reaction: Annotated[str, "Reaction to investigate in ALL CAPS"],
) -> str:
    """
    Check for drug-drug interaction (DDI) signal: compare the reaction rate
    in reports that mention BOTH drugs vs primary drug alone.
    A much higher co-occurrence rate suggests the reaction may be caused
    by the drug combination, not the primary drug alone.
    """
    client = _client()
    try:
        # Reports with primary drug + reaction
        alone_resp = await client.search(index=INDEX, body={
            "size": 0, "track_total_hits": True,
            "query": {"bool": {"must": [
                {"term": {"drug_names": primary_drug}},
                {"term": {"reactions": reaction}},
            ]}},
        })
        alone_count = alone_resp["hits"]["total"]["value"]

        # Reports with BOTH drugs + reaction
        combo_resp = await client.search(index=INDEX, body={
            "size": 0, "track_total_hits": True,
            "query": {"bool": {"must": [
                {"term": {"drug_names": primary_drug}},
                {"term": {"drug_names": suspect_drug}},
                {"term": {"reactions": reaction}},
            ]}},
        })
        combo_count = combo_resp["hits"]["total"]["value"]

        # Primary drug total reports
        total_resp = await client.search(index=INDEX, body={
            "size": 0, "track_total_hits": True,
            "query": {"term": {"drug_names": primary_drug}},
        })
        drug_total = total_resp["hits"]["total"]["value"]

        rate_alone = alone_count / drug_total if drug_total else 0
        rate_combo = combo_count / drug_total if drug_total else 0
        ddi_ratio  = rate_combo / rate_alone if rate_alone else 0

        return json.dumps({
            "primary_drug":      primary_drug,
            "suspect_drug":      suspect_drug,
            "reaction":          reaction,
            "alone_count":       alone_count,
            "combo_count":       combo_count,
            "ddi_ratio":         round(ddi_ratio, 2),
            "ddi_likely":        ddi_ratio >= 2.0 and combo_count >= 3,
            "interpretation": (
                f"DDI LIKELY — co-occurrence {ddi_ratio:.1f}x higher with {suspect_drug}"
                if ddi_ratio >= 2.0 and combo_count >= 3 else
                f"DDI UNLIKELY — co-occurrence not significantly elevated"
            ),
        })
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Tool 4: Temporal trend — when did the signal emerge?
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

        buckets = resp["aggregations"]["by_year"]["buckets"]
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
            "timeline": timeline[-6:],  # last 6 years
        })
    finally:
        await client.close()
