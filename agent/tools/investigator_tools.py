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

import httpx
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
    Saves one network round-trip vs the previous two-query implementation.
    """
    client = _client()
    try:
        rxn_upper = reaction.upper()

        # Single query: drug filter gives drug_total + drug_count;
        # global_rxn sub-agg gives baseline across all drugs.
        resp = await client.search(index=INDEX, body={
            "size": 0,
            "track_total_hits": True,
            "query": {"terms": {"drug_names": drug_names}},
            "aggs": {
                # Reaction count for this drug
                "drug_rxn": {"filter": {"term": {"reactions": rxn_upper}}},
                # Baseline: reaction across ALL drugs (global filter, no drug query)
                "global_rxn": {
                    "global": {},
                    "aggs": {
                        "rxn": {"filter": {"term": {"reactions": rxn_upper}}}
                    }
                },
            },
        })

        drug_total  = resp["hits"]["total"]["value"]
        drug_count  = resp["aggregations"]["drug_rxn"]["doc_count"]
        faers_total = resp["aggregations"]["global_rxn"]["doc_count"]
        baseline    = resp["aggregations"]["global_rxn"]["rxn"]["doc_count"]

        if drug_total == 0 or baseline == 0 or faers_total == 0:
            return {"prr": 0, "drug_count": drug_count, "drug_total": drug_total}

        # Textbook 2×2: comparator arm is the NON-drug population
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


# ---------------------------------------------------------------------------
# Tool 5: DataDistributionTool (OpenSearch 3.3+ ML Commons built-in)
# Compares reaction distribution between two time periods — identifies WHEN
# a signal emerged without pre-computing class_ratio.
# ---------------------------------------------------------------------------

_OS_BASE = os.getenv("OPENSEARCH_URL", "https://localhost:9200")
_OS_AUTH = (os.getenv("OPENSEARCH_USER", "admin"),
            os.getenv("OPENSEARCH_PASSWORD", "Pharma@2024!"))


@tool
async def compare_time_periods(
    drug: Annotated[str, "Drug name in ALL CAPS"],
    reaction: Annotated[str, "MedDRA reaction term in ALL CAPS"],
    recent_start: Annotated[str, "ISO date for recent period start, e.g. 2023-01-01T00:00:00.000Z"],
    recent_end:   Annotated[str, "ISO date for recent period end, e.g. 2026-01-01T00:00:00.000Z"],
    baseline_start: Annotated[str, "ISO date for baseline period start, e.g. 2018-01-01T00:00:00.000Z"],
    baseline_end:   Annotated[str, "ISO date for baseline period end, e.g. 2022-01-01T00:00:00.000Z"],
) -> str:
    """
    Use OpenSearch DataDistributionTool (ML Commons 3.3+) to compare how
    the distribution of reactions changed between a baseline and recent period.

    Returns divergence scores and top-changed fields — identifies whether a
    signal is NEW (absent in baseline, present recently) or GROWING (present
    in both but higher recently). More powerful than a simple count comparison
    because it analyses the full field distribution, not just one reaction.

    Use this when you need to understand the temporal emergence of a signal.
    """
    params = {
        "index":                  INDEX,
        "timeField":              "receivedate",
        "selectionTimeRangeStart": recent_start,
        "selectionTimeRangeEnd":   recent_end,
        "baselineTimeRangeStart":  baseline_start,
        "baselineTimeRangeEnd":    baseline_end,
        "size": 500,
    }
    try:
        async with httpx.AsyncClient(verify=False, timeout=30) as client:
            r = await client.post(
                f"{_OS_BASE}/_plugins/_ml/tools/_execute/DataDistributionTool",
                auth=_OS_AUTH,
                headers={"Content-Type": "application/json"},
                json={"parameters": params},
            )
            if r.status_code != 200:
                return json.dumps({"error": f"HTTP {r.status_code}: {r.text[:200]}"})

            raw = r.json()
            # Parse the nested result string
            result_str = (raw.get("inference_results", [{}])[0]
                            .get("output", [{}])[0]
                            .get("result", "{}"))
            result = json.loads(result_str)

            # Focus on the reactions field — extract signal-relevant findings
            reaction_changes = []
            for field_analysis in result.get("comparisonAnalysis", []):
                if field_analysis.get("field") != "reactions":
                    continue
                for change in field_analysis.get("topChanges", []):
                    if reaction.upper() in change.get("value", "").upper():
                        reaction_changes.append({
                            "reaction":              change["value"],
                            "recent_pct":            change.get("selectionPercentage", 0),
                            "baseline_pct":          change.get("baselinePercentage", 0),
                            "divergence":            field_analysis.get("divergence", 0),
                            "interpretation": (
                                "EMERGING — not in baseline, appeared recently"
                                if change.get("baselinePercentage", 0) == 0 else
                                "GROWING — increased vs baseline"
                                if change.get("selectionPercentage", 0) > change.get("baselinePercentage", 0) else
                                "STABLE"
                            )
                        })

            return json.dumps({
                "drug": drug, "reaction": reaction,
                "periods": {"recent": f"{recent_start[:10]} → {recent_end[:10]}",
                            "baseline": f"{baseline_start[:10]} → {baseline_end[:10]}"},
                "reaction_changes": reaction_changes or [{"note": "reaction not in top distribution changes"}],
                "overall_divergence": result.get("comparisonAnalysis", [{}])[0].get("divergence", 0)
                    if result.get("comparisonAnalysis") else 0,
            })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool 6: SearchIndexTool — flexible DSL search (OpenSearch MCP built-in)
# Lets the investigator run custom queries: by demographics, reporter type,
# serious outcomes, concomitant drugs — anything DSL supports.
# ---------------------------------------------------------------------------

@tool
async def search_faers(
    query_description: Annotated[str, "Plain English description of what you're looking for"],
    drug: Annotated[str, "Drug name in ALL CAPS"],
    reaction: Annotated[str, "MedDRA reaction term in ALL CAPS (optional, use empty string if not filtering)"] = "",
    filters: Annotated[str, "Optional JSON filter string e.g. '{\"term\":{\"serious\":\"1\"}}' (leave empty for no filter)"] = "",
    size: Annotated[int, "Number of results to return (max 20)"] = 10,
) -> str:
    """
    Flexible search of the FAERS index using OpenSearch Query DSL.

    Use this when you need to investigate a signal from a specific angle
    that get_prr or check_class_effect don't cover:
      - 'How many serious outcomes reported for this reaction?'
      - 'Is this reaction reported more in elderly patients?'
      - 'Which concomitant drugs appear most often with this reaction?'
      - 'What reporter types (physician/consumer) report this most?'

    Returns aggregation summary, not raw documents.
    """
    client = _client()
    try:
        must_clauses = [{"terms": {"drug_names": [drug]}}]
        if reaction:
            must_clauses.append({"term": {"reactions": reaction.upper()}})
        if filters:
            import json as _json
            try:
                must_clauses.append(_json.loads(filters))
            except Exception:
                pass  # ignore malformed filter JSON

        body = {
            "size": 0,
            "track_total_hits": True,
            "query": {"bool": {"must": must_clauses}},
            "aggs": {
                "serious":       {"terms": {"field": "serious",       "size": 5}},
                "reporter_type": {"terms": {"field": "reporter_type", "size": 5}},
                "country":       {"terms": {"field": "country",       "size": 10}},
                "sex":           {"terms": {"field": "patient_sex",   "size": 3}},
                "by_year": {
                    "date_histogram": {
                        "field": "receivedate",
                        "calendar_interval": "year",
                        "format": "yyyy",
                    }
                },
            },
        }

        resp = await client.search(index=INDEX, body=body)
        total = resp["hits"]["total"]["value"]
        aggs  = resp["aggregations"]

        return json.dumps({
            "query_description": query_description,
            "drug": drug,
            "reaction": reaction or "all reactions",
            "total_reports": total,
            "serious_breakdown": {b["key"]: b["doc_count"] for b in aggs["serious"]["buckets"]},
            "reporter_breakdown": {b["key"]: b["doc_count"] for b in aggs["reporter_type"]["buckets"]},
            "sex_breakdown": {b["key"]: b["doc_count"] for b in aggs["sex"]["buckets"]},
            "top_countries": {b["key"]: b["doc_count"] for b in aggs["country"]["buckets"][:5]},
            "yearly_trend": [{"year": b["key_as_string"], "count": b["doc_count"]}
                             for b in aggs["by_year"]["buckets"] if b["doc_count"] > 0],
        })
    finally:
        await client.close()
