"""
PRR (Proportional Reporting Ratio) calculation via OpenSearch aggregations.

Formula: PRR = (drug_count / drug_total) / (non_drug_count / non_drug_total)
         (textbook 2×2 contingency table — non-exposed denominator)

EMA threshold: PRR >= 2.0 AND count >= 3

Baseline: fetched per-reaction via `filters` aggregation — not truncated to
a global top-N. Rare/novel signals are never silently dropped.

Reference: EMA/813938/2011 Guideline on statistical signal detection methods.
"""

import os
from typing import Any
from agent.os_client import client as _client
from dotenv import load_dotenv

load_dotenv()

INDEX = os.getenv("OPENSEARCH_INDEX", "faers_reports")


def _yates_chi2(a: int, b: int, c: int, d: int) -> float:
    """
    Yates continuity-corrected Pearson χ² for a 2×2 contingency table.

        χ² = N · (|ad − bc| − N/2)² / ((a+b)(c+d)(a+c)(b+d))

    Yates correction is used (vs uncorrected Pearson) because PRR signals
    are screened at small exposed-cell counts (n ≥ 3). Uncorrected χ²
    over-rejects at small counts; Yates is conservative and matches the
    EMA-style PRR threshold calibration (PRR≥2 / χ²≥4 / n≥3).
    """
    n = a + b + c + d
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    denom = row1 * row2 * col1 * col2
    if denom == 0 or n == 0:
        return 0.0
    numer = n * (abs(a * d - b * c) - n / 2.0) ** 2
    return numer / denom


async def calculate_prr(
    drug_names: list[str],
    top_n: int = 50,
    min_count: int = 3,
    min_prr: float = 2.0,
) -> dict[str, Any]:
    """
    Calculate PRR signals for a drug against the full FAERS population.

    Args:
        drug_names: List of drug name variants in ALL CAPS
        top_n:      Number of top reactions to check for the drug
        min_count:  Minimum drug reports (EMA: n >= 3)
        min_prr:    Minimum PRR threshold (EMA: >= 2.0)
    """
    client = _client()
    try:
        # Step 1: drug total + top-N reactions
        drug_resp = await client.search(
            index=INDEX,
            body={
                "size": 0,
                "track_total_hits": True,
                "query": {"terms": {"drug_names": drug_names}},
                "aggs": {"reactions": {"terms": {"field": "reactions", "size": top_n}}},
            },
        )
        drug_total   = drug_resp["hits"]["total"]["value"]
        drug_buckets = drug_resp["aggregations"]["reactions"]["buckets"]

        # Step 2: FAERS total
        faers_total = (await client.count(index=INDEX))["count"]

        # Step 3: per-reaction baseline via filters agg — no rank truncation
        reaction_keys = [b["key"] for b in drug_buckets]
        baselines: dict[str, int] = {}
        if reaction_keys:
            base_resp = await client.search(
                index=INDEX,
                body={
                    "size": 0,
                    "aggs": {
                        "per_reaction": {
                            "filters": {
                                "filters": {
                                    rxn: {"term": {"reactions": rxn}}
                                    for rxn in reaction_keys
                                }
                            }
                        }
                    },
                },
            )
            baselines = {
                rxn: bucket["doc_count"]
                for rxn, bucket in
                base_resp["aggregations"]["per_reaction"]["buckets"].items()
            }

        # Step 4: compute PRR using textbook 2×2 table
        non_drug_total = faers_total - drug_total
        signals = []
        for bucket in drug_buckets:
            reaction   = bucket["key"]
            drug_count = bucket["doc_count"]
            baseline   = baselines.get(reaction, 0)

            if drug_count < min_count or baseline == 0:
                continue

            non_drug_count = baseline - drug_count
            if non_drug_total <= 0 or non_drug_count <= 0:
                continue

            prr = (drug_count / drug_total) / (non_drug_count / non_drug_total)
            if prr >= min_prr:
                # Annotate with Yates χ² — do NOT filter on it.
                # A PRR=15 on n=4 is statistically weak but clinically interesting;
                # we surface it with significant=False for human triage.
                chi2 = _yates_chi2(
                    a=drug_count,
                    b=drug_total - drug_count,
                    c=non_drug_count,
                    d=non_drug_total - non_drug_count,
                )
                signals.append({
                    "reaction":    reaction,
                    "prr":         round(prr, 2),
                    "drug_count":  drug_count,
                    "baseline":    baseline,
                    "chi2":        round(chi2, 2),
                    "significant": chi2 >= 4.0,   # EMA standard gate
                })

        signals.sort(key=lambda x: -x["prr"])
        return {
            "drug_names":   drug_names,
            "drug_total":   drug_total,
            "faers_total":  faers_total,
            "signals":      signals,
            "signal_count": len(signals),
        }

    finally:
        await client.close()


async def get_drug_names(drug_name: str) -> dict[str, Any]:
    """Resolve a drug name to all FAERS variants using RxNorm + fallback dict."""
    import httpx

    RXNORM = "https://rxnav.nlm.nih.gov/REST"
    all_names: set[str] = {drug_name.upper()}

    try:
        async with httpx.AsyncClient(timeout=10) as http:
            r = await http.get(f"{RXNORM}/rxcui.json",
                               params={"name": drug_name, "search": "1"})
            rxcui = r.json().get("idGroup", {}).get("rxnormId", [None])[0]
            if rxcui:
                r2 = await http.get(f"{RXNORM}/rxcui/{rxcui}/related.json",
                                    params={"tty": "BN"})
                for grp in r2.json().get("relatedGroup", {}).get("conceptGroup", []):
                    for prop in grp.get("conceptProperties", []):
                        name = prop.get("name", "").upper().strip()
                        if name and not any(c.isdigit() for c in name):
                            all_names.add(name)
    except Exception:
        pass

    FALLBACKS = {
        "ROFECOXIB": ["VIOXX"], "CERIVASTATIN": ["BAYCOL"],
        "CISAPRIDE": ["PROPULSID"], "TERFENADINE": ["SELDANE"],
    }
    all_names.update(FALLBACKS.get(drug_name.upper(), []))
    return {"query": drug_name, "found_names": sorted(all_names)}


async def get_signal_timeline(
    drug_names: list[str], reaction: str
) -> dict[str, Any]:
    """Get quarterly report counts for a drug+reaction pair over time."""
    client = _client()
    try:
        resp = await client.search(
            index=INDEX,
            body={
                "size": 0,
                "query": {"bool": {"must": [
                    {"terms": {"drug_names": drug_names}},
                    {"term":  {"reactions": reaction.upper()}},
                ]}},
                "aggs": {"by_quarter": {"date_histogram": {
                    "field":             "receivedate",
                    "calendar_interval": "quarter",
                    "format":            "yyyy-QQQ",
                }}},
            },
        )
        timeline = [
            {"period": b["key_as_string"], "count": b["doc_count"]}
            for b in resp["aggregations"]["by_quarter"]["buckets"]
            if b["doc_count"] > 0
        ]
        return {"drug_names": drug_names, "reaction": reaction, "timeline": timeline}
    finally:
        await client.close()
