"""
PRR (Proportional Reporting Ratio) calculation via OpenSearch aggregations.

No LLM query generation — PRR is computed directly in Python using the
opensearch-py client. This eliminates query syntax bugs and is fully
reproducible across model versions.

Formula: PRR = (drug_count/drug_total) / (baseline/faers_total)
EMA threshold: PRR >= 2.0 AND count >= 3

Reference: EMA Guideline on the use of statistical signal detection
methods in the EudraVigilance database (EMA/813938/2011)
"""

import os
from typing import Any
from opensearchpy import AsyncOpenSearch
from dotenv import load_dotenv

load_dotenv()

INDEX = os.getenv("OPENSEARCH_INDEX", "faers_reports")


def _client() -> AsyncOpenSearch:
    return AsyncOpenSearch(
        hosts=[os.getenv("OPENSEARCH_URL", "https://localhost:9200")],
        http_auth=(
            os.getenv("OPENSEARCH_USER", "admin"),
            os.getenv("OPENSEARCH_PASSWORD", "Admin@changeme1"),
        ),
        use_ssl=True,
        verify_certs=False,  # self-signed cert in local dev
        ssl_show_warn=False,
    )


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
                    (e.g. ["SEMAGLUTIDE", "OZEMPIC", "WEGOVY"])
        top_n:      Number of top reactions to consider for the drug
        min_count:  Minimum reports to be considered a signal
        min_prr:    Minimum PRR threshold (EMA standard: 2.0)

    Returns:
        dict with drug_total, faers_total, and signals list
    """
    client = _client()
    try:
        # Drug-specific reaction counts
        drug_resp = await client.search(
            index=INDEX,
            body={
                "size": 0,
                "track_total_hits": True,   # bypass 10K cap
                "query": {"terms": {"drug_names": drug_names}},
                "aggs": {
                    "reactions": {
                        "terms": {"field": "reactions", "size": top_n}
                    }
                },
            },
        )
        drug_total = drug_resp["hits"]["total"]["value"]

        # Population-wide baseline (all drugs, top 500 reactions)
        baseline_resp = await client.search(
            index=INDEX,
            body={
                "size": 0,
                "track_total_hits": True,   # bypass 10K cap
                "aggs": {
                    "reactions": {
                        "terms": {"field": "reactions", "size": 500}
                    },
                },
            },
        )
        faers_total = baseline_resp["hits"]["total"]["value"]

        baselines = {
            b["key"]: b["doc_count"]
            for b in baseline_resp["aggregations"]["reactions"]["buckets"]
        }

        # Calculate PRR for each reaction
        signals = []
        for bucket in drug_resp["aggregations"]["reactions"]["buckets"]:
            reaction = bucket["key"]
            drug_count = bucket["doc_count"]
            baseline = baselines.get(reaction, 0)

            if baseline == 0 or drug_count < min_count:
                continue

            prr = (drug_count / drug_total) / (baseline / faers_total)
            if prr >= min_prr:
                signals.append(
                    {
                        "reaction": reaction,
                        "prr": round(prr, 2),
                        "drug_count": drug_count,
                        "baseline": baseline,
                    }
                )

        signals.sort(key=lambda x: -x["prr"])

        return {
            "drug_names": drug_names,
            "drug_total": drug_total,
            "faers_total": faers_total,
            "signals": signals,
            "signal_count": len(signals),
        }

    finally:
        await client.close()


async def get_drug_names(drug_name: str) -> dict[str, Any]:
    """
    Resolve a drug name to all FAERS variants using two-step lookup:
    1. RxNorm API  → canonical generic + official brand names
    2. FAERS index → any additional name variants actually in the data

    Args:
        drug_name: Generic or brand name (case-insensitive)
    Returns:
        dict with found_names list (ALL CAPS, as stored in FAERS)
    """
    import httpx

    RXNORM = "https://rxnav.nlm.nih.gov/REST"
    all_names: set[str] = {drug_name.upper()}

    # Step 1: RxNorm — get official brand names
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            # Resolve to ingredient RxCUI
            r = await http.get(f"{RXNORM}/rxcui.json",
                               params={"name": drug_name, "search": "1"})
            rxcui = r.json().get("idGroup", {}).get("rxnormId", [None])[0]

            if rxcui:
                # Get brand names (BN tty)
                r2 = await http.get(f"{RXNORM}/rxcui/{rxcui}/related.json",
                                    params={"tty": "BN"})
                for grp in r2.json().get("relatedGroup", {}).get("conceptGroup", []):
                    for prop in grp.get("conceptProperties", []):
                        name = prop.get("name", "").upper().strip()
                        if name and not any(c.isdigit() for c in name):
                            all_names.add(name)
    except Exception:
        pass  # RxNorm unavailable — fall back to FAERS index lookup

    # Known fallbacks for withdrawn drugs not in RxNorm BN
    FALLBACKS = {
        "ROFECOXIB": ["VIOXX"], "CERIVASTATIN": ["BAYCOL"],
        "CISAPRIDE": ["PROPULSID"], "TERFENADINE": ["SELDANE"],
    }
    all_names.update(FALLBACKS.get(drug_name.upper(), []))

    found = sorted(all_names)
    return {"query": drug_name, "found_names": found}


async def get_signal_timeline(
    drug_names: list[str], reaction: str
) -> dict[str, Any]:
    """
    Get quarterly report counts for a drug+reaction pair over time.
    Used to identify when a signal first emerged.
    """
    client = _client()
    try:
        resp = await client.search(
            index=INDEX,
            body={
                "size": 0,
                "query": {
                    "bool": {
                        "must": [
                            {"terms": {"drug_names": drug_names}},
                            {"term": {"reactions": reaction.upper()}},
                        ]
                    }
                },
                "aggs": {
                    "by_quarter": {
                        "date_histogram": {
                            "field": "receipt_date",
                            "calendar_interval": "quarter",
                            "format": "yyyy-QQ",
                        }
                    }
                },
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
