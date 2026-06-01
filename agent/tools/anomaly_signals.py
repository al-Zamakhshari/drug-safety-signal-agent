"""
Query class_ratio anomaly signals for drug safety from faers_ml_rates index.

class_ratio = drug_rate / mean(comparator_class_rate)

class_ratio > 3           → strong drug-specific signal
class_ratio > 1           → mild elevation above class baseline
class_ratio < 1           → reaction is LESS common than class average
class_ratio = 999.0 (sentinel) → reaction absent from ALL comparators;
                              shown as no_class_baseline=True (undefined ratio)

No AD detector training needed — class_ratio is pre-computed by
compute_class_ratio.py and is immediately queryable.
"""

import os
from typing import Any
from agent.os_client import client as _client
from dotenv import load_dotenv

load_dotenv()

RATES_INDEX = "faers_ml_rates"


async def get_anomaly_signals(
    drug: str,
    min_ratio: float = 2.0,
    min_count: int = 5,
    top_n: int = 15,
) -> dict[str, Any]:
    """
    Get class_ratio anomaly signals for a drug from faers_ml_rates.

    Args:
        drug:       Drug name in ALL CAPS (as stored in faers_ml_rates)
        min_ratio:  Minimum class_ratio to consider a signal (default 2.0)
        min_count:  Minimum drug_count per reaction
        top_n:      Max signals to return
    """
    client = _client()
    try:
        try:
            count_r = await client.count(
                index=RATES_INDEX,
                body={"query": {"term": {"drug": drug}}}
            )
            doc_count = count_r["count"]
        except Exception:
            doc_count = 0

        if doc_count == 0:
            return {
                "drug": drug,
                "signals": [],
                "note": f"No class_ratio data for {drug}. "
                        f"Run: uv run python -m ingestion.compute_class_ratio",
            }

        resp = await client.search(
            index=RATES_INDEX,
            body={
                "size": 0,
                "query": {"term": {"drug": drug}},
                "aggs": {
                    "by_reaction": {
                        "terms": {"field": "reaction", "size": 300},
                        "aggs": {
                            "max_ratio":     {"max":   {"field": "class_ratio"}},
                            "avg_ratio":     {"avg":   {"field": "class_ratio"}},
                            "max_count":     {"max":   {"field": "drug_count"}},
                            "quarters_seen": {"value_count": {"field": "quarter"}},
                            "recent": {
                                "filter": {"range": {"quarter": {"gte": "2023-01-01"}}},
                                "aggs": {"avg_recent": {"avg": {"field": "class_ratio"}}}
                            },
                            "early": {
                                "filter": {"range": {"quarter": {
                                    "gte": "2020-01-01", "lt": "2022-01-01"
                                }}},
                                "aggs": {"avg_early": {"avg": {"field": "class_ratio"}}}
                            },
                        }
                    }
                }
            }
        )

        signals = []
        for b in resp["aggregations"]["by_reaction"]["buckets"]:
            rxn       = b["key"]
            max_ratio = b["max_ratio"]["value"] or 0
            avg_ratio = b["avg_ratio"]["value"] or 0
            max_count = int(b["max_count"]["value"] or 0)
            quarters  = b["quarters_seen"]["value"]

            if max_count < min_count:
                continue

            # 999.0 sentinel: reaction appears in this drug but in ZERO comparators.
            # Show it with a NO_CLASS_BASELINE tag — it may be genuinely novel but
            # the class comparison is undefined (not a ratio signal).
            no_class_baseline = max_ratio >= 999.0

            if not no_class_baseline and max_ratio < min_ratio:
                continue

            avg_recent = b["recent"]["avg_recent"]["value"] or 0
            avg_early  = b["early"]["avg_early"]["value"] or 0
            if avg_early > 0 and avg_recent > avg_early * 1.5:
                trend = "GROWING"
            elif avg_recent > 0 and avg_early == 0:
                trend = "EMERGING"
            else:
                trend = "STABLE"

            signals.append({
                "reaction":          rxn,
                "max_ratio":         None if no_class_baseline else round(max_ratio, 2),
                "avg_ratio":         None if no_class_baseline else round(avg_ratio, 2),
                "max_count":         max_count,
                "quarters":          quarters,
                "trend":             trend,
                "persistent":        quarters >= 3,
                "no_class_baseline": no_class_baseline,
            })

        signals.sort(key=lambda x: -x["max_ratio"])
        return {
            "drug":         drug,
            "signal_count": len(signals),
            "signals":      signals[:top_n],
        }

    finally:
        await client.close()
