"""
Query class_ratio anomaly signals for drug safety from faers_ml_rates index.

The class_ratio = drug_rate / mean(comparator_class_rate) directly measures
how much a drug over-reports a reaction compared to its therapeutic class.

class_ratio > 3  → strong drug-specific signal
class_ratio > 1  → mild elevation above class baseline
class_ratio < 1  → reaction is LESS common than class average

No AD detector training needed — class_ratio is computed and indexed by
compute_class_ratio.py and is immediately queryable.
"""

import os
from typing import Any
from opensearchpy import AsyncOpenSearch
from dotenv import load_dotenv

load_dotenv()

RATES_INDEX = "faers_ml_rates"


def _client() -> AsyncOpenSearch:
    return AsyncOpenSearch(
        hosts=[os.getenv("OPENSEARCH_URL", "https://localhost:9200")],
        http_auth=(
            os.getenv("OPENSEARCH_USER", "admin"),
            os.getenv("OPENSEARCH_PASSWORD", "Pharma@2024!"),
        ),
        use_ssl=True, verify_certs=False, ssl_show_warn=False,
    )


async def get_anomaly_signals(
    drug: str,
    min_ratio: float = 2.0,
    min_count: int = 5,
    top_n: int = 15,
) -> dict[str, Any]:
    """
    Get class_ratio anomaly signals for a drug from faers_ml_rates.

    A signal is flagged when:
      max_class_ratio >= min_ratio  (drug rate elevated vs class)
      AND max_count >= min_count    (enough reports to be reliable)

    Also computes:
      trend: GROWING if recent quarters show accelerating class_ratio
      persistent: True if signal appears in 3+ quarters

    Args:
        drug:       Drug name in ALL CAPS (as stored in faers_ml_rates)
        min_ratio:  Minimum class_ratio to consider a signal (default 2.0)
        min_count:  Minimum drug_count per reaction
        top_n:      Max signals to return

    Returns:
        dict with signals list ranked by max_class_ratio
    """
    client = _client()
    try:
        # Check index has data for this drug
        count_r = await client.count(
            index=RATES_INDEX,
            body={"query": {"term": {"drug": drug}}}
        )
        if count_r["count"] == 0:
            return {
                "drug": drug,
                "signals": [],
                "note": f"No class_ratio data for {drug}. "
                        f"Run: uv run python -m ingestion.compute_class_ratio",
            }

        # Get per-reaction stats: max_ratio, avg_ratio, count, quarters active
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
                            # Recent quarters (last 4): trend detection
                            "recent": {
                                "filter": {
                                    "range": {
                                        "quarter": {"gte": "2023-01-01"}
                                    }
                                },
                                "aggs": {
                                    "avg_recent": {"avg": {"field": "class_ratio"}}
                                }
                            },
                            "early": {
                                "filter": {
                                    "range": {
                                        "quarter": {
                                            "gte": "2020-01-01",
                                            "lt":  "2022-01-01"
                                        }
                                    }
                                },
                                "aggs": {
                                    "avg_early": {"avg": {"field": "class_ratio"}}
                                }
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

            if max_ratio < min_ratio or max_count < min_count:
                continue

            # Trend: compare recent (2023+) vs early (2020-2021) avg_ratio
            avg_recent = b["recent"]["avg_recent"]["value"] or 0
            avg_early  = b["early"]["avg_early"]["value"] or 0
            if avg_early > 0 and avg_recent > avg_early * 1.5:
                trend = "GROWING"
            elif avg_recent > 0 and avg_early == 0:
                trend = "EMERGING"
            else:
                trend = "STABLE"

            signals.append({
                "reaction":    rxn,
                "max_ratio":   round(max_ratio, 2),
                "avg_ratio":   round(avg_ratio, 2),
                "max_count":   max_count,
                "quarters":    quarters,
                "trend":       trend,
                "persistent":  quarters >= 3,
            })

        # Sort by max_ratio descending
        signals.sort(key=lambda x: -x["max_ratio"])
        signals = signals[:top_n]

        return {
            "drug":         drug,
            "signal_count": len(signals),
            "signals":      signals,
        }

    finally:
        await client.close()
