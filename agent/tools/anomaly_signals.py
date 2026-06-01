"""
Within-class disproportionality screening from faers_ml_rates index.

Method:
  class_ratio = (drug_count / drug_total) / (comp_count / comp_total)
  where comp_* are pooled across all active comparator groups for a quarter.

Counts are aggregated (summed) across all available quarters to produce a
single pooled rate ratio and its 95% CI per drug×reaction pair.

Signal gate: class_ratio_lower (lower 95% CI bound) > 1.0 AND drug_count ≥ min_count.
This penalises small-n estimates correctly — a ratio of 8 on n=4 will have a
lower bound near 0.8 and NOT pass, while the same ratio on n=500 will be robust.

The 999.0 sentinel from the old schema (reaction absent from all comparators) has
been replaced by Haldane–Anscombe +0.5 continuity correction in compute_class_ratio.py.
Reactions with comp_count=0 now appear with a finite but large ratio and a wide CI;
they may or may not pass the class_ratio_lower > 1.0 gate depending on sample size.

Trend detection: GROWING / EMERGING / STABLE based on whether the rolling
recent-quarter CI lower bound has been persistently > 1.0 or only recently emerged.
This is advisory and informational — not a statistical gate.
"""

import math
import os
from typing import Any

from agent.os_client import client as _client
from dotenv import load_dotenv

load_dotenv()

RATES_INDEX = "faers_ml_rates"


def _rate_ratio_ci(a: float, n1: int, c: float, n2: int) -> tuple[float, float, float]:
    """
    95% CI on the rate ratio (a/n1)/(c/n2) using log-normal approximation.

    Applies Haldane–Anscombe +0.5 continuity to zero cells to avoid log(0).

    Returns (class_ratio, lower, upper). Wide CI from +0.5 correction correctly
    signals unreliable estimates from sparse comparator data.

    class_ratio_robust = True when lower >= 1.0.
    """
    a_adj = max(a, 0.5)
    c_adj = max(c, 0.5)
    if n1 <= 0 or n2 <= 0:
        return 0.0, 0.0, float("inf")

    rr = (a_adj / n1) / (c_adj / n2)
    try:
        se    = math.sqrt(1/a_adj - 1/n1 + 1/c_adj - 1/n2)
        lower = math.exp(math.log(rr) - 1.96 * se)
        upper = math.exp(math.log(rr) + 1.96 * se)
    except (ValueError, ZeroDivisionError):
        lower, upper = 0.0, float("inf")

    return round(rr, 3), round(lower, 3), round(upper, 3)


async def get_anomaly_signals(
    drug: str,
    min_ratio_lower: float = 1.0,  # class_ratio_lower > this to pass gate
    min_count: int = 5,
    top_n: int = 15,
) -> dict[str, Any]:
    """
    Get within-class disproportionality signals for a drug from faers_ml_rates.

    Counts are pooled (summed) across all quarters and the 95% CI is computed
    on the pooled ratio. This is more stable than per-quarter max/avg.

    Args:
        drug:            Drug name in ALL CAPS (as stored in faers_ml_rates)
        min_ratio_lower: Minimum class_ratio LOWER CI bound to consider a signal (default 1.0)
        min_count:       Minimum total drug_count across all quarters
        top_n:           Max signals to return
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

        # Aggregate by reaction: sum raw counts across all quarters.
        # Pool drug_count, drug_total, comp_count, comp_total — then compute
        # a single pooled rate ratio and CI from the sums.
        # Also keep the recent/early sub-aggregations for trend detection.
        resp = await client.search(
            index=RATES_INDEX,
            body={
                "size": 0,
                "query": {"term": {"drug": drug}},
                "aggs": {
                    "by_reaction": {
                        "terms": {"field": "reaction", "size": 300},
                        "aggs": {
                            # Pooled counts across ALL quarters
                            "sum_drug_count":  {"sum": {"field": "drug_count"}},
                            "sum_drug_total":  {"sum": {"field": "drug_total"}},
                            "sum_comp_count":  {"sum": {"field": "comp_count"}},
                            "sum_comp_total":  {"sum": {"field": "comp_total"}},
                            "quarters_seen":   {"value_count": {"field": "quarter"}},
                            # Recent sub-bucket for trend detection (advisory only)
                            "recent": {
                                "filter": {"range": {"quarter": {"gte": "2023-01-01"}}},
                                "aggs": {
                                    "sum_drug_count_r": {"sum": {"field": "drug_count"}},
                                    "sum_drug_total_r": {"sum": {"field": "drug_total"}},
                                    "sum_comp_count_r": {"sum": {"field": "comp_count"}},
                                    "sum_comp_total_r": {"sum": {"field": "comp_total"}},
                                }
                            },
                            "early": {
                                "filter": {"range": {"quarter": {
                                    "gte": "2020-01-01", "lt": "2022-01-01"
                                }}},
                                "aggs": {
                                    "sum_drug_count_e": {"sum": {"field": "drug_count"}},
                                    "sum_drug_total_e": {"sum": {"field": "drug_total"}},
                                    "sum_comp_count_e": {"sum": {"field": "comp_count"}},
                                    "sum_comp_total_e": {"sum": {"field": "comp_total"}},
                                }
                            },
                        }
                    }
                }
            }
        )

        # Check whether the new comp_count/comp_total fields exist in the index.
        # If not (old schema), surface a graceful note rather than silently failing.
        has_new_schema = True

        signals = []
        for b in resp["aggregations"]["by_reaction"]["buckets"]:
            rxn            = b["key"]
            drug_count_sum = int(b["sum_drug_count"]["value"] or 0)
            drug_total_sum = int(b["sum_drug_total"]["value"] or 0)
            comp_count_sum = b["sum_comp_count"]["value"]   # None if field absent
            comp_total_sum = b["sum_comp_total"]["value"]
            quarters       = b["quarters_seen"]["value"]

            if drug_count_sum < min_count:
                continue

            # Detect old schema: comp_count/comp_total fields absent
            if comp_count_sum is None or comp_total_sum is None or int(comp_total_sum or 0) == 0:
                has_new_schema = False
                continue

            comp_count_sum = int(comp_count_sum)
            comp_total_sum = int(comp_total_sum)

            # Pooled rate-ratio CI across all quarters
            class_ratio, lower, upper = _rate_ratio_ci(
                drug_count_sum, drug_total_sum,
                comp_count_sum, comp_total_sum,
            )

            # Gate: lower CI bound must exceed 1.0 (excludes null association)
            if lower <= min_ratio_lower:
                continue

            # Trend detection using recent vs early sub-aggregations (advisory)
            r = b["recent"]
            e = b["early"]
            r_dc = int(r["sum_drug_count_r"]["value"] or 0)
            r_dt = int(r["sum_drug_total_r"]["value"] or 0)
            r_cc = int(r["sum_comp_count_r"]["value"] or 0)
            r_ct = int(r["sum_comp_total_r"]["value"] or 0)
            e_dc = int(e["sum_drug_count_e"]["value"] or 0)
            e_dt = int(e["sum_drug_total_e"]["value"] or 0)
            e_cc = int(e["sum_comp_count_e"]["value"] or 0)
            e_ct = int(e["sum_comp_total_e"]["value"] or 0)

            r_rr, r_lo, _ = _rate_ratio_ci(r_dc, r_dt, r_cc, r_ct) if r_ct > 0 else (0, 0, 0)
            e_rr, e_lo, _ = _rate_ratio_ci(e_dc, e_dt, e_cc, e_ct) if e_ct > 0 else (0, 0, 0)

            if e_lo <= 1.0 and r_lo > 1.0:
                trend = "EMERGING"
            elif r_lo > 1.0 and r_rr > e_rr * 1.5 and e_rr > 0:
                trend = "GROWING"
            else:
                trend = "STABLE"

            signals.append({
                "reaction":             rxn,
                "class_ratio":          class_ratio,
                "class_ratio_lower":    lower,
                "class_ratio_upper":    upper,
                "class_ratio_robust":   lower > 1.0,
                "drug_count":           drug_count_sum,
                "comp_count":           comp_count_sum,
                "quarters":             quarters,
                "trend":                trend,
                "persistent":           quarters >= 3,
            })

        if not has_new_schema and not signals:
            return {
                "drug":    drug,
                "signals": [],
                "note":    (
                    "faers_ml_rates index uses old schema (missing comp_count/comp_total). "
                    "Re-run: uv run python -m ingestion.compute_class_ratio"
                ),
            }

        signals.sort(key=lambda x: -x["class_ratio_lower"])
        return {
            "drug":         drug,
            "signal_count": len(signals),
            "signals":      signals[:top_n],
        }

    finally:
        await client.close()
