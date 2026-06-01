"""
Within-class disproportionality screening from faers_ml_rates index.

Method: Mantel–Haenszel stratified rate ratio (Robins, Breslow, Greenland 1986)

  Each quarter k is treated as a stratum. Within stratum k:
    a_k = drug_count_k        n1_k = drug_total_k
    c_k = comp_count_k        n2_k = comp_total_k
    N_k = n1_k + n2_k

  MH rate ratio:
    RR_MH = Σ_k (a_k × n2_k / N_k)  /  Σ_k (c_k × n1_k / N_k)

  Robins–Breslow–Greenland variance of ln(RR_MH):
    Let R_k = a_k × n2_k / N_k,  S_k = c_k × n1_k / N_k
        P_k = (a_k + c_k) / N_k,  Q_k = 1 − P_k
    Var = Σ P_k R_k / (2R²) + Σ (P_k S_k + Q_k R_k) / (2RS) + Σ Q_k S_k / (2S²)

  95% CI = exp(ln(RR_MH) ± 1.96 × sqrt(Var))

Why MH instead of naive pooling:
  The previous version summed all counts across quarters and computed a single
  log-normal CI. This treats 33 quarters × 5 comparator groups as one homogeneous
  pool, ignoring between-quarter variation in drug reporting rates and comparator
  mix. MH stratification controls for temporal confounding — each quarter's
  contribution is weighted by its information content.

Trend detection: quarterly MH sub-estimates for GROWING/EMERGING/STABLE
  (advisory only — not a gate).

Zero cells: Haldane–Anscombe +0.5 continuity applied per-stratum when
  comp_count_k = 0 or drug_count_k = 0.
"""

import math
import os
from typing import Any

from agent.os_client import client as _client
from dotenv import load_dotenv

load_dotenv()

RATES_INDEX = "faers_ml_rates"


def _mh_rate_ratio(
    strata: list[dict],
) -> tuple[float, float, float]:
    """
    Mantel–Haenszel stratified rate ratio and 95% CI.

    Each stratum dict must have keys:
      drug_count, drug_total, comp_count, comp_total

    Returns (rr_mh, lower_95, upper_95).
    Returns (0.0, 0.0, inf) if no valid strata.

    Haldane–Anscombe +0.5 applied to zero cells per stratum.
    """
    R = 0.0  # Σ a_k × n2_k / N_k
    S = 0.0  # Σ c_k × n1_k / N_k
    # RBG variance components
    sum_P_R = 0.0  # Σ P_k × R_k
    sum_PSkQRk = 0.0  # Σ (P_k × S_k + Q_k × R_k)
    sum_Q_S = 0.0  # Σ Q_k × S_k

    valid = 0
    for s in strata:
        a_raw  = s.get("drug_count", 0) or 0
        n1     = s.get("drug_total", 0) or 0
        c_raw  = s.get("comp_count", 0) or 0
        n2     = s.get("comp_total", 0) or 0

        if n1 <= 0 or n2 <= 0:
            continue

        # Haldane–Anscombe continuity for zero cells
        a = a_raw if a_raw > 0 else 0.5
        c = c_raw if c_raw > 0 else 0.5

        N = n1 + n2
        R_k = a * n2 / N
        S_k = c * n1 / N
        P_k = (a + c) / N
        Q_k = 1.0 - P_k

        R += R_k
        S += S_k
        sum_P_R    += P_k * R_k
        sum_PSkQRk += P_k * S_k + Q_k * R_k
        sum_Q_S    += Q_k * S_k
        valid += 1

    if valid == 0 or R <= 0 or S <= 0:
        return 0.0, 0.0, float("inf")

    rr_mh = R / S

    # Robins–Breslow–Greenland variance of ln(RR_MH)
    try:
        var_ln = (sum_P_R / (2 * R * R)
                  + sum_PSkQRk / (2 * R * S)
                  + sum_Q_S / (2 * S * S))
        se = math.sqrt(max(var_ln, 0.0))
        lower = math.exp(math.log(rr_mh) - 1.96 * se)
        upper = math.exp(math.log(rr_mh) + 1.96 * se)
    except (ValueError, ZeroDivisionError):
        lower, upper = 0.0, float("inf")

    return round(rr_mh, 3), round(lower, 3), round(upper, 3)


def _rate_ratio_ci(a: float, n1: int, c: float, n2: int) -> tuple[float, float, float]:
    """
    Single-stratum rate-ratio CI (log-normal approximation).
    Used for individual-quarter sub-estimates in trend detection,
    and by test_prr_ci.py for formula verification.

    Returns (rr, lower, upper).
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
    min_ratio_lower: float = 1.0,
    min_count: int = 5,
    top_n: int = 15,
) -> dict[str, Any]:
    """
    Get within-class disproportionality signals using Mantel–Haenszel stratification.

    Each quarter is treated as a stratum. The MH rate ratio and its 95% CI
    are computed from per-quarter counts, controlling for temporal variation.

    Args:
        drug:            Drug name in ALL CAPS (as stored in faers_ml_rates)
        min_ratio_lower: Minimum MH lower CI bound to pass gate (default 1.0)
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
                "note": (f"No class_ratio data for {drug}. "
                         f"Run: uv run python -m ingestion.compute_class_ratio"),
            }

        # Per-quarter counts for each reaction — strata for MH estimator.
        # Also aggregate total drug_count for the min_count gate.
        resp = await client.search(
            index=RATES_INDEX,
            body={
                "size": 0,
                "query": {"term": {"drug": drug}},
                "aggs": {
                    "by_reaction": {
                        "terms": {"field": "reaction", "size": 300},
                        "aggs": {
                            "total_drug_count": {"sum": {"field": "drug_count"}},
                            "quarters_seen":    {"value_count": {"field": "quarter"}},
                            # Per-quarter strata for MH
                            "by_quarter": {
                                "terms": {
                                    "field": "quarter",
                                    "size":  50,
                                    "order": {"_key": "asc"},
                                },
                                "aggs": {
                                    "drug_count":  {"sum": {"field": "drug_count"}},
                                    "drug_total":  {"sum": {"field": "drug_total"}},
                                    "comp_count":  {"sum": {"field": "comp_count"}},
                                    "comp_total":  {"sum": {"field": "comp_total"}},
                                }
                            },
                        }
                    }
                }
            }
        )

        # Detect old schema (comp_count/comp_total absent)
        has_new_schema = True

        signals = []
        for b in resp["aggregations"]["by_reaction"]["buckets"]:
            rxn             = b["key"]
            total_drug_cnt  = int(b["total_drug_count"]["value"] or 0)
            quarters        = b["quarters_seen"]["value"]

            if total_drug_cnt < min_count:
                continue

            # Build per-quarter strata list
            strata = []
            for qb in b["by_quarter"]["buckets"]:
                dc = qb["drug_count"]["value"]
                dt = qb["drug_total"]["value"]
                cc = qb["comp_count"]["value"]
                ct = qb["comp_total"]["value"]
                if ct is None:
                    has_new_schema = False
                    continue
                strata.append({
                    "quarter":    qb["key_as_string"],
                    "drug_count": int(dc or 0),
                    "drug_total": int(dt or 0),
                    "comp_count": int(cc or 0),
                    "comp_total": int(ct or 0),
                })

            if not strata:
                continue

            # MH rate ratio across all quarters
            rr_mh, mh_lower, mh_upper = _mh_rate_ratio(strata)

            if mh_lower <= min_ratio_lower:
                continue

            # Trend: compare last-third vs first-third of the drug's observed quarter range.
            # Data-relative windows so rofecoxib (2001-2004) and semaglutide (2018-2026)
            # both get meaningful GROWING/EMERGING/STABLE labels.
            # Fixed 2020/2023 cutoffs silently returned STABLE for any pre-2020 drug.
            quarters_sorted = sorted(s["quarter"] for s in strata)
            n_q = len(quarters_sorted)
            if n_q >= 3:
                early_cutoff  = quarters_sorted[n_q // 3]      # end of first third
                recent_cutoff = quarters_sorted[2 * n_q // 3]  # start of last third
                early_strata  = [s for s in strata if s["quarter"] <  early_cutoff]
                recent_strata = [s for s in strata if s["quarter"] >= recent_cutoff]
            elif n_q == 2:
                early_strata  = [strata[0]]
                recent_strata = [strata[1]]
            else:
                early_strata  = []
                recent_strata = strata

            r_rr, r_lo, _ = _mh_rate_ratio(recent_strata) if recent_strata else (0, 0, 0)
            e_rr, e_lo, _ = _mh_rate_ratio(early_strata)  if early_strata  else (0, 0, 0)

            if e_lo <= 1.0 and r_lo > 1.0:
                trend = "EMERGING"
            elif r_lo > 1.0 and r_rr > e_rr * 1.5 and e_rr > 0:
                trend = "GROWING"
            else:
                trend = "STABLE"

            total_comp_count = sum(s["comp_count"] for s in strata)

            signals.append({
                "reaction":          rxn,
                "class_ratio":       rr_mh,
                "class_ratio_lower": mh_lower,
                "class_ratio_upper": mh_upper,
                "class_ratio_robust": mh_lower > 1.0,
                "mh_strata":         len(strata),   # number of quarterly strata used
                "drug_count":        total_drug_cnt,
                "comp_count":        total_comp_count,
                "quarters":          quarters,
                "trend":             trend,
                "persistent":        quarters >= 3,
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
            "method":       "Mantel-Haenszel stratified rate ratio (quarterly strata)",
        }

    finally:
        await client.close()
