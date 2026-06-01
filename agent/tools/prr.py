"""
PRR (Proportional Reporting Ratio) calculation via OpenSearch aggregations.

Formula: PRR = (a / (a+b)) / (c / (c+d))
         2×2 contingency table, non-exposed comparator denominator.

         a = drug reports with reaction        b = drug reports without reaction
         c = non-drug reports with reaction    d = non-drug reports without reaction

Signal criteria (EMA/813938/2011):
  - PRR ≥ 2.0
  - n (drug reports with reaction) ≥ 3
  - Yates χ² ≥ 4.0  (per-test, annotated as `significant`)
  - PRR lower 95% CI > 1.0  (annotated as `robust` — small-n penalised)
  - BH-FDR q < 0.05 across all m reactions tested (annotated as `fdr_significant`)

Baseline: fetched per-reaction via `filters` aggregation — not truncated to
a global top-N. Rare/novel signals are never silently dropped from the baseline.
Note: the drug's own top-N reactions are still capped at `top_n` (default 50).

References:
  EMA/813938/2011 Guideline on statistical signal detection methods
  Evans et al. (2001) Use of proportional reporting ratios (PRRs) for signal
  generation from spontaneous adverse drug reaction reports — Pharmacoepidemiology
  and Drug Safety 10:483-486
"""

import math
import os
from typing import Any

import numpy as np
from scipy.stats import chi2 as _chi2_dist

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


def _prr_ci(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """
    95% confidence interval on PRR using the log-normal approximation (Evans 2001).

        SE      = sqrt(1/a − 1/(a+b) + 1/c − 1/(c+d))
        PRR_lo  = exp(ln(PRR) − 1.96·SE)
        PRR_hi  = exp(ln(PRR) + 1.96·SE)

    Returns (lower, upper). Uses Haldane–Anscombe +0.5 continuity when a or c
    is zero to avoid log(0). Wide CI from +0.5 correction correctly signals
    unreliable estimates.

    A signal is `robust` when PRR_lower ≥ 1.0 — i.e. the lower bound excludes
    null association even under the conservative small-n penalty.
    """
    a_adj = a if a > 0 else 0.5
    c_adj = c if c > 0 else 0.5
    n1    = a + b   # drug total
    n2    = c + d   # non-drug total

    prr = (a_adj / n1) / (c_adj / n2)
    try:
        se    = math.sqrt(1/a_adj - 1/n1 + 1/c_adj - 1/n2)
        lower = math.exp(math.log(prr) - 1.96 * se)
        upper = math.exp(math.log(prr) + 1.96 * se)
    except (ValueError, ZeroDivisionError):
        lower, upper = 0.0, float("inf")

    return round(lower, 2), round(upper, 2)


async def calculate_prr(
    drug_names: list[str],
    top_n: int = 50,
    min_count: int = 3,
    min_prr: float = 2.0,
) -> dict[str, Any]:
    """
    Calculate PRR signals for a drug against the full FAERS population.

    Steps:
      1. Fetch drug total + top-N reactions by frequency.
      2. Fetch FAERS total (for non-drug denominator).
      3. Fetch per-reaction baseline via `filters` agg — no rank truncation.
      4. Compute 2×2 table, PRR, Yates χ², 95% CI for ALL reactions with n≥3.
      5. Apply Benjamini–Hochberg FDR correction across all m reactions tested.
      6. Return signals with PRR ≥ min_prr, annotated with robust/fdr fields.

    Args:
        drug_names: List of drug name variants in ALL CAPS
        top_n:      Number of top reactions to check for the drug (default 50)
        min_count:  Minimum drug reports per reaction (EMA: n ≥ 3)
        min_prr:    Minimum PRR threshold (EMA: ≥ 2.0)
    """
    client = _client()
    try:
        # Step 1: drug total + top-N reactions by frequency
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

        # Step 4: compute 2×2 + PRR + χ² + CI for ALL reactions qualifying n≥3.
        # We collect ALL tested reactions (not just PRR≥2) so BH operates on the
        # full multiple-comparison burden m = all reactions actually tested.
        non_drug_total = faers_total - drug_total
        tested: list[dict] = []

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

            a, b = drug_count, drug_total - drug_count
            c, d = non_drug_count, non_drug_total - non_drug_count

            chi2        = _yates_chi2(a, b, c, d)
            lower, upper = _prr_ci(a, b, c, d)

            tested.append({
                "reaction":    reaction,
                "prr":         round(prr, 2),
                "prr_lower":   lower,
                "prr_upper":   upper,
                "drug_count":  drug_count,
                "baseline":    baseline,
                "chi2":        round(chi2, 2),
                "significant": chi2 >= 4.0,      # EMA per-test gate
                "robust":      lower >= 1.0,      # lower CI > null (small-n penalised)
                "_a": a, "_b": b, "_c": c, "_d": d,  # scratch — removed before return
            })

        # Step 5: Benjamini–Hochberg FDR across all m reactions tested.
        # m = len(tested) — every reaction that got a 2×2 computed, including those
        # with PRR < 2.0. Using the full tested set is the correct BH denominator.
        m = len(tested)
        if m > 0:
            p_vals = np.array([float(_chi2_dist.sf(t["chi2"], 1)) for t in tested])
            order  = np.argsort(p_vals)
            ranks  = np.empty(m, int)
            ranks[order] = np.arange(1, m + 1)
            # Raw BH q-values, then enforce monotonicity via reverse cumulative min
            q_raw = p_vals * m / ranks
            q_adj = np.empty(m)
            q_adj[order] = np.minimum.accumulate(q_raw[order][::-1])[::-1]
            for t, qi in zip(tested, q_adj.tolist()):
                t["q_value"]        = round(float(min(qi, 1.0)), 4)
                t["fdr_significant"] = float(min(qi, 1.0)) < 0.05

        # Step 6: return signals with PRR ≥ min_prr; strip scratch cells
        signals = []
        for t in tested:
            t.pop("_a", None); t.pop("_b", None)
            t.pop("_c", None); t.pop("_d", None)
            if t["prr"] >= min_prr:
                signals.append(t)

        signals.sort(key=lambda x: -x["prr"])
        return {
            "drug_names":   drug_names,
            "drug_total":   drug_total,
            "faers_total":  faers_total,
            "signals":      signals,
            "signal_count": len(signals),
            "tested_count": m,   # total reactions tested (BH denominator)
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
