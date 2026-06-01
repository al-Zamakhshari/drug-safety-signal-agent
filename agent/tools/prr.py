"""
PRR (Proportional Reporting Ratio) calculation via OpenSearch aggregations.

Formula: PRR = (a / (a+b)) / (c / (c+d))
         2×2 contingency table, non-exposed comparator denominator.

         a = drug reports with reaction        b = drug reports without reaction
         c = non-drug reports with reaction    d = non-drug reports without reaction

Two estimators are computed for every signal:

  PRR = (a/(a+b)) / (c/(c+d))   — Proportional Reporting Ratio (EMA standard)
  ROR = (a·d) / (b·c)           — Reporting Odds Ratio (WHO/Uppsala standard)

Both use the same 2×2 table. ROR is asymptotically equivalent to PRR when the
reaction is rare, but diverges at high prevalence — reporting both lets comparison
against external tools (e.g. OpenVigil 2 reports both PRR and ROR).

Signal criteria (EMA/813938/2011):
  - PRR ≥ 2.0  AND  ROR ≥ 2.0
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
  Rothman et al. (2004) The reporting odds ratio and its advantages over the
  proportional reporting ratio — Pharmacoepidemiology and Drug Safety 13:519-523
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

# ── Stratification helpers ──────────────────────────────────────────────────

# Age bands for stratified PRR (FAERS standard, EMA guidance)
# patient_age is stored in years (after age_cod normalization in faers_zip_indexer.py)
AGE_BANDS = [
    {"key": "<18",   "to":   18.0},
    {"key": "18-44", "from": 18.0, "to":  45.0},
    {"key": "45-64", "from": 45.0, "to":  65.0},
    {"key": "65-74", "from": 65.0, "to":  75.0},
    {"key": "75+",   "from": 75.0},
]

# FAERS patient_sex code → readable label
# ZIP path: raw FAERS codes ("0","1","2"); API path: same numeric strings
SEX_LABELS = {"0": "Unknown", "1": "Male", "2": "Female"}

# FAERS reporter_type codes → readable label
# ZIP path: rept_cod / i_f_cod codes; API path: primarysource.qualification codes
REPORTER_LABELS = {
    # FAERS (2012+) rept_cod
    "EXP": "Expedited",    "DIR": "Direct",
    "PER": "Periodic",     "15DAY": "15-day",
    # openFDA qualification codes (API path)
    "1": "Physician",      "2": "Pharmacist",
    "3": "Other HCP",      "4": "Lawyer",
    "5": "Consumer/Patient",
    # AERS (pre-2012) i_f_cod
    "I": "Initial",        "F": "Follow-up",
}


def _stratification_agg(stratify_by: str) -> dict:
    """
    Build the outer OpenSearch aggregation for the chosen stratification field.
    Returns a dict suitable for use as the 'strata' agg value.
    """
    if stratify_by == "age":
        return {
            "range": {
                "field": "patient_age",
                "ranges": AGE_BANDS,
            }
        }
    elif stratify_by in ("sex", "reporter_type"):
        return {
            "terms": {
                "field": stratify_by,
                "size": 10,
                "missing": "Unknown",
            }
        }
    else:
        raise ValueError(f"stratify_by must be 'age', 'sex', or 'reporter_type', got {stratify_by!r}")


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


def _ror_ci(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    """
    Reporting Odds Ratio and its 95% CI using log-normal approximation.

        ROR = (a·d) / (b·c)
        SE  = sqrt(1/a + 1/b + 1/c + 1/d)
        CI  = exp(ln(ROR) ± 1.96·SE)

    ROR is the WHO/Uppsala Monitoring Centre standard alongside PRR.
    It equals the odds that a reaction is reported for the drug vs not,
    relative to the same odds for all other drugs.

    Uses Haldane–Anscombe +0.5 for zero cells.
    Returns (ror, lower, upper).
    """
    a_adj = max(a, 0.5)
    b_adj = max(b, 0.5)
    c_adj = max(c, 0.5)
    d_adj = max(d, 0.5)

    ror = (a_adj * d_adj) / (b_adj * c_adj)
    try:
        se    = math.sqrt(1/a_adj + 1/b_adj + 1/c_adj + 1/d_adj)
        lower = math.exp(math.log(ror) - 1.96 * se)
        upper = math.exp(math.log(ror) + 1.96 * se)
    except (ValueError, ZeroDivisionError):
        lower, upper = 0.0, float("inf")

    return round(ror, 2), round(lower, 2), round(upper, 2)


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
    stratify_by: str | None = None,
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
      7. Annotate with EBGM/EB05 (GPS, DuMouchel 1999).
      8. Annotate with IC/IC025 (BCPNN, Bate 1998 / Norén 2006).
      9. If stratify_by is set, compute Mantel-Haenszel stratified PRR using
         the chosen stratification variable (age | sex | reporter_type).

    Args:
        drug_names:   List of drug name variants in ALL CAPS
        top_n:        Number of top reactions to check for the drug (default 50)
        min_count:    Minimum drug reports per reaction (EMA: n ≥ 3)
        min_prr:      Minimum PRR threshold (EMA: ≥ 2.0)
        stratify_by:  Optional — one of 'age', 'sex', 'reporter_type'.
                      Adds prr_mh, prr_mh_lower, prr_mh_upper to each signal.
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

            chi2             = _yates_chi2(a, b, c, d)
            prr_lower, prr_upper = _prr_ci(a, b, c, d)
            ror, ror_lower, ror_upper = _ror_ci(a, b, c, d)

            tested.append({
                "reaction":    reaction,
                "prr":         round(prr, 2),
                "prr_lower":   prr_lower,
                "prr_upper":   prr_upper,
                "ror":         ror,
                "ror_lower":   ror_lower,
                "ror_upper":   ror_upper,
                "drug_count":  drug_count,
                "baseline":    baseline,
                "chi2":        round(chi2, 2),
                "significant": chi2 >= 4.0,         # EMA per-test gate
                "robust":      prr_lower >= 1.0,    # lower CI > null (small-n penalised)
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

        # Step 7: annotate with EBGM / EB05 (Gamma-Poisson Shrinker, DuMouchel 1999)
        # FDA MGPS standard — shrinks PRR=15 on n=3 down to EB05≈1.1.
        try:
            from agent.tools.ebgm import annotate_signals_with_ebgm
            signals = annotate_signals_with_ebgm(signals, drug_total, faers_total)
        except Exception:
            pass   # EBGM is supplemental — never block the main PRR result

        # Step 8: annotate with BCPNN IC (WHO Uppsala standard, Bate 1998 / Norén 2006)
        # Complements EBGM via a Beta-Binomial prior.  IC025 > 0 is the WHO signal flag.
        try:
            from agent.tools.bcpnn import annotate_signals_with_bcpnn
            signals = annotate_signals_with_bcpnn(signals, drug_total, faers_total)
        except Exception:
            pass   # BCPNN is supplemental — never block the main PRR result

        # Step 9: Mantel–Haenszel stratified PRR (optional, gated on stratify_by)
        # Adds prr_mh, prr_mh_lower, prr_mh_upper, prr_strat_field to each signal.
        # Uses the same MH estimator as anomaly_signals.py (quarterly strata → here
        # age bands / sex / reporter type). Detects Simpson's paradox confounding.
        if stratify_by and signals:
            try:
                signals = await _calculate_stratified_prr(
                    client_ref=client,
                    drug_names=drug_names,
                    reaction_keys=[s["reaction"] for s in signals],
                    signals=signals,
                    stratify_by=stratify_by,
                )
            except Exception as exc:
                # Stratified PRR is supplemental — log but don't block
                for s in signals:
                    s["prr_mh"]       = None
                    s["prr_mh_lower"] = None
                    s["prr_mh_upper"] = None
                    s["prr_strat_field"] = stratify_by

        return {
            "drug_names":    drug_names,
            "drug_total":    drug_total,
            "faers_total":   faers_total,
            "signals":       signals,
            "signal_count":  len(signals),
            "tested_count":  m,   # total reactions tested (BH denominator)
            "stratify_by":   stratify_by,
        }

    finally:
        await client.close()


async def _calculate_stratified_prr(
    client_ref,
    drug_names: list[str],
    reaction_keys: list[str],
    signals: list[dict],
    stratify_by: str,
) -> list[dict]:
    """
    Compute Mantel–Haenszel stratified PRR for each signal reaction.

    For each stratum k (age band / sex / reporter type) and each reaction R:
      a_k  = drug reports with R in stratum k
      n1_k = drug total in stratum k
      c_k  = (all reports with R in stratum k) − a_k
      n2_k = (all reports in stratum k) − n1_k

    Pass all strata to _mh_rate_ratio (imported from anomaly_signals) to
    get the Mantel–Haenszel estimate and Robins–Breslow–Greenland CI.

    Adds to each signal: prr_mh, prr_mh_lower, prr_mh_upper, prr_strat_field,
    and prr_strata_detail (list of per-stratum crude PRRs for inspection).
    """
    from agent.tools.anomaly_signals import _mh_rate_ratio

    strat_agg = _stratification_agg(stratify_by)

    # Query 1: drug-subset counts per stratum per reaction
    drug_resp = await client_ref.search(
        index=INDEX,
        body={
            "size": 0,
            "query": {"terms": {"drug_names": drug_names}},
            "aggs": {
                "strata": {
                    **strat_agg,
                    "aggs": {
                        "stratum_total": {"value_count": {"field": "safetyreportid"}},
                        "per_reaction": {
                            "filters": {
                                "filters": {
                                    rxn: {"term": {"reactions": rxn}}
                                    for rxn in reaction_keys
                                }
                            }
                        },
                    },
                }
            },
        },
    )

    # Query 2: population counts per stratum per reaction
    pop_resp = await client_ref.search(
        index=INDEX,
        body={
            "size": 0,
            "aggs": {
                "strata": {
                    **strat_agg,
                    "aggs": {
                        "stratum_total": {"value_count": {"field": "safetyreportid"}},
                        "per_reaction": {
                            "filters": {
                                "filters": {
                                    rxn: {"term": {"reactions": rxn}}
                                    for rxn in reaction_keys
                                }
                            }
                        },
                    },
                }
            },
        },
    )

    # Parse stratum buckets from both queries
    def _parse_strata(resp):
        buckets = resp["aggregations"]["strata"].get("buckets", {})
        if isinstance(buckets, dict):           # range/filters agg → dict
            return {k: v for k, v in buckets.items()}
        return {b.get("key_as_string", str(b.get("key", ""))): b for b in buckets}

    drug_strata = _parse_strata(drug_resp)
    pop_strata  = _parse_strata(pop_resp)

    # Code → label maps for readable stratum names
    label_map = SEX_LABELS if stratify_by == "sex" else (
        REPORTER_LABELS if stratify_by == "reporter_type" else {}
    )

    # Build per-reaction MH inputs
    signal_map = {s["reaction"]: s for s in signals}

    for rxn in reaction_keys:
        s = signal_map.get(rxn)
        if s is None:
            continue

        mh_strata      = []
        strata_detail  = []

        for stratum_key, drug_bucket in drug_strata.items():
            pop_bucket = pop_strata.get(stratum_key)
            if pop_bucket is None:
                continue

            n1_k = drug_bucket.get("stratum_total", {}).get("value", 0) or 0
            N_k  = pop_bucket.get("stratum_total",  {}).get("value", 0) or 0

            drug_rxn_count = (
                drug_bucket.get("per_reaction", {})
                           .get("buckets", {})
                           .get(rxn, {})
                           .get("doc_count", 0)
            ) or 0
            pop_rxn_count  = (
                pop_bucket.get("per_reaction", {})
                          .get("buckets", {})
                          .get(rxn, {})
                          .get("doc_count", 0)
            ) or 0

            if n1_k <= 0 or N_k <= 0:
                continue

            a_k  = drug_rxn_count
            n2_k = N_k - n1_k
            c_k  = pop_rxn_count - a_k  # non-drug reports with reaction

            if n2_k <= 0:
                continue

            mh_strata.append({
                "drug_count": a_k,
                "drug_total": n1_k,
                "comp_count": max(c_k, 0),
                "comp_total": n2_k,
            })

            # Per-stratum crude PRR for transparency
            crude_prr = None
            if n1_k > 0 and n2_k > 0 and c_k > 0 and a_k > 0:
                crude_prr = round((a_k / n1_k) / (c_k / n2_k), 2)
            label = label_map.get(str(stratum_key), stratum_key)
            strata_detail.append({"stratum": label, "n": a_k, "prr": crude_prr})

        # MH estimate from all strata
        rr_mh, mh_lo, mh_hi = _mh_rate_ratio(mh_strata) if mh_strata else (None, None, None)

        s["prr_mh"]          = round(rr_mh, 2) if rr_mh else None
        s["prr_mh_lower"]    = round(mh_lo, 2) if mh_lo else None
        s["prr_mh_upper"]    = round(mh_hi, 2) if mh_hi else None
        s["prr_strat_field"] = stratify_by
        s["prr_strata"]      = strata_detail

    return signals


async def get_drug_names(drug_name: str) -> dict[str, Any]:
    """
    Resolve a drug name to all FAERS variants using RxNorm BN tty.
    Falls back to config/brand_aliases.yaml for withdrawn drugs that
    have no BN entries in RxNorm.
    """
    import httpx
    import yaml
    from pathlib import Path

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

    # Brand fallbacks from config/brand_aliases.yaml (withdrawn/niche drugs)
    try:
        _aliases_file = Path(__file__).parent.parent.parent / "config" / "brand_aliases.yaml"
        if _aliases_file.exists():
            _aliases = yaml.safe_load(_aliases_file.read_text()) or {}
            fallback = _aliases.get(drug_name.upper(), {}).get("brands", [])
            all_names.update(fallback)
    except Exception:
        pass

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
