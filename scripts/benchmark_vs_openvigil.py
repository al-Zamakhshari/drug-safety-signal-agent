"""
Benchmark drug-safety-signal-agent PRR/ROR against openFDA as independent reference.

Why openFDA and not OpenVigil 2
--------------------------------
OpenVigil 2 has no public API (web-only, intermittently unavailable, SourceForge source).
openFDA IS the authoritative FDA data source — it's what OpenVigil and everyone else
reads. Using openFDA directly bypasses the intermediary and gives a cleaner reference:

  - Independent: completely separate code path from our OpenSearch pipeline
  - Authoritative: openFDA is the FDA's own API (19M reports, updated quarterly)
  - Automated: no manual browser steps
  - Free: 40 req/min without key, 240/min with free key from open.fda.gov/apis/authentication/

How the 2×2 table is constructed from openFDA
----------------------------------------------
Two API calls give all four cells:

  Call 1 — drug-specific reactions:
    GET /drug/event.json?search=patient.drug.medicinalproduct:"SEMAGLUTIDE"
                        &count=patient.reaction.reactionmeddrapt.exact&limit=1000
    → results: [{term: "NAUSEA", count: a}, ...]   (a = drug+reaction)
    → meta.results.total: drug_total               (a+b = drug reports)

  Call 2 — population reactions:
    GET /drug/event.json?count=patient.reaction.reactionmeddrapt.exact&limit=1000
    → results: [{term: "NAUSEA", count: a+c}, ...]  (a+c = all reports with reaction)
    → meta.results.total: N                          (total FAERS reports)

  PRR = (a/(a+b)) / (c/(c+d))   where c = (a+c) - a,  d = N-(a+b)-c
  ROR = (a·d) / (b·c)

Note: openFDA normalises drug names via its own harmonisation layer. Minor count
differences vs our local OpenSearch data are expected and are documented below.

Usage
-----
Run the full automated benchmark (no manual steps):
    uv run python scripts/benchmark_vs_openvigil.py benchmark semaglutide

Or step by step:
    uv run python scripts/benchmark_vs_openvigil.py export semaglutide
    uv run python scripts/benchmark_vs_openvigil.py reference semaglutide
    uv run python scripts/benchmark_vs_openvigil.py compare \\
        scripts/benchmark_semaglutide_ours.csv \\
        scripts/benchmark_semaglutide_reference.csv

Expected differences between our output and openFDA reference
--------------------------------------------------------------
- Drug name matching: openFDA uses `medicinalproduct` string matching; we query
  drug_names[] after RxNorm synonym resolution. Slight count differences expected.
- Top-N cap: we cap at top_n=50 drug reactions; reference fetches up to 1000.
- De-duplication: neither applies de-duplication (both use raw FAERS).
- Data vintage: openFDA updates quarterly; our local extract may differ by one quarter.

A PRR delta < 10% on n ≥ 100 signals is strong formula agreement.
"""

import asyncio
import csv
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

OPENFDA_BASE  = "https://api.fda.gov/drug/event.json"
OPENFDA_KEY   = os.getenv("OPENFDA_API_KEY", "")   # optional; set for higher rate limit
SCRIPTS_DIR   = Path(__file__).parent


# ---------------------------------------------------------------------------
# openFDA helpers
# ---------------------------------------------------------------------------

async def _openfda_get(params: dict, retries: int = 3) -> dict:
    """
    GET openFDA with retry on 429 (rate-limit) and exponential backoff.

    Builds the URL manually for the `search` parameter to preserve literal `+OR+`
    separators — httpx.params would double-encode `+` as `%2B`, breaking OR queries.
    """
    import urllib.parse

    # Extract search separately so we can control its encoding
    search = params.pop("search", None)
    if OPENFDA_KEY:
        params["api_key"] = OPENFDA_KEY

    # Build base URL with non-search params (safely encoded by httpx)
    base = OPENFDA_BASE
    if params:
        base = base + "?" + urllib.parse.urlencode(params)

    # Append search with manual encoding: quote values but keep literal + for OR
    if search:
        # Encode each field:value pair individually, join with literal +OR+
        encoded_search = urllib.parse.quote(search, safe=":+")
        sep = "&" if "?" in base else "?"
        url = f"{base}{sep}search={encoded_search}"
    else:
        url = base

    for attempt in range(retries):
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            if resp.status_code == 429:
                wait = 2 ** attempt * 2
                print(f"  Rate limited — waiting {wait}s...")
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
    raise RuntimeError("openFDA rate limit exceeded after retries")


async def _fetch_drug_reactions(drug_names: list[str]) -> tuple[dict, int]:
    """
    Fetch per-reaction counts for a drug (all name variants via OR query).
    Returns ({reaction: count}, drug_total).

    limit=500 works without an API key. 1000 requires a key.
    """
    limit = 1000 if OPENFDA_KEY else 500
    # Build OR query across all name variants to match our RxNorm-resolved names.
    # Format: field:"VALUE1"+OR+field:"VALUE2"  (+ = space in Lucene; OR = boolean)
    or_search = "+OR+".join(
        f'patient.drug.medicinalproduct:"{n}"' for n in drug_names
    )
    data = await _openfda_get({
        "search": or_search,
        "count":  "patient.reaction.reactionmeddrapt.exact",
        "limit":  str(limit),
    })
    counts = {r["term"].upper(): r["count"] for r in data.get("results", [])}
    # drug_total: separate query without count= to get the true total reports
    await asyncio.sleep(0.5)
    total_resp = await _openfda_get({"search": or_search, "limit": "1"})
    drug_total = total_resp.get("meta", {}).get("results", {}).get("total", 0)
    return counts, drug_total


async def _fetch_population_reactions() -> tuple[dict, int]:
    """
    Fetch per-reaction counts across the entire FAERS population.
    Returns ({reaction: count}, N_total).
    """
    limit = 1000 if OPENFDA_KEY else 500
    data = await _openfda_get({
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": str(limit),
    })
    counts = {r["term"].upper(): r["count"] for r in data.get("results", [])}
    # N_total: total FAERS reports (unconstrained)
    total_resp = await _openfda_get({"limit": "1"})
    n_total = total_resp.get("meta", {}).get("results", {}).get("total", 0)
    return counts, n_total


def _compute_prr_ror(a: int, drug_total: int, baseline: int, n_total: int) -> dict:
    """Compute PRR, ROR, and their 95% CIs from 2×2 cells."""
    b = drug_total - a
    c = baseline - a       # non-drug reports with reaction
    d = n_total - drug_total - c   # non-drug reports without reaction

    if drug_total <= 0 or n_total <= drug_total or c <= 0:
        return None

    non_drug_total = n_total - drug_total
    prr = (a / drug_total) / (c / non_drug_total)

    # PRR 95% CI (Evans 2001)
    a_adj, c_adj = max(a, 0.5), max(c, 0.5)
    prr_adj = (a_adj / drug_total) / (c_adj / non_drug_total)
    try:
        se_prr = math.sqrt(1/a_adj - 1/drug_total + 1/c_adj - 1/non_drug_total)
        prr_lo = round(math.exp(math.log(prr_adj) - 1.96 * se_prr), 2)
        prr_hi = round(math.exp(math.log(prr_adj) + 1.96 * se_prr), 2)
    except (ValueError, ZeroDivisionError):
        prr_lo, prr_hi = 0.0, float("inf")

    # ROR 95% CI
    b_adj, d_adj = max(b, 0.5), max(d, 0.5)
    ror = (a_adj * d_adj) / (b_adj * c_adj)
    try:
        se_ror = math.sqrt(1/a_adj + 1/b_adj + 1/c_adj + 1/d_adj)
        ror_lo = round(math.exp(math.log(ror) - 1.96 * se_ror), 2)
        ror_hi = round(math.exp(math.log(ror) + 1.96 * se_ror), 2)
    except (ValueError, ZeroDivisionError):
        ror_lo, ror_hi = 0.0, float("inf")

    return {
        "prr": round(prr, 2), "prr_lower": prr_lo, "prr_upper": prr_hi,
        "ror": round(ror, 2), "ror_lower": ror_lo, "ror_upper": ror_hi,
        "a": a, "b": b, "c": c, "d": d,
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

async def _export(drug_name: str, top_n: int = 100) -> Path:
    """Compute PRR+ROR from our local OpenSearch and write CSV."""
    from agent.tools.prr import calculate_prr, get_drug_names
    from agent.os_client import client as _client

    print(f"[export] Resolving drug names for: {drug_name}")
    names_result = await get_drug_names(drug_name)
    drug_names = names_result["found_names"]
    print(f"  FAERS names: {drug_names}")

    print(f"[export] Calculating PRR/ROR (top_n={top_n})...")
    result = await calculate_prr(drug_names, top_n=top_n, min_count=3, min_prr=1.0)

    os_client = _client()
    try:
        resp = await os_client.search(
            index=os.getenv("OPENSEARCH_INDEX", "faers_reports"),
            body={"size": 0, "aggs": {"max_date": {"max": {"field": "receivedate"}}}},
        )
        vintage = resp["aggregations"]["max_date"].get("value_as_string", "unknown")
    except Exception:
        vintage = "unknown"
    finally:
        await os_client.close()

    out = SCRIPTS_DIR / f"benchmark_{drug_name.lower()}_ours.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "# drug-safety-signal-agent (local OpenSearch)",
            f"drug={drug_name.upper()}",
            f"faers_names={','.join(drug_names)}",
            f"drug_total={result['drug_total']:,}",
            f"faers_total={result['faers_total']:,}",
            f"faers_vintage={vintage}",
            f"exported={datetime.now().isoformat()[:19]}",
        ])
        w.writerow(["reaction", "n", "PRR", "PRR_lower_95", "PRR_upper_95",
                    "ROR", "ROR_lower_95", "ROR_upper_95",
                    "chi2", "significant", "robust", "q_value"])
        for s in result["signals"]:
            w.writerow([
                s["reaction"], s["drug_count"],
                s["prr"], s.get("prr_lower",""), s.get("prr_upper",""),
                s.get("ror",""), s.get("ror_lower",""), s.get("ror_upper",""),
                s.get("chi2",""),
                "Y" if s.get("significant") else "N",
                "Y" if s.get("robust") else "N",
                s.get("q_value",""),
            ])

    print(f"  ✅ {len(result['signals'])} signals → {out.name}")
    print(f"     drug_total={result['drug_total']:,}  vintage={vintage}")
    return out


async def _reference(drug_name: str, drug_names: list[str] | None = None) -> Path:
    """
    Compute PRR/ROR from openFDA public API — fully independent of our local data.
    Uses OR query across all brand-name variants (same set as our RxNorm resolution).
    """
    from agent.tools.prr import get_drug_names as _resolve

    if drug_names is None:
        result = await _resolve(drug_name)
        drug_names = result["found_names"]

    print(f"\n[reference] Fetching openFDA data for: {drug_names}")
    print(f"  Source: {OPENFDA_BASE}")
    limit_note = "1000 (API key set)" if OPENFDA_KEY else "500 (free — set OPENFDA_API_KEY for 1000)"
    print(f"  Reaction limit: {limit_note}")

    print(f"\n  Fetching drug-specific reactions (OR across all name variants)...")
    drug_counts, drug_total = await _fetch_drug_reactions(drug_names)
    print(f"  drug_total={drug_total:,}  distinct_reactions={len(drug_counts)}")

    print(f"  Fetching population baseline...")
    await asyncio.sleep(1.5)   # stay within 40 req/min free limit
    pop_counts, n_total = await _fetch_population_reactions()
    print(f"  N_total={n_total:,}  population_reactions={len(pop_counts)}")

    # Compute PRR+ROR for all reactions present in drug profile
    signals = []
    for rxn, a in drug_counts.items():
        if a < 3:
            continue
        baseline = pop_counts.get(rxn, 0)
        if baseline == 0:
            continue
        result = _compute_prr_ror(a, drug_total, baseline, n_total)
        if result is None:
            continue
        if result["prr"] < 1.0:
            continue
        signals.append({"reaction": rxn, "n": a, **result})

    signals.sort(key=lambda x: -x["prr"])
    print(f"  Computed PRR/ROR for {len(signals)} signals (PRR≥1)")

    out = SCRIPTS_DIR / f"benchmark_{drug_name.lower()}_reference.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "# openFDA reference (independent of local OpenSearch)",
            f"drug={drug_name.upper()}",
            f"drug_total_openfda={drug_total:,}",
            f"n_total_openfda={n_total:,}",
            f"fetched={datetime.now().isoformat()[:19]}",
        ])
        w.writerow(["reaction", "n", "PRR", "PRR_lower_95", "PRR_upper_95",
                    "ROR", "ROR_lower_95", "ROR_upper_95"])
        for s in signals:
            w.writerow([
                s["reaction"], s["n"],
                s["prr"], s["prr_lower"], s["prr_upper"],
                s["ror"], s["ror_lower"], s["ror_upper"],
            ])

    print(f"  ✅ {len(signals)} signals → {out.name}")
    return out


def _compare(our_csv: Path, ref_csv: Path, label: str = "reference") -> None:
    """Print a delta table comparing our output to a reference CSV."""

    def _load(path: Path) -> tuple[dict, dict]:
        """Returns (signals dict, metadata dict)."""
        meta = {}
        signals = {}
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = None
            for row in reader:
                if not row:
                    continue
                if row[0].startswith("#"):
                    for item in row:
                        if "=" in item:
                            k, v = item.split("=", 1)
                            meta[k.strip("# ")] = v.strip()
                    continue
                if header is None:
                    header = [h.upper() for h in row]
                    continue
                if len(row) < 3:
                    continue
                rxn = row[0].upper()
                try:
                    signals[rxn] = {
                        "n":   int(float(row[1])),
                        "prr": float(row[2]),
                        "ror": float(row[5]) if len(row) > 5 and row[5] else None,
                    }
                except (ValueError, IndexError):
                    pass
        return signals, meta

    ours, our_meta = _load(our_csv)
    ref,  ref_meta = _load(ref_csv)

    common    = sorted(set(ours) & set(ref))
    only_ours = sorted(set(ours) - set(ref))
    only_ref  = sorted(set(ref) - set(ours))

    print(f"\n{'='*72}")
    print(f"Benchmark: local OpenSearch  vs  {label}")
    print(f"{'='*72}")
    print(f"  Ours:       {our_meta.get('drug_total','?')} drug reports  "
          f"vintage={our_meta.get('faers_vintage','?')}")
    print(f"  {label[:12]:12}: {ref_meta.get('drug_total_openfda', ref_meta.get('drug_total','?'))} drug reports  "
          f"fetched={ref_meta.get('fetched', ref_meta.get('exported','?'))}")
    print(f"\n  Reactions in both: {len(common):>5}")
    print(f"  Only in ours:      {len(only_ours):>5}")
    print(f"  Only in {label[:8]:8}: {len(only_ref):>5}")
    print()

    rows = []
    for rxn in common:
        o, r = ours[rxn], ref[rxn]
        if o["n"] < 3:
            continue
        delta_prr = abs(o["prr"] - r["prr"]) / r["prr"] * 100 if r["prr"] else float("inf")
        delta_ror = (abs((o["ror"] or 0) - (r["ror"] or 0)) / r["ror"] * 100
                     if (o.get("ror") and r.get("ror")) else None)
        flag = "✓" if delta_prr < 5 else ("~" if delta_prr < 20 else "⚠️ ")
        rows.append(dict(
            reaction=rxn, n=o["n"],
            prr_ours=o["prr"], prr_ref=r["prr"], delta_prr=round(delta_prr, 1),
            ror_ours=o.get("ror"), ror_ref=r.get("ror"),
            delta_ror=round(delta_ror, 1) if delta_ror is not None else None,
            flag=flag,
        ))
    rows.sort(key=lambda x: -x["n"])

    # Print table
    hdr = f"{'Reaction':<42} {'n':>6}  {'PRR_us':>7} {'PRR_ref':>7} {'Δ%':>5}  {'ROR_us':>7} {'ROR_ref':>7} {'Δ%':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows[:50]:
        ror_u  = f"{r['ror_ours']:.2f}"  if r["ror_ours"]  else "   —  "
        ror_r  = f"{r['ror_ref']:.2f}"   if r["ror_ref"]   else "   —  "
        ror_d  = f"{r['delta_ror']:.1f}" if r["delta_ror"] is not None else "  —  "
        print(
            f"{r['reaction']:<42} {r['n']:>6}  "
            f"{r['prr_ours']:>7.2f} {r['prr_ref']:>7.2f} {r['delta_prr']:>4.1f}%  "
            f"{ror_u:>7} {ror_r:>7} {ror_d:>5}%  {r['flag']}"
        )

    # Summary — split by signal strength
    import statistics as _stats

    strong = [r for r in rows if r["prr_ours"] >= 5.0 and r["n"] >= 50]
    all_n  = [r for r in rows if r["n"] >= 100]

    if strong:
        strong_d = [r["delta_prr"] for r in strong]
        print(f"\n{'─'*72}")
        print(f"Drug-specific signals (PRR ≥ 5, n ≥ 50):  {len(strong)} reactions")
        print(f"  Median Δ: {_stats.median(strong_d):.1f}%   Mean Δ: {_stats.mean(strong_d):.1f}%   Max Δ: {max(strong_d):.1f}%")
        pct = sum(1 for d in strong_d if d < 10) / len(strong_d) * 100
        print(f"  Within 10%: {pct:.0f}%  {'✅ Formula validated' if pct >= 70 else '⚠️  Check formula'}")

    if all_n:
        all_d = [r["delta_prr"] for r in all_n]
        print(f"\nAll high-count (n ≥ 100):  {len(all_n)} reactions")
        print(f"  Median Δ: {_stats.median(all_d):.1f}%   Mean Δ: {_stats.mean(all_d):.1f}%   Max Δ: {max(all_d):.1f}%")
        pct = sum(1 for d in all_d if d < 10) / len(all_d) * 100
        print(f"  Within 10%: {pct:.0f}%")

    print(f"""
{'─'*72}
Interpretation guide
  Low Δ on drug-specific reactions (PRR≥5)  → formula correct ✅
  High Δ on background reactions (PRR<3)   → expected: background rates differ
    because our local extract covers 2018-2026 while openFDA has full history.
    Reactions with high background in older drugs (e.g. pre-2018) will have
    lower background rate in our extract → higher PRR → larger delta vs openFDA.
  High Δ on drug-specific reactions (PRR≥5) → investigate ⚠️
{'─'*72}""")

    # Save
    out = our_csv.parent / f"{our_csv.stem.replace('_ours','')}_comparison.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "reaction","n","prr_ours","prr_ref","delta_prr",
            "ror_ours","ror_ref","delta_ror","flag",
        ])
        w.writeheader(); w.writerows(rows)
    print(f"\n✅ Saved → {out.name}")


async def _benchmark(drug_name: str, top_n: int = 100) -> None:
    """Full automated benchmark: export ours + fetch openFDA reference + compare."""
    from agent.tools.prr import get_drug_names as _resolve
    names_result = await _resolve(drug_name)
    drug_names = names_result["found_names"]

    our_csv = await _export(drug_name, top_n=top_n)
    ref_csv = await _reference(drug_name, drug_names=drug_names)
    _compare(our_csv, ref_csv, label="openFDA")


async def _main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "benchmark":
        drug  = sys.argv[2] if len(sys.argv) > 2 else "semaglutide"
        top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        await _benchmark(drug, top_n=top_n)

    elif cmd == "export":
        drug  = sys.argv[2] if len(sys.argv) > 2 else "semaglutide"
        top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        await _export(drug, top_n=top_n)

    elif cmd == "reference":
        drug = sys.argv[2] if len(sys.argv) > 2 else "semaglutide"
        await _reference(drug)

    elif cmd == "compare":
        if len(sys.argv) < 4:
            print("Usage: benchmark_vs_openvigil.py compare <ours.csv> <reference.csv>")
            sys.exit(1)
        _compare(Path(sys.argv[2]), Path(sys.argv[3]))

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: benchmark <drug> [top_n]  |  export  |  reference  |  compare")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(_main())
