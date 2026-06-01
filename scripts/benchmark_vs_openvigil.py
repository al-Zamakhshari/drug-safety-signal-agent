"""
Benchmark drug-safety-signal-agent PRR/ROR against OpenVigil 2.

OpenVigil 2 is the peer-reviewed FAERS-based pharmacovigilance tool from
Böhm et al. (Pharmacoepidemiology and Drug Safety, 2012, 2016). It uses the
same FAERS source and the same 2×2 PRR/ROR formulas — making it the natural
reference implementation for validating our output.

Usage
-----
Step 1: Export our PRR/ROR output for a drug:
    uv run python scripts/benchmark_vs_openvigil.py export semaglutide

    → Writes scripts/benchmark_semaglutide_ours.csv

Step 2: Get OpenVigil 2 data for the same drug:
    a. Go to https://openvigil.pharmacology.uni-kiel.de/openvigil2/
    b. Enter drug name: "SEMAGLUTIDE" (use the FAERS spelling shown in our export)
    c. Set the time window to match our FAERS vintage (see "faers_vintage" in our CSV)
    d. Click "Calculate" → wait for results
    e. Click "Download CSV" → save as scripts/benchmark_semaglutide_openvigil.csv

Step 3: Compare:
    uv run python scripts/benchmark_vs_openvigil.py compare \\
        scripts/benchmark_semaglutide_ours.csv \\
        scripts/benchmark_semaglutide_openvigil.csv

    → Prints a delta table and saves scripts/benchmark_semaglutide_comparison.csv

Why differences are expected
-----------------------------
- FAERS vintage: OpenVigil may use a different quarterly cut-off.
- De-duplication: OpenVigil applies their own de-duplication; ours is FDA-raw.
- Multi-drug reports: both count reports (not patients), but handling of
  records listing many drugs may differ slightly.
- Top-N reactions: we cap at top_n=50 per drug; OpenVigil uses all reactions.

A PRR delta < 10% on high-count signals (n ≥ 100) is strong agreement.
Larger deltas on rare reactions (n < 20) are expected from de-duplication
differences and are not a formula error.
"""

import asyncio
import csv
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


async def _export(drug_name: str, top_n: int = 100) -> Path:
    """Compute PRR+ROR for a drug and write our output CSV."""
    from agent.tools.prr import calculate_prr, get_drug_names
    from agent.os_client import client as _client

    print(f"Resolving drug names for: {drug_name}")
    names_result = await get_drug_names(drug_name)
    drug_names = names_result["found_names"]
    print(f"  FAERS names: {drug_names}")

    print(f"Calculating PRR/ROR (top_n={top_n}, all reactions tested for BH)...")
    result = await calculate_prr(drug_names, top_n=top_n, min_count=3, min_prr=1.0)
    # min_prr=1.0 to export everything ≥ 1.0, not just ≥ 2.0
    # (OpenVigil shows all reactions including sub-threshold ones)

    # Get FAERS vintage from OpenSearch
    os_client = _client()
    try:
        resp = await os_client.search(
            index=os.getenv("OPENSEARCH_INDEX", "faers_reports"),
            body={"size": 0, "aggs": {"max_date": {"max": {"field": "receivedate"}}}},
        )
        max_date = resp["aggregations"]["max_date"].get("value_as_string", "unknown")
    except Exception:
        max_date = "unknown"
    finally:
        await os_client.close()

    out_path = Path(__file__).parent / f"benchmark_{drug_name.lower()}_ours.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "# drug-safety-signal-agent export",
            f"drug={drug_name.upper()}",
            f"faers_names={','.join(drug_names)}",
            f"drug_total={result['drug_total']:,}",
            f"faers_total={result['faers_total']:,}",
            f"faers_vintage={max_date}",
            f"exported={datetime.now().isoformat()[:19]}",
            f"tested_reactions={result.get('tested_count', '?')}",
        ])
        writer.writerow([
            "reaction", "n",
            "PRR", "PRR_lower_95", "PRR_upper_95",
            "ROR", "ROR_lower_95", "ROR_upper_95",
            "chi2", "significant", "robust", "q_value", "fdr_significant",
        ])
        for s in result["signals"]:
            writer.writerow([
                s["reaction"],
                s["drug_count"],
                s["prr"],
                s.get("prr_lower", ""),
                s.get("prr_upper", ""),
                s.get("ror", ""),
                s.get("ror_lower", ""),
                s.get("ror_upper", ""),
                s.get("chi2", ""),
                "Y" if s.get("significant") else "N",
                "Y" if s.get("robust") else "N",
                s.get("q_value", ""),
                "Y" if s.get("fdr_significant") else "N",
            ])

    print(f"\n✅ Exported {len(result['signals'])} signals → {out_path}")
    print(f"   drug_total={result['drug_total']:,}  faers_total={result['faers_total']:,}")
    print(f"   faers_vintage={max_date}")
    print()
    print("Next step:")
    print("  1. Go to https://openvigil.pharmacology.uni-kiel.de/openvigil2/")
    print(f"  2. Search: {drug_names[0]}")
    print(f"  3. Download CSV → save as:")
    print(f"     {Path(__file__).parent / f'benchmark_{drug_name.lower()}_openvigil.csv'}")
    print(f"  4. Run: uv run python scripts/benchmark_vs_openvigil.py compare \\")
    print(f"         {out_path} \\")
    print(f"         {Path(__file__).parent / f'benchmark_{drug_name.lower()}_openvigil.csv'}")
    return out_path


def _compare(our_csv: Path, openvigil_csv: Path) -> None:
    """Compare our output with OpenVigil 2 CSV and print a delta table."""

    # --- Load ours ---
    ours: dict[str, dict] = {}
    with open(our_csv, newline="") as f:
        reader = csv.reader(f)
        header_row = None
        for row in reader:
            if row and row[0].startswith("#"):
                continue  # metadata comment row
            if header_row is None:
                header_row = row
                continue
            if len(row) < 3:
                continue
            try:
                ours[row[0].upper()] = {
                    "n":       int(row[1]),
                    "prr":     float(row[2]),
                    "ror":     float(row[5]) if row[5] else None,
                }
            except (ValueError, IndexError):
                pass

    # --- Load OpenVigil ---
    # OpenVigil 2 CSV columns vary by export format.
    # Common format: reaction | n | PRR | PRR_lower | PRR_upper | ROR | ROR_lower | ROR_upper | ...
    # We auto-detect the column positions by header name.
    openvigil: dict[str, dict] = {}
    with open(openvigil_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Normalise headers to upper
        fieldnames = [str(fn).strip().upper() for fn in (reader.fieldnames or [])]

        # Detect reaction column
        rxn_col  = _find_col(fieldnames, ["REACTION", "ADVERSE EVENT", "MED", "MEDDRA", "PREFERRED TERM", "PT"])
        n_col    = _find_col(fieldnames, ["N_DRUG", "N", "COUNT", "REPORTS", "DRUG_COUNT", "CASES"])
        prr_col  = _find_col(fieldnames, ["PRR", "PROPORTIONAL REPORTING RATIO"])
        ror_col  = _find_col(fieldnames, ["ROR", "REPORTING ODDS RATIO"])

        if rxn_col is None or prr_col is None:
            print(f"⚠️  Could not auto-detect columns in {openvigil_csv}")
            print(f"   Detected headers: {fieldnames}")
            print("   Please check the OpenVigil CSV format and adjust the script.")
            return

        for row in reader:
            raw_row = {str(k).strip().upper(): v for k, v in row.items()}
            rxn = raw_row.get(rxn_col, "").strip().upper()
            if not rxn:
                continue
            try:
                prr_val = float(raw_row.get(prr_col, "") or 0)
                n_val   = int(float(raw_row.get(n_col, "") or 0)) if n_col else None
                ror_val = float(raw_row.get(ror_col, "") or 0) if ror_col else None
                openvigil[rxn] = {"n": n_val, "prr": prr_val, "ror": ror_val}
            except (ValueError, TypeError):
                pass

    # --- Compare ---
    common = sorted(set(ours) & set(openvigil))
    only_ours = sorted(set(ours) - set(openvigil))
    only_ov   = sorted(set(openvigil) - set(ours))

    print(f"\n{'='*70}")
    print(f"Benchmark: drug-safety-signal-agent  vs  OpenVigil 2")
    print(f"{'='*70}")
    print(f"Reactions in both:       {len(common):>5}")
    print(f"Only in ours:            {len(only_ours):>5}  (may be above our top_n cutoff in OV)")
    print(f"Only in OpenVigil:       {len(only_ov):>5}  (below our top_n=50 or de-dup difference)")
    print()

    # Delta table for reactions with n ≥ 10 in our output
    rows = []
    for rxn in common:
        o = ours[rxn]
        v = openvigil[rxn]
        if o["n"] < 3:
            continue
        prr_delta_pct = abs(o["prr"] - v["prr"]) / v["prr"] * 100 if v["prr"] else float("inf")
        ror_delta_pct = (abs(o["ror"] - v["ror"]) / v["ror"] * 100
                         if (o.get("ror") and v.get("ror")) else None)
        flag = "⚠️ " if prr_delta_pct > 20 else ("✓" if prr_delta_pct < 5 else "~")
        rows.append({
            "reaction": rxn,
            "n_ours":        o["n"],
            "prr_ours":      o["prr"],
            "prr_ov":        v["prr"],
            "prr_delta_pct": round(prr_delta_pct, 1),
            "ror_ours":      o.get("ror"),
            "ror_ov":        v.get("ror"),
            "ror_delta_pct": round(ror_delta_pct, 1) if ror_delta_pct is not None else None,
            "flag":          flag,
        })

    # Sort by n descending (high-count = most meaningful comparison)
    rows.sort(key=lambda x: -x["n_ours"])

    # Print table
    print(f"{'Reaction':<45} {'n':>6}  {'PRR_ours':>8} {'PRR_OV':>8} {'Δ%':>6}  {'ROR_ours':>8} {'ROR_OV':>8} {'Δ%':>6}  {'':>3}")
    print("-" * 110)
    for r in rows[:40]:  # top 40 by count
        ror_o = f"{r['ror_ours']:.2f}" if r["ror_ours"] else "  —  "
        ror_v = f"{r['ror_ov']:.2f}"   if r["ror_ov"]   else "  —  "
        ror_d = f"{r['ror_delta_pct']:.1f}" if r["ror_delta_pct"] is not None else "  —"
        print(
            f"{r['reaction']:<45} {r['n_ours']:>6}  "
            f"{r['prr_ours']:>8.2f} {r['prr_ov']:>8.2f} {r['prr_delta_pct']:>5.1f}%  "
            f"{ror_o:>8} {ror_v:>8} {ror_d:>5}%  {r['flag']}"
        )

    # Summary stats
    deltas = [r["prr_delta_pct"] for r in rows if r["n_ours"] >= 100]
    if deltas:
        import statistics
        print()
        print(f"High-count signals (n≥100, {len(deltas)} reactions):")
        print(f"  Median PRR delta:  {statistics.median(deltas):.1f}%")
        print(f"  Mean PRR delta:    {statistics.mean(deltas):.1f}%")
        print(f"  Max PRR delta:     {max(deltas):.1f}%")
        agreement = sum(1 for d in deltas if d < 10) / len(deltas) * 100
        print(f"  Within 10%:        {agreement:.0f}% of reactions")

    # Save comparison CSV
    out_path = openvigil_csv.parent / openvigil_csv.name.replace("_openvigil", "_comparison")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "reaction", "n_ours", "prr_ours", "prr_ov", "prr_delta_pct",
            "ror_ours", "ror_ov", "ror_delta_pct", "flag"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ Comparison saved → {out_path}")


def _find_col(fieldnames: list[str], candidates: list[str]) -> str | None:
    """Return the first fieldname that contains any of the candidate strings."""
    for fn in fieldnames:
        for c in candidates:
            if c in fn:
                return fn
    return None


async def _main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "export":
        drug = sys.argv[2] if len(sys.argv) > 2 else "semaglutide"
        top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        await _export(drug, top_n=top_n)

    elif cmd == "compare":
        if len(sys.argv) < 4:
            print("Usage: benchmark_vs_openvigil.py compare <ours.csv> <openvigil.csv>")
            sys.exit(1)
        our_csv = Path(sys.argv[2])
        ov_csv  = Path(sys.argv[3])
        if not our_csv.exists():
            print(f"File not found: {our_csv}")
            sys.exit(1)
        if not ov_csv.exists():
            print(f"File not found: {ov_csv}")
            sys.exit(1)
        _compare(our_csv, ov_csv)

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: export <drug> [top_n]  |  compare <ours.csv> <openvigil.csv>")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(_main())
