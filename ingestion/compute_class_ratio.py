"""
Compute class-ratio disproportionality time series for within-class comparison.

Method:
  class_ratio = (drug_count / drug_total) / (comp_count / comp_total)
  drug_rate(D, R, Q) = count(D+R+Q) / count(D+Q)
  comp_rate(class, R, Q) = Σ count(C+R+Q) / Σ count(C+Q)  [pooled across comparators]

This is a within-class rate ratio — it answers: "Is this drug's reporting rate for
reaction R higher than its therapeutic class's pooled rate, after controlling for
class-level effects?"

Zero cells: when a reaction appears in the drug but not the comparator pool,
Haldane–Anscombe continuity (+0.5) is applied to avoid divide-by-zero.
The raw comp_count=0 is preserved in the index; the adjusted ratio is large-but-finite,
and the downstream 95% CI (in anomaly_signals.py) will be appropriately wide.
This replaces the previous 999.0 sentinel, which was statistically undefined.

Output index: faers_ml_rates — one doc per (drug, reaction, quarter).

Usage:
    uv run python -m ingestion.compute_class_ratio
"""

import asyncio, os, math
from opensearchpy import helpers
from agent.os_client import client as _client
from dotenv import load_dotenv

load_dotenv()

SOURCE_INDEX = os.getenv("OPENSEARCH_INDEX", "faers_reports")
RATES_INDEX  = "faers_ml_rates"

DRUG_GROUPS = {
    "SEMAGLUTIDE": {
        "names":       ["SEMAGLUTIDE", "OZEMPIC", "WEGOVY", "RYBELSUS"],
        "comparators": [
            ["LIRAGLUTIDE", "VICTOZA", "SAXENDA"],
            ["DULAGLUTIDE", "TRULICITY"],
            ["TIRZEPATIDE", "MOUNJARO", "ZEPBOUND"],
            ["EMPAGLIFLOZIN", "JARDIANCE"],
            ["SITAGLIPTIN", "JANUVIA"],
        ],
    },
    "ROFECOXIB": {
        "names":       ["ROFECOXIB", "VIOXX"],
        "comparators": [["CELECOXIB", "CELEBREX"], ["IBUPROFEN"], ["NAPROXEN"]],
    },
    "LIRAGLUTIDE": {
        "names":       ["LIRAGLUTIDE", "VICTOZA", "SAXENDA"],
        "comparators": [
            ["SEMAGLUTIDE", "OZEMPIC"],
            ["DULAGLUTIDE", "TRULICITY"],
            ["TIRZEPATIDE", "MOUNJARO"],
        ],
    },
}

RATES_MAPPING = {
    "mappings": {
        "properties": {
            "drug":         {"type": "keyword"},
            "reaction":     {"type": "keyword"},
            "quarter":      {"type": "date", "format": "yyyy-MM-dd"},
            "drug_count":   {"type": "integer"},
            "drug_total":   {"type": "integer"},
            "drug_rate":    {"type": "float"},
            # Comparator pooled counts — stored raw (unadjusted) so downstream
            # callers can see the real data and apply corrections explicitly.
            "comp_count":   {"type": "integer"},   # Σ reaction counts across comparator groups
            "comp_total":   {"type": "integer"},   # Σ quarter totals across comparator groups
            "class_rate":   {"type": "float"},     # comp_count_adj / comp_total (Haldane-adj if 0)
            "class_ratio":  {"type": "float"},     # drug_rate / class_rate
        }
    },
    "settings": {"number_of_shards": 1, "number_of_replicas": 0}
}

# Quarter number → first day of quarter (ISO date)
QUARTER_STARTS = {"1": "01-01", "2": "04-01", "3": "07-01", "4": "10-01"}


def _quarter_to_date(q_str: str) -> str:
    """'2021-Q3' → '2021-07-01'"""
    try:
        year, q = q_str.split("-Q")
        return f"{year}-{QUARTER_STARTS[q]}"
    except Exception:
        return None


async def _get_quarterly_counts(client, drug_names: list[str]) -> dict:
    """
    Returns {quarter_date_str: {reaction: raw_count, "__total__": total_count}}

    Values are raw integer counts (not rates) so the caller can pool counts
    across comparator groups correctly before computing rates/ratios.
    """
    resp = await client.search(index=SOURCE_INDEX, body={
        "size": 0,
        "track_total_hits": True,
        "query": {"terms": {"drug_names": drug_names}},
        "aggs": {
            "by_quarter": {
                "date_histogram": {
                    "field":             "receivedate",
                    "calendar_interval": "quarter",
                    "format":            "yyyy-'Q'Q",   # "2021-Q3"
                },
                "aggs": {
                    "by_reaction": {
                        "terms": {"field": "reactions", "size": 300}
                    }
                }
            }
        }
    })

    result = {}
    for bucket in resp["aggregations"]["by_quarter"]["buckets"]:
        q_str  = bucket["key_as_string"]   # "2021-Q3"
        q_date = _quarter_to_date(q_str)   # "2021-07-01"
        if not q_date:
            continue
        total = bucket["doc_count"]
        if total < 10:
            continue
        result[q_date] = {"__total__": total}
        for rbucket in bucket["by_reaction"]["buckets"]:
            result[q_date][rbucket["key"]] = rbucket["doc_count"]   # raw count

    return result


async def compute_and_index(drug_key: str, group: dict) -> int:
    client = _client()
    try:
        print(f"\n{'='*55}")
        print(f"Drug: {drug_key} | names: {group['names']}")

        drug_counts = await _get_quarterly_counts(client, group["names"])
        print(f"  Quarters: {len(drug_counts)}")

        # Get counts for each comparator group
        comp_group_counts: list[dict] = []
        for comp_names in group["comparators"]:
            counts = await _get_quarterly_counts(client, comp_names)
            if counts:
                comp_group_counts.append(counts)
                print(f"  Comparator {comp_names[0]}: {len(counts)} quarters")

        if not comp_group_counts:
            print("  No comparator data — skip")
            return 0

        # Build class_ratio docs
        docs = []
        for q_date, rxn_counts in drug_counts.items():
            drug_total = rxn_counts.get("__total__", 0)
            if drug_total < 10:
                continue

            for rxn, drug_rxn_count in rxn_counts.items():
                if rxn == "__total__":
                    continue

                drug_rate  = drug_rxn_count / drug_total
                drug_count = drug_rxn_count   # integer

                # Pool raw counts across ALL active comparator groups for this quarter.
                # Using counts (not rates) ensures correct variance estimation downstream.
                active_groups = [g for g in comp_group_counts if q_date in g]
                if not active_groups:
                    continue

                comp_count = sum(g[q_date].get(rxn, 0) for g in active_groups)
                comp_total = sum(g[q_date]["__total__"] for g in active_groups)

                if comp_total == 0:
                    continue

                # Haldane–Anscombe continuity correction for zero comparator cells.
                # When comp_count == 0, the ratio is undefined (divide-by-zero).
                # Adding 0.5 gives a finite, conservative estimate while keeping
                # comp_total correct. The stored comp_count is the unadjusted integer;
                # the adjustment is applied only for computing class_ratio/class_rate
                # here and in the CI downstream. This replaces the old 999 sentinel
                # which was statistically meaningless.
                comp_count_adj = comp_count if comp_count > 0 else 0.5
                class_rate     = comp_count_adj / comp_total
                class_ratio    = round(drug_rate / class_rate, 4)

                if drug_count < 3:
                    continue

                doc_id = f"{drug_key}__{rxn}__{q_date}".replace(" ", "_")
                docs.append({
                    "_index":   RATES_INDEX,
                    "_id":      doc_id,
                    "_source": {
                        "drug":        drug_key,
                        "reaction":    rxn,
                        "quarter":     q_date,
                        "drug_count":  drug_count,
                        "drug_total":  drug_total,
                        "drug_rate":   round(drug_rate, 6),
                        "comp_count":  comp_count,      # raw, unadjusted (may be 0)
                        "comp_total":  comp_total,
                        "class_rate":  round(class_rate, 6),
                        "class_ratio": class_ratio,
                    }
                })

        if docs:
            ok, errs = await helpers.async_bulk(
                client, docs, chunk_size=2000, raise_on_error=False
            )
            await client.indices.refresh(index=RATES_INDEX)
            print(f"  Indexed {ok:,} docs | {len(errs)} errors")
            if errs:
                print(f"  First error: {errs[0]}")
            return ok
        print("  No docs generated")
        return 0

    finally:
        await client.close()


async def main():
    client = _client()
    try:
        await client.indices.create(index=RATES_INDEX, body=RATES_MAPPING)
        print(f"Created index: {RATES_INDEX}")
    except Exception:
        print(f"Index {RATES_INDEX} already exists — dropping and recreating")
        await client.indices.delete(index=RATES_INDEX)
        await client.indices.create(index=RATES_INDEX, body=RATES_MAPPING)
        print(f"Recreated index: {RATES_INDEX}")
    finally:
        await client.close()

    total = 0
    for drug_key, group in DRUG_GROUPS.items():
        total += await compute_and_index(drug_key, group)

    print(f"\n✅ Done — {total:,} class_ratio docs indexed")


if __name__ == "__main__":
    asyncio.run(main())
