"""
Compute class_ratio time series for OpenSearch Anomaly Detection.

Mirrors the hackathon's Elastic ML approach:
  class_ratio = drug_rate / mean(comparator_drug_rates)
  drug_rate(D, R, Q) = count(D+R+Q) / count(D+Q)

Output index: faers_ml_rates — one doc per (drug, reaction, quarter).
OpenSearch AD then runs RCF on class_ratio over time.

Usage:
    uv run python -m ingestion.compute_class_ratio
"""

import asyncio, os
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
            "class_rate":   {"type": "float"},
            "class_ratio":  {"type": "float"},
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


    return AsyncOpenSearch(
        hosts=[os.getenv("OPENSEARCH_URL", "https://localhost:9200")],
        http_auth=(os.getenv("OPENSEARCH_USER", "admin"),
                   os.getenv("OPENSEARCH_PASSWORD", "Pharma@2024!")),
        use_ssl=True, verify_certs=False, ssl_show_warn=False,
    )


async def _get_quarterly_rates(client, drug_names: list[str]) -> dict:
    """
    Returns {quarter_date_str: {reaction: rate, "__total__": count}}
    quarter_date_str is ISO format e.g. "2021-07-01"
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
                    "format":            "yyyy-'Q'Q",  # gives "2021-Q3"
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
        q_str  = bucket["key_as_string"]     # e.g. "2021-Q3"
        q_date = _quarter_to_date(q_str)     # e.g. "2021-07-01"
        if not q_date:
            continue
        total = bucket["doc_count"]
        if total < 10:
            continue
        result[q_date] = {"__total__": total}
        for rbucket in bucket["by_reaction"]["buckets"]:
            rxn   = rbucket["key"]
            count = rbucket["doc_count"]
            result[q_date][rxn] = count / total   # drug_rate

    return result


async def compute_and_index(drug_key: str, group: dict) -> int:
    client = _client()
    try:
        print(f"\n{'='*55}")
        print(f"Drug: {drug_key} | names: {group['names']}")

        drug_rates = await _get_quarterly_rates(client, group["names"])
        print(f"  Quarters: {len(drug_rates)}")

        # Get rates for each comparator group
        comp_group_rates: list[dict] = []
        for comp_names in group["comparators"]:
            rates = await _get_quarterly_rates(client, comp_names)
            if rates:
                comp_group_rates.append(rates)
                print(f"  Comparator {comp_names[0]}: {len(rates)} quarters")

        if not comp_group_rates:
            print("  No comparator data — skip")
            return 0

        # Build class_ratio docs
        docs = []
        for q_date, rxn_rates in drug_rates.items():
            drug_total = rxn_rates.get("__total__", 0)
            if drug_total < 10:
                continue

            for rxn, drug_rate in rxn_rates.items():
                if rxn == "__total__":
                    continue

                # Mean rate across ALL active comparator groups for this quarter.
                # Missing reaction in a group = rate 0.0 (not dropped).
                # Dropping zeros was the old bug: it inflated class_rate and
                # excluded reactions absent from all comparators (the most
                # drug-specific signals — exactly what we want to detect).
                active_groups = [g for g in comp_group_rates if q_date in g]
                if not active_groups:
                    continue

                comp_vals   = [g[q_date].get(rxn, 0.0) for g in active_groups]
                class_rate  = sum(comp_vals) / len(active_groups)

                if class_rate > 0:
                    class_ratio = round(drug_rate / class_rate, 4)
                else:
                    # Reaction absent from entire drug class → strong drug-specific
                    # signal. Cap at 999 instead of divide-by-zero.
                    class_ratio = 999.0
                drug_count  = int(drug_rate * drug_total)

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
