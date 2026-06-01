"""
Index FAERS adverse event reports into OpenSearch.

Usage:
    uv run python -m ingestion.faers_indexer --drug <drug-name> --limit 5000
"""

import asyncio
import argparse
import os
import httpx
from opensearchpy import AsyncOpenSearch, helpers
from dotenv import load_dotenv

load_dotenv()

OPENSEARCH_URL  = os.getenv("OPENSEARCH_URL", "https://localhost:9200")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "admin")
OPENSEARCH_PASS = os.getenv("OPENSEARCH_PASSWORD", "Pharma@2024!")
INDEX           = os.getenv("OPENSEARCH_INDEX", "faers_reports")
OPENFDA_BASE    = "https://api.fda.gov/drug/event.json"

MAPPING = {
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "safetyreportid": {"type": "keyword"},
            "receivedate":    {"type": "date", "format": "yyyyMMdd||yyyy-MM-dd"},
            "drug_names":     {"type": "keyword", "eager_global_ordinals": True},
            "reactions":      {"type": "keyword", "eager_global_ordinals": True},
            "serious":        {"type": "keyword"},
            "outcomes":       {"type": "keyword"},
            "patient_sex":    {"type": "keyword"},
            "reporter_type":  {"type": "keyword"},
            "country":        {"type": "keyword"},
            "patient_age":    {"type": "float"},
            "narrative":      {"type": "text", "index": False},
        },
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "refresh_interval": "30s",
    },
}


def _client() -> AsyncOpenSearch:
    return AsyncOpenSearch(
        hosts=[OPENSEARCH_URL],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
    )


def _parse_report(raw: dict) -> dict:
    patient   = raw.get("patient", {})
    drugs     = patient.get("drug", [])
    reactions = patient.get("reaction", [])

    return {
        "safetyreportid": raw.get("safetyreportid"),
        "receivedate":    raw.get("receivedate"),
        "serious":        raw.get("serious"),
        "drug_names":     list({d.get("medicinalproduct", "").upper() for d in drugs if d.get("medicinalproduct")}),
        "reactions":      [r.get("reactionmeddrapt", "").upper() for r in reactions if r.get("reactionmeddrapt")],
        "outcomes":       list({r.get("reactionoutcome", "") for r in reactions if r.get("reactionoutcome")}),
        "patient_age":    float(patient["patientonsetage"]) if patient.get("patientonsetage") else None,
        "patient_sex":    patient.get("patientsex"),
        "reporter_type":  (raw.get("primarysource") or {}).get("qualification"),
        "country":        raw.get("occurcountry"),
        "narrative":      raw.get("narrativeincludeclinical", ""),
    }


async def fetch_reports(drug_name: str, limit: int = 1000) -> list[dict]:
    reports, skip = [], 0
    async with httpx.AsyncClient(timeout=30) as client:
        while skip < limit:
            r = await client.get(OPENFDA_BASE, params={
                "search": f'patient.drug.medicinalproduct:"{drug_name}"',
                "limit": min(100, limit - skip),
                "skip": skip,
            })
            if r.status_code != 200:
                break
            data = r.json()
            batch = data.get("results", [])
            if not batch:
                break
            reports.extend(batch)
            skip += len(batch)
            total = data.get("meta", {}).get("results", {}).get("total", 0)
            print(f"  Fetched {skip}/{min(total, limit)} reports for {drug_name}...")
            if skip >= total:
                break
    return reports


async def index_drug(drug_name: str, limit: int = 1000):
    client = _client()
    try:
        # Create index (ignore if exists)
        try:
            await client.indices.create(index=INDEX, body=MAPPING)
            print(f"Created index: {INDEX}")
        except Exception:
            pass

        print(f"Fetching FAERS reports for: {drug_name}")
        raw_reports = await fetch_reports(drug_name, limit=limit)
        docs = [_parse_report(r) for r in raw_reports]
        print(f"Indexing {len(docs)} reports...")

        # Use helpers.async_bulk for efficient batch indexing
        actions = [
            {"_index": INDEX, "_id": doc["safetyreportid"], "_source": doc}
            for doc in docs if doc["safetyreportid"]
        ]
        success, errors = await helpers.async_bulk(client, actions, raise_on_error=False)
        print(f"✅ Indexed {success} docs | {len(errors)} errors")

        # Force refresh so data is immediately queryable
        await client.indices.refresh(index=INDEX)
    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser(description="Index FAERS data into OpenSearch")
    parser.add_argument("--drug",  required=True, help="Drug name (generic or brand, any case)")
    parser.add_argument("--limit", type=int, default=1000, help="Max reports to fetch")
    args = parser.parse_args()
    asyncio.run(index_drug(args.drug, args.limit))


if __name__ == "__main__":
    main()
