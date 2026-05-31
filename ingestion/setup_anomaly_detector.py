"""
Set up OpenSearch Anomaly Detection for drug safety signals.

OpenSearch uses Random Cut Forest (RCF) — same family as Elastic ML.
This creates a detector that flags when a drug's reaction rate
deviates from the population baseline.

Usage:
    uv run python -m ingestion.setup_anomaly_detector
"""

import asyncio, os, json
import httpx
from dotenv import load_dotenv

load_dotenv()

OPENSEARCH_URL  = os.getenv("OPENSEARCH_URL", "https://localhost:9200")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "admin")
OPENSEARCH_PASS = os.getenv("OPENSEARCH_PASSWORD", "Pharma@2024!")
INDEX           = os.getenv("OPENSEARCH_INDEX", "faers_reports")
DETECTOR_NAME   = "drug_reaction_anomaly_detector"

# Drugs to monitor (same comparator class as hackathon ML job)
MONITORED_DRUGS = [
    "SEMAGLUTIDE", "LIRAGLUTIDE", "DULAGLUTIDE",
    "EMPAGLIFLOZIN", "DAPAGLIFLOZIN", "SITAGLIPTIN", "METFORMIN",
]


async def create_detector():
    """
    Create an OpenSearch anomaly detector for drug reaction rates.

    Uses category_field (partition by drug_name) so one detector
    monitors all drugs simultaneously — same concept as Elastic ML
    partition_field.
    """
    auth = (OPENSEARCH_USER, OPENSEARCH_PASS)
    headers = {"Content-Type": "application/json"}
    base = OPENSEARCH_URL

    async with httpx.AsyncClient(verify=False, timeout=30) as client:
        # Check if detector already exists
        r = await client.get(
            f"{base}/_plugins/_anomaly_detection/detectors/_search",
            auth=auth, headers=headers,
            json={"query": {"term": {"name.keyword": DETECTOR_NAME}}}
        )
        if r.status_code == 200 and r.json().get("hits", {}).get("total", {}).get("value", 0) > 0:
            detector_id = r.json()["hits"]["hits"][0]["_id"]
            print(f"Detector already exists: {detector_id}")
            return detector_id

        # Detector config — monitors report counts per drug per reaction
        # RCF will learn the baseline pattern and flag deviations
        detector_config = {
            "name": DETECTOR_NAME,
            "description": "Detect anomalous drug-reaction reporting rates (pharmacovigilance)",
            "time_field": "receivedate",
            "indices": [INDEX],
            "feature_attributes": [
                {
                    "feature_name": "report_count",
                    "feature_enabled": True,
                    "aggregation_query": {
                        "report_count": {"value_count": {"field": "safetyreportid"}}
                    }
                }
            ],
            # Partition by drug — one model per drug (like Elastic ML partition_field)
            "category_field": ["drug_names"],
            "detection_interval": {"period": {"interval": 1, "unit": "MONTHS"}},
            "window_delay":        {"period": {"interval": 1, "unit": "DAYS"}},
            # Filter to monitored drugs only
            "filter_query": {
                "terms": {"drug_names": MONITORED_DRUGS}
            },
            "shingle_size": 8,  # RCF uses last 8 months as context window
        }

        r = await client.post(
            f"{base}/_plugins/_anomaly_detection/detectors",
            auth=auth, headers=headers, json=detector_config
        )
        if r.status_code not in (200, 201):
            print(f"Error creating detector: {r.status_code} {r.text}")
            return None

        detector_id = r.json()["_id"]
        print(f"✅ Created detector: {detector_id}")

        # Start the detector (historical analysis)
        r = await client.post(
            f"{base}/_plugins/_anomaly_detection/detectors/{detector_id}/_start",
            auth=auth, headers=headers
        )
        if r.status_code == 200:
            print(f"✅ Detector started — historical analysis running in background")
            print(f"   Monitor at: http://localhost:5601 → Anomaly Detection")
        else:
            print(f"Start response: {r.status_code} {r.text}")

        return detector_id


async def get_anomaly_results(detector_id: str, top_n: int = 20) -> list[dict]:
    """
    Query anomaly results from a running detector.
    Returns top anomalies sorted by anomaly_grade (0-1 scale).
    """
    auth = (OPENSEARCH_USER, OPENSEARCH_PASS)
    base = OPENSEARCH_URL

    async with httpx.AsyncClient(verify=False, timeout=30) as client:
        r = await client.post(
            f"{base}/_plugins/_anomaly_detection/detectors/{detector_id}/results/_search",
            auth=auth,
            headers={"Content-Type": "application/json"},
            json={
                "size": top_n,
                "query": {"range": {"anomaly_grade": {"gte": 0.3}}},
                "sort": [{"anomaly_grade": "desc"}],
            }
        )
        if r.status_code != 200:
            return []

        results = []
        for hit in r.json().get("hits", {}).get("hits", []):
            src = hit["_source"]
            results.append({
                "drug":          src.get("entity", [{}])[0].get("value", "UNKNOWN"),
                "period":        src.get("data_start_time", ""),
                "anomaly_grade": round(src.get("anomaly_grade", 0), 3),
                "confidence":    round(src.get("confidence", 0), 3),
                "count":         src.get("feature_data", [{}])[0].get("data", 0),
            })
        return results


def main():
    asyncio.run(create_detector())


if __name__ == "__main__":
    main()
