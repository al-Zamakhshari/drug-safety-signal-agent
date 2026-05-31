"""
Set up OpenSearch Anomaly Detection on the class_ratio time series.

Mirrors the hackathon's Elastic ML job:
  - Detector: high_mean(class_ratio) partitioned by reaction, over drug
  - Weekly buckets (OpenSearch max: 10080 minutes = 7 days)
  - Runs historical analysis on faers_ml_rates index

Usage:
    uv run python -m ingestion.setup_anomaly_detector
"""

import asyncio, os, json
import httpx
from dotenv import load_dotenv

load_dotenv()

BASE     = os.getenv("OPENSEARCH_URL", "https://localhost:9200")
USER     = os.getenv("OPENSEARCH_USER", "admin")
PASS     = os.getenv("OPENSEARCH_PASSWORD", "Pharma@2024!")
DETECTOR = "drug_class_ratio_detector"


async def delete_old_detectors(client: httpx.AsyncClient, auth):
    """Remove old detectors to start fresh."""
    r = await client.post(
        f"{BASE}/_plugins/_anomaly_detection/detectors/_search",
        auth=auth, headers={"Content-Type": "application/json"},
        json={"query": {"match_all": {}}}
    )
    for hit in r.json().get("hits", {}).get("hits", []):
        did = hit["_id"]
        name = hit["_source"].get("name", "")
        # Stop then delete
        await client.post(
            f"{BASE}/_plugins/_anomaly_detection/detectors/{did}/_stop",
            auth=auth, headers={"Content-Type": "application/json"}
        )
        dr = await client.delete(
            f"{BASE}/_plugins/_anomaly_detection/detectors/{did}",
            auth=auth
        )
        print(f"  Deleted detector: {name} ({did}) → {dr.status_code}")


async def create_detector():
    auth = (USER, PASS)
    headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient(verify=False, timeout=30) as client:
        print("Cleaning up old detectors...")
        await delete_old_detectors(client, auth)

        # Create detector on class_ratio index
        # category_field partitions by (drug, reaction) — one RCF model per pair
        config = {
            "name": DETECTOR,
            "description": "Detect anomalous class_ratio (drug vs comparator class rate)",
            "time_field":  "quarter",
            "indices":     ["faers_ml_rates"],
            "feature_attributes": [{
                "feature_name":    "class_ratio",
                "feature_enabled": True,
                "aggregation_query": {
                    "class_ratio": {"avg": {"field": "class_ratio"}}
                }
            }],
            # Partition by reaction+drug pair: one model per combination
            "category_field": ["drug", "reaction"],
            # 10080 min = 7 days (max supported by OpenSearch AD)
            "detection_interval": {"period": {"interval": 10080, "unit": "MINUTES"}},
            "window_delay":       {"period": {"interval": 1440,  "unit": "MINUTES"}},
            "shingle_size": 8,  # look back 8 windows (8 quarters) for pattern
            "filter_query": {
                "range": {"class_ratio": {"gte": 0.01}}  # exclude near-zero
            }
        }

        r = await client.post(
            f"{BASE}/_plugins/_anomaly_detection/detectors",
            auth=auth, headers=headers, json=config
        )
        if r.status_code not in (200, 201):
            print(f"Error: {r.status_code} {r.text[:200]}")
            return None

        detector_id = r.json()["_id"]
        print(f"✅ Created detector: {detector_id}")

        # Start historical analysis
        r2 = await client.post(
            f"{BASE}/_plugins/_anomaly_detection/detectors/{detector_id}/_start",
            auth=auth, headers=headers
        )
        print(f"✅ Started historical analysis: {r2.status_code}")
        print(f"   Monitor at: http://localhost:5601 → Anomaly Detection")
        print(f"   Detector ID: {detector_id}")
        return detector_id


async def get_top_anomalies(min_grade: float = 0.3, top_n: int = 20) -> list[dict]:
    """
    Query anomaly results. Returns top anomalies sorted by anomaly_grade.
    Run AFTER the detector has completed historical analysis (~few minutes).
    """
    auth = (USER, PASS)

    async with httpx.AsyncClient(verify=False, timeout=30) as client:
        # Find detector ID
        r = await client.post(
            f"{BASE}/_plugins/_anomaly_detection/detectors/_search",
            auth=auth, headers={"Content-Type": "application/json"},
            json={"query": {"term": {"name.keyword": DETECTOR}}}
        )
        hits = r.json().get("hits", {}).get("hits", [])
        if not hits:
            print("Detector not found — run setup first")
            return []

        detector_id = hits[0]["_id"]

        # Query results
        r2 = await client.post(
            f"{BASE}/_plugins/_anomaly_detection/detectors/{detector_id}/results/_search",
            auth=auth,
            headers={"Content-Type": "application/json"},
            json={
                "size": top_n,
                "query": {"range": {"anomaly_grade": {"gte": min_grade}}},
                "sort": [{"anomaly_grade": "desc"}],
            }
        )
        if r2.status_code != 200:
            print(f"Results query failed: {r2.status_code}")
            return []

        results = []
        for hit in r2.json().get("hits", {}).get("hits", []):
            src = hit["_source"]
            entities = {e["name"]: e["value"] for e in src.get("entity", [])}
            results.append({
                "drug":          entities.get("drug", "UNKNOWN"),
                "reaction":      entities.get("reaction", "UNKNOWN"),
                "quarter":       src.get("data_start_time", ""),
                "anomaly_grade": round(src.get("anomaly_grade", 0), 3),
                "confidence":    round(src.get("confidence", 0), 3),
                "class_ratio":   round(
                    src.get("feature_data", [{}])[0].get("data", 0), 2
                ),
            })
        return results


def main():
    asyncio.run(create_detector())


if __name__ == "__main__":
    main()
