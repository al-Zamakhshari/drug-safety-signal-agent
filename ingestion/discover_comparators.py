"""
Auto-discover therapeutic comparators for a drug via RxClass ATC API.

Given a drug name:
  1. Resolve to RxNorm RxCUI
  2. Look up ATC class memberships via RxClass
  3. List all class members (drugs in the same ATC class)
  4. Filter to drugs actually present in the faers_reports index
  5. Resolve brand names for each comparator (RxNorm BN tty)
  6. Append the generated entry to config/comparators.yaml for review

Usage:
    uv run python -m ingestion.discover_comparators --drug semaglutide
    uv run python -m ingestion.discover_comparators --drug metformin --atc-level 4
    # Then review config/comparators.yaml and rebuild the index:
    uv run python -m ingestion.compute_class_ratio --drug SEMAGLUTIDE
"""

import argparse
import asyncio
import os
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv

from agent.os_client import client as _client

load_dotenv()

RXNORM_BASE  = "https://rxnav.nlm.nih.gov/REST"
RXCLASS_BASE = "https://rxnav.nlm.nih.gov/REST/rxclass"
CONFIG_FILE  = Path(__file__).parent.parent / "config" / "comparators.yaml"
FAERS_INDEX  = os.getenv("OPENSEARCH_INDEX", "faers_reports")


async def resolve_rxcui(drug_name: str) -> str | None:
    """Resolve a drug name to its ingredient-level RxCUI."""
    async with httpx.AsyncClient(timeout=10) as http:
        r = await http.get(f"{RXNORM_BASE}/rxcui.json",
                           params={"name": drug_name, "search": "1"})
        data = r.json()
        rxcui = data.get("idGroup", {}).get("rxnormId", [None])[0]
        if rxcui:
            return rxcui
        # Fallback: ingredient concept from drugs.json
        r2 = await http.get(f"{RXNORM_BASE}/drugs.json", params={"name": drug_name})
        for grp in r2.json().get("drugGroup", {}).get("conceptGroup", []):
            if grp.get("tty") == "IN":
                for prop in grp.get("conceptProperties", []):
                    return prop.get("rxcui")
    return None


async def get_atc_classes(rxcui: str, atc_level: int = 4) -> list[dict]:
    """Return ATC classes for a given RxCUI. atc_level 4 = subgroup (e.g. A10BJ)."""
    async with httpx.AsyncClient(timeout=10) as http:
        r = await http.get(
            f"{RXCLASS_BASE}/class/byRxcui.json",
            params={"rxcui": rxcui, "relaSource": "ATC"},
        )
        classes = []
        for item in r.json().get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", []):
            cls = item.get("rxclassMinConceptItem", {})
            class_id   = cls.get("classId", "")
            class_name = cls.get("className", "")
            # Filter by ATC level (length: 1=A, 3=A10, 4=A10B, 5=A10BJ, 7=A10BJ02)
            if len(class_id) == atc_level:
                classes.append({"class_id": class_id, "class_name": class_name})
        return classes


async def get_class_members(class_id: str) -> list[str]:
    """Return all RxCUIs that belong to an ATC class."""
    async with httpx.AsyncClient(timeout=15) as http:
        r = await http.get(
            f"{RXCLASS_BASE}/classMembers.json",
            params={"classId": class_id, "relaSource": "ATC"},
        )
        members = []
        for item in r.json().get("drugMemberGroup", {}).get("drugMember", []):
            rxcui = item.get("minConcept", {}).get("rxcui")
            if rxcui:
                members.append(rxcui)
        return members


async def get_brand_names(rxcui: str) -> list[str]:
    """Return brand names (BN tty) for a given ingredient RxCUI."""
    async with httpx.AsyncClient(timeout=10) as http:
        r = await http.get(
            f"{RXNORM_BASE}/rxcui/{rxcui}/related.json",
            params={"tty": "BN"},
        )
        brands = []
        for grp in r.json().get("relatedGroup", {}).get("conceptGroup", []):
            for prop in grp.get("conceptProperties", []):
                name = prop.get("name", "").upper().strip()
                if name and not any(c.isdigit() for c in name):
                    brands.append(name)
        return brands


async def get_ingredient_name(rxcui: str) -> str | None:
    """Return the canonical ingredient name for a RxCUI."""
    async with httpx.AsyncClient(timeout=10) as http:
        r = await http.get(f"{RXNORM_BASE}/rxcui/{rxcui}/properties.json")
        return r.json().get("properties", {}).get("name", "").upper() or None


async def filter_to_indexed_drugs(
    candidates: list[dict], min_reports: int = 100
) -> list[dict]:
    """Keep only drugs that have ≥ min_reports in faers_reports."""
    os_client = _client()
    try:
        result = []
        for cand in candidates:
            all_names = [cand["generic"]] + cand.get("brands", [])
            try:
                resp = await os_client.count(
                    index=FAERS_INDEX,
                    body={"query": {"terms": {"drug_names": all_names}}},
                )
                count = resp.get("count", 0)
            except Exception:
                count = 0
            if count >= min_reports:
                cand["faers_count"] = count
                result.append(cand)
        return result
    finally:
        await os_client.close()


async def discover(
    drug_name: str,
    atc_level: int = 4,
    min_reports: int = 100,
) -> dict | None:
    """
    Full discovery pipeline for one drug.
    Returns a comparator-group dict ready for comparators.yaml, or None.
    """
    print(f"\n[discover] Drug: {drug_name}")

    rxcui = await resolve_rxcui(drug_name)
    if not rxcui:
        print(f"  ✗ Could not resolve RxCUI for '{drug_name}'")
        return None
    print(f"  RxCUI: {rxcui}")

    classes = await get_atc_classes(rxcui, atc_level=atc_level)
    if not classes:
        print(f"  ✗ No ATC level-{atc_level} class found")
        return None
    print(f"  ATC classes: {[c['class_id'] + ' ' + c['class_name'] for c in classes]}")

    # Use the first ATC class (most drugs have one at level 4)
    atc_class = classes[0]
    member_rxcuis = await get_class_members(atc_class["class_id"])
    print(f"  ATC class members: {len(member_rxcuis)} RxCUIs")

    # Resolve names + brands for each member (excluding the index drug itself)
    candidates = []
    for member_rxcui in member_rxcuis:
        if member_rxcui == rxcui:
            continue
        generic = await get_ingredient_name(member_rxcui)
        if not generic:
            continue
        brands = await get_brand_names(member_rxcui)
        candidates.append({
            "rxcui":   member_rxcui,
            "generic": generic,
            "brands":  brands,
        })

    print(f"  Candidate comparators: {len(candidates)}")
    if not candidates:
        return None

    # Filter to drugs actually in faers_reports
    indexed = await filter_to_indexed_drugs(candidates, min_reports=min_reports)
    print(f"  With ≥{min_reports} FAERS reports: {len(indexed)}")
    for d in indexed:
        print(f"    {d['generic']} — {d['faers_count']:,} reports")

    if not indexed:
        print("  ✗ No comparators found in index")
        return None

    # Resolve index drug's own brand names
    index_brands = await get_brand_names(rxcui)
    index_generic = await get_ingredient_name(rxcui) or drug_name.upper()

    # Build the comparators.yaml entry
    comparators = []
    for d in indexed:
        group = [d["generic"]] + d["brands"][:3]   # generic + up to 3 brand names
        comparators.append(group)

    entry = {
        "names":       [index_generic] + index_brands[:4],
        "comparators": comparators,
    }
    drug_key = index_generic

    print(f"\n  Generated entry for {drug_key}:")
    print(f"    names: {entry['names']}")
    print(f"    comparators: {entry['comparators']}")

    return drug_key, entry


def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(cfg: dict) -> None:
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


async def main(drug_name: str, atc_level: int, min_reports: int, dry_run: bool) -> None:
    result = await discover(drug_name, atc_level=atc_level, min_reports=min_reports)
    if result is None:
        print("\n✗ Discovery failed — no entry written")
        return

    drug_key, entry = result

    if dry_run:
        print("\n[dry-run] Would write to config/comparators.yaml:")
        print(yaml.dump({drug_key: entry}, default_flow_style=False))
        return

    cfg = load_config()
    if drug_key in cfg:
        print(f"\n⚠️  {drug_key} already exists in comparators.yaml — overwriting.")
    cfg[drug_key] = entry
    save_config(cfg)
    print(f"\n✅ Wrote {drug_key} → {CONFIG_FILE}")
    print(f"\nNext step: rebuild the within-class index:")
    print(f"  uv run python -m ingestion.compute_class_ratio --drug {drug_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-discover ATC comparators via RxClass")
    parser.add_argument("--drug",        required=True, help="Drug name to discover comparators for")
    parser.add_argument("--atc-level",   type=int, default=4,   help="ATC level depth (default: 4)")
    parser.add_argument("--min-reports", type=int, default=100, help="Min FAERS reports to include comparator")
    parser.add_argument("--dry-run",     action="store_true",   help="Print entry without writing")
    args = parser.parse_args()

    asyncio.run(main(
        drug_name   = args.drug,
        atc_level   = args.atc_level,
        min_reports = args.min_reports,
        dry_run     = args.dry_run,
    ))
