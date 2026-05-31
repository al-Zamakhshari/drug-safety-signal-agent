"""
Ingest FAERS quarterly ZIP archives into OpenSearch.

Processes all drugs in a single pass — much faster than the openFDA API.
Files inside ZIPs live under ascii/ subdirectory, pipe-delimited ($).

Usage:
    uv run python -m ingestion.faers_zip_indexer --dir ~/faers_data --all-drugs
    uv run python -m ingestion.faers_zip_indexer --dir ~/faers_data --drugs semaglutide,rofecoxib
"""

import asyncio, argparse, os, zipfile, csv, io, glob, time
from pathlib import Path
from opensearchpy import AsyncOpenSearch, helpers
from dotenv import load_dotenv
from ingestion.faers_indexer import MAPPING, INDEX, _client

load_dotenv()

# Default drug list for pharmacovigilance studies
DEFAULT_DRUGS = {
    # GLP-1 / diabetes class — generics AND brand names (FAERS uses both)
    "SEMAGLUTIDE", "OZEMPIC", "WEGOVY", "RYBELSUS",   # semaglutide brands
    "LIRAGLUTIDE", "VICTOZA", "SAXENDA",               # liraglutide brands
    "DULAGLUTIDE", "TRULICITY",
    "TIRZEPATIDE", "MOUNJARO", "ZEPBOUND",
    "EXENATIDE", "BYETTA", "BYDUREON",
    "EMPAGLIFLOZIN", "JARDIANCE",
    "DAPAGLIFLOZIN", "FARXIGA",
    "SITAGLIPTIN", "JANUVIA",
    "METFORMIN", "GLUCOPHAGE",
    # COX-2 / NSAIDs (retrospective validation)
    "ROFECOXIB", "CELECOXIB", "IBUPROFEN", "NAPROXEN",
    # Common baseline drugs (improves PRR denominator)
    "ATORVASTATIN", "LISINOPRIL", "AMLODIPINE", "LOSARTAN",
    "METOPROLOL", "ASPIRIN", "WARFARIN", "OMEPRAZOLE",
    "LEVOTHYROXINE", "GABAPENTIN", "SERTRALINE", "AMOXICILLIN",
}


def _read_table(zf: zipfile.ZipFile, prefix: str) -> dict[str, list[dict]]:
    """Read a pipe-delimited ($) table from the ZIP, return dict keyed by primaryid."""
    for name in zf.namelist():
        basename = os.path.basename(name).upper()
        if basename.startswith(prefix.upper()) and basename.endswith(".TXT"):
            with zf.open(name) as f:
                content = f.read().decode("latin-1", errors="replace")
            reader = csv.DictReader(io.StringIO(content), delimiter="$")
            rows: dict[str, list[dict]] = {}
            for row in reader:
                pid = row.get("primaryid", row.get("PRIMARYID", ""))
                if pid:
                    rows.setdefault(pid, []).append(row)
            return rows
    return {}


def _parse_zip(zf: zipfile.ZipFile, target_drugs: set[str]) -> list[dict]:
    """Parse one quarterly ZIP → documents for target drugs."""
    print(f"    Reading DRUG table...")
    drug_table = _read_table(zf, "DRUG")

    target_pids: set[str] = set()
    for pid, drugs in drug_table.items():
        names = {d.get("drugname", d.get("DRUGNAME", "")).upper().strip() for d in drugs}
        if not target_drugs or names & target_drugs:
            target_pids.add(pid)

    if not target_pids:
        print(f"    No matching reports found")
        return []

    print(f"    Found {len(target_pids):,} matching reports — reading DEMO + REAC...")
    demo_table = _read_table(zf, "DEMO")
    reac_table = _read_table(zf, "REAC")

    docs = []
    for pid in target_pids:
        demo = (demo_table.get(pid) or [{}])[0]
        drug_names = list({
            d.get("drugname", d.get("DRUGNAME", "")).upper().strip()
            for d in drug_table.get(pid, [])
            if d.get("drugname", d.get("DRUGNAME", ""))
        })
        reactions = [
            r.get("pt", r.get("PT", "")).upper().strip()
            for r in reac_table.get(pid, [])
            if r.get("pt", r.get("PT", ""))
        ]
        receivedate = demo.get("fda_dt", demo.get("FDA_DT",
                     demo.get("event_dt", demo.get("EVENT_DT", ""))))
        age = None
        try:
            age_val = demo.get("age", demo.get("AGE", ""))
            age = float(age_val) if age_val else None
        except (ValueError, TypeError):
            pass

        doc = {
            "safetyreportid": demo.get("caseid", demo.get("CASEID", pid)),
            "receivedate":    receivedate[:8] if receivedate else None,
            "serious":        demo.get("serious", demo.get("SERIOUS", "")),
            "drug_names":     drug_names,
            "reactions":      reactions,
            "outcomes":       [],
            "patient_age":    age,
            "patient_sex":    demo.get("sex", demo.get("SEX", demo.get("gndr_cod", ""))),
            "country":        demo.get("occr_country", demo.get("OCCR_COUNTRY", "")),
            "reporter_type":  demo.get("rept_cod", demo.get("REPT_COD", "")),
            "narrative":      "",
        }
        if doc["safetyreportid"] and doc["receivedate"]:
            docs.append(doc)
    return docs


async def index_docs(client: AsyncOpenSearch, docs: list[dict]) -> int:
    if not docs:
        return 0
    actions = [
        {"_index": INDEX, "_id": doc["safetyreportid"], "_source": doc}
        for doc in docs if doc.get("safetyreportid")
    ]
    success, errors = await helpers.async_bulk(client, actions, chunk_size=5_000, raise_on_error=False)
    return success


async def process_zip(client: AsyncOpenSearch, zip_path: str, target_drugs: set[str]) -> int:
    print(f"\n  Processing {Path(zip_path).name}...")
    t0 = time.time()
    with zipfile.ZipFile(zip_path, "r") as zf:
        docs = _parse_zip(zf, target_drugs)
    if not docs:
        return 0
    print(f"    Indexing {len(docs):,} documents...")
    indexed = await index_docs(client, docs)
    elapsed = time.time() - t0
    print(f"    ✅ {indexed:,} indexed in {elapsed:.0f}s")
    return indexed


async def run(zip_paths: list[str], target_drugs: set[str]):
    client = _client()
    try:
        # Ensure index exists
        try:
            await client.indices.create(index=INDEX, body=MAPPING)
            print(f"Created index: {INDEX}")
        except Exception:
            print(f"Index {INDEX} already exists — appending")

        total = 0
        for path in sorted(zip_paths):
            indexed = await process_zip(client, path, target_drugs)
            total += indexed

            # Progress after each ZIP
            stats = await client.indices.stats(index=INDEX)
            idx_stats = stats["indices"][INDEX]["total"]
            size_mb = idx_stats["store"]["size_in_bytes"] / 1024 / 1024
            doc_count = idx_stats["docs"]["count"]
            print(f"    Total: {doc_count:,} docs | {size_mb:.0f} MB")

        # Final refresh
        await client.indices.refresh(index=INDEX)
        print(f"\n✅ Ingestion complete — {total:,} new documents indexed")
        print(f"   Run: uv run python main.py semaglutide")
    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser(description="Index FAERS ZIP archives into OpenSearch")
    parser.add_argument("--dir",       help="Directory containing FAERS ZIP files")
    parser.add_argument("--file",      help="Single ZIP file to process")
    parser.add_argument("--drugs",     help="Comma-separated drug names (default: 30 common drugs)")
    parser.add_argument("--all-drugs", action="store_true", help="Index all drugs (slow, ~11.9M docs)")
    args = parser.parse_args()

    if args.all_drugs:
        target_drugs: set[str] = set()  # empty = all
        print("Indexing ALL drugs (11.9M+ docs, ~1 hour)")
    elif args.drugs:
        target_drugs = {d.strip().upper() for d in args.drugs.split(",")}
        print(f"Indexing drugs: {target_drugs}")
    else:
        target_drugs = DEFAULT_DRUGS
        print(f"Indexing {len(DEFAULT_DRUGS)} default drugs")

    if args.file:
        zip_paths = [args.file]
    elif args.dir:
        zip_paths = glob.glob(os.path.join(os.path.expanduser(args.dir), "*.zip"))
        if not zip_paths:
            print(f"No ZIP files found in {args.dir}")
            return
        print(f"Found {len(zip_paths)} ZIP files")
    else:
        parser.print_help()
        return

    asyncio.run(run(zip_paths, target_drugs))


if __name__ == "__main__":
    main()
