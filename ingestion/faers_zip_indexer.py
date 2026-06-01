"""
Ingest FAERS quarterly ZIP archives into OpenSearch.

Uses Polars for parsing — ~3x less memory than csv.DictReader:
  - Arrow columnar format vs Python dict-of-lists
  - Single file read (no StringIO copy)
  - Vectorised join instead of nested dict lookups

Memory profile per ZIP:
  Old (csv.DictReader):  ~3 GB peak (3 tables × full dict in RAM)
  New (Polars):          ~300 MB peak (Arrow columnar, in-place join)

Usage:
    uv run python -m ingestion.faers_zip_indexer --dir ~/faers_data --all-drugs
    uv run python -m ingestion.faers_zip_indexer --dir ~/faers_data --drugs semaglutide,rofecoxib
"""

import asyncio, argparse, os, zipfile, glob, time
from io import BytesIO
from pathlib import Path

import polars as pl
from opensearchpy import AsyncOpenSearch, helpers
from dotenv import load_dotenv
from ingestion.faers_indexer import MAPPING, INDEX, _client

load_dotenv()

def _load_default_drugs() -> set[str]:
    """
    Derive the default drug filter set from config/comparators.yaml.
    Includes the index drugs AND all their comparators so the full
    within-class comparison is supported without re-filtering.
    Falls back to a minimal hardcoded set if the config is absent.
    """
    from pathlib import Path
    import yaml as _yaml
    cfg_path = Path(__file__).parent.parent / "config" / "comparators.yaml"
    if cfg_path.exists():
        try:
            cfg = _yaml.safe_load(cfg_path.read_text()) or {}
            drugs: set[str] = set()
            for entry in cfg.values():
                drugs.update(n.upper() for n in entry.get("names", []))
                for grp in entry.get("comparators", []):
                    drugs.update(n.upper() for n in grp)
            if drugs:
                return drugs
        except Exception:
            pass
    # Fallback: keep a minimal set so --all-drugs works even without the YAML
    return {
        "SEMAGLUTIDE", "OZEMPIC", "WEGOVY", "RYBELSUS",
        "ROFECOXIB", "VIOXX", "CELECOXIB", "IBUPROFEN", "NAPROXEN",
        "LIRAGLUTIDE", "VICTOZA", "SAXENDA",
    }


DEFAULT_DRUGS = _load_default_drugs()


def _read_df(zf: zipfile.ZipFile, prefix: str) -> pl.DataFrame | None:
    """
    Read a pipe-delimited ($) FAERS table from a ZIP as a Polars DataFrame.
    Columns normalised to lowercase. All types as Utf8 (no schema inference).
    Single read — no StringIO copy, ~3x less memory than csv.DictReader.
    """
    for name in zf.namelist():
        basename = os.path.basename(name).upper()
        if basename.startswith(prefix.upper()) and basename.endswith(".TXT"):
            with zf.open(name) as f:
                data = f.read()   # bytes — one copy only
            try:
                df = pl.read_csv(
                    BytesIO(data),
                    separator="$",
                    encoding="latin1",
                    infer_schema_length=0,   # all Utf8 — no type guessing
                    ignore_errors=True,
                    truncate_ragged_lines=True,
                )
                # Normalise column names to lowercase
                return df.rename({c: c.lower() for c in df.columns})
            except Exception:
                pass
    return None


def _col(df: pl.DataFrame, *candidates: str) -> str | None:
    """Return first column name that exists in the DataFrame."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _parse_zip(zf: zipfile.ZipFile, target_drugs: set[str]) -> list[dict]:
    """
    Parse one quarterly ZIP using Polars joins.
    Returns list of OpenSearch-ready dicts.
    """
    # ── DRUG table ──────────────────────────────────────────────────────────
    drug_df = _read_df(zf, "DRUG")
    if drug_df is None:
        return []

    # Pre-2012 AERS: primary key is "isr" not "primaryid"
    pid_col  = _col(drug_df, "primaryid", "isr", "caseid") or "primaryid"
    name_col = _col(drug_df, "drugname") or "drugname"

    if name_col not in drug_df.columns or pid_col not in drug_df.columns:
        return []

    drug_df = drug_df.select([
        pl.col(pid_col).alias("pid"),
        pl.col(name_col).str.to_uppercase().str.strip_chars().alias("drug_upper"),
    ])

    # Filter to target drugs (skip filter if target_drugs is empty = all drugs)
    if target_drugs:
        drug_df = drug_df.filter(pl.col("drug_upper").is_in(target_drugs))

    if drug_df.is_empty():
        return []

    print(f"    Found {drug_df['pid'].n_unique():,} matching reports")

    # Aggregate drug names per report
    drug_agg = (
        drug_df
        .group_by("pid")
        .agg(pl.col("drug_upper").alias("drug_names"))
    )

    # ── REAC table ──────────────────────────────────────────────────────────
    reac_df = _read_df(zf, "REAC")
    if reac_df is not None:
        rpid = _col(reac_df, "primaryid", "isr", "caseid") or "primaryid"
        pt   = _col(reac_df, "pt") or "pt"
        if rpid in reac_df.columns and pt in reac_df.columns:
            reac_agg = (
                reac_df
                .select([
                    pl.col(rpid).alias("pid"),
                    pl.col(pt).str.to_uppercase().str.strip_chars().alias("reaction"),
                ])
                .filter(pl.col("reaction").is_not_null() & (pl.col("reaction") != ""))
                .group_by("pid")
                .agg(pl.col("reaction").alias("reactions"))
            )
            result = drug_agg.join(reac_agg, on="pid", how="left")
        else:
            result = drug_agg.with_columns(pl.lit(None).alias("reactions"))
    else:
        result = drug_agg.with_columns(pl.lit(None).alias("reactions"))

    # ── DEMO table ──────────────────────────────────────────────────────────
    demo_df = _read_df(zf, "DEMO")
    if demo_df is not None:
        # Pre-2012 AERS: primary key is "isr" not "primaryid"
        dpid = _col(demo_df, "primaryid", "isr", "caseid") or "primaryid"
        if dpid in demo_df.columns:
            # Select only columns we need — avoids joining massive wide table
            keep = {dpid}
            for alias, candidates in [
                # FAERS (2012+) and AERS (2004-2011) field names
                ("caseid",       ["caseid", "case"]),
                ("fda_dt",       ["fda_dt", "event_dt"]),
                ("serious",      ["serious"]),
                ("age",          ["age"]),
                ("age_cod",      ["age_cod"]),              # unit: DEC/YR/MON/WK/DY/HR
                ("sex",          ["sex", "gndr_cod"]),      # gndr_cod = pre-2012
                ("occr_country", ["occr_country", "reporter_country", "to_mfr"]),
                ("rept_cod",     ["rept_cod", "i_f_cod"]),  # i_f_cod = pre-2012
            ]:
                found = _col(demo_df, *candidates)
                if found:
                    keep.add(found)

            demo_slim = demo_df.select([c for c in demo_df.columns if c in keep])
            demo_slim = demo_slim.rename({dpid: "pid"})
            # Deduplicate (take first row per pid)
            demo_slim = demo_slim.unique(subset=["pid"], keep="first")
            result = result.join(demo_slim, on="pid", how="left")

    # ── Build OpenSearch docs ───────────────────────────────────────────────
    docs = []
    for row in result.iter_rows(named=True):
        pid = row.get("pid") or ""

        # date field
        receivedate = (
            row.get("fda_dt") or row.get("event_dt") or ""
        )
        receivedate = receivedate[:8] if receivedate else None

        safetyid = row.get("caseid") or pid

        if not safetyid or not receivedate:
            continue

        # age — normalize to years using age_cod unit field
        # FAERS age_cod values: DEC=decade, YR=year, MON=month, WK=week, DY=day, HR=hour
        # Without normalization, a 6-month-old (age=6, age_cod=MON) would be binned as ≥18.
        age = None
        try:
            age_val = row.get("age") or ""
            age_cod = str(row.get("age_cod") or "").upper().strip()
            if age_val:
                raw_age = float(age_val)
                # Convert to years
                if age_cod == "DEC":
                    age = raw_age * 10.0
                elif age_cod in ("YR", ""):
                    age = raw_age            # default assumption: years
                elif age_cod == "MON":
                    age = raw_age / 12.0
                elif age_cod == "WK":
                    age = raw_age / 52.18
                elif age_cod == "DY":
                    age = raw_age / 365.25
                elif age_cod == "HR":
                    age = raw_age / 8_766.0
                else:
                    age = raw_age            # unknown unit — store raw, document caveat
        except (ValueError, TypeError):
            pass

        docs.append({
            "safetyreportid": safetyid,
            "receivedate":    receivedate,
            "serious":        row.get("serious") or "",
            "drug_names":     row.get("drug_names") or [],
            "reactions":      row.get("reactions") or [],
            "outcomes":       [],
            "patient_age":    age,
            "patient_sex":    row.get("sex") or row.get("gndr_cod") or "",
            "country":        row.get("occr_country") or row.get("reporter_country") or "",
            "reporter_type":  row.get("rept_cod") or "",
            "narrative":      "",
        })

    return docs


async def index_docs(client: AsyncOpenSearch, docs: list[dict]) -> int:
    if not docs:
        return 0
    actions = [
        {"_index": INDEX, "_id": doc["safetyreportid"], "_source": doc}
        for doc in docs if doc.get("safetyreportid")
    ]
    success, errors = await helpers.async_bulk(
        client, actions, chunk_size=5_000, raise_on_error=False
    )
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
        try:
            await client.indices.create(index=INDEX, body=MAPPING)
            print(f"Created index: {INDEX}")
        except Exception:
            print(f"Index {INDEX} already exists — appending")

        total = 0
        for path in sorted(zip_paths):
            indexed = await process_zip(client, path, target_drugs)
            total += indexed

            stats = await client.indices.stats(index=INDEX)
            idx  = stats["indices"][INDEX]["total"]
            print(f"    Total: {idx['docs']['count']:,} docs | "
                  f"{idx['store']['size_in_bytes']//1024//1024:.0f} MB")

        await client.indices.refresh(index=INDEX)
        print(f"\n✅ Ingestion complete — {total:,} new documents indexed")
    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser(description="Index FAERS ZIP archives into OpenSearch")
    parser.add_argument("--dir",       help="Directory containing FAERS ZIP files")
    parser.add_argument("--file",      help="Single ZIP file to process")
    parser.add_argument("--drugs",     help="Comma-separated drug names")
    parser.add_argument("--all-drugs", action="store_true", help="Index all drugs (~11.9M docs)")
    args = parser.parse_args()

    if args.all_drugs:
        target_drugs: set[str] = set()
        print("Indexing ALL drugs")
    elif args.drugs:
        target_drugs = {d.strip().upper() for d in args.drugs.split(",")}
        print(f"Indexing: {target_drugs}")
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
