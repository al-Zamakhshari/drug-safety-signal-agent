"""Tools for querying the openFDA public API."""

import asyncio
import httpx
import json
import os
import re
import yaml
from functools import lru_cache
from pathlib import Path
from typing import Any

_BRAND_ALIASES_FILE = Path(__file__).parent.parent.parent / "config" / "brand_aliases.yaml"


@lru_cache(maxsize=1)
def _load_brand_aliases() -> dict:
    """Load brand aliases from config/brand_aliases.yaml (cached at first call)."""
    if _BRAND_ALIASES_FILE.exists():
        try:
            return yaml.safe_load(_BRAND_ALIASES_FILE.read_text()) or {}
        except Exception:
            pass
    return {}

OPENFDA_BASE = "https://api.fda.gov/drug"
RXNORM_BASE  = "https://rxnav.nlm.nih.gov/REST"

# Local cache for MedDRA PT→LLT mappings — populated on first use, persists across runs.
# This keeps the tool offline after the first fetch of each drug's reactions.
_LLT_CACHE_FILE = Path(__file__).parent.parent.parent / ".meddra_llt_cache.json"
_llt_cache: dict[str, list[str]] | None = None


def _load_llt_cache() -> dict[str, list[str]]:
    global _llt_cache
    if _llt_cache is None:
        if _LLT_CACHE_FILE.exists():
            try:
                _llt_cache = json.loads(_LLT_CACHE_FILE.read_text())
            except Exception:
                _llt_cache = {}
        else:
            _llt_cache = {}
    return _llt_cache


def _save_llt_cache(cache: dict[str, list[str]]) -> None:
    try:
        _LLT_CACHE_FILE.write_text(json.dumps(cache, indent=2))
    except Exception:
        pass


async def get_meddra_llts(pt: str) -> list[str]:
    """
    Return MedDRA Lower-Level Terms (LLTs) for a Preferred Term (PT).

    LLTs are official MedDRA synonyms for a PT — e.g. "myocardial infarction"
    has LLTs including "heart attack", "MI", etc. Using LLTs prevents false-novel
    flags in `_is_labeled` when FDA label text uses a synonym not in the PT.

    Data source: openFDA drug/event API (counts by LLT for a given PT reaction).
    Results are cached locally in .meddra_llt_cache.json so subsequent runs
    work fully offline.

    Returns list of clean LLT strings in ALL CAPS. Falls back to [pt] on any error.
    """
    cache = _load_llt_cache()
    pt_upper = pt.upper()

    if pt_upper in cache:
        return cache[pt_upper]

    # Query openFDA for all LLTs that map to this PT via the reactions count endpoint.
    # The LLT field is reactionmeddrallt — filtering by PT and counting by LLT
    # gives us the LLT vocabulary for this PT.
    try:
        data = await _get(
            f"{OPENFDA_BASE}/event.json",
            {
                "search": f'patient.reaction.reactionmeddrapt.exact:"{pt_upper}"',
                "count":  "patient.reaction.reactionmeddrallt.exact",
                "limit":  "20",
            },
        )
        llts = []
        for item in data.get("results", []):
            term = item.get("term", "").upper().strip()
            pt_tokens  = set(re.findall(r"[a-z]+", pt_upper.lower()))
            llt_tokens = set(re.findall(r"[a-z]+", term.lower()))
            if pt_tokens and len(pt_tokens & llt_tokens) >= max(1, len(pt_tokens) // 2):
                if term != pt_upper:
                    llts.append(term)
        cache[pt_upper] = llts
        _save_llt_cache(cache)
        return llts
    except Exception:
        cache[pt_upper] = []
        _save_llt_cache(cache)
        return []


_RETRY_STATUS = {429, 500, 502, 503, 504}
_RETRY_DELAYS = (1.0, 2.0, 4.0)   # seconds; 3 attempts total


async def _get(url: str, params: dict) -> dict:
    """GET with exponential backoff on transient errors (429/5xx, timeouts)."""
    last_exc: Exception | None = None
    for attempt, delay in enumerate((*_RETRY_DELAYS, None), start=1):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(url, params=params)
                if r.status_code in _RETRY_STATUS and delay is not None:
                    await asyncio.sleep(delay)
                    continue
                r.raise_for_status()
                return r.json()
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            last_exc = exc
            if delay is not None:
                await asyncio.sleep(delay)
            continue
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in _RETRY_STATUS and delay is not None:
                last_exc = exc
                await asyncio.sleep(delay)
                continue
            raise
    raise last_exc or RuntimeError(f"All retries failed for {url}")


async def normalize_drug_name(drug_name: str) -> dict[str, Any]:
    """
    Resolve a drug name to its RxNorm CUI and return clean brand-name tokens
    suitable for FAERS drug_names field matching (ALL CAPS).

    Uses RxNorm related.json?tty=BN (Brand Names) — returns short trademarked
    names, not verbose product descriptions. The old drugs.json endpoint returns
    strings like "0.25 MG Pen Injector [Ozempic]" which do not match FAERS.

    Args:
        drug_name: Generic or brand name

    Returns:
        dict with:
          rxcui       — RxNorm concept identifier
          generic     — canonical generic name in FAERS ALL-CAPS format
          brand_names — list of brand name tokens (ALL CAPS) for FAERS IN query
          all_names   — combined list: generic + brand_names (use for FAERS queries)
    """
    try:
        # Step 1: resolve ingredient-level RxCUI.
        # Use /rxcui.json?name= which returns the canonical ingredient concept (IN tty),
        # NOT approximateTerm or drugs.json which return product-level CUIs with dose info.
        rxcui = None
        generic = drug_name.upper()

        rxcui_resp = await _get(f"{RXNORM_BASE}/rxcui.json",
                                {"name": drug_name, "search": "1"})
        rxcui = rxcui_resp.get("idGroup", {}).get("rxnormId", [None])[0]

        if not rxcui:
            # Fallback: use drugs.json and take the first IN (ingredient) concept
            data = await _get(f"{RXNORM_BASE}/drugs.json", {"name": drug_name})
            for grp in data.get("drugGroup", {}).get("conceptGroup", []):
                if grp.get("tty") == "IN":   # ingredient, not product
                    for prop in grp.get("conceptProperties", []):
                        rxcui = prop.get("rxcui")
                        generic = prop.get("name", drug_name).upper()
                        break
                if rxcui:
                    break

        if not rxcui:
            return {
                "rxcui": None, "generic": generic,
                "brand_names": [], "all_names": [generic],
                "warning": "RxCUI not found — using provided name only"
            }

        # Step 2: get brand names (tty=BN returns short trademarked brand tokens).
        # These match FAERS drug_names field exactly (ALL CAPS, no dose information).
        # Withdrawn drugs may have no BN in RxNorm — config/brand_aliases.yaml
        # provides overrides that users can extend.
        related = await _get(f"{RXNORM_BASE}/rxcui/{rxcui}/related.json",
                             {"tty": "BN"})
        brand_names: list[str] = []
        for grp in related.get("relatedGroup", {}).get("conceptGroup", []):
            for prop in grp.get("conceptProperties", []):
                name = prop.get("name", "").upper().strip()
                # Skip names with dose info (contain digits or ML/MG)
                if name and name != generic and not any(c.isdigit() for c in name):
                    brand_names.append(name)

        # Load brand fallbacks from config/brand_aliases.yaml for withdrawn/niche drugs
        aliases = _load_brand_aliases()
        fallback = aliases.get(generic.upper(), {}).get("brands", [])
        brand_names.extend(fallback)

        # Deduplicate, preserve order
        seen: set[str] = {generic}
        unique_brands: list[str] = []
        for b in brand_names:
            if b not in seen:
                seen.add(b)
                unique_brands.append(b)

        return {
            "rxcui": rxcui,
            "generic": generic,
            "brand_names": unique_brands,
            "all_names": [generic] + unique_brands,
        }
    except Exception as e:
        return {
            "rxcui": None, "generic": drug_name.upper(),
            "brand_names": [], "all_names": [drug_name.upper()],
            "error": str(e)
        }


async def get_reaction_counts_for_drug(drug_name: str, limit: int = 50) -> dict[str, Any]:
    """
    Get the top adverse reactions reported for a drug in FAERS, with counts.
    This is the primary input for PRR signal calculation.

    Args:
        drug_name: Drug name to search for
        limit: Max number of reaction types to return

    Returns:
        dict with drug_name, total_reports, and reactions list [{term, count}]
    """
    try:
        data = await _get(
            f"{OPENFDA_BASE}/event.json",
            {
                "search": f'patient.drug.medicinalproduct:"{drug_name}"',
                "count": "patient.reaction.reactionmeddrapt.exact",
                "limit": limit,
            },
        )
        total_resp = await _get(
            f"{OPENFDA_BASE}/event.json",
            {
                "search": f'patient.drug.medicinalproduct:"{drug_name}"',
                "limit": 1,
            },
        )
        total = total_resp.get("meta", {}).get("results", {}).get("total", 0)
        reactions = [
            {"term": r["term"], "count": r["count"]}
            for r in data.get("results", [])
        ]
        return {"drug_name": drug_name, "total_reports": total, "reactions": reactions}
    except Exception as e:
        return {"drug_name": drug_name, "total_reports": 0, "reactions": [], "error": str(e)}


async def get_reaction_baseline_count(reaction_term: str) -> dict[str, Any]:
    """
    Get the total number of FAERS reports containing a specific adverse reaction across ALL drugs.
    Used as the denominator in PRR calculation.

    Args:
        reaction_term: MedDRA preferred term (e.g. "NAUSEA", "CARDIAC ARREST")

    Returns:
        dict with reaction_term and total_count
    """
    try:
        data = await _get(
            f"{OPENFDA_BASE}/event.json",
            {
                "search": f'patient.reaction.reactionmeddrapt:"{reaction_term}"',
                "limit": 1,
            },
        )
        total = data.get("meta", {}).get("results", {}).get("total", 0)
        return {"reaction_term": reaction_term, "total_count": total}
    except Exception as e:
        return {"reaction_term": reaction_term, "total_count": 0, "error": str(e)}


async def get_total_faers_reports() -> dict[str, Any]:
    """
    Get the total number of reports in the FAERS database.
    Used as the overall denominator for PRR calculation.

    Returns:
        dict with total_reports
    """
    try:
        data = await _get(f"{OPENFDA_BASE}/event.json", {"limit": 1})
        total = data.get("meta", {}).get("results", {}).get("total", 0)
        return {"total_reports": total}
    except Exception as e:
        return {"total_reports": 0, "error": str(e)}


async def get_signal_timeline(drug_name: str, reaction_term: str) -> dict[str, Any]:
    """
    Get the monthly report count for a specific drug+reaction pair over time.
    Used to identify WHEN a signal first emerged.

    Args:
        drug_name: Drug name
        reaction_term: MedDRA reaction term

    Returns:
        dict with timeline list [{year_month, count}]
    """
    try:
        data = await _get(
            f"{OPENFDA_BASE}/event.json",
            {
                "search": (
                    f'patient.drug.medicinalproduct:"{drug_name}"'
                    f'+AND+patient.reaction.reactionmeddrapt:"{reaction_term}"'
                ),
                "count": "receivedate",
            },
        )
        timeline = [{"date": r["time"], "count": r["count"]} for r in data.get("results", [])]
        return {"drug_name": drug_name, "reaction_term": reaction_term, "timeline": timeline}
    except Exception as e:
        return {"drug_name": drug_name, "reaction_term": reaction_term, "timeline": [], "error": str(e)}


async def get_drug_label(drug_name: str) -> dict[str, Any]:
    """
    Fetch the current FDA drug label for a drug.
    Tries the generic name, then brand names from config/brand_aliases.yaml
    (for withdrawn drugs not indexed in openFDA by generic name).

    Args:
        drug_name: Generic or brand name

    Returns:
        dict with warnings, adverse_reactions, contraindications, boxed_warning
    """
    # Build the list of names to try from config/brand_aliases.yaml
    aliases = _load_brand_aliases()
    label_names = aliases.get(drug_name.upper(), {}).get("label_names", [])
    names_to_try = [drug_name] + label_names

    for name in names_to_try:
        try:
            # Use separate queries — the + encoding breaks OR in httpx
            data = await _get(
                f"{OPENFDA_BASE}/label.json",
                {"search": f'openfda.brand_name:"{name}"', "limit": 1},
            )
            if not data.get("results"):
                data = await _get(
                    f"{OPENFDA_BASE}/label.json",
                    {"search": f'openfda.generic_name:"{name}"', "limit": 1},
                )
            results = data.get("results", [])
            if results:
                label = results[0]
                return {
                    "drug_name": drug_name,
                    "resolved_as": name,
                    "found": True,
                    "boxed_warning":        label.get("boxed_warning", []),
                    "warnings":             label.get("warnings", []),
                    "adverse_reactions":    label.get("adverse_reactions", []),
                    "contraindications":    label.get("contraindications", []),
                    "warnings_and_cautions": label.get("warnings_and_cautions", []),
                }
        except Exception:
            continue

    return {"drug_name": drug_name, "found": False, "error": "Label not found under any known name"}
