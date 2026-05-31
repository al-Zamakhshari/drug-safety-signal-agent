"""Tools for searching biomedical literature via PubMed."""

import httpx
from typing import Any

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


async def _get(url: str, params: dict) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params={**params, "retmode": "json"})
        r.raise_for_status()
        return r.json()


async def search_literature(drug_name: str, reaction_term: str, max_results: int = 5) -> dict[str, Any]:
    """
    Search PubMed for papers about a specific drug and adverse reaction.
    Returns titles, abstracts, and publication years to support or contradict a signal.

    Args:
        drug_name: Drug name
        reaction_term: Adverse reaction term
        max_results: Maximum papers to return

    Returns:
        dict with papers list [{pmid, title, abstract, year}]
    """
    query = f"{drug_name}[Title/Abstract] AND {reaction_term}[Title/Abstract] AND (adverse[Title/Abstract] OR safety[Title/Abstract])"

    try:
        search = await _get(
            f"{NCBI_BASE}/esearch.fcgi",
            {"db": "pubmed", "term": query, "retmax": max_results, "sort": "relevance"},
        )
        ids = search.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {"drug_name": drug_name, "reaction_term": reaction_term, "papers": []}

        summary = await _get(
            f"{NCBI_BASE}/esummary.fcgi",
            {"db": "pubmed", "id": ",".join(ids)},
        )
        result_map = summary.get("result", {})
        papers = []
        for pmid in ids:
            article = result_map.get(pmid, {})
            papers.append({
                "pmid": pmid,
                "title": article.get("title", ""),
                "year": article.get("pubdate", "")[:4],
                "journal": article.get("fulljournalname", ""),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })

        return {"drug_name": drug_name, "reaction_term": reaction_term, "papers": papers}
    except Exception as e:
        return {"drug_name": drug_name, "reaction_term": reaction_term, "papers": [], "error": str(e)}
