"""
Drug Safety Signal Detection — LangGraph pipeline

Design principle for small local models (Gemma4 E4B):
  - Python handles ALL data retrieval and computation
  - LLM has exactly ONE job: write the final report
  - Context passed to LLM: ~500 tokens (structured tables, not raw JSON)

Graph:
  resolve_names → calculate_prr → fetch_label → [search_literature?] → write_report

Conditional edge: literature search only runs if unlabeled signals exist.
"""

import os
import json
from typing import TypedDict, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from agent.tools.prr import calculate_prr, get_drug_names
from agent.tools.openfda import get_drug_label
from agent.tools.pubmed import search_literature

load_dotenv()

LOCAL_MODEL_URL = os.getenv("LOCAL_MODEL_URL", "http://localhost:12434/v1")


# ---------------------------------------------------------------------------
# State — typed dict passed between nodes
# ---------------------------------------------------------------------------

class DrugSafetyState(TypedDict):
    drug_name:          str
    drug_names:         list[str]          # all brand+generic names resolved
    prr_signals:        list[dict]         # [{reaction, prr, drug_count}]
    drug_total:         int
    faers_total:        int
    labeled_reactions:  list[str]          # from FDA label (ALL CAPS)
    literature:         list[dict]         # [{signal, papers, supports}]
    briefing:           str                # final LLM output
    error:              Optional[str]


# ---------------------------------------------------------------------------
# Nodes — pure Python (no LLM except write_report)
# ---------------------------------------------------------------------------

async def resolve_names(state: DrugSafetyState) -> dict:
    """Resolve drug name to all FAERS variants (generic + brand names)."""
    result = await get_drug_names(state["drug_name"])
    names = result.get("found_names", [state["drug_name"].upper()])
    print(f"  [resolve_names] {state['drug_name']} → {names}")
    return {"drug_names": names}


async def calculate_prr_signals(state: DrugSafetyState) -> dict:
    """Calculate PRR for all reactions. Pure Python — no LLM."""
    result = await calculate_prr(state["drug_names"], top_n=50)
    signals = result.get("signals", [])
    print(f"  [calculate_prr] {result['drug_total']:,} drug reports | "
          f"{result['faers_total']:,} total | {len(signals)} signals (PRR≥2)")
    return {
        "prr_signals":  signals,
        "drug_total":   result["drug_total"],
        "faers_total":  result["faers_total"],
    }


async def fetch_label(state: DrugSafetyState) -> dict:
    """Fetch FDA label reactions. Pure Python — no LLM."""
    drug = state["drug_name"].lower()
    label = await get_drug_label(drug)
    reactions: list[str] = []
    for section in ("boxed_warning", "warnings_and_cautions", "adverse_reactions"):
        text = " ".join(label.get(section, []))
        # Quick extraction: find capitalized medical terms
        import re
        found = re.findall(r'\b[A-Z][A-Z\s]{3,}\b', text.upper())
        reactions.extend(found)
    # Clean up
    reactions = list({r.strip() for r in reactions if 3 < len(r.strip()) < 60})
    print(f"  [fetch_label] {len(reactions)} labeled reactions")
    return {"labeled_reactions": reactions}


async def search_lit(state: DrugSafetyState) -> dict:
    """Search PubMed for top unlabeled signals. Pure Python — no LLM."""
    labeled = set(state.get("labeled_reactions", []))
    unlabeled = [
        s for s in state["prr_signals"]
        if s["reaction"] not in labeled
    ][:3]  # top 3 by PRR

    literature = []
    for signal in unlabeled:
        result = await search_literature(state["drug_name"], signal["reaction"])
        papers = result.get("papers", [])
        literature.append({
            "signal":   signal["reaction"],
            "prr":      signal["prr"],
            "papers":   len(papers),
            "pmids":    [p.get("pmid", "") for p in papers[:3]],
            "supports": len(papers) > 0,
        })
        print(f"  [literature] {signal['reaction']}: {len(papers)} papers")

    return {"literature": literature}


async def write_report(state: DrugSafetyState) -> dict:
    """
    ONE LLM call — Gemma4 E4B writes the final report from structured data.
    Context is ~500 tokens: just the tables it needs to format.
    """
    import litellm

    labeled = set(state.get("labeled_reactions", []))
    lit_map = {l["signal"]: l for l in state.get("literature", [])}

    # Build compact PRR table string
    prr_rows = []
    for s in state["prr_signals"][:15]:
        rxn = s["reaction"]
        is_labeled = "Yes" if rxn in labeled else "**No ⚠️**"
        papers = lit_map.get(rxn, {}).get("papers", "-")
        prr_rows.append(f"| {rxn} | {s['prr']} | {s['drug_count']} | {is_labeled} | {papers} |")

    prr_table = "\n".join(prr_rows) if prr_rows else "| No signals detected | - | - | - | - |"

    prompt = f"""Write a drug safety briefing for {state['drug_name'].upper()}.

DATA:
- FAERS reports for drug: {state['drug_total']:,}
- Total FAERS database: {state['faers_total']:,}
- PRR signals detected: {len(state['prr_signals'])}

PRR SIGNALS TABLE (already calculated — just format it):
| Reaction | PRR | Reports | Labeled? | Papers |
|----------|-----|---------|----------|--------|
{prr_table}

LABELED REACTIONS (from FDA label): {', '.join(list(labeled)[:10])}

Write this exact format:
## Drug Safety Briefing: {state['drug_name'].upper()}
**FAERS reports analysed**: {state['drug_total']:,}  |  **Total index**: {state['faers_total']:,}

### Signals Detected (PRR ≥ 2.0)
| Reaction | PRR | Reports | In FDA Label? | Literature |
[copy the table above]

### Key Findings
2-3 bullet points on the most important signals, focusing on unlabeled ones.

**Risk**: LOW/MEDIUM/HIGH
**Action**: MONITOR/INVESTIGATE/ESCALATE

> Research only. Requires clinical validation."""

    print(f"  [write_report] calling Gemma4 E4B (~{len(prompt)//4} tokens)...")

    try:
        resp = await litellm.acompletion(
            model="openai/docker.io/ai/gemma4:E4B",
            base_url=LOCAL_MODEL_URL,
            api_key="docker",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        briefing = resp.choices[0].message.content or ""
        print(f"  [write_report] got {len(briefing)} chars, finish={resp.choices[0].finish_reason}")
        # Strip thinking tokens if present
        if "<think>" in briefing:
            briefing = briefing.split("</think>")[-1].strip()
        if not briefing:
            briefing = f"[Model returned empty response — finish_reason: {resp.choices[0].finish_reason}]"
    except Exception as e:
        print(f"  [write_report] ERROR: {e}")
        briefing = f"[Report generation failed: {e}]"

    return {"briefing": briefing}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def should_search_literature(state: DrugSafetyState) -> str:
    """Only search PubMed if there are unlabeled signals worth investigating."""
    labeled = set(state.get("labeled_reactions", []))
    unlabeled = [s for s in state["prr_signals"] if s["reaction"] not in labeled]
    has_strong = any(s["prr"] >= 3.0 and s["drug_count"] >= 10 for s in unlabeled)
    result = "search_lit" if (unlabeled and has_strong) else "write_report"
    print(f"  [route] {len(unlabeled)} unlabeled signals → {result}")
    return result


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_pipeline() -> StateGraph:
    graph = StateGraph(DrugSafetyState)

    graph.add_node("resolve_names",  resolve_names)
    graph.add_node("calculate_prr",  calculate_prr_signals)
    graph.add_node("fetch_label",    fetch_label)
    graph.add_node("search_lit",     search_lit)
    graph.add_node("write_report",   write_report)

    graph.set_entry_point("resolve_names")
    graph.add_edge("resolve_names", "calculate_prr")
    graph.add_edge("calculate_prr", "fetch_label")
    graph.add_conditional_edges(
        "fetch_label",
        should_search_literature,
        {"search_lit": "search_lit", "write_report": "write_report"}
    )
    graph.add_edge("search_lit",  "write_report")
    graph.add_edge("write_report", END)

    return graph.compile()


# Singleton — compiled once at import
pipeline = build_pipeline()
