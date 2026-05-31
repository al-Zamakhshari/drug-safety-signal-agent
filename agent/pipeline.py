"""
Drug Safety Signal Detection — LangGraph pipeline

Node responsibilities:
  Python nodes  → all data retrieval and computation (deterministic)
  Gemma4 E4B   → two roles:
    1. investigator_node: function calling — detects class effects / DDI / trends
    2. write_report: formats all findings into clinical prose

Graph:
  resolve_names → calculate_prr → anomaly_detection → fetch_label
       → [search_literature?]
       → [investigator?]
       → write_report
"""

import os
import json
import re
from typing import TypedDict, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from agent.tools.prr import calculate_prr, get_drug_names
from agent.tools.openfda import get_drug_label
from agent.tools.pubmed import search_literature
from agent.tools.anomaly_signals import get_anomaly_signals
from agent.tools.investigator_tools import (
    get_prr, check_class_effect, get_signal_trend
)

load_dotenv()

LOCAL_MODEL_URL  = os.getenv("LOCAL_MODEL_URL", "http://localhost:12434/v1")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "docker.io/ai/gemma4:E2B")

# ---------------------------------------------------------------------------
# Shared model — ChatOpenAI pointing at Docker Model Runner
# ---------------------------------------------------------------------------

def _model(max_tokens: int = 800) -> ChatOpenAI:
    return ChatOpenAI(
        model=LOCAL_MODEL_NAME,
        base_url=LOCAL_MODEL_URL,
        api_key="docker",
        max_tokens=max_tokens,
        temperature=0,          # deterministic — critical for reliable tool calling
    )


# ---------------------------------------------------------------------------
# Investigator sub-agent (create_react_agent)
# Gemma4 E4B with 4 tools — handles its own multi-turn tool-call loop
# ---------------------------------------------------------------------------

_investigator_agent = create_react_agent(
    _model(max_tokens=1000),    # extra headroom for thinking + tool call JSON
    tools=[get_prr, check_class_effect, get_signal_trend],
    # check_ddi removed — tool never invoked in prompt, metric was incorrect
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class DrugSafetyState(TypedDict):
    drug_name:          str
    drug_names:         list[str]
    prr_signals:        list[dict]   # [{reaction, prr, drug_count}]
    drug_total:         int
    faers_total:        int
    anomaly_signals:    list[dict]   # [{reaction, max_ratio, anomaly_grade}]
    label_text:         str          # raw label text for token-overlap matching
    literature:         list[dict]
    investigation:      list[dict]
    briefing:           str
    error:              Optional[str]


# ---------------------------------------------------------------------------
# Label matching — token-overlap (no MedDRA ontology required)
# ---------------------------------------------------------------------------
_LABEL_STOP = {
    "acute", "chronic", "disorder", "syndrome", "disease", "reaction",
    "increased", "decreased", "abnormal", "nos", "unspecified", "type",
    "associated", "related", "induced", "mediated", "and", "the", "with",
    "due", "from", "that", "this", "following", "including", "severe",
}

def _label_tokens(term: str) -> set[str]:
    """Extract significant tokens from a MedDRA PT or label text."""
    return {
        w for w in re.findall(r"[a-z]+", term.lower())
        if len(w) > 3 and w not in _LABEL_STOP
    }

def _is_labeled(reaction: str, label_text: str) -> bool:
    """
    Returns True if the MedDRA PT is likely documented in the FDA label.

    Matches by token overlap — handles:
      - Word order differences: PT 'PANCREATITIS ACUTE' vs label 'acute pancreatitis'
      - Case: MedDRA ALL CAPS vs sentence case in labels
      - Partial matches: PT 'INTESTINAL OBSTRUCTION' matched by 'obstruction'

    Falls back to substring match for single-token PTs.
    """
    toks = _label_tokens(reaction)
    if not toks:
        return reaction.lower() in label_text
    return all(t in label_text for t in toks)


# ---------------------------------------------------------------------------
# Python nodes — no LLM
# ---------------------------------------------------------------------------

async def resolve_names(state: DrugSafetyState) -> dict:
    result = await get_drug_names(state["drug_name"])
    names = result.get("found_names", [state["drug_name"].upper()])
    print(f"  [names]  {state['drug_name']} → {names}")
    return {"drug_names": names}


async def calculate_prr_signals(state: DrugSafetyState) -> dict:
    result = await calculate_prr(state["drug_names"], top_n=50)
    signals = result.get("signals", [])
    print(f"  [PRR]    {result['drug_total']:,} reports | "
          f"{result['faers_total']:,} total | {len(signals)} signals")
    return {
        "prr_signals": signals,
        "drug_total":  result["drug_total"],
        "faers_total": result["faers_total"],
    }


async def run_anomaly_detection(state: DrugSafetyState) -> dict:
    """Query OpenSearch AD for class_ratio anomalies. Pure Python — no LLM."""
    # Use canonical drug name (as indexed in faers_ml_rates), not brand names
    drug = state["drug_name"].upper()
    result = await get_anomaly_signals(drug, min_ratio=2.0, min_count=5, top_n=15)
    signals = result.get("signals", [])
    state_info = result.get("detector_state", "UNKNOWN")
    print(f"  [AD]     {len(signals)} anomaly signals | detector: {state_info}")
    if signals:
        top3 = [(s["reaction"], s["max_ratio"]) for s in signals[:3]]
        print(f"           top: {top3}")
    return {"anomaly_signals": signals}


async def fetch_label(state: DrugSafetyState) -> dict:
    """
    Fetch FDA label and store raw text for token-overlap matching.
    Reads all sections including 'warnings' (not just warnings_and_cautions).
    """
    label = await get_drug_label(state["drug_name"].lower())
    # Concatenate all safety-relevant sections — lowercase for matching
    label_text = " ".join(
        " ".join(label.get(s, []))
        for s in ("boxed_warning", "warnings", "warnings_and_cautions",
                  "adverse_reactions", "contraindications",
                  "warnings_and_precautions")
    ).lower()
    print(f"  [label]  {len(label_text):,} chars | "
          f"found={label.get('found', False)} resolved_as={label.get('resolved_as','?')}")
    return {"label_text": label_text}


async def search_lit(state: DrugSafetyState) -> dict:
    label_text = state.get("label_text", "")
    # Top 3 unlabeled signals by PRR
    targets = [s for s in state["prr_signals"] if not _is_labeled(s["reaction"], label_text)][:3]
    literature = []
    for signal in targets:
        result = await search_literature(state["drug_name"], signal["reaction"])
        papers = result.get("papers", [])
        literature.append({
            "signal":   signal["reaction"],
            "prr":      signal["prr"],
            "papers":   len(papers),
            "pmids":    [p.get("pmid", "") for p in papers[:3]],
            "supports": len(papers) > 0,
        })
        print(f"  [lit]    {signal['reaction']}: {len(papers)} papers")
    return {"literature": literature}


# ---------------------------------------------------------------------------
# Investigator node — Gemma4 E4B with function calling
# ---------------------------------------------------------------------------

async def investigate(state: DrugSafetyState) -> dict:
    """
    Gemma4 E4B investigates the top novel signals using function calling.
    Runs get_prr, check_class_effect, check_ddi, get_signal_trend autonomously.
    Returns structured classification for each investigated signal.
    """
    label_text = state.get("label_text", "")
    # Only investigate strong unlabeled signals (PRR≥5, n≥10)
    targets = [
        s for s in state["prr_signals"]
        if not _is_labeled(s["reaction"], label_text)
        and s["prr"] >= 5.0
        and s["drug_count"] >= 10
    ][:3]

    if not targets:
        print("  [invest] no strong unlabeled signals to investigate")
        return {"investigation": []}

    drug = state["drug_names"][0]
    reactions_str = ", ".join(f"{s['reaction']} (PRR={s['prr']})" for s in targets)

    # GLP-1 comparators — hardcoded for now, could be dynamic
    comparators = ["LIRAGLUTIDE", "DULAGLUTIDE", "TIRZEPATIDE", "EXENATIDE"]
    comparators = [c for c in comparators if c not in state["drug_names"]][:3]

    prompt = (
        f"Investigate these novel safety signals for {drug}: {reactions_str}\n\n"
        f"For each signal:\n"
        f"1. Use get_prr to confirm the PRR\n"
        f"2. Use check_class_effect with comparators {comparators} "
        f"to determine if it's class-wide or drug-specific\n"
        f"3. If signal is strong and drug-specific, use get_signal_trend to see "
        f"when it emerged\n\n"
        f"Then classify each as: CLASS_EFFECT | DRUG_SPECIFIC | GROWING | DDI_SUSPECT\n"
        f"Be concise. One classification per signal."
    )

    print(f"  [invest] investigating {len(targets)} signals: "
          f"{[s['reaction'] for s in targets]}")

    result = await _investigator_agent.ainvoke({"messages": [("user", prompt)]})

    # Extract final text response
    final_msg = result["messages"][-1]
    investigation_text = final_msg.content if hasattr(final_msg, "content") else str(final_msg)

    # Count tool calls made
    tool_calls = sum(
        1 for m in result["messages"]
        if hasattr(m, "tool_calls") and m.tool_calls
    )
    print(f"  [invest] {tool_calls} tool calls → classification done")

    # Structure the output
    investigation = [{
        "signals_investigated": [s["reaction"] for s in targets],
        "tool_calls_made":      tool_calls,
        "findings":             investigation_text,
    }]
    return {"investigation": investigation}


# ---------------------------------------------------------------------------
# Report writer — Gemma4 E4B formats everything into clinical prose
# ---------------------------------------------------------------------------

_VALID_RISK   = {"LOW", "MEDIUM", "HIGH"}
_VALID_ACTION = {"MONITOR", "INVESTIGATE", "ESCALATE"}
_DISCLAIMER   = "> Research only. Requires clinical validation before any regulatory action."


async def write_report(state: DrugSafetyState) -> dict:
    """
    Phase 3.1 fix: deterministic sections emitted by Python, LLM writes prose only.
    Clinical numbers (PRR, counts) are never re-typed by the model.
    """
    label_text = state.get("label_text", "")
    lit_map    = {l["signal"]: l for l in state.get("literature", [])}
    drug       = state["drug_name"].upper()

    # ── Deterministic header (Python, not LLM) ──────────────────────────────
    header = (
        f"## Drug Safety Briefing: {drug}\n"
        f"**FAERS reports analysed**: {state['drug_total']:,}  |  "
        f"**Index**: {state['faers_total']:,}\n"
    )

    # ── Deterministic PRR table (Python, not LLM) ───────────────────────────
    prr_rows = []
    for s in state["prr_signals"][:15]:
        rxn        = s["reaction"]
        is_labeled = "Yes" if _is_labeled(rxn, label_text) else "**No ⚠️**"
        papers     = lit_map.get(rxn, {}).get("papers", "—")
        prr_rows.append(
            f"| {rxn} | {s['prr']} | {s['drug_count']} | {is_labeled} | {papers} |"
        )
    prr_block = (
        "### PRR Signals (EMA standard: PRR ≥ 2.0)\n"
        "| Reaction | PRR | Reports | In FDA Label? | Literature |\n"
        "|----------|-----|---------|---------------|------------|\n"
        + ("\n".join(prr_rows) if prr_rows else "| No signals detected | — | — | — | — |")
    )

    # ── Deterministic anomaly table (Python, not LLM) ───────────────────────
    anomaly_rows = []
    for s in state.get("anomaly_signals", [])[:8]:
        trend = s.get("trend", "—")
        anomaly_rows.append(
            f"| {s['reaction']} | {s['max_ratio']} | {s['max_count']} | {trend} |"
        )
    anomaly_block = (
        "### Anomaly Detection (class_ratio vs drug class)\n"
        "| Reaction | Max class_ratio | Count | Trend |\n"
        "|----------|----------------|-------|-------|\n"
        + ("\n".join(anomaly_rows) if anomaly_rows
           else "| (run: uv run python -m ingestion.compute_class_ratio) | — | — | — |")
    )

    # ── Investigation findings (structured, from investigate node) ───────────
    invest_block = ""
    if state.get("investigation"):
        inv      = state["investigation"][0]
        findings = inv.get("findings", "").strip()
        n_tools  = inv.get("tool_calls_made", 0)
        if findings and n_tools > 0:
            invest_block = (
                f"### Investigation Results ({n_tools} tool calls)\n"
                f"{findings}\n"
            )
        elif findings:
            invest_block = f"### Investigation Results\n{findings}\n"

    # ── Ask LLM for narrative only — no numbers, no tables ─────────────────
    # Feed structured JSON so the model doesn't need to re-parse the tables
    signals_json = json.dumps([
        {"reaction": s["reaction"], "prr": s["prr"],
         "labeled": _is_labeled(s["reaction"], label_text),
         "papers": lit_map.get(s["reaction"], {}).get("papers", 0)}
        for s in state["prr_signals"][:10]
    ], indent=2)

    narrative_prompt = (
        f"Write 2-3 Key Findings bullet points and a Risk/Action line for {drug}.\n\n"
        f"PRR signals (pre-computed, do NOT repeat the numbers):\n{signals_json}\n\n"
        f"Focus on: unlabeled signals (In FDA Label = false), signals with literature "
        f"support, signals appearing in both PRR and anomaly detection.\n\n"
        f"Format exactly:\n"
        f"### Key Findings\n* bullet 1\n* bullet 2\n\n"
        f"**Risk**: LOW/MEDIUM/HIGH\n"
        f"**Action**: MONITOR/INVESTIGATE/ESCALATE"
    )

    print(f"  [report] LLM narrative only (~{len(narrative_prompt)//4} tokens)...")

    narrative = ""
    try:
        resp = await _model(max_tokens=500).ainvoke(narrative_prompt)
        narrative = resp.content or ""
        if "<think>" in narrative:
            narrative = narrative.split("</think>")[-1].strip()
    except Exception as e:
        narrative = f"### Key Findings\n* Report generation error: {e}"

    # Validate Risk/Action — derive from PRR if LLM output is garbled
    if not narrative or "**Risk**" not in narrative:
        top_prr = state["prr_signals"][0]["prr"] if state["prr_signals"] else 0
        risk   = "HIGH" if top_prr >= 10 else "MEDIUM" if top_prr >= 5 else "LOW"
        action = "ESCALATE" if top_prr >= 10 else "INVESTIGATE" if top_prr >= 5 else "MONITOR"
        narrative = (
            f"### Key Findings\n"
            f"* {len([s for s in state['prr_signals'] if not _is_labeled(s['reaction'], label_text)])} "
            f"unlabeled signals detected (PRR ≥ 2.0).\n"
            f"* Top signal: {state['prr_signals'][0]['reaction'] if state['prr_signals'] else 'none'} "
            f"(PRR={top_prr})\n\n"
            f"**Risk**: {risk}\n**Action**: {action}"
        )

    # ── Assemble final briefing deterministically ────────────────────────────
    sections = [header, prr_block, anomaly_block]
    if invest_block:
        sections.append(invest_block)
    sections.append(narrative)
    sections.append(_DISCLAIMER)

    briefing = "\n\n".join(sections)
    return {"briefing": briefing}


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def should_search_literature(state: DrugSafetyState) -> str:
    label_text = state.get("label_text", "")
    unlabeled = [s for s in state["prr_signals"] if not _is_labeled(s["reaction"], label_text)]
    needs_lit = unlabeled and any(s["prr"] >= 3.0 and s["drug_count"] >= 10 for s in unlabeled)
    result = "search_lit" if needs_lit else "investigate"
    print(f"  [route]  {len(unlabeled)} unlabeled → {result}")
    return result


def should_investigate(state: DrugSafetyState) -> str:
    label_text = state.get("label_text", "")
    strong_unlabeled = [
        s for s in state["prr_signals"]
        if not _is_labeled(s["reaction"], label_text)
        and s["prr"] >= 5.0
        and s["drug_count"] >= 10
    ]
    result = "investigate" if strong_unlabeled else "write_report"
    print(f"  [route]  {len(strong_unlabeled)} strong unlabeled → {result}")
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
    graph.add_node("investigate",    investigate)
    graph.add_node("write_report",   write_report)

    graph.add_node("anomaly_detection", run_anomaly_detection)

    graph.set_entry_point("resolve_names")
    graph.add_edge("resolve_names",    "calculate_prr")
    graph.add_edge("calculate_prr",    "anomaly_detection")
    graph.add_edge("anomaly_detection", "fetch_label")

    # After label: search literature if strong unlabeled signals exist
    graph.add_conditional_edges(
        "fetch_label",
        should_search_literature,
        {"search_lit": "search_lit", "investigate": "investigate"},
    )

    # After literature: investigate if strong novel signals remain
    graph.add_conditional_edges(
        "search_lit",
        should_investigate,
        {"investigate": "investigate", "write_report": "write_report"},
    )

    graph.add_edge("investigate",  "write_report")
    graph.add_edge("write_report", END)

    return graph.compile()


pipeline = build_pipeline()
