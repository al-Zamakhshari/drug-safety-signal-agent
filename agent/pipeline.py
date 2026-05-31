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
    get_prr, check_class_effect, check_ddi, get_signal_trend
)

load_dotenv()

LOCAL_MODEL_URL = os.getenv("LOCAL_MODEL_URL", "http://localhost:12434/v1")

# ---------------------------------------------------------------------------
# Shared model — ChatOpenAI pointing at Docker Model Runner
# ---------------------------------------------------------------------------

def _model(max_tokens: int = 800) -> ChatOpenAI:
    return ChatOpenAI(
        model="docker.io/ai/gemma4:E4B",
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
    tools=[get_prr, check_class_effect, check_ddi, get_signal_trend],
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
    labeled_reactions:  list[str]
    literature:         list[dict]
    investigation:      list[dict]
    briefing:           str
    error:              Optional[str]


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
    label = await get_drug_label(state["drug_name"].lower())
    reactions: list[str] = []
    for section in ("boxed_warning", "warnings_and_cautions", "adverse_reactions"):
        text = " ".join(label.get(section, []))
        reactions.extend(re.findall(r'\b[A-Z][A-Z\s]{3,}\b', text.upper()))
    reactions = list({r.strip() for r in reactions if 3 < len(r.strip()) < 60})
    print(f"  [label]  {len(reactions)} labeled reactions")
    return {"labeled_reactions": reactions}


async def search_lit(state: DrugSafetyState) -> dict:
    labeled = set(state.get("labeled_reactions", []))
    # Top 3 unlabeled signals by PRR
    targets = [s for s in state["prr_signals"] if s["reaction"] not in labeled][:3]
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
    labeled = set(state.get("labeled_reactions", []))
    # Only investigate strong unlabeled signals (PRR≥5, n≥10)
    targets = [
        s for s in state["prr_signals"]
        if s["reaction"] not in labeled
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

async def write_report(state: DrugSafetyState) -> dict:
    labeled = set(state.get("labeled_reactions", []))
    lit_map = {l["signal"]: l for l in state.get("literature", [])}

    # Build PRR table
    rows = []
    for s in state["prr_signals"][:15]:
        rxn = s["reaction"]
        is_labeled = "Yes" if rxn in labeled else "**No ⚠️**"
        papers = lit_map.get(rxn, {}).get("papers", "—")
        rows.append(f"| {rxn} | {s['prr']} | {s['drug_count']} | {is_labeled} | {papers} |")

    prr_table = "\n".join(rows) if rows else "| No signals | — | — | — | — |"

    # Investigation findings (if any)
    invest_section = ""
    if state.get("investigation"):
        inv = state["investigation"][0]
        findings = inv.get("findings", "").strip()
        tools_used = inv.get("tool_calls_made", 0)
        signals_inv = inv.get("signals_investigated", [])
        if findings and findings != "[Empty response from model]":
            invest_section = (
                f"\n\nINVESTIGATION ({tools_used} tool calls on {signals_inv}):\n"
                f"{findings}\n"
                f"Use this to fill the Investigation Results section."
            )

    # Build anomaly signals summary
    anomaly_rows = ""
    for s in state.get("anomaly_signals", [])[:8]:
        grade = f"{s['anomaly_grade']:.2f}" if s.get("anomaly_grade") else "pending"
        anomaly_rows += f"| {s['reaction']} | {s['max_ratio']} | {s['max_count']} | {grade} |\n"

    prompt = f"""Write a drug safety briefing for {state['drug_name'].upper()}.

DATA:
- Drug reports in FAERS: {state['drug_total']:,}
- Total FAERS index: {state['faers_total']:,}
- PRR signals (≥2.0): {len(state['prr_signals'])}
- Anomaly signals (class_ratio): {len(state.get('anomaly_signals', []))}
{invest_section}

PRR TABLE (pre-computed — copy into briefing):
| Reaction | PRR | Reports | In FDA Label? | Literature |
|----------|-----|---------|---------------|------------|
{prr_table}

ANOMALY DETECTION (class_ratio vs comparator class — higher = drug-specific):
| Reaction | Max class_ratio | Count | Anomaly grade |
|----------|----------------|-------|---------------|
{anomaly_rows if anomaly_rows else "| (detector still training) | — | — | — |"}

WRITE THIS FORMAT:
## Drug Safety Briefing: {state['drug_name'].upper()}
**FAERS reports analysed**: {state['drug_total']:,}  |  **Index**: {state['faers_total']:,}

### PRR Signals (EMA standard: PRR ≥ 2.0)
[copy PRR table above]

### Anomaly Detection (class_ratio vs drug class)
[copy anomaly table above — note which reactions appear in BOTH PRR and AD]

### Investigation Results
[summarise investigation findings if available]

### Key Findings
2-3 bullets: highlight signals appearing in BOTH PRR AND anomaly detection (highest confidence).

**Risk**: LOW/MEDIUM/HIGH
**Action**: MONITOR/INVESTIGATE/ESCALATE

> Research only. Requires clinical validation."""

    print(f"  [report] writing briefing (~{len(prompt)//4} tokens)...")

    try:
        resp = await _model(max_tokens=2000).ainvoke(prompt)
        briefing = resp.content or ""
        if "<think>" in briefing:
            briefing = briefing.split("</think>")[-1].strip()
        if not briefing:
            briefing = "[Empty response from model]"
    except Exception as e:
        briefing = f"[Report generation failed: {e}]"

    return {"briefing": briefing}


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def should_search_literature(state: DrugSafetyState) -> str:
    labeled = set(state.get("labeled_reactions", []))
    unlabeled = [s for s in state["prr_signals"] if s["reaction"] not in labeled]
    needs_lit = unlabeled and any(s["prr"] >= 3.0 and s["drug_count"] >= 10 for s in unlabeled)
    result = "search_lit" if needs_lit else "investigate"
    print(f"  [route]  {len(unlabeled)} unlabeled → {result}")
    return result


def should_investigate(state: DrugSafetyState) -> str:
    labeled = set(state.get("labeled_reactions", []))
    strong_unlabeled = [
        s for s in state["prr_signals"]
        if s["reaction"] not in labeled
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
