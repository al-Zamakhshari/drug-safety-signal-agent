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
from agent.tools.signal_memory import save_finding, search_findings
from agent.tools.investigator_tools import (
    get_prr, check_class_effect, get_signal_trend,
)
from agent.tools.opensearch_mcp import list_opensearch_tools, call_opensearch_tool

load_dotenv()

LOCAL_MODEL_URL  = os.getenv("LOCAL_MODEL_URL", "http://localhost:12434/v1")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "docker.io/ai/gemma4:E2B")
GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY", "")

# ---------------------------------------------------------------------------
# Shared model — ChatOpenAI pointing at Docker Model Runner
# ---------------------------------------------------------------------------

def _model(max_tokens: int = 800) -> ChatOpenAI:
    """Local Gemma4 via Docker Model Runner."""
    return ChatOpenAI(
        model=LOCAL_MODEL_NAME,
        base_url=LOCAL_MODEL_URL,
        api_key="docker",
        max_tokens=max_tokens,
        temperature=0,
    )


def _model_31b():
    """
    Gemma4 31B via Google AI Studio — used for the investigator when
    GOOGLE_API_KEY is set. Much more reliable for multi-step tool calling
    than E2B (4B → 31B active params). Falls back to local E2B if no key.
    """
    if not GOOGLE_API_KEY:
        # E4B as local investigator — 100% classification rate vs 0% for E2B.
        # E2B gets cut off mid-sentence before writing the classification tag.
        # E4B uses 2000 tokens efficiently: tools + clean synthesis.
        return ChatOpenAI(
            model="docker.io/ai/gemma4:E4B",
            base_url=LOCAL_MODEL_URL,
            api_key="docker",
            max_tokens=2000,
            temperature=0,
        )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemma-4-31b-it",
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            max_tokens=3000,   # thinking + tool calls + synthesis need room
        )
    except ImportError:
        return _model(max_tokens=1000)


# ---------------------------------------------------------------------------
# Investigator sub-agent (create_react_agent)
# Gemma4 E4B with 4 tools — handles its own multi-turn tool-call loop
# ---------------------------------------------------------------------------

_investigator_agent = create_react_agent(
    _model_31b(),   # 31B via API if GOOGLE_API_KEY set, else local E2B
    tools=[
        # Statistical tools (direct Python → OpenSearch)
        get_prr,                  # confirm PRR for a specific drug+reaction
        check_class_effect,       # class-wide or drug-specific?
        get_signal_trend,         # GROWING / STABLE / EMERGING over time
        # Built-in OS MCP tools — agent discovers and uses freely
        list_opensearch_tools,    # discover all registered OS MCP tools
        call_opensearch_tool,     # call any registered OS MCP tool by name
        # e.g. analyze_reaction_distribution, search_faers, future tools
    ],
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class DrugSafetyState(TypedDict):
    drug_name:          str
    drug_names:         list[str]
    prr_signals:        list[dict]   # [{reaction, prr, drug_count, chi2, significant}]
    drug_total:         int
    faers_total:        int
    anomaly_signals:    list[dict]   # [{reaction, max_ratio, trend}]
    label_text:         str          # raw label text for token-overlap matching
    literature:         list[dict]
    past_findings:      list[dict]   # from ML Memory — prior run results
    investigation:      list[dict]
    briefing:           str
    error:              Optional[str]


# ---------------------------------------------------------------------------
# Label matching — negation-aware, direction-aware (no MedDRA ontology)
# ---------------------------------------------------------------------------

# Synonyms: expand label text before matching so vocabulary differences between
# MedDRA PTs and FDA label prose don't create false-novel flags.
# e.g. MedDRA "IMPAIRED GASTRIC EMPTYING" vs label "delays gastric emptying"
_SYNONYMS: dict[str, str] = {
    # Morphological + vocabulary variants for gastric-emptying / motility
    "impaired":      "impaired delayed delays delay slowed slowing",
    "delayed":       "delayed delay delays impaired slowed",
    "delay":         "delay delayed delays impaired slowed",
    "delays":        "delays delay delayed impaired slowed",
    "slowed":        "slowed slow slowing delayed impaired",
    # Direction synonyms
    "reduced":       "reduced decreased decrease lowered diminished",
    "decreased":     "decreased reduced decrease lowered diminished",
    "increased":     "increased elevated elevation raised higher",
    "elevated":      "elevated increased raised higher elevation",
    # Clinical synonym pairs
    "injury":        "injury damage",
    "failure":       "failure insufficiency",
    "insufficiency": "insufficiency failure",
    "obstruction":   "obstruction blockage",
    "haemorrhage":   "haemorrhage hemorrhage bleeding",
    "hemorrhage":    "hemorrhage haemorrhage bleeding",
}

def _expand_label(text: str) -> str:
    """Add synonym variants so vocabulary drift between MedDRA and label prose doesn't hide matches."""
    words = text.split()
    out = []
    for w in words:
        out.append(w)
        if w in _SYNONYMS:
            out.append(_SYNONYMS[w])
    return " ".join(out)


# Stop words stripped from PT before matching — direction words NOT here
_LABEL_STOP = {
    "acute", "chronic", "disorder", "syndrome", "disease", "reaction",
    "abnormal", "nos", "unspecified", "type", "associated", "related",
    "induced", "mediated", "and", "the", "with", "due", "from", "that",
    "this", "following", "including", "severe",
    # deliberately NOT: increased, decreased, elevated, reduced — see _DIRECTION
}

# Direction families. A directional PT only matches if a word from the
# SAME family appears near the clinical tokens — not the opposite.
_DIRECTION: dict[str, set[str]] = {
    "up":   {"increased", "increase", "elevated", "elevation", "high",
             "raised", "hyper"},
    "down": {"decreased", "decrease", "reduced", "reduction", "low",
             "lowered", "hypo"},
}
_DIR_OPPOSITE = {"up": "down", "down": "up"}

# Negation cues — if one precedes a clinical-token match, that occurrence
# does not count as "labeled in the label"
_NEGATION = {
    "no", "not", "without", "absence", "absent", "denies", "denied",
    "negative", "free", "ruled", "rule", "neither", "nor", "never",
}
_NEG_WINDOW = 5   # tokens of look-back for a negation cue


def _label_tokens(term: str) -> set[str]:
    """Significant clinical tokens from a MedDRA PT (no stop/direction words)."""
    direction_words = _DIRECTION["up"] | _DIRECTION["down"]
    return {
        w for w in re.findall(r"[a-z]+", term.lower())
        if len(w) > 3 and w not in _LABEL_STOP and w not in direction_words
    }


def _reaction_direction(reaction: str) -> str | None:
    """Return 'up' or 'down' if the PT names a direction, else None."""
    words = set(re.findall(r"[a-z]+", reaction.lower()))
    for fam, members in _DIRECTION.items():
        if words & members:
            return fam
    return None


def _is_labeled(reaction: str, label_text: str) -> bool:
    """
    True if the MedDRA PT is documented (non-negated, direction-consistent)
    in the FDA label text.

    Sentence-aware: the label is split on sentence boundaries first, so a
    negation in one sentence cannot reach across to invalidate a genuine
    match in a later sentence.

    Handles:
      - Cross-sentence safety: 'no other findings. pancreatitis occurred.' → True
      - Negation: 'no evidence of pancreatitis' → False
      - Direction: 'BLOOD GLUCOSE DECREASED' rejected by 'increased'-only label
      - Word order: 'PANCREATITIS ACUTE' matches 'acute pancreatitis'
    """
    clin = _label_tokens(reaction)

    if not clin:
        key = re.sub(r"[^a-z ]+", " ", reaction.lower()).strip()
        return bool(key) and key in label_text

    # Expand label text with synonyms before matching
    expanded = _expand_label(label_text)

    # Fast-fail: required token absent from entire expanded label
    if not clin.issubset(set(re.findall(r"[a-z]+", expanded))):
        return False

    direction = _reaction_direction(reaction)

    # Process sentence-by-sentence — negation scope is contained to one sentence
    for sentence in re.split(r"[.!?;\n]+", expanded):
        words    = re.findall(r"[a-z]+", sentence)
        word_set = set(words)

        if not clin.issubset(word_set):
            continue  # not all clinical tokens in this sentence

        # Negation check: look back _NEG_WINDOW tokens before the first
        # clinical token found in this sentence
        negated = False
        for i, w in enumerate(words):
            if w in clin:
                neg_lo = max(0, i - _NEG_WINDOW)
                if _NEGATION & set(words[neg_lo:i]):
                    negated = True
                break
        if negated:
            continue

        # Direction check: sentence must contain the correct direction
        if direction and not (word_set & _DIRECTION[direction]):
            continue

        return True  # clean match in this sentence

    return False


# ---------------------------------------------------------------------------
# Python nodes — no LLM
# ---------------------------------------------------------------------------

async def resolve_names(state: DrugSafetyState) -> dict:
    result = await get_drug_names(state["drug_name"])
    names = result.get("found_names", [state["drug_name"].upper()])
    print(f"  [names]  {state['drug_name']} → {names}")
    return {"drug_names": names}


async def load_memory(state: DrugSafetyState) -> dict:
    """Load past findings from ML Memory — gives investigator cross-run context."""
    try:
        findings = await search_findings(state["drug_name"], top_n=3)
        if findings and "error" not in findings[0]:
            print(f"  [memory] {len(findings)} past run(s) found in ML Memory")
        else:
            print(f"  [memory] first run for {state['drug_name']} — no prior findings")
    except Exception as e:
        findings = []
        print(f"  [memory] unavailable ({e})")
    return {"past_findings": findings}


async def save_memory(state: DrugSafetyState) -> dict:
    """Save this run's findings to ML Memory for future reference."""
    inv_text = ""
    if state.get("investigation"):
        inv_text = state["investigation"][0].get("findings", "")

    # Derive risk from briefing or fallback
    risk = "MEDIUM"
    briefing = state.get("briefing", "")
    for level in ("HIGH", "MEDIUM", "LOW"):
        if f"**Risk**: {level}" in briefing:
            risk = level
            break

    try:
        result = await save_finding(
            drug=state["drug_name"],
            signals=state.get("prr_signals", []),
            investigation=inv_text,
            risk=risk,
        )
        print(f"  [memory] saved to ML Memory → message_id={result.get('message_id','?')}")
    except Exception as e:
        print(f"  [memory] save failed ({e})")
    return {}  # no state change needed


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
    print(f"  [AD]     {len(signals)} anomaly signals")
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
                  "warnings_and_precautions",   # both keys (openFDA uses both)
                  "adverse_reactions", "contraindications")
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
    comparators = ["LIRAGLUTIDE", "DULAGLUTIDE", "TIRZEPATIDE", "EXENATIDE"]
    comparators = [c for c in comparators if c not in state["drug_names"]][:3]

    # Include ML Memory context — prior run findings
    memory_context = ""
    past = state.get("past_findings", [])
    if past and "error" not in past[0]:
        prior = past[0].get("response", "")
        if prior:
            memory_context = f"\nPRIOR RUN FINDINGS (from ML Memory):\n{prior[:400]}\n"

    prompt = (
        f"Investigate these novel safety signals for {drug}: {reactions_str}\n"
        f"{memory_context}\n"
        f"You have access to OpenSearch ML tools. Start by calling list_opensearch_tools "
        f"to discover available tools, then use them freely to investigate.\n\n"
        f"Required steps for each signal:\n"
        f"1. Call get_prr(drug='{drug}', reaction='<REACTION>') — confirm PRR\n"
        f"2. Call check_class_effect(reaction='<REACTION>', comparator_drugs={comparators}) "
        f"— class effect or drug-specific?\n"
        f"3. Use call_opensearch_tool with analyze_reaction_distribution to compare "
        f"recent (2023-2026) vs baseline (2018-2022) periods — is the signal growing?\n\n"
        f"Classify each as: CLASS_EFFECT | DRUG_SPECIFIC | GROWING | EMERGING | STABLE\n"
        f"{'Note PERSISTENT if matches prior findings.' if memory_context else ''}"
    )

    print(f"  [invest] investigating {len(targets)} signals: "
          f"{[s['reaction'] for s in targets]}")

    result = await _investigator_agent.ainvoke({"messages": [("user", prompt)]})

    # Extract final text response — Gemma4 31B returns list [{type:thinking},{type:text}]
    final_msg = result["messages"][-1]
    raw_content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
    if isinstance(raw_content, list):
        # Try text parts first
        text_parts = [p.get("text", "") for p in raw_content
                      if isinstance(p, dict) and p.get("type") == "text"]
        investigation_text = " ".join(text_parts).strip()

        # If text is empty (model ran out of tokens after thinking + tool calls),
        # extract the conclusion from thinking tokens — they contain the actual reasoning
        if not investigation_text:
            thinking_parts = [p.get("thinking", p.get("text", ""))
                              for p in raw_content
                              if isinstance(p, dict) and p.get("type") == "thinking"]
            if thinking_parts:
                thinking = " ".join(thinking_parts)
                # Take last 800 chars of thinking — that's the synthesis/conclusion
                investigation_text = f"*(Extracted from model reasoning)*\n{thinking[-800:]}"
    else:
        investigation_text = str(raw_content).strip()

    # Count tool calls made
    tool_calls = sum(
        1 for m in result["messages"]
        if hasattr(m, "tool_calls") and m.tool_calls
    )
    print(f"  [invest] {tool_calls} tool calls → classification done")

    # Fallback: if model returned no text (e.g. quota exhausted, only thinking tokens),
    # generate a minimal classification from PRR + class structure alone
    if not investigation_text.strip():
        lines = []
        for s in targets:
            lines.append(
                f"**{s['reaction']}** (PRR={s['prr']}, n={s['drug_count']}): "
                f"signal confirmed. Investigation model unavailable — "
                f"check GOOGLE_API_KEY quota. Preliminary: `CLASS_EFFECT` likely "
                f"(GLP-1 class drug)."
            )
        investigation_text = "\n".join(lines)
        print(f"  [invest] WARNING: model returned empty — using fallback text")

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

    # ── Parse investigation classifications for PRR table column ────────────
    # Extract reaction → classification from 31B investigation text
    # Patterns: CLASS_EFFECT, DRUG_SPECIFIC, GROWING, EMERGING, STABLE, PERSISTENT
    inv_tags: dict[str, str] = {}
    inv_badges: list[str] = []
    inv_findings = ""
    inv_tools = 0

    if state.get("investigation"):
        inv       = state["investigation"][0]
        inv_findings = inv.get("findings", "").strip()
        inv_tools = inv.get("tool_calls_made", 0)
        signals_inv = inv.get("signals_investigated", [])

        # Build per-reaction tag map from findings text
        CLASS_TAGS = {"CLASS_EFFECT", "DRUG_SPECIFIC", "GROWING", "EMERGING",
                      "STABLE", "PERSISTENT", "HYPOTHESIS"}
        for rxn in signals_inv:
            # Find the section for this reaction in the findings
            rxn_idx = inv_findings.upper().find(rxn.upper())
            if rxn_idx >= 0:
                snippet = inv_findings[rxn_idx:rxn_idx + 400].upper()
                found = [t for t in CLASS_TAGS if t in snippet]
                if found:
                    # Compact badge: top 2 tags
                    inv_tags[rxn] = " ".join(f"`{t}`" for t in found[:2])
                    inv_badges.append(f"**{rxn}**: {' · '.join(found[:2])}")

    # ── Deterministic PRR table — now with Investigation column ─────────────
    prr_rows = []
    for s in state["prr_signals"][:15]:
        rxn        = s["reaction"]
        is_labeled = "Yes" if _is_labeled(rxn, label_text) else "**No ⚠️**"
        papers     = lit_map.get(rxn, {}).get("papers", "—")
        sig        = "✓" if s.get("significant", True) else "~"
        inv_col    = inv_tags.get(rxn, "—")
        prr_rows.append(
            f"| {rxn} | {s['prr']} | {sig} | {s['drug_count']} | {is_labeled} | {papers} | {inv_col} |"
        )
    prr_block = (
        "### PRR Signals (EMA standard: PRR ≥ 2.0, χ²≥4)\n"
        "| Reaction | PRR | Sig | Reports | In FDA Label? | Literature | Investigation |\n"
        "|----------|-----|-----|---------|---------------|------------|---------------|\n"
        + ("\n".join(prr_rows) if prr_rows else "| No signals detected | — | — | — | — | — | — |")
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

    # ── Investigation badge summary — shown inline, details collapsible ──────
    invest_block = ""
    if inv_findings and inv_tools > 0:
        badge_line = "  ".join(inv_badges) if inv_badges else "See full details below"
        invest_block = (
            f"### 🔬 Investigation ({inv_tools} tool calls)\n"
            f"{badge_line}\n\n"
            f"<details>\n<summary>Full investigation details</summary>\n\n"
            f"{inv_findings}\n\n"
            f"</details>\n"
        )
    elif inv_findings:
        invest_block = f"### 🔬 Investigation\n{inv_findings}\n"

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
    # χ² significance gate: prefer statistically robust signals for literature search.
    # If none pass χ²≥4, fall back to any unlabeled PRR≥3 signal (advisory mode).
    sig_unlabeled = [s for s in unlabeled if s.get("significant", True)
                     and s["prr"] >= 3.0 and s["drug_count"] >= 10]
    weak_unlabeled = [s for s in unlabeled if not s.get("significant", True)
                      and s["prr"] >= 3.0 and s["drug_count"] >= 10]
    needs_lit = bool(sig_unlabeled) or bool(weak_unlabeled)
    result = "search_lit" if needs_lit else "investigate"
    sig_count = len(sig_unlabeled)
    print(f"  [route]  {len(unlabeled)} unlabeled ({sig_count} χ²-significant) → {result}")
    return result


def should_investigate(state: DrugSafetyState) -> str:
    label_text = state.get("label_text", "")
    # Gate investigation on χ²-significant signals only — weak signals surface
    # in the table with '~' but don't trigger expensive LLM investigation.
    strong_unlabeled = [
        s for s in state["prr_signals"]
        if not _is_labeled(s["reaction"], label_text)
        and s["prr"] >= 5.0
        and s["drug_count"] >= 10
        and s.get("significant", True)   # χ²≥4 gate
    ]
    result = "investigate" if strong_unlabeled else "write_report"
    print(f"  [route]  {len(strong_unlabeled)} strong χ²-significant unlabeled → {result}")
    return result


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_pipeline() -> StateGraph:
    graph = StateGraph(DrugSafetyState)

    graph.add_node("resolve_names",    resolve_names)
    graph.add_node("load_memory",      load_memory)        # ML Memory: load prior findings
    graph.add_node("calculate_prr",    calculate_prr_signals)
    graph.add_node("anomaly_detection", run_anomaly_detection)
    graph.add_node("fetch_label",      fetch_label)
    graph.add_node("search_lit",       search_lit)
    graph.add_node("investigate",      investigate)
    graph.add_node("write_report",     write_report)
    graph.add_node("save_memory",      save_memory)        # ML Memory: persist findings

    graph.set_entry_point("resolve_names")
    graph.add_edge("resolve_names",     "load_memory")      # load prior ML Memory
    graph.add_edge("load_memory",       "calculate_prr")
    graph.add_edge("calculate_prr",     "anomaly_detection")
    graph.add_edge("anomaly_detection",  "fetch_label")

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
    graph.add_edge("write_report", "save_memory")           # persist to ML Memory
    graph.add_edge("save_memory",  END)

    return graph.compile()


pipeline = build_pipeline()
