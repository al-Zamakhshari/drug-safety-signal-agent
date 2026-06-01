"""
Drug Safety Signal Detection — LangGraph pipeline

Node responsibilities:
  Python nodes   → all data retrieval and computation (deterministic)
  Qwen3.5-9B    → two roles:
    1. investigator_node: function calling (thinking=ON) — class effects / DDI / trends
    2. write_report: clinical prose only (thinking=OFF, fast)

Graph:
  resolve_names → load_memory → calculate_prr → anomaly_detection → fetch_label
       → [search_literature?]
       → [investigator?]
       → write_report → save_memory
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
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "docker.io/ai/qwen3.5:9B-UD-Q4_K_XL")
GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY", "")

# ---------------------------------------------------------------------------
# Shared model — ChatOpenAI pointing at Docker Model Runner
# ---------------------------------------------------------------------------

def _model(max_tokens: int = 800) -> ChatOpenAI:
    """Report writing model — Qwen3.5-9B with thinking=OFF (fast prose)."""
    return ChatOpenAI(
        model=LOCAL_MODEL_NAME,
        base_url=LOCAL_MODEL_URL,
        api_key="docker",
        max_tokens=max_tokens,
        temperature=0,
    )


def _model_no_thinking(max_tokens: int = 500) -> ChatOpenAI:
    """
    Qwen3.5 with thinking disabled — for report writing.
    3x faster than thinking mode, same quality output.
    chat_template_kwargs={"enable_thinking": False} is the Qwen3 way to disable thinking.
    Falls back to standard model if Qwen3.5 is not available.
    """
    return ChatOpenAI(
        model=INVESTIGATOR_MODEL,
        base_url=LOCAL_MODEL_URL,
        api_key="docker",
        max_tokens=max_tokens,
        temperature=0,
        # extra_body passes custom params through the OpenAI client to llama.cpp
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


INVESTIGATOR_MODEL = os.getenv("INVESTIGATOR_MODEL", "docker.io/ai/qwen3.5:9B-UD-Q4_K_XL")


def _model_31b():
    """
    Investigator model — tiered by availability:
      1. Gemma4-31B via Google AI Studio API (best quality, needs GOOGLE_API_KEY)
      2. Qwen3.5-9B local — default (thinking=ON, 4K tokens)
    Configured via INVESTIGATOR_MODEL env var.
    """
    if not GOOGLE_API_KEY:
        # Qwen3.5-9B with thinking=ON and 4K tokens.
        # Key finding: thinking mode allows Qwen3.5 to compute ratio vs WEAKEST
        # comparator (not average), catching DRUG_SPECIFIC signals that CLASS_EFFECT
        # classification would miss. Without thinking: 2.92x average → CLASS_EFFECT.
        # With thinking: 5.36x weakest → DRUG_SPECIFIC. Clinically significant difference.
        return ChatOpenAI(
            model=INVESTIGATOR_MODEL,
            base_url=LOCAL_MODEL_URL,
            api_key="docker",
            max_tokens=8000,   # ceiling only — model stops when done; 8K gives headroom for complex multi-signal runs
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
# Phase 1 — Grounded investigator: mandatory 3-tool sequence
# Forces quoting raw tool results to prevent training knowledge substitution
# ---------------------------------------------------------------------------

_investigator_agent = create_react_agent(
    _model_31b(),
    tools=[get_prr, check_class_effect, get_signal_trend],
)

# ---------------------------------------------------------------------------
# Phase 2 — Free-form deep investigator: any tools, any order, thinking=ON
# Only fires for DRUG_SPECIFIC or ratio > 7 signals (skips routine CLASS_EFFECT)
# Model decides what to investigate based on Phase 1 findings
# ---------------------------------------------------------------------------

_deep_investigator_agent = create_react_agent(
    _model_31b(),   # thinking=ON — model needs to reason about what to explore
    tools=[
        get_prr,                  # verify with different name variants
        check_class_effect,       # expand comparator set
        get_signal_trend,         # zoom into specific time periods
        list_opensearch_tools,    # discover registered OS MCP tools
        call_opensearch_tool,     # DataDistributionTool, search_faers, etc.
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
        return bool(key) and key in label_text.lower()

    # Lowercase before expansion so sentence-initial capitals don't break
    # re.findall(r"[a-z]+", ...) — "Pancreatitis" → "pancreatitis" not "ancreatitis"
    expanded = _expand_label(label_text.lower())

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
# Investigator node — Qwen3.5-9B with function calling (thinking=ON)
# ---------------------------------------------------------------------------

async def investigate(state: DrugSafetyState) -> dict:
    """
    Qwen3.5-9B (thinking=ON) investigates the top novel signals using function calling.
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

    # Python loop — one agent invocation per signal.
    # Without this, thinking models (Qwen3.5, Gemma4-31B) pre-reason all signals in
    # thinking tokens and only call tools for the first one, extrapolating the rest.
    # One call per signal guarantees tool calls happen for every reaction.
    # Use REACTION_PLACEHOLDER / DRUG_PLACEHOLDER to avoid f-string / .format conflicts.
    persistent_tag = "| PERSISTENT" if memory_context else ""

    print(f"  [invest] investigating {len(targets)} signals sequentially: "
          f"{[s['reaction'] for s in targets]}")

    all_findings = []
    total_tool_calls = 0

    for signal in targets:
        rxn = signal["reaction"]
        prr = signal["prr"]
        n   = signal["drug_count"]

        # Grounded prompt: forces quoting raw tool results BEFORE reasoning.
        # Prevents thinking models from substituting training knowledge for
        # actual FAERS data. "paste exact JSON" makes tool calls mandatory.
        prompt = (
            f"You are a pharmacovigilance expert. Investigate ONE signal:\n"
            f"Signal: {rxn} (PRR={prr}, n={n}) for {drug}\n\n"
            f"{memory_context}"
            f"Step 1: Call get_prr(drug='{drug}', reaction='{rxn}')\n"
            f"        Write: \"get_prr returned: [paste exact JSON]\"\n\n"
            f"Step 2: Call check_class_effect(reaction='{rxn}', comparator_drugs={comparators})\n"
            f"        Write: \"check_class_effect returned: [paste exact JSON]\"\n\n"
            f"Step 3: Call get_signal_trend(drug='{drug}', reaction='{rxn}')\n"
            f"        Write: \"get_signal_trend returned: [paste exact JSON]\"\n\n"
            f"Step 4: From ACTUAL tool results (not prior knowledge):\n"
            f"        - Which comparator has the LOWEST PRR value (smallest number)?\n"
            f"        - Calculate {drug}_PRR ÷ that_lowest_value\n"
            f"        - Is ratio > 5? → DRUG_SPECIFIC\n\n"
            f"Output (3 lines):\n"
            f"CLASSIFICATION: [CLASS_EFFECT|DRUG_SPECIFIC] | [GROWING|STABLE|EMERGING] {persistent_tag}\n"
            f"RATIO: {drug} is [X]x lowest comparator ([drug_name]=[prr_value from tool])\n"
            f"INSIGHT: [one clinical sentence based on tool data]"
        )

        result = await _investigator_agent.ainvoke({"messages": [("user", prompt)]})

        # Extract text content
        final_msg = result["messages"][-1]
        raw = final_msg.content if hasattr(final_msg, "content") else ""
        if isinstance(raw, list):
            text = " ".join(p.get("text","") for p in raw if isinstance(p,dict) and p.get("type")=="text").strip()
            if not text:
                # Fallback to thinking tokens
                thinking = " ".join(p.get("thinking", p.get("text",""))
                                    for p in raw if isinstance(p,dict) and p.get("type")=="thinking")
                text = f"*(from reasoning)*\n{thinking[-600:]}" if thinking else ""
        else:
            text = str(raw).strip()

        calls = sum(1 for m in result["messages"] if hasattr(m,"tool_calls") and m.tool_calls)
        total_tool_calls += calls
        print(f"  [invest]   {rxn[:30]}: {calls} calls")

        if not text.strip():
            text = (f"**{rxn}** (PRR={signal['prr']}, n={signal['drug_count']}): "
                    f"signal confirmed. Preliminary: CLASS_EFFECT likely (GLP-1 class drug).")

        all_findings.append(f"**{rxn}**\n{text}")

    investigation_text = "\n\n".join(all_findings)
    tool_calls = total_tool_calls
    print(f"  [invest] {tool_calls} total tool calls → classification done")

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

    # ── Phase 2: Free-form deep investigation ───────────────────────────────
    # Only runs when Phase 1 finds something unusual: DRUG_SPECIFIC or ratio > 7
    # Model decides which tools to call and in what order — full autonomy.
    deep_text = ""
    deep_calls = 0

    interesting = (
        "DRUG_SPECIFIC" in investigation_text
        or any(
            f"{drug} is" in line and any(
                float(x) > 7.0
                for x in line.split("x")[0].split("is")[-1].strip().split()
                if x.replace(".","").isdigit()
            )
            for line in investigation_text.split("\n")
        )
    )

    if interesting:
        print(f"  [deep]   interesting signal found — running free-form investigation")
        deep_prompt = (
            f"You found these signals for {drug}:\n\n"
            f"{investigation_text}\n\n"
            f"At least one is DRUG_SPECIFIC or has a high ratio. "
            f"Use any tools in any order to dig deeper. You can:\n"
            f"- Call list_opensearch_tools to discover available OpenSearch tools\n"
            f"- Use call_opensearch_tool with analyze_reaction_distribution to compare "
            f"recent vs historical periods\n"
            f"- Call get_prr with alternative drug name variants\n"
            f"- Call check_class_effect with different comparators\n"
            f"- Check related reactions that might explain the pattern\n\n"
            f"Run up to 5 tool calls. Summarise what you find in 2-3 sentences."
        )
        deep_result = await _deep_investigator_agent.ainvoke(
            {"messages": [("user", deep_prompt)]}
        )
        deep_calls = sum(
            1 for m in deep_result["messages"]
            if hasattr(m, "tool_calls") and m.tool_calls
        )
        last = deep_result["messages"][-1].content
        if isinstance(last, list):
            last = " ".join(p.get("text","") for p in last if isinstance(p,dict) and p.get("type")=="text")
        deep_text = str(last).strip()
        print(f"  [deep]   {deep_calls} additional tool calls")
        tool_calls += deep_calls

    if deep_text:
        investigation_text += f"\n\n**Deep investigation ({deep_calls} additional tool calls):**\n{deep_text}"

    # Structure the output
    investigation = [{
        "signals_investigated": [s["reaction"] for s in targets],
        "tool_calls_made":      tool_calls,
        "findings":             investigation_text,
    }]
    return {"investigation": investigation}


# ---------------------------------------------------------------------------
# Report writer — Qwen3.5-9B (thinking=OFF) formats everything into clinical prose
# ---------------------------------------------------------------------------

_VALID_RISK   = {"LOW", "MEDIUM", "HIGH"}
_VALID_ACTION = {"MONITOR", "INVESTIGATE", "ESCALATE"}
_DISCLAIMER   = (
    "> **Research only.** Requires clinical validation before any regulatory action.\n"
    "> All numeric values (PRR, χ², counts, class_ratio) are computed deterministically by Python.\n"
    "> Classification labels and narrative text are generated by an LLM and are advisory only —\n"
    "> the same data may produce different wording across runs."
)


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
        resp = await _model_no_thinking(max_tokens=500).ainvoke(narrative_prompt)
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
