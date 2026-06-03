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
from langchain.agents import create_agent as create_react_agent   # LangGraph V2 migration
from langchain_openai import ChatOpenAI

from agent.tools.prr import calculate_prr, get_drug_names
from agent.tools.openfda import get_drug_label
from agent.tools.pubmed import search_literature
from agent.tools.anomaly_signals import get_anomaly_signals
from agent.tools.signal_memory import (
    save_finding, search_findings,
    load_last_run, save_run_signals, build_memory_context,
)
from agent.tools.investigator_tools import (
    get_prr, check_class_effect, get_signal_trend, compare_time_periods,
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
# Only fires for DRUG_SPECIFIC or ratio > 5 signals (skips routine CLASS_EFFECT)
# Model decides what to investigate based on Phase 1 findings
# ---------------------------------------------------------------------------

_deep_investigator_agent = create_react_agent(
    _model_31b(),   # thinking=ON — model needs to reason about what to explore
    tools=[
        get_prr,                  # verify with different name variants
        check_class_effect,       # expand comparator set
        get_signal_trend,         # zoom into specific time periods
        compare_time_periods,     # typed DataDistributionTool wrapper — EMERGING/GROWING test
        list_opensearch_tools,    # discover registered OS MCP tools
        call_opensearch_tool,     # search_faers, AD tools, etc.
    ],
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class DrugSafetyState(TypedDict):
    drug_name:          str
    drug_names:         list[str]
    prr_signals:        list[dict]   # [{reaction, prr, prr_lower, prr_upper, robust, q_value, fdr_significant, ebgm, eb05, ic, ic025, ...}]
    drug_total:         int
    faers_total:        int
    prr_strat_field:    Optional[str]  # which field was used for stratified PRR (None = unstratified)
    anomaly_signals:    list[dict]   # [{reaction, class_ratio, class_ratio_lower, class_ratio_upper, ...}]
    label_text:         str          # raw label text for token-overlap matching
    llt_expansions:     dict         # PT → [LLT synonyms], pre-fetched from openFDA
    literature:         list[dict]
    past_findings:      list[dict]   # from ML Memory — prior run results
    investigation:      list[dict]
    classifications:    list[dict]   # per-reaction parsed Phase-1 output [{reaction, effect, trend, ratio, persistent}]
    signal_status:      list[dict]   # cross-run status [{reaction, status, prr, prr_prior, effect, trend}]
    _prior_run:         Optional[dict]  # raw prior run doc from agent-signal-runs (internal)
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


def _is_labeled(
    reaction: str,
    label_text: str,
    llt_expansions: dict | None = None,
) -> bool:
    """
    True if the MedDRA PT is documented (non-negated, direction-consistent)
    in the FDA label text.

    Sentence-aware: the label is split on sentence boundaries first, so a
    negation in one sentence cannot reach across to invalidate a genuine
    match in a later sentence.

    Handles:
      - MedDRA LLT expansion: if llt_expansions[PT] is provided, also tries
        matching each LLT synonym. E.g. "MYOCARDIAL INFARCTION" → also tries
        "HEART ATTACK", "MI" etc. Fixes false-novel flags for off-demo drugs.
      - Cross-sentence safety: 'no other findings. pancreatitis occurred.' → True
      - Negation: 'no evidence of pancreatitis' → False
      - Direction: 'BLOOD GLUCOSE DECREASED' rejected by 'increased'-only label
      - Word order: 'PANCREATITIS ACUTE' matches 'acute pancreatitis'
    """
    # Try the PT first, then any LLT synonyms
    candidates = [reaction]
    if llt_expansions:
        candidates += llt_expansions.get(reaction.upper(), [])

    for candidate in candidates:
        if _is_labeled_single(candidate, label_text):
            return True
    return False


def _is_labeled_single(reaction: str, label_text: str) -> bool:
    """Core single-term label match (no LLT expansion)."""
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


def _label_match_display(
    reaction: str,
    label_text: str,
    llt_expansions: dict | None = None,
) -> str:
    """
    Three-state label match display for the report table.
    Yes          → exact match (non-negated, direction-consistent, incl. LLT synonyms)
    **Possible** → partial match (fewer than all clinical tokens matched)
    **No ⚠️**   → no match found
    """
    if _is_labeled(reaction, label_text, llt_expansions):
        return "Yes"
    # Partial check: does ANY single clinical token appear in the label?
    clin = _label_tokens(reaction)
    if clin:
        expanded = _expand_label(label_text.lower())
        label_words = set(re.findall(r"[a-z]+", expanded))
        if clin & label_words:   # at least one token present
            return "**Possible**"
    return "**No ⚠️**"


def _lifecycle_status(
    c_prr: float,
    c_lo: float | None,
    c_up: float | None,
    p_prr: float,
    p_lo: float | None,
    p_up: float | None,
) -> str:
    """
    Determine cross-run signal lifecycle status.

    Extracted as a pure function so tests call the real implementation rather
    than duplicating the logic. Used by classify_signals.

    Returns: "VALIDATED" | "DISMISSED"  (NEW is handled upstream when p is None)

    Logic:
      DISMISSED:  current upper CI entirely below prior lower CI (genuine collapse)
      VALIDATED:  CIs overlap, or any CI missing and PRR ≥ 50% of prior
    """
    if None in (c_lo, c_up, p_lo, p_up):
        # CI missing on one run — 50% point-estimate fallback
        return "VALIDATED" if c_prr >= p_prr * 0.5 else "DISMISSED"
    if c_up < p_lo:
        # Current CI entirely below prior CI — genuine signal collapse
        return "DISMISSED"
    # CIs overlap — change is within sampling noise
    return "VALIDATED"


# ---------------------------------------------------------------------------
# Phase-1 output parser — robust structured extraction
# ---------------------------------------------------------------------------

# Tokens expected on the CLASSIFICATION line.
# Ordered tuples — DRUG_SPECIFIC first so it wins when both tokens appear in one line.
_EFFECT_TOKENS = ("DRUG_SPECIFIC", "CLASS_EFFECT")
_TREND_TOKENS  = ("GROWING", "EMERGING", "STABLE")


def _parse_classification(reaction: str, text: str) -> dict:
    """
    Parse the structured Phase-1 investigator output for ONE reaction section.

    Expected format (from the investigation prompt):
      CLASSIFICATION: [CLASS_EFFECT|DRUG_SPECIFIC] | [GROWING|STABLE|EMERGING] | PERSISTENT
      RATIO: {drug} is [X]x lowest comparator ([name]=[value])
      INSIGHT: ...

    Returns {effect, trend, ratio, persistent} with None where not found.
    Rejects template echoes (lines containing '[' — model echoed the prompt).
    """
    out: dict = {"reaction": reaction, "effect": None, "trend": None,
                 "ratio": None, "persistent": False}

    m = re.search(r"CLASSIFICATION:\s*([^\n]+)", text, re.IGNORECASE)
    if m:
        line = m.group(1).upper()
        # Reject template echoes: the alternation pattern [X|Y|Z] (with | inside brackets)
        # but allow real output that incidentally contains [] e.g. "DRUG_SPECIFIC [confirmed]"
        if not re.search(r"\[[A-Z_]+\|", line):
            for tok in _EFFECT_TOKENS:
                if re.search(rf"\b{tok}\b", line):
                    out["effect"] = tok
                    break
            for tok in _TREND_TOKENS:
                if re.search(rf"\b{tok}\b", line):
                    out["trend"] = tok
                    break
            out["persistent"] = bool(re.search(r"\bPERSISTENT\b", line))

    # RATIO line: capture the numeric value immediately before 'x'
    # Handles: "5.36x", "5.36 x", "is 5.36x lowest", "drug is 7x"
    rm = re.search(r"RATIO:[^\n]*?(\d+(?:\.\d+)?)\s*x\b", text, re.IGNORECASE)
    if rm:
        try:
            out["ratio"] = float(rm.group(1))
        except ValueError:
            pass

    return out


# ---------------------------------------------------------------------------
# Python nodes — no LLM
# ---------------------------------------------------------------------------

async def resolve_names(state: DrugSafetyState) -> dict:
    result = await get_drug_names(state["drug_name"])
    names = result.get("found_names", [state["drug_name"].upper()])
    print(f"  [names]  {state['drug_name']} → {names}")
    return {"drug_names": names}


async def load_memory(state: DrugSafetyState) -> dict:
    """
    Load past findings from both memory stores:
      - ML Memory text trail (for investigator prose context)
      - Structured run store (for per-reaction PRR delta / trajectory)
    """
    findings: list[dict] = []
    prior_run: dict | None = None
    try:
        findings = await search_findings(state["drug_name"], top_n=3)
        if findings and "error" not in findings[0]:
            print(f"  [memory] {len(findings)} past run(s) found in ML Memory")
        else:
            print(f"  [memory] first run for {state['drug_name']} — no prior findings")
            findings = []
    except Exception as e:
        print(f"  [memory] unavailable ({e})")

    try:
        prior_run = await load_last_run(state["drug_name"])
        if prior_run:
            n_prior = len(prior_run.get("signals", []))
            print(f"  [memory] prior run loaded ({n_prior} signals for trajectory diff)")
    except Exception:
        pass

    return {"past_findings": findings, "_prior_run": prior_run}


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
    # stratify_by can be set via env var for advanced usage
    # Default: "reporter_type" — highest-yield FAERS confounder (consumer vs physician)
    # Override: STRATIFY_PRR=age | sex | reporter_type | "" (empty = disable)
    import os as _os
    stratify_by_env = _os.getenv("STRATIFY_PRR", "reporter_type").strip() or None
    result = await calculate_prr(state["drug_names"], top_n=50, stratify_by=stratify_by_env)
    signals = result.get("signals", [])
    strat = result.get("stratify_by")
    print(f"  [PRR]    {result['drug_total']:,} reports | "
          f"{result['faers_total']:,} total | {len(signals)} signals"
          + (f" | stratified by {strat}" if strat else ""))
    return {
        "prr_signals":     signals,
        "drug_total":      result["drug_total"],
        "faers_total":     result["faers_total"],
        "prr_strat_field": strat,
    }


async def run_anomaly_detection(state: DrugSafetyState) -> dict:
    """Query OpenSearch AD for class_ratio anomalies. Pure Python — no LLM."""
    # Use canonical drug name (as indexed in faers_ml_rates), not brand names
    drug = state["drug_name"].upper()
    result = await get_anomaly_signals(drug, min_ratio_lower=1.0, min_count=5, top_n=15)
    signals = result.get("signals", [])
    print(f"  [AD]     {len(signals)} within-class signals")
    if signals:
        top3 = [(s["reaction"], s.get("class_ratio_lower", s.get("class_ratio")))
                for s in signals[:3]]
        print(f"           top (by CI lower): {top3}")
    return {"anomaly_signals": signals}


async def fetch_label(state: DrugSafetyState) -> dict:
    """
    Fetch FDA label and store raw text for token-overlap matching.
    Also pre-fetches MedDRA LLT synonyms for the top PRR signals so that
    _is_labeled can match FDA label vocabulary without requiring a full MedDRA license.
    """
    from agent.tools.openfda import get_meddra_llts

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

    # Pre-fetch LLT synonyms for top signals (cached locally after first fetch)
    llt_expansions: dict[str, list[str]] = {}
    top_reactions = [s["reaction"] for s in state.get("prr_signals", [])[:20]]
    for rxn in top_reactions:
        llts = await get_meddra_llts(rxn)
        if llts:
            llt_expansions[rxn] = llts
    if llt_expansions:
        print(f"  [label]  LLT expansions loaded for {len(llt_expansions)} reactions")

    return {"label_text": label_text, "llt_expansions": llt_expansions}


async def search_lit(state: DrugSafetyState) -> dict:
    label_text     = state.get("label_text", "")
    llt_expansions = state.get("llt_expansions", {})
    # Top 3 unlabeled signals by PRR
    targets = [s for s in state["prr_signals"]
               if not _is_labeled(s["reaction"], label_text, llt_expansions)][:3]
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
    Runs get_prr, check_class_effect, get_signal_trend autonomously (Phase 1).
    Phase 2 adds compare_time_periods, list_opensearch_tools, call_opensearch_tool.
    Returns structured classification for each investigated signal.
    """
    label_text     = state.get("label_text", "")
    llt_expansions = state.get("llt_expansions", {})
    # Only investigate strong unlabeled signals (PRR≥5, n≥10)
    targets = [
        s for s in state["prr_signals"]
        if not _is_labeled(s["reaction"], label_text, llt_expansions)
        and s["prr"] >= 5.0
        and s["drug_count"] >= 10
    ][:3]

    if not targets:
        print("  [invest] no strong unlabeled signals to investigate")
        return {"investigation": []}

    drug = state["drug_names"][0]
    reactions_str = ", ".join(f"{s['reaction']} (PRR={s['prr']})" for s in targets)

    # Derive comparators from config/comparators.yaml for this drug.
    # Falls back to an empty list (the investigator will still run — it just
    # won't have a class_effect check context). Far better than hardcoding
    # GLP-1 comparators for every drug, which would compare rofecoxib against
    # semaglutide (scientifically wrong).
    comparators: list[str] = []
    try:
        from pathlib import Path
        import yaml as _yaml
        _cfg_path = Path(__file__).parent.parent / "config" / "comparators.yaml"
        if _cfg_path.exists():
            _cfg = _yaml.safe_load(_cfg_path.read_text()) or {}
            # Find the entry whose names list contains this drug
            drug_upper = drug.upper()
            for _entry in _cfg.values():
                if drug_upper in [n.upper() for n in _entry.get("names", [])]:
                    # Flatten comparator groups → unique names, exclude this drug
                    seen: set[str] = set(n.upper() for n in _entry.get("names", []))
                    for grp in _entry.get("comparators", []):
                        for n in grp:
                            nu = n.upper()
                            if nu not in seen:
                                comparators.append(nu)
                                seen.add(nu)
                    break
    except Exception:
        pass
    comparators = comparators[:3]  # top 3 to keep the prompt concise

    # Build memory context — prefer structured PRR trajectory from prior run,
    # fall back to truncated text from ML Memory if no structured data available.
    prior_run = state.get("_prior_run")
    memory_context = build_memory_context(state["prr_signals"], prior_run)
    if not memory_context:
        past = state.get("past_findings", [])
        if past and "error" not in past[0]:
            prose = past[0].get("response", "")
            if prose:
                memory_context = f"\nPRIOR RUN FINDINGS:\n{prose[:400]}\n"

    # Python loop — one agent invocation per signal.
    # Without this, thinking models (Qwen3.5, Gemma4-31B) pre-reason all signals in
    # thinking tokens and only call tools for the first one, extrapolating the rest.
    # One call per signal guarantees tool calls happen for every reaction.
    persistent_tag = "| PERSISTENT" if memory_context else ""

    print(f"  [invest] investigating {len(targets)} signals sequentially: "
          f"{[s['reaction'] for s in targets]}")

    all_findings = []
    total_tool_calls = 0
    classifications: list[dict] = []   # structured per-reaction parse results

    for signal in targets:
        rxn       = signal["reaction"]
        prr       = signal["prr"]
        prr_lo    = signal.get("prr_lower", "?")
        prr_hi    = signal.get("prr_upper", "?")
        q_val     = signal.get("q_value", "?")
        n         = signal["drug_count"]

        # Grounded prompt: forces quoting raw tool results BEFORE reasoning.
        # Prevents thinking models from substituting training knowledge for
        # actual FAERS data. "paste exact JSON" makes tool calls mandatory.
        # New: also surface the 95% CI and BH q-value so the model's reasoning
        # is grounded in the corrected statistics (not just the point PRR).
        prompt = (
            f"You are a pharmacovigilance expert. Investigate ONE signal:\n"
            f"Signal: {rxn} for {drug}\n"
            f"  PRR={prr} (95% CI: {prr_lo}–{prr_hi})  n={n}  BH q={q_val}\n\n"
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
            f"INSIGHT: [one clinical sentence based on tool data and the CI/q provided above]"
        )

        # Try primary investigator (Gemma4-31B or local); fall back to local on API errors.
        # Google 500/503 errors are ServerError — not retried by tenacity, so we catch here.
        try:
            result = await _investigator_agent.ainvoke({"messages": [("user", prompt)]})
        except Exception as api_err:
            err_str = str(api_err).lower()
            if any(k in err_str for k in ("500", "503", "internal", "unavailable", "resource_exhausted", "429")):
                print(f"  [invest]   {rxn[:30]}: API error ({type(api_err).__name__}) — falling back to local model")
                _local_inv = create_react_agent(_model(max_tokens=2000), tools=[get_prr, check_class_effect, get_signal_trend])
                result = await _local_inv.ainvoke({"messages": [("user", prompt)]})
            else:
                raise

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
            ci_str = (f"95% CI: {signal.get('prr_lower','?')}–{signal.get('prr_upper','?')}"
                      if signal.get("prr_lower") else "")
            text = (f"**{rxn}** (PRR={signal['prr']}{', ' + ci_str if ci_str else ''}, "
                    f"n={signal['drug_count']}): signal confirmed. "
                    f"Investigation inconclusive — see PRR/CI table for numeric evidence.")

        all_findings.append(f"**{rxn}**\n{text}")

        # Parse structured output immediately while section text is in scope
        parsed = _parse_classification(rxn, text)
        classifications.append(parsed)

    investigation_text = "\n\n".join(all_findings)
    tool_calls = total_tool_calls
    print(f"  [invest] {tool_calls} total tool calls → classification done")

    # Fallback: if model returned no text (e.g. quota exhausted, only thinking tokens),
    # generate a minimal classification from PRR + class structure alone
    if not investigation_text.strip():
        lines = []
        for s in targets:
            ci_str = (f", 95% CI: {s.get('prr_lower','?')}–{s.get('prr_upper','?')}"
                      if s.get("prr_lower") else "")
            lines.append(
                f"**{s['reaction']}** (PRR={s['prr']}{ci_str}, n={s['drug_count']}): "
                f"signal confirmed. Investigation model unavailable — "
                f"see PRR/CI table for numeric evidence."
            )
        investigation_text = "\n".join(lines)
        print(f"  [invest] WARNING: model returned empty — using fallback text")

    # ── Phase 2: Free-form deep investigation ───────────────────────────────
    # Fires when Phase 1 finds DRUG_SPECIFIC OR ratio > 5 (same cutoff as the
    # Phase-1 prompt's own DRUG_SPECIFIC threshold — avoids the old dead band
    # where ratio 5–7 was DRUG_SPECIFIC by label but skipped Phase 2).
    # Uses the structured _parse_classification results, not fragile text scraping.
    deep_text = ""
    deep_calls = 0

    interesting = any(
        c["effect"] == "DRUG_SPECIFIC"
        or (c["ratio"] is not None and c["ratio"] > 5.0)
        for c in classifications
    )

    if interesting:
        print(f"  [deep]   interesting signal found — running free-form investigation")
        deep_prompt = (
            f"You found these signals for {drug}:\n\n"
            f"{investigation_text}\n\n"
            f"At least one is DRUG_SPECIFIC or has a high ratio. "
            f"Use any tools in any order to dig deeper. Suggested approaches:\n"
            f"- Call compare_time_periods(drug, reaction, recent_start, recent_end, "
            f"baseline_start, baseline_end) to test whether the signal is EMERGING "
            f"(absent in baseline) or GROWING. Use ISO dates, e.g. 2024-01-01.\n"
            f"- Call call_opensearch_tool('list_anomaly_detectors', '{{}}') then "
            f"call_opensearch_tool('get_anomaly_results', '{{\"anomalyGradeThreshold\": 0.7}}') "
            f"to confirm WHICH time window the class-ratio anomaly peaked.\n"
            f"- Call get_prr with alternative drug name variants\n"
            f"- Call check_class_effect with different comparator drugs\n"
            f"- Search related reactions that might explain the pattern\n\n"
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
    return {"investigation": investigation, "classifications": classifications}


async def classify_signals(state: DrugSafetyState) -> dict:
    """
    Cross-run signal lifecycle classification.

    Diffs current PRR signals against the prior run to assign:
      NEW        — reaction not seen in any prior run
      VALIDATED  — seen in prior run, PRR still elevated (≥50% of prior value)
      DISMISSED  — prior signal confirmed gone: current upper CI is entirely below
                   the prior lower CI (genuine collapse, not sampling noise)

    Uses a CI-overlap test when both runs have confidence intervals:
      VALIDATED when CIs overlap (change is within sampling noise)
      DISMISSED only when current upper CI < prior lower CI (genuine collapse)
      Falls back to a 50% point-estimate rule when CIs are absent.

    Persists the structured run document (including prr_upper) for the next
    run to diff against. Never raises — logs its own failures.
    """
    prior_run = state.get("_prior_run")
    prior_map = {s["reaction"]: s for s in (prior_run.get("signals", []) if prior_run else [])}
    cls_map   = {c["reaction"]: c for c in state.get("classifications", [])}
    label_text = state.get("label_text", "")
    llt_expansions = state.get("llt_expansions", {})

    signal_status: list[dict] = []
    for s in state["prr_signals"]:
        rxn   = s["reaction"]
        p     = prior_map.get(rxn)
        c     = cls_map.get(rxn, {})
        c_prr = s["prr"]
        p_prr = (p.get("prr") or 0) if p else 0

        if p is None:
            status = "NEW"
        else:
            # CI-overlap test via _lifecycle_status — the pure helper tests call directly
            status = _lifecycle_status(
                c_prr = c_prr,
                c_lo  = s.get("prr_lower"),
                c_up  = s.get("prr_upper"),
                p_prr = p_prr,
                p_lo  = p.get("prr_lower"),
                p_up  = p.get("prr_upper"),
            )

        signal_status.append({
            "reaction":   rxn,
            "prr":        c_prr,
            "prr_lower":  s.get("prr_lower"),
            "prr_upper":  s.get("prr_upper"),   # persisted so next run can do CI overlap
            "drug_count": s["drug_count"],
            "effect":     c.get("effect"),
            "trend":      c.get("trend"),
            "labeled":    _is_labeled(rxn, label_text, llt_expansions),
            "status":     status,
            "prr_prior":  p_prr if p else None,
        })

    # Persist structured run doc — save_run_signals logs its own failures, never raises
    await save_run_signals(state["drug_name"], signal_status)

    new_count  = sum(1 for s in signal_status if s["status"] == "NEW")
    val_count  = sum(1 for s in signal_status if s["status"] == "VALIDATED")
    dism_count = sum(1 for s in signal_status if s["status"] == "DISMISSED")
    print(f"  [status] NEW={new_count} VALIDATED={val_count} DISMISSED={dism_count}")

    return {"signal_status": signal_status}


# ---------------------------------------------------------------------------
# Report writer — Qwen3.5-9B (thinking=OFF) formats everything into clinical prose
# ---------------------------------------------------------------------------

_VALID_RISK   = {"LOW", "MEDIUM", "HIGH"}
_VALID_ACTION = {"MONITOR", "INVESTIGATE", "ESCALATE"}
_DISCLAIMER   = (
    "> **Research only.** Requires clinical validation before any regulatory action.\n"
    "> All numeric values (PRR, ROR, EBGM, EB05, 95% CI, BH q-value, MH rate ratio, counts) are "
    "computed deterministically by Python and are fully reproducible. "
    "Formulas follow EMA/813938/2011 and DuMouchel (1999).\n"
    "> Classification labels (DRUG_SPECIFIC / CLASS_EFFECT) and Key Findings text are generated by "
    "an LLM and are advisory only — the same data may produce different wording across runs."
)


async def write_report(state: DrugSafetyState) -> dict:
    """
    Phase 3.1 fix: deterministic sections emitted by Python, LLM writes prose only.
    Clinical numbers (PRR, counts) are never re-typed by the model.
    """
    label_text     = state.get("label_text", "")
    llt_expansions = state.get("llt_expansions", {})
    lit_map        = {l["signal"]: l for l in state.get("literature", [])}
    drug           = state["drug_name"].upper()

    # ── Deterministic header (Python, not LLM) ──────────────────────────────
    # Include cross-run signal status summary if available
    status_map = {s["reaction"]: s["status"] for s in state.get("signal_status", [])}
    new_count   = sum(1 for v in status_map.values() if v == "NEW")
    val_count   = sum(1 for v in status_map.values() if v == "VALIDATED")
    dism_count  = sum(1 for v in status_map.values() if v == "DISMISSED")
    status_note = ""
    if status_map:
        status_note = (f"  |  **Signals**: {new_count} NEW · "
                       f"{val_count} VALIDATED · {dism_count} DISMISSED")

    header = (
        f"## Drug Safety Briefing: {drug}\n"
        f"**FAERS reports analysed**: {state['drug_total']:,}  |  "
        f"**Index**: {state['faers_total']:,}{status_note}\n"
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

        # Build per-reaction tag map from findings text.
        # Split on **REACTION** section headers (written by the investigate loop
        # at all_findings.append(f"**{rxn}**\n{text}")) so each reaction's tags
        # are extracted only from its own section — not from adjacent reactions.
        CLASS_TAGS = {"CLASS_EFFECT", "DRUG_SPECIFIC", "GROWING", "EMERGING",
                      "STABLE", "PERSISTENT", "HYPOTHESIS"}
        import re as _re
        # Split the full findings into per-reaction sections
        # Headers look like: **BLOOD GLUCOSE INCREASED** or **PANCREATITIS**
        sections = _re.split(r"\*\*([A-Z][A-Z0-9 /()-]+)\*\*", inv_findings)
        # sections alternates: [preamble, rxn1, text1, rxn2, text2, ...]
        section_map: dict[str, str] = {}
        for i in range(1, len(sections) - 1, 2):
            section_map[sections[i].strip()] = sections[i + 1].upper()

        for rxn in signals_inv:
            # Look for exact match, then case-insensitive fallback
            snippet = section_map.get(rxn, section_map.get(rxn.upper(), ""))
            if not snippet:
                # Fallback: first 400 chars after the reaction name in full text
                idx = inv_findings.upper().find(rxn.upper())
                snippet = inv_findings[idx:idx + 400].upper() if idx >= 0 else ""
            found = [t for t in CLASS_TAGS if t in snippet]
            if found:
                inv_tags[rxn] = " ".join(f"`{t}`" for t in found[:2])
                inv_badges.append(f"**{rxn}**: {' · '.join(found[:2])}")

    # ── Deterministic PRR table — PRR + 95% CI + BH q-value + Investigation ──
    prr_rows = []
    for s in state["prr_signals"][:15]:
        rxn        = s["reaction"]
        is_labeled = _label_match_display(rxn, label_text, llt_expansions)
        papers     = lit_map.get(rxn, {}).get("papers", "—")
        # Significance badges: χ² + robust (CI lower>1) + FDR
        chi2_badge = "✓" if s.get("significant") else "~"
        robust_col = "✓" if s.get("robust") else "~"
        fdr_col    = "✓" if s.get("fdr_significant") else "~"
        prr_ci_col = (f"{s.get('prr_lower','?')}–{s.get('prr_upper','?')}"
                      if s.get("prr_lower") is not None else "—")
        ror_col    = str(s.get("ror", "—"))
        ror_ci_col = (f"{s.get('ror_lower','?')}–{s.get('ror_upper','?')}"
                      if s.get("ror_lower") is not None else "—")
        q_col      = str(s.get("q_value", "—"))
        ebgm_col   = str(s.get("ebgm", "—"))
        eb05_col   = str(s.get("eb05", "—"))
        eb05_flag  = "✓" if s.get("eb05_signal") else ("~" if s.get("eb05") else "—")
        ic_col     = str(s.get("ic",   "—"))
        ic025_col  = str(s.get("ic025","—"))
        ic_flag    = "✓" if s.get("ic_signal") else ("~" if s.get("ic") else "—")
        inv_col    = inv_tags.get(rxn, "—")
        # Cross-run status badge: NEW / VALIDATED / DISMISSED
        run_status = status_map.get(rxn, "")
        status_badge = {"NEW": "🆕", "VALIDATED": "✅", "DISMISSED": "📉"}.get(run_status, "")
        # Stratified PRR (MH) — only shown when stratify_by was requested
        mh_col = ""
        if s.get("prr_strat_field"):
            mh_lo = s.get("prr_mh_lower")
            mh_hi = s.get("prr_mh_upper")
            mh_ci = f"{mh_lo}–{mh_hi}" if mh_lo is not None else "—"
            mh_val = str(s.get("prr_mh", "—"))
            mh_col = f" | {mh_val} ({mh_ci})"
        prr_rows.append(
            f"| {rxn} {status_badge} | {s['prr']} ({prr_ci_col}) | {ror_col} ({ror_ci_col}) | "
            f"{ebgm_col} / {eb05_col} {eb05_flag} | "
            f"{ic_col} / {ic025_col} {ic_flag} | "
            f"{chi2_badge}{robust_col}{fdr_col} | "
            f"{s['drug_count']} | {is_labeled} | {papers} | {inv_col}{mh_col} |"
        )
    # Determine if stratified PRR is available and which field was used
    strat_field = state.get("prr_strat_field") or (
        state["prr_signals"][0].get("prr_strat_field") if state["prr_signals"] else None
    )
    strat_header = f" | PRR_MH ({strat_field})" if strat_field else ""
    strat_sep    = " |------" if strat_field else ""

    prr_block = (
        "### PRR + ROR + EBGM + BCPNN Signals (EMA/FDA/WHO standards)\n"
        f"| Reaction | PRR (95% CI) | ROR (95% CI) | EBGM / EB05 | IC / IC025 | χ²/CI/FDR | Reports | In FDA Label? | Lit | Investigation{strat_header} |\n"
        f"|----------|-------------|-------------|-------------|-----------|-----------|---------|---------------|-----|---------------{strat_sep}|\n"
        + ("\n".join(prr_rows) if prr_rows else "| No signals detected | — | — | — | — | — | — | — | — | — |")
    )

    # ── Deterministic within-class disproportionality table (Python, not LLM) ─
    anomaly_rows = []
    for s in state.get("anomaly_signals", [])[:8]:
        trend   = s.get("trend", "—")
        rr      = s.get("class_ratio", "—")
        rr_lo   = s.get("class_ratio_lower", "?")
        rr_hi   = s.get("class_ratio_upper", "?")
        ci_str  = f"{rr_lo}–{rr_hi}" if rr_lo != "?" else "—"
        robust  = "✓" if s.get("class_ratio_robust") else "~"
        anomaly_rows.append(
            f"| {s['reaction']} | {rr} | {ci_str} | {robust} | {s.get('drug_count','—')} | {trend} |"
        )
    anomaly_block = (
        "### Within-class Disproportionality (pooled rate ratio vs comparator class)\n"
        "| Reaction | Rate Ratio | 95% CI | Robust | Count | Trend |\n"
        "|----------|-----------|--------|--------|-------|-------|\n"
        + ("\n".join(anomaly_rows) if anomaly_rows
           else "| (run: uv run python -m ingestion.compute_class_ratio) | — | — | — | — | — |")
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
         "prr_lower": s.get("prr_lower"), "prr_upper": s.get("prr_upper"),
         "q_value": s.get("q_value"),
         "labeled": _is_labeled(s["reaction"], label_text, llt_expansions),
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
            f"* {len([s for s in state['prr_signals'] if not _is_labeled(s['reaction'], label_text, llt_expansions)])} "
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
    label_text     = state.get("label_text", "")
    llt_expansions = state.get("llt_expansions", {})
    unlabeled = [s for s in state["prr_signals"]
                 if not _is_labeled(s["reaction"], label_text, llt_expansions)]
    # Primary gate: robust signals (lower 95% CI > 1.0) — small-n penalised.
    # Fallback: χ²≥4 significant signals when no CI available (old schema).
    robust_unlabeled = [s for s in unlabeled
                        if s.get("robust", s.get("significant", True))
                        and s["prr"] >= 3.0 and s["drug_count"] >= 10]
    needs_lit = bool(robust_unlabeled)
    result = "search_lit" if needs_lit else "investigate"
    print(f"  [route]  {len(unlabeled)} unlabeled ({len(robust_unlabeled)} robust) → {result}")
    return result


def should_investigate(state: DrugSafetyState) -> str:
    label_text     = state.get("label_text", "")
    llt_expansions = state.get("llt_expansions", {})
    # Gate investigation on signals that pass BOTH:
    #   1. robust (PRR lower CI > 1.0) — excludes small-n noise
    #   2. fdr_significant (BH q < 0.05) — controls family-wise false positives
    # Fall back to χ²≥4 gate when CI/FDR fields absent (old schema or m=0).
    strong_unlabeled = [
        s for s in state["prr_signals"]
        if not _is_labeled(s["reaction"], label_text, llt_expansions)
        and s["prr"] >= 5.0
        and s["drug_count"] >= 10
        and s.get("robust", s.get("significant", True))           # CI lower > 1.0
        and s.get("fdr_significant", s.get("significant", True))  # BH q < 0.05
    ]
    result = "investigate" if strong_unlabeled else "write_report"
    print(f"  [route]  {len(strong_unlabeled)} robust+FDR-significant unlabeled → {result}")
    return result


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_pipeline() -> StateGraph:
    graph = StateGraph(DrugSafetyState)

    graph.add_node("resolve_names",     resolve_names)
    graph.add_node("load_memory",       load_memory)          # ML Memory: load prior findings + structured run
    graph.add_node("calculate_prr",     calculate_prr_signals)
    graph.add_node("anomaly_detection", run_anomaly_detection)
    graph.add_node("fetch_label",       fetch_label)
    graph.add_node("search_lit",        search_lit)
    graph.add_node("investigate",       investigate)
    graph.add_node("classify_signals",  classify_signals)     # cross-run NEW/VALIDATED/DISMISSED
    graph.add_node("write_report",      write_report)
    graph.add_node("save_memory",       save_memory)          # ML Memory: persist findings

    graph.set_entry_point("resolve_names")
    graph.add_edge("resolve_names",     "load_memory")
    graph.add_edge("load_memory",       "calculate_prr")
    graph.add_edge("calculate_prr",     "anomaly_detection")
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
        {"investigate": "investigate", "write_report": "classify_signals"},
    )

    # classify_signals always runs (whether or not investigation fired)
    graph.add_edge("investigate",       "classify_signals")
    graph.add_edge("classify_signals",  "write_report")
    graph.add_edge("write_report",      "save_memory")        # persist to ML Memory
    graph.add_edge("save_memory",       END)

    return graph.compile()


pipeline = build_pipeline()
