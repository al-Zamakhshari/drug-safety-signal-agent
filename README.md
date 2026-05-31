# Drug Safety Signal Agent — Local Edition

> Detect pharmacovigilance signals from FDA FAERS adverse event data.  
> Runs entirely on your laptop. No API keys. No cloud. No licenses.

```bash
git clone https://github.com/al-Zamakhshari/drug-safety-signal-agent
cd drug-safety-signal-agent
docker compose up -d
uv sync
uv run python -m ingestion.faers_zip_indexer --dir ~/faers_data --all-drugs
uv run python main.py semaglutide
```

---

## What It Does

Analyzes FDA adverse event reports (FAERS) to detect drug safety signals:

1. **PRR (Proportional Reporting Ratio)** — EMA gold-standard, correct 2×2 contingency table, per-reaction baseline (no rank truncation)
2. **Class-ratio anomaly detection** — compares drug reaction rate against therapeutic class baseline
3. **FDA label cross-reference** — token-overlap matching distinguishes known vs novel signals
4. **PubMed literature search** — evidence for unlabeled signals
5. **LLM investigation** — Gemma4 autonomously calls tools to classify signals (class effect / drug-specific / growing)

Example output (semaglutide, 82,699 reports, 11.9M baseline):
```
### PRR Signals (EMA standard: PRR ≥ 2.0)
| Reaction                          | PRR   | Reports | In FDA Label? | Literature |
|-----------------------------------|-------|---------|---------------|------------|
| IMPAIRED GASTRIC EMPTYING         | 82.72 | 3,057   | Yes           | —          |
| GLYCOSYLATED HAEMOGLOBIN INCREASED| 11.54 | 1,111   | No ⚠️         | 0 papers   |
| PANCREATITIS                      | 10.38 | 1,504   | Yes           | 5 papers   |
| BLOOD GLUCOSE DECREASED           | 9.97  | 1,311   | No ⚠️         | 1 paper    |

Risk: HIGH  |  Action: ESCALATE
```

---

## Stack

Everything runs locally via Docker. Zero external dependencies.

| Component | Technology | License |
|-----------|-----------|---------|
| Database | [OpenSearch 3.6.0](https://opensearch.org) | Apache 2.0 |
| LLM | Gemma4 E2B (2B active params) via Docker Model Runner | [Gemma](https://ai.google.dev/gemma/terms) |
| Agent framework | [LangGraph](https://langchain-ai.github.io/langgraph/) | MIT |
| Ingestion | [Polars](https://pola.rs) — 3× less memory than pandas | MIT |
| Observability | [Arize Phoenix](https://phoenix.arize.com) | Apache 2.0 |
| Data sources | FDA openFDA API + PubMed + FDA FAERS ZIPs | Public domain |

---

## Requirements

- **Docker Desktop** with [Model Runner](https://docs.docker.com/desktop/features/model-runner/) enabled
- **Python 3.11+** with [uv](https://docs.astral.sh/uv/)
- **16GB RAM** recommended (OpenSearch 1.5GB + Gemma4 E2B ~3GB)

---

## Setup

```bash
# 1. Install Python dependencies
uv sync

# 2. Start infrastructure (first run downloads Gemma4 ~3GB)
docker compose up -d

# 3. Load FAERS data

# Quick demo (~8k reports via API, 5 min)
uv run python -m ingestion.faers_indexer --drug semaglutide --limit 6000
uv run python -m ingestion.faers_indexer --drug rofecoxib --limit 2000

# Full dataset (11.9M reports, 2018-2026, ~1 hour)
./ingestion/download_faers.sh
uv run python -m ingestion.faers_zip_indexer --dir ~/faers_data --all-drugs

# Full history including 2004-2017 (adds rofecoxib peak, ~2.8GB more)
./ingestion/download_faers_historical.sh
uv run python -m ingestion.faers_zip_indexer --dir ~/faers_data --all-drugs

# 4. (Optional) Compute class-ratio for anomaly detection
uv run python -m ingestion.compute_class_ratio

# 5. Run
uv run python main.py semaglutide
uv run python main.py rofecoxib   # retrospective: recalled 2004 for MI risk

# Use E4B for richer prose (E2B default, 1.5x faster)
LOCAL_MODEL_NAME=docker.io/ai/gemma4:E4B uv run python main.py semaglutide
```

---

## How PRR Works

PRR uses the correct 2×2 contingency table (EMA/813938/2011):

```
PRR = (a / (a+b)) / (c / (c+d))

a = drug reports with reaction        b = drug reports without reaction
c = non-drug reports with reaction    d = non-drug reports without reaction

Signal threshold: PRR ≥ 2.0 AND n ≥ 3  (EMA standard)
```

The baseline is fetched **per-reaction** via an OpenSearch `filters` aggregation — not truncated to a global top-N. Rare/novel signals (the ones that matter most) are never silently dropped.

---

## Architecture

```
LangGraph StateGraph
│
├── resolve_names       Python — RxNorm API → brand names        no LLM
├── calculate_prr       Python — OpenSearch filters agg → PRR    no LLM
├── anomaly_detection   Python — faers_ml_rates → class_ratio    no LLM
├── fetch_label         Python — openFDA API + token matcher      no LLM
│
├── [search_lit?]       Python — PubMed API                      no LLM
│   (only if PRR≥3 unlabeled signals)
│
├── [investigate?]      Gemma4 E2B + 4 tools                     ✅ LLM
│   (only if PRR≥5 unlabeled signals)
│   tools: get_prr, check_class_effect, get_signal_trend
│
└── write_report        Gemma4 E2B — narrative prose only         ✅ LLM
    (PRR/anomaly tables emitted by Python — never re-typed by model)
```

**Design principle:** Python computes all statistics; LLM writes prose and investigates.
The reported PRR values are guaranteed exact — the model only writes the narrative around them.

---

## Class-Ratio Anomaly Detection

After ingestion, compute the class-ratio time series:

```bash
uv run python -m ingestion.compute_class_ratio
```

This computes quarterly `class_ratio = drug_rate / mean(comparator_rates)` for each (drug, reaction) pair. A `class_ratio >> 1` indicates a drug-specific signal not explained by class effects. Results are stored in the `faers_ml_rates` index and queried by the pipeline alongside PRR.

---

## Observability

Phoenix traces every LLM call at `http://localhost:6006` (if running).  
OpenSearch Dashboards at `http://localhost:5601` (admin / Pharma@2024!).

---

## Roadmap

- [x] PRR signal detection — correct 2×2 table, per-reaction baseline
- [x] FDA label cross-reference — token-overlap, no MedDRA ontology needed
- [x] PubMed literature evidence
- [x] Gemma4 local inference (E2B default, E4B option)
- [x] Class-ratio anomaly detection vs therapeutic class
- [x] LLM investigation with function calling (class effect / DDI / trend)
- [x] Deterministic table rendering (numbers guaranteed exact)
- [x] Polars-based ingestion (3× less memory than csv.DictReader)
- [x] Full FAERS 2004-2026 support (AERS + FAERS format)
- [ ] Signal registry (NEW/VALIDATED/DISMISSED status tracking)
- [ ] Web UI
- [ ] Significance gating (χ² / confidence interval)

---

## Related

**Hackathon version** (Elasticsearch + Elastic ML + Kibana MCP + Gemini API):  
→ [google-cloud-rapid-agent-hackathon](https://github.com/al-Zamakhshari/google-cloud-rapid-agent-hackathon)

---

## Disclaimer

For **research purposes only**. PRR signals are statistical associations, not causal evidence. No regulatory decisions should be made based solely on this tool's output.
