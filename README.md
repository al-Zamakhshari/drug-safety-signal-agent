# Drug Safety Signal Agent — Local Edition

> Detect pharmacovigilance signals from FDA FAERS adverse event data.  
> Runs entirely on your laptop. No API keys. No cloud. No licenses.

```bash
git clone https://github.com/al-Zamakhshari/drug-safety-signal-agent
cd drug-safety-signal-agent
docker compose up -d
uv sync
uv run python -m ingestion.faers_zip_indexer --dir ~/faers_data --all-drugs
uv run python -m app.server          # → http://localhost:8080
```

---

## What It Does

Detects drug safety signals from FDA FAERS adverse event reports using a fully local pipeline — no cloud services, no API keys, no licenses.

| Stage | Method | Technology |
|-------|--------|-----------|
| Signal detection | PRR (Proportional Reporting Ratio) | OpenSearch aggregations |
| Anomaly detection | class_ratio vs therapeutic class | OpenSearch faers_ml_rates |
| Label cross-reference | Synonym-aware token-overlap | openFDA API |
| Literature evidence | PubMed search | NCBI eUtils |
| Investigation | Function calling — class effect / trend / DDI | Gemma4 E2B |
| Signal memory | Cross-run persistence | OpenSearch ML Memory (3.6+) |
| Web interface | Real-time streaming briefing | FastAPI + SSE |

**Example output — semaglutide, 82,699 reports, 11.9M baseline:**

```
### PRR Signals (EMA standard: PRR ≥ 2.0, χ²≥4)
| Reaction                           | PRR   | Sig | Reports | In FDA Label? | Literature |
|------------------------------------|-------|-----|---------|---------------|------------|
| IMPAIRED GASTRIC EMPTYING          | 82.72 | ✓   | 3,057   | Yes           | —          |
| GLYCOSYLATED HAEMOGLOBIN INCREASED | 11.54 | ✓   | 1,111   | No ⚠️         | 0 papers   |
| PANCREATITIS                       | 10.38 | ✓   | 1,504   | Yes           | 5 papers   |
| BLOOD GLUCOSE DECREASED            | 9.97  | ✓   | 1,311   | No ⚠️         | 1 paper    |

Risk: HIGH  |  Action: ESCALATE
```

---

## Stack

Everything runs locally via Docker. Zero external dependencies.

| Component | Technology | License |
|-----------|-----------|---------|
| Database | [OpenSearch 3.6.0](https://opensearch.org) | Apache 2.0 |
| LLM | Gemma4 E2B (2B active, ~3GB) via Docker Model Runner | [Gemma](https://ai.google.dev/gemma/terms) |
| Agent framework | [LangGraph](https://langchain-ai.github.io/langgraph/) | MIT |
| Web UI | FastAPI + SSE streaming | MIT |
| Ingestion | [Polars](https://pola.rs) — 3× less memory than pandas | MIT |
| Observability | [Arize Phoenix](https://phoenix.arize.com) | Apache 2.0 |
| Data | FDA openFDA API + PubMed + FDA FAERS ZIPs | Public domain |

---

## Requirements

- **Docker Desktop** with [Model Runner](https://docs.docker.com/desktop/features/model-runner/) enabled
- **Python 3.11+** with [uv](https://docs.astral.sh/uv/)
- **16GB RAM** recommended (OpenSearch 1.5GB + Gemma4 E2B ~3GB)
- **~10GB disk** for full FAERS 2018–2026 dataset

---

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Start infrastructure (downloads Gemma4 ~3GB on first run)
docker compose up -d

# 3. Load FAERS data

# Quick demo — 5 min via openFDA API
uv run python -m ingestion.faers_indexer --drug semaglutide --limit 6000
uv run python -m ingestion.faers_indexer --drug rofecoxib --limit 2000

# Full dataset — 2018–2026, ~11.9M reports, ~1 hour
./ingestion/download_faers.sh
uv run python -m ingestion.faers_zip_indexer --dir ~/faers_data --all-drugs

# Full history — adds 2004–2017 (rofecoxib peak period, ~2.8GB more)
./ingestion/download_faers_historical.sh
uv run python -m ingestion.faers_zip_indexer --dir ~/faers_data --all-drugs

# 4. Compute class-ratio (one-time, enables anomaly detection)
uv run python -m ingestion.compute_class_ratio

# 5a. Web UI
uv run python -m app.server          # → http://localhost:8080

# 5b. CLI
uv run python main.py semaglutide
uv run python main.py rofecoxib      # retrospective: recalled 2004 for MI risk

# Switch to E4B for richer prose (default is E2B — faster, smaller)
LOCAL_MODEL_NAME=docker.io/ai/gemma4:E4B uv run python main.py semaglutide
```

---

## How PRR Works

PRR uses the correct 2×2 contingency table (EMA/813938/2011):

```
PRR = (a / (a+b)) / (c / (c+d))

a = drug reports with reaction        b = drug reports without reaction
c = non-drug reports with reaction    d = non-drug reports without reaction

Signal threshold: PRR ≥ 2.0  AND  Yates χ² ≥ 4  AND  n ≥ 3  (EMA standard)
```

Key implementation details:
- Baseline fetched **per-reaction** via OpenSearch `filters` agg — not truncated to a global top-N. Rare/novel signals are never silently dropped.
- Annotated with Yates-corrected χ² (`significant: bool`) — weak signals still appear in the table with `~` badge for human review.
- Same formula runs for every drug. No drug-specific code. Independently audited by Claude Opus for overfitting/rigging.

---

## Architecture

```
LangGraph StateGraph — 10 nodes
│
├── resolve_names        RxNorm API → all brand/generic names          Python
├── load_memory          OpenSearch ML Memory → prior run findings      Python
├── calculate_prr        OpenSearch filters agg → PRR + χ²             Python
├── anomaly_detection    faers_ml_rates → class_ratio signals           Python
├── fetch_label          openFDA + synonym expansion → label text       Python
│
├── [search_lit?]        PubMed API                                     Python
│   (PRR≥3 AND χ²-significant AND unlabeled)
│
├── [investigate?]       Gemma4 E2B — function calling, temp=0          LLM
│   (PRR≥5 AND χ²-significant AND unlabeled)
│   tools: get_prr, check_class_effect, get_signal_trend,
│          compare_time_periods (DataDistributionTool),
│          search_faers (flexible DSL queries)
│
├── write_report         Gemma4 E2B — narrative prose only              LLM
│   (PRR/anomaly tables emitted by Python — numbers never re-typed by model)
│
└── save_memory          OpenSearch ML Memory → persist findings         Python
```

**Design principle:** Python owns all statistics; LLM writes prose and investigates.

---

## OpenSearch 3.6.0 Features Used

| Feature | API | Purpose |
|---------|-----|---------|
| ML Memory | `/_plugins/_ml/memory` | Signal registry — persists findings across runs |
| DataDistributionTool | `/_plugins/_ml/tools/_execute/DataDistributionTool` | Time-period divergence — finds EMERGING vs GROWING signals |
| Anomaly Detection | `/_plugins/_anomaly_detection` | RCF on class_ratio time series |
| KNN / Neural Search | `/_plugins/_knn` | Available for semantic reaction matching |
| `filters` aggregation | standard | Per-reaction baseline (no top-N truncation) |

---

## Signal Registry (ML Memory)

Every pipeline run persists its findings to OpenSearch ML Memory — one container per drug. The next run loads prior findings and the investigator agent is told what signals appeared in previous runs, enabling it to classify signals as **PERSISTENT** (seen across multiple runs = higher confidence).

```bash
# Memory is automatic — no setup needed.
# To inspect a drug's history:
curl -u admin:Pharma@2024! -k https://localhost:9200/_plugins/_ml/memory
```

---

## Label Matching

FDA labels and MedDRA PTs use different vocabulary. The matcher handles:

- **Synonym expansion**: `delays`↔`impaired`, `reduced`↔`decreased`, etc. so "delays gastric emptying" matches "IMPAIRED GASTRIC EMPTYING"
- **Sentence-aware negation**: "no evidence of pancreatitis" → `False`; "no other findings. pancreatitis occurred." → `True`
- **Direction-aware**: "BLOOD GLUCOSE DECREASED" not matched by a label that only says "blood glucose increased"
- **Word-order independent**: "PANCREATITIS ACUTE" matches "acute pancreatitis"

No MedDRA ontology required — pure string matching.

---

## Scientific Validity (Independent Audit)

The system was audited by Claude Opus for overfitting and rigging. Key findings:

| Concern | Verdict |
|---------|---------|
| PRR formula | ✅ Correct 2×2, no drug-specific code |
| Thresholds | ✅ EMA published standards (PRR≥2, χ²≥4, n≥3) — not tuned on semaglutide |
| Novel signal detection | ✅ Would detect rofecoxib MI signal with pre-2004 data |
| Label matching | ✅ Fixed — synonym expansion prevents false-novel flags |
| Class-ratio comparators | ✅ Defensible, conservative |
| Baseline self-reference | ✅ Correctly subtracted (negligible bias) |

---

## Observability

- **Web UI**: `http://localhost:8080` — real-time streaming briefing
- **Phoenix traces**: `http://localhost:6006` (start with `docker compose up phoenix`)
- **OpenSearch Dashboards**: `http://localhost:5601` (admin / Pharma@2024!)

---

## Roadmap

- [x] PRR — correct 2×2 table, per-reaction baseline, no rank truncation
- [x] Yates χ² significance annotation
- [x] FDA label cross-reference — synonym-aware, negation-aware, sentence-scoped
- [x] PubMed literature evidence
- [x] Class-ratio anomaly detection vs therapeutic class
- [x] LLM investigation — function calling with 5 tools
- [x] Deterministic table rendering — numbers never re-typed by model
- [x] Signal registry — OpenSearch ML Memory (cross-run persistence)
- [x] DataDistributionTool — time-period emergence detection
- [x] Polars ingestion — 3× less memory, handles AERS + FAERS formats
- [x] Full FAERS 2004–2026 (historical + current)
- [x] Web UI — FastAPI + SSE streaming, dark-mode
- [x] Independent scientific audit (no overfitting confirmed)
- [ ] Significance gating in routing (currently advisory only)
- [ ] Stratified PRR (by age / sex / reporter type)
- [ ] BCPNN / EBGM Bayesian shrinkage estimators

---

## Related

**Hackathon version** (Elasticsearch + Elastic ML + Kibana MCP + Gemini API):  
→ [google-cloud-rapid-agent-hackathon](https://github.com/al-Zamakhshari/google-cloud-rapid-agent-hackathon)

---

## Disclaimer

For **research purposes only**. PRR signals are statistical associations, not causal evidence. No regulatory decisions should be made based solely on this tool's output. Requires clinical validation before any regulatory action.
