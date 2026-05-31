# Drug Safety Signal Agent — Local Edition

> Detect pharmacovigilance signals from FDA FAERS data.  
> Runs entirely on your laptop. No API keys. No cloud. No licenses.

```bash
git clone https://github.com/akhallaf/drug-safety-signal-agent
cd drug-safety-signal-agent
docker compose up -d          # starts OpenSearch + Phoenix
uv run python main.py semaglutide
```

---

## What It Does

Analyzes FDA adverse event reports (FAERS) to detect drug safety signals using:

- **PRR (Proportional Reporting Ratio)** — EMA gold-standard statistical method
- **FDA label cross-reference** — distinguishes known vs novel signals  
- **PubMed literature search** — evidence for unlabeled signals

Example output:
```
## Drug Safety Briefing: SEMAGLUTIDE
FAERS reports analysed: 47,453  |  Method: PRR (EMA standard)

### Signals Detected (PRR ≥ 2.0, n ≥ 3)
| Reaction                  | PRR  | Reports | In FDA Label? | Literature |
|---------------------------|------|---------|---------------|------------|
| OPTIC ISCHAEMIC NEUROPATHY| 99.2 | 31      | No ⚠️         | 4 papers   |
| PANCREATITIS              | 11.4 | 847     | Yes           | 28 papers  |
| INTESTINAL OBSTRUCTION    | 5.1  | 124     | No ⚠️         | 2 papers   |

Risk: HIGH  |  Action: INVESTIGATE
```

---

## Stack

Everything runs locally via Docker. Zero external dependencies.

| Component | Technology | License |
|-----------|-----------|---------|
| Database | [OpenSearch 2.x](https://opensearch.org) | Apache 2.0 |
| Dashboard | OpenSearch Dashboards | Apache 2.0 |
| LLM | Gemma4 E4B (4B params) via Docker Model Runner | [Gemma](https://ai.google.dev/gemma/terms) |
| Agent framework | [Google ADK](https://github.com/google/adk-python) | Apache 2.0 |
| Observability | [Arize Phoenix](https://phoenix.arize.com) | Apache 2.0 |
| Data sources | FDA openFDA API + PubMed | Public domain |

---

## Requirements

- **Docker Desktop** with [Model Runner](https://docs.docker.com/desktop/features/model-runner/) enabled
- **Python 3.11+** with [uv](https://docs.astral.sh/uv/)
- **16GB RAM** recommended (OpenSearch 512MB + Gemma4 ~4GB)

---

## Setup

```bash
# 1. Install Python dependencies
uv sync

# 2. Start infrastructure (first run downloads Gemma4 ~4GB)
docker compose up -d

# 3. Load FAERS data — pick a tier:

# Tier 1: Quick demo (~8k reports via API, 5 min)
uv run python -m ingestion.faers_indexer --drug semaglutide --limit 6000
uv run python -m ingestion.faers_indexer --drug rofecoxib --limit 2000

# Tier 2: Full dataset (11.9M reports, 2018-2026, ~2GB, accurate PRR)
./ingestion/download_faers.sh
uv run python -m ingestion.faers_zip_indexer --dir ~/faers_data --all-drugs

# 4. Run
uv run python main.py semaglutide
uv run python main.py rofecoxib   # retrospective: recalled 2004 for MI risk
```

---

## How PRR Works

PRR measures whether a drug is over-represented in adverse event reports:

```
PRR = (drug_count / drug_total) / (baseline_count / faers_total)

Signal threshold: PRR ≥ 2.0 AND count ≥ 3  (EMA standard)
```

Unlike the cloud version, PRR is computed directly via OpenSearch aggregations — no LLM query generation means no syntax bugs and fully reproducible results.

---

## Architecture

```
User query
    │
    ▼
drug_safety_pipeline (SequentialAgent)
    ├── prr_analyst       → OpenSearch aggregations → PRR table
    ├── label_analyst     → openFDA API → labeled reactions
    ├── literature_analyst → PubMed API → evidence for novel signals
    └── report_writer     → Gemma4 E4B → final briefing
```

The LLM (Gemma4) acts as coordinator and report writer. All statistical work — PRR calculation, FDR correction, anomaly detection — happens in OpenSearch or Python code, making results model-independent and reproducible.

---

## Observability

Phoenix traces every LLM call at `http://localhost:6006`.  
OpenSearch Dashboards at `http://localhost:5601` (admin / Admin@changeme1).

---

## Roadmap

- [x] PRR signal detection via OpenSearch aggregations
- [x] FDA label cross-reference
- [x] PubMed literature evidence
- [x] Gemma4 E4B local inference
- [ ] OpenSearch Anomaly Detection (ML signals)
- [ ] Signal registry with status tracking (NEW/VALIDATED/DISMISSED)
- [ ] Web UI

---

## Related

**Hackathon version** (Elasticsearch + Elastic ML + Kibana MCP + Gemini API):  
→ [google-cloud-rapid-agent-hackathon](https://github.com/akhallaf/google-cloud-rapid-agent-hackathon)

---

## Disclaimer

This tool is for **research purposes only**. PRR signals are statistical associations, not causal evidence. No regulatory decisions should be made based solely on this tool's output.
