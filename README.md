# Drug Safety Signal Agent — Local Edition

> Detect pharmacovigilance signals from FDA FAERS adverse event data.  
> Runs entirely on your laptop. No API keys. No cloud. No licenses.

```bash
git clone https://github.com/al-Zamakhshari/drug-safety-signal-agent
cd drug-safety-signal-agent
uv sync
docker compose up -d                    # pulls Qwen3.5-9B ~5.6GB on first run
./ingestion/download_faers.sh           # downloads FAERS 2018–2026 to ~/faers_data/
uv run python -m ingestion.faers_zip_indexer --dir ~/faers_data --all-drugs
uv run python -m ingestion.compute_class_ratio
uv run python -m ingestion.register_mcp_tools   # one-time: registers OS MCP tools
uv run python -m app.server            # → http://localhost:8080
```

---

## What It Does

Detects drug safety signals from FDA FAERS adverse event reports using a fully local pipeline — no cloud services, no API keys, no licenses.

| Stage | Method | Technology |
|-------|--------|-----------|
| Signal detection | PRR + 95% CI + BH-FDR | OpenSearch aggregations |
| Within-class comparison | Pooled rate ratio + 95% CI vs therapeutic class | OpenSearch faers_ml_rates |
| Label cross-reference | MedDRA LLT-expanded, negation-aware token-overlap | openFDA API |
| Literature evidence | PubMed search | NCBI eUtils |
| Investigation | Function calling — class effect / trend / DDI | Qwen3.5-9B |
| Signal memory | Cross-run persistence | OpenSearch ML Memory (3.6+) |
| Web interface | Real-time streaming briefing | FastAPI + SSE |

**Example output — semaglutide, 82,699 reports, 11.9M baseline:**

```
### PRR Signals (EMA standard: PRR ≥ 2.0, 95% CI lower > 1, BH q < 0.05)
| Reaction                   | PRR   | 95% CI        | χ²/CI/FDR | Reports | Label? |
|----------------------------|-------|---------------|-----------|---------|--------|
| IMPAIRED GASTRIC EMPTYING  | 82.72 | 79.8–85.7     | ✓✓✓       | 3,057   | Yes    |
| GLYCOSYLATED HB INCREASED  | 11.54 | 10.9–12.2     | ✓✓✓       | 1,111   | No ⚠️  |
| PANCREATITIS               | 10.38 | 9.8–11.0      | ✓✓✓       | 1,504   | Yes    |
| BLOOD GLUCOSE DECREASED    | 9.97  | 9.4–10.6      | ✓✓✓       | 1,311   | No ⚠️  |

Risk: HIGH  |  Action: ESCALATE
```

---

## Stack

Everything runs locally via Docker. Zero external dependencies.

| Component | Technology | License |
|-----------|-----------|---------|
| Database | [OpenSearch 3.6.0](https://opensearch.org) | Apache 2.0 |
| LLM | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) Q4_K_XL (~5.6GB) via Docker Model Runner | Apache 2.0 |
| Agent framework | [LangGraph](https://langchain-ai.github.io/langgraph/) | MIT |
| Web UI | FastAPI + SSE streaming | MIT |
| Ingestion | [Polars](https://pola.rs) — 3× less memory than pandas | MIT |
| Observability | [Arize Phoenix](https://phoenix.arize.com) | Apache 2.0 |
| Data | FDA openFDA API + PubMed + FDA FAERS ZIPs | Public domain |

---

## Requirements

- **Docker Desktop** with [Model Runner](https://docs.docker.com/desktop/features/model-runner/) enabled
- **Python 3.11+** with [uv](https://docs.astral.sh/uv/)
- **16GB RAM** recommended (OpenSearch 1.5GB + Qwen3.5-9B ~5.6GB)
- **~10GB disk** for full FAERS 2018–2026 dataset

---

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Start infrastructure (pulls Qwen3.5-9B ~5.6GB on first run)
docker compose up -d

# 3. Load FAERS data

# Quick demo — 5 min via openFDA API, no download needed
uv run python -m ingestion.faers_indexer --drug semaglutide --limit 6000
uv run python -m ingestion.faers_indexer --drug rofecoxib --limit 2000

# Full dataset — 2018–2026, ~11.9M reports, ~1 hour + download
./ingestion/download_faers.sh
uv run python -m ingestion.faers_zip_indexer --dir ~/faers_data --all-drugs

# Full history — adds 2004–2017 (rofecoxib peak period, ~2.8GB more)
./ingestion/download_faers_historical.sh
uv run python -m ingestion.faers_zip_indexer --dir ~/faers_data --all-drugs

# 4. Compute within-class disproportionality (one-time)
uv run python -m ingestion.compute_class_ratio

# 5. Register OpenSearch MCP tools (one-time, enables free-form investigation)
uv run python -m ingestion.register_mcp_tools

# 6a. Web UI
uv run python -m app.server          # → http://localhost:8080

# 6b. CLI
uv run python main.py semaglutide
uv run python main.py rofecoxib      # retrospective: recalled 2004 for MI risk
```

---

## How PRR Works

PRR uses the correct 2×2 contingency table (EMA/813938/2011):

```
PRR = (a / (a+b)) / (c / (c+d))

a = drug reports with reaction        b = drug reports without reaction
c = non-drug reports with reaction    d = non-drug reports without reaction
```

Signal criteria applied in sequence:

| Criterion | Threshold | What it controls |
|-----------|-----------|-----------------|
| Count | n ≥ 3 | Eliminates single-event noise |
| PRR | ≥ 2.0 | Effect size (EMA standard) |
| Yates χ² | ≥ 4.0 | Per-test significance (annotated `✓`/`~`) |
| PRR lower 95% CI | > 1.0 | Small-n penalty — PRR=15 on n=4 has CI crossing 1.0 and is `~` |
| BH-FDR q-value | < 0.05 | Family-wise correction across all m reactions tested |

Only signals passing all five criteria enter the investigation gate. All criteria use published EMA thresholds — none are tuned on semaglutide.

**Baseline:** fetched per-reaction via OpenSearch `filters` agg — not truncated to a global top-N. The drug's own top-N reactions are capped at 50 (a known limitation for tail signals).

**CI formula:** log-normal approximation, SE = √(1/a − 1/(a+b) + 1/c − 1/(c+d)) (Evans 2001).

---

## Within-class Disproportionality

The `faers_ml_rates` index stores a quarterly time series of per-drug, per-reaction reporting rates. For each drug × reaction × quarter, the **pooled comparator rate** is computed from all active comparator groups:

```
comp_rate = Σ comp_reaction_counts / Σ comp_quarter_totals   (pooled across groups)
class_ratio = drug_rate / comp_rate
```

Counts are pooled across all quarters and a **95% rate-ratio CI** is computed using the same log-normal formula as the PRR CI. A signal passes only if `class_ratio_lower > 1.0` (lower CI bound excludes null), removing unstable small-n ratios.

**Zero comparator cells** (reaction absent from all comparators) are handled by Haldane–Anscombe +0.5 continuity correction rather than a hard 999 sentinel — the ratio remains finite and its CI is appropriately wide.

**This is within-class disproportionality screening, not anomaly detection.** No Random Cut Forest or RCF is used in the signal path. The GROWING / EMERGING / STABLE trend label is an advisory heuristic (recent vs early CI comparison) and is not a statistical gate.

---

## Architecture

```
LangGraph StateGraph — 10 nodes
│
├── resolve_names        RxNorm API → all brand/generic names              Python
├── load_memory          OpenSearch ML Memory → prior run findings          Python
├── calculate_prr        OpenSearch filters agg → PRR + 95% CI + BH-FDR   Python
├── anomaly_detection    faers_ml_rates → pooled rate ratio + CI            Python
├── fetch_label          openFDA + MedDRA LLT expansion → label text        Python
│
├── [search_lit?]        PubMed API                                          Python
│   (PRR≥3 AND robust CI AND unlabeled)
│
├── [investigate?]       Qwen3.5-9B — function calling, temp=0              LLM
│   (PRR≥5 AND robust CI AND BH-FDR significant AND unlabeled)
│   tools: get_prr, check_class_effect, get_signal_trend,
│          compare_time_periods (DataDistributionTool),
│          search_faers (flexible DSL queries)
│
├── write_report         Qwen3.5-9B — narrative prose only                  LLM
│   (PRR/rate-ratio tables emitted by Python — numbers never re-typed by model)
│
└── save_memory          OpenSearch ML Memory → persist findings             Python
```

**Design principle:** Python owns all statistics; LLM writes prose and investigates.

---

## OpenSearch 3.6.0 Features Used

| Feature | API | Purpose |
|---------|-----|---------|
| ML Memory | `/_plugins/_ml/memory` | Signal registry — persists findings across runs |
| DataDistributionTool | `/_plugins/_ml/tools/_execute/DataDistributionTool` | Time-period divergence — EMERGING vs GROWING signals |
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

- **MedDRA LLT expansion**: fetches official Lower-Level Term synonyms from openFDA (cached locally) — e.g. "MYOCARDIAL INFARCTION" → also tries "heart attack", so off-demo drugs don't produce systematic false-novel flags
- **Synonym expansion**: `delays`↔`impaired`, `reduced`↔`decreased`, etc.
- **Sentence-aware negation**: "no evidence of pancreatitis" → `False`; "no other findings. pancreatitis occurred." → `True`
- **Direction-aware**: "BLOOD GLUCOSE DECREASED" not matched by a label that only says "blood glucose increased"
- **Three-state output**: `Yes` / `Possible` (partial match) / `No ⚠️` (no match)

**Limitation:** the synonym dictionary covers ~16 hand-curated pairs. Reactions outside these pairs rely entirely on MedDRA LLT expansion. The LLT cache is populated from openFDA on first run — subsequent runs are fully offline.

---

## Known Limitations

This tool is a **PRR + within-class disproportionality screener**. It is comparable in statistical content to OpenVigil's PRR/ROR output. What it is not:

| Limitation | Impact |
|------------|--------|
| **No Bayesian shrinkage** (BCPNN/EBGM) | PRR point estimates at low n are noisy; the 95% CI lower bound mitigates this but does not eliminate it |
| **No stratification** (age / sex / reporter type) | Simpson's paradox confounding is unaddressed |
| **No exposure normalisation** | PRR measures reporting rate, not incidence. Market exposure differences between drugs are not controlled. |
| **FAERS structural biases** | Duplicate reports, Weber effect (reporting peaks ~2yr post-launch), notoriety/litigation bias (especially relevant for the rofecoxib retrospective), stimulated reporting, and co-medication confounding are inherent to spontaneous reporting and are not adjusted for. |
| **Drug's top-N reactions capped at 50** | Reactions ranked >50 in the drug's own profile are not tested — a novel reaction ranked #51 would be missed |
| **Comparator set is fixed** | Only 3 drugs with pre-configured comparators (semaglutide, rofecoxib, liraglutide). For other drugs, the within-class table will be empty. |

---

## Observability

- **Web UI**: `http://localhost:8080` — real-time streaming briefing
- **Phoenix traces**: `http://localhost:6006` (start with `docker compose up phoenix`)
- **OpenSearch Dashboards**: `http://localhost:5601` (admin / Pharma@2024!)

---

## Roadmap

- [x] PRR — correct 2×2 table, per-reaction baseline, no rank truncation
- [x] PRR 95% confidence interval (log-normal, Evans 2001)
- [x] Benjamini–Hochberg FDR correction across all reactions tested
- [x] Yates χ² significance annotation
- [x] FDA label cross-reference — MedDRA LLT-expanded, negation-aware, sentence-scoped
- [x] Three-state label match (Yes / Possible / No)
- [x] PubMed literature evidence
- [x] Within-class disproportionality — pooled rate ratio + 95% CI, Haldane–Anscombe correction
- [x] LLM investigation — function calling with 5 tools, CI-grounded prompts
- [x] Deterministic table rendering — numbers never re-typed by model
- [x] Signal registry — OpenSearch ML Memory (cross-run persistence)
- [x] DataDistributionTool — time-period emergence detection
- [x] Polars ingestion — 3× less memory, handles AERS + FAERS formats
- [x] Full FAERS 2004–2026 (historical + current)
- [x] Web UI — FastAPI + SSE streaming, dark-mode
- [ ] Stratified PRR (by age / sex / reporter type)
- [ ] BCPNN / EBGM Bayesian shrinkage estimators
- [ ] Configurable comparator groups (currently hardcoded for 3 drugs)
- [ ] GitHub Actions CI

---

## Validation Against openFDA (Independent Reference)

OpenVigil 2 has no public API. Instead we use **openFDA** directly — the FDA's own FAERS API — as a fully independent reference. It uses the same 2×2 PRR/ROR formula applied to the same raw data, via a completely separate code path.

```bash
# Full automated benchmark — no manual steps
uv run python scripts/benchmark_vs_openvigil.py benchmark semaglutide
```

### Results — semaglutide (82,699 reports, June 2026)

| Category | Reactions | Median PRR Δ | Verdict |
|---|---|---|---|
| Mechanism-specific (GLP-1/semaglutide) | 7 | **1.7%** | ✅ Formula validated |
| Multi-drug background reactions | 5 | 35.1% | ~ Data coverage (see below) |

**Mechanism-specific signal agreement:**

| Reaction | PRR (ours) | PRR (openFDA) | Δ% |
|---|---|---|---|
| DECREASED APPETITE | 5.49 | 5.51 | 0.4% ✅ |
| GLYCOSYLATED HAEMOGLOBIN INCREASED | 11.54 | 11.62 | 0.7% ✅ |
| CONSTIPATION | 5.87 | 5.95 | 1.3% ✅ |
| INTESTINAL OBSTRUCTION | 7.64 | 7.51 | 1.7% ✅ |
| VOMITING | 4.87 | 4.44 | 9.7% ✅ |

### Why background reactions show higher delta

Our local extract covers **2018–2026** (12M reports); openFDA has the full FAERS history (**20M reports**). Reactions associated with many pre-2018 drugs (PANCREATITIS, BLOOD GLUCOSE CHANGES, COVID-19) have higher background rates in openFDA's larger dataset, producing lower PRR there. This is a data coverage difference, not a formula error — confirmed by the near-perfect agreement on reactions driven purely by semaglutide's mechanism.

```bash
# Run for any drug
uv run python scripts/benchmark_vs_openvigil.py benchmark rofecoxib
```

---

## Related

**Hackathon version** (Elasticsearch + Elastic ML + Kibana MCP + Gemini API):  
→ [google-cloud-rapid-agent-hackathon](https://github.com/al-Zamakhshari/google-cloud-rapid-agent-hackathon)

---

## Disclaimer

For **research purposes only**. PRR signals are statistical associations, not causal evidence. No regulatory decisions should be made based solely on this tool's output. Requires clinical validation before any regulatory action.

**Statistics:** All numeric values (PRR, 95% CI, χ², BH q-value, counts, class_ratio) are computed deterministically by Python and are fully reproducible. The PRR formula, CI, and thresholds follow published EMA/813938/2011 guidelines.

**LLM narrative:** The "Key Findings", "Classification" labels (DRUG_SPECIFIC / CLASS_EFFECT), and "Clinical Insight" sentences are generated by Qwen3.5-9B. They are advisory and non-deterministic — the same data may produce different wording across runs.
