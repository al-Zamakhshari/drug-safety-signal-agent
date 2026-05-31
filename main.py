"""Entry point — analyze drug safety signals locally.

Uses LangGraph pipeline:
  resolve_names → calculate_prr → fetch_label → [search_literature?] → write_report

Python handles all data retrieval. LLM (Gemma4 E4B) only writes the final report.
"""

import asyncio
import sys
from dotenv import load_dotenv

load_dotenv()


async def run(drug_name: str):
    from agent.pipeline import pipeline, DrugSafetyState
    from agent.agent import setup_tracing, _tracing_active

    print(f"\nAnalyzing safety signals for: {drug_name}")
    print("─" * 60)

    initial_state: DrugSafetyState = {
        "drug_name":       drug_name,
        "drug_names":      [],
        "prr_signals":     [],
        "drug_total":      0,
        "faers_total":     0,
        "anomaly_signals": [],
        "label_text":      "",
        "literature":      [],
        "past_findings":   [],
        "investigation":   [],
        "briefing":        "",
        "error":           None,
    }

    final_state = await pipeline.ainvoke(initial_state)

    print("\n" + "─" * 60)
    print(final_state["briefing"])

    if _tracing_active:
        _tracing_active.force_flush()


def main():
    drug = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Drug name: ").strip()
    if not drug:
        print("Usage: uv run python main.py <drug_name>")
        sys.exit(1)
    asyncio.run(run(drug))


if __name__ == "__main__":
    main()
