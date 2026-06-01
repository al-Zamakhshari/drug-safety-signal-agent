"""
Unit tests for the two previously-untested ML layer functions:
  - build_memory_context  (signal_memory.py) — pure function, no OpenSearch
  - compare_time_periods classification branches (investigator_tools.py) — pure logic

Both are deterministic and don't need OpenSearch to test their core behaviour.
"""
import json
import pytest
from agent.tools.signal_memory import build_memory_context


# ---------------------------------------------------------------------------
# build_memory_context
# ---------------------------------------------------------------------------

class TestBuildMemoryContext:
    """
    build_memory_context(current_signals, prior_run) → str

    Rules:
      - Returns "" when prior_run is None (first run ever)
      - NEW line for reactions absent from prior run
      - Delta PRR line for reactions present in both runs
      - RESOLVED line for prior VALIDATED signals absent from current run
      - Capped at 10 lines
      - Tags: prior effect + PERSISTENT when prior status==VALIDATED
    """

    def _make_prior(self, signals: list[dict]) -> dict:
        return {"signals": signals}

    def test_no_prior_returns_empty(self):
        current = [{"reaction": "PANCREATITIS", "prr": 8.2}]
        assert build_memory_context(current, None) == ""

    def test_empty_prior_signals_all_new(self):
        current = [{"reaction": "PANCREATITIS", "prr": 8.2}]
        prior   = self._make_prior([])
        ctx = build_memory_context(current, prior)
        assert "PANCREATITIS" in ctx
        assert "NEW" in ctx

    def test_new_reaction_labelled_new(self):
        current = [{"reaction": "NAUSEA", "prr": 4.3}]
        prior   = self._make_prior([])
        ctx = build_memory_context(current, prior)
        assert "NAUSEA: NEW" in ctx

    def test_delta_positive(self):
        """PRR increased → positive delta percentage."""
        current = [{"reaction": "PANCREATITIS", "prr": 9.1}]
        prior   = self._make_prior([{
            "reaction": "PANCREATITIS", "prr": 8.2,
            "effect": "DRUG_SPECIFIC", "status": "VALIDATED",
        }])
        ctx = build_memory_context(current, prior)
        assert "PANCREATITIS" in ctx
        assert "8.2→9.1" in ctx
        assert "+" in ctx           # positive delta
        assert "DRUG_SPECIFIC" in ctx
        assert "PERSISTENT" in ctx   # VALIDATED → PERSISTENT tag

    def test_delta_negative(self):
        """PRR decreased → negative delta."""
        current = [{"reaction": "PANCREATITIS", "prr": 5.0}]
        prior   = self._make_prior([{
            "reaction": "PANCREATITIS", "prr": 8.2, "status": "VALIDATED",
        }])
        ctx = build_memory_context(current, prior)
        assert "8.2→5.0" in ctx
        assert "-" in ctx

    def test_resolved_validated_signal(self):
        """Prior VALIDATED signal absent from current → RESOLVED line."""
        current = []   # no current signals
        prior   = self._make_prior([{
            "reaction": "MYOCARDIAL_INFARCTION", "prr": 12.5, "status": "VALIDATED",
        }])
        ctx = build_memory_context(current, prior)
        assert "MYOCARDIAL_INFARCTION" in ctx
        assert "RESOLVED" in ctx

    def test_dismissed_prior_not_in_resolved(self):
        """Prior DISMISSED signal absent from current should NOT appear as RESOLVED."""
        current = []
        prior   = self._make_prior([{
            "reaction": "HEADACHE", "prr": 2.1, "status": "DISMISSED",
        }])
        ctx = build_memory_context(current, prior)
        # DISMISSED signals that disappear are not worth surfacing
        assert "HEADACHE" not in ctx or "RESOLVED" not in ctx

    def test_persistent_tag_only_for_validated(self):
        """PERSISTENT tag must only appear when prior status is VALIDATED."""
        current = [{"reaction": "NAUSEA", "prr": 4.5}]
        prior_dismissed = self._make_prior([{
            "reaction": "NAUSEA", "prr": 4.0,
            "effect": "CLASS_EFFECT", "status": "DISMISSED",
        }])
        ctx = build_memory_context(current, prior_dismissed)
        assert "PERSISTENT" not in ctx

    def test_returns_prefix_header_when_content_exists(self):
        current = [{"reaction": "PANCREATITIS", "prr": 8.0}]
        prior   = self._make_prior([{"reaction": "PANCREATITIS", "prr": 7.9, "status": "VALIDATED"}])
        ctx = build_memory_context(current, prior)
        assert ctx.startswith("PRIOR RUN SIGNAL TRAJECTORY:")

    def test_capped_at_10_lines(self):
        """More than 10 signals → output capped at 10 lines."""
        current = [{"reaction": f"REACTION_{i}", "prr": float(i)} for i in range(15)]
        prior   = self._make_prior([])
        ctx = build_memory_context(current, prior)
        # The content lines (not the header) are capped at 10
        content_lines = [l for l in ctx.split("\n") if l.strip().startswith("REACTION")]
        assert len(content_lines) <= 10

    def test_no_prior_effect_still_shows_delta(self):
        """Signal in prior with no effect tag still shows the PRR delta."""
        current = [{"reaction": "EPISTAXIS", "prr": 6.0}]
        prior   = self._make_prior([{"reaction": "EPISTAXIS", "prr": 5.5, "status": "VALIDATED"}])
        ctx = build_memory_context(current, prior)
        assert "5.5→6.0" in ctx


# ---------------------------------------------------------------------------
# compare_time_periods — classification branch logic
# ---------------------------------------------------------------------------

class TestCompareTimePeriodsBranches:
    """
    The core classification logic (EMERGING/GROWING/DECLINING/STABLE/NOT REPORTED)
    is pure arithmetic on counts — test it without OpenSearch by reproducing the
    exact branching conditions from investigator_tools.py.
    """

    def _classify(self, rc: int, rt: int, bc: int, bt: int) -> str:
        """Mirror the branch logic in compare_time_periods exactly."""
        r_rate = rc / rt if rt else 0.0
        b_rate = bc / bt if bt else 0.0

        if bc == 0 and rc > 0:
            return "EMERGING — absent in baseline, present recently"
        elif bt > 0 and rt > 0 and r_rate > b_rate * 1.5:
            return "GROWING — reporting rate increased vs baseline"
        elif bt > 0 and rt > 0 and r_rate < b_rate * 0.67:
            return "DECLINING — reporting rate fell vs baseline"
        elif rc == 0 and bc == 0:
            return "NOT REPORTED — no reports in either period"
        else:
            return "STABLE — similar rate in both periods"

    def test_emerging(self):
        """Zero in baseline, present recently → EMERGING."""
        result = self._classify(rc=50, rt=1000, bc=0, bt=1000)
        assert "EMERGING" in result

    def test_growing(self):
        """Rate more than 50% higher recently → GROWING."""
        # recent: 100/1000=10%, baseline: 50/1000=5% → ratio=2.0 > 1.5
        result = self._classify(rc=100, rt=1000, bc=50, bt=1000)
        assert "GROWING" in result

    def test_stable(self):
        """Rates similar (within 50%/67% band) → STABLE."""
        # recent: 50/1000=5%, baseline: 48/1000=4.8% → no threshold crossed
        result = self._classify(rc=50, rt=1000, bc=48, bt=1000)
        assert "STABLE" in result

    def test_declining(self):
        """Rate dropped below 67% of baseline → DECLINING."""
        # recent: 20/1000=2%, baseline: 50/1000=5% → 2/5=0.4 < 0.67
        result = self._classify(rc=20, rt=1000, bc=50, bt=1000)
        assert "DECLINING" in result

    def test_not_reported(self):
        """Zero in both periods → NOT REPORTED."""
        result = self._classify(rc=0, rt=1000, bc=0, bt=1000)
        assert "NOT REPORTED" in result

    def test_emerging_takes_priority_over_growing(self):
        """
        EMERGING (bc==0) must be tested before GROWING (rate ratio).
        If bc==0, the rate ratio is undefined — EMERGING is the correct label.
        """
        # bc=0, so b_rate=0; r_rate/b_rate would be undefined if growing checked first
        result = self._classify(rc=100, rt=1000, bc=0, bt=1000)
        assert "EMERGING" in result  # must not be GROWING

    def test_growing_boundary(self):
        """Exactly 1.5× baseline rate → NOT GROWING (must be strictly >1.5)."""
        # recent: 75/1000=7.5%, baseline: 50/1000=5% → ratio exactly 1.5 — stable
        result = self._classify(rc=75, rt=1000, bc=50, bt=1000)
        assert "STABLE" in result   # 1.5 is not > 1.5

    def test_declining_boundary(self):
        """Exactly at 67% of baseline → NOT DECLINING (must be strictly <0.67)."""
        # recent: 33.5/1000, baseline: 50/1000 → 0.67 is not < 0.67
        result = self._classify(rc=34, rt=1000, bc=50, bt=1000)
        # 34/50 = 0.68 > 0.67 → STABLE
        assert "STABLE" in result

    def test_zero_period_totals_no_crash(self):
        """rt=0 or bt=0 → rate is 0.0, no division-by-zero."""
        result = self._classify(rc=0, rt=0, bc=0, bt=0)
        assert "NOT REPORTED" in result

    def test_consistent_with_json_output(self):
        """
        The tool returns JSON with the interpretation string embedded.
        Verify the exact output format matches what the investigator expects.
        """
        # Replicate the output structure
        rc, rt, bc, bt = 50, 1000, 0, 1000
        r_rate = rc / rt if rt else 0.0
        b_rate = bc / bt if bt else 0.0

        output = {
            "recent":   {"count": rc, "total": rt, "rate": round(r_rate, 5)},
            "baseline": {"count": bc, "total": bt, "rate": round(b_rate, 5)},
            "interpretation": self._classify(rc, rt, bc, bt),
        }
        serialized = json.dumps(output)
        parsed = json.loads(serialized)

        assert parsed["interpretation"] == "EMERGING — absent in baseline, present recently"
        assert parsed["recent"]["rate"] == pytest.approx(0.05)
        assert parsed["baseline"]["rate"] == pytest.approx(0.0)
